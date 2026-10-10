import numpy as np

from .common import relative_error
from .equations import Equation, TERMS
from .pinn import tensor


def evaluate(field, equation, data, reference=None, true_equation=None, seed=210):
    rng = np.random.default_rng(seed)
    end, length = float(data["time_end"]), float(data["length"])
    x, t = rng.uniform(0, length, 1200), rng.uniform(0, end, 1200)
    u, ut, ux, uxx = field.derivatives(tensor(x), tensor(t, gradient=True), create_graph=False)
    arrays = [a.detach().numpy().ravel() for a in (u, ut, ux, uxx)]
    values, time_derivative, gradient, laplacian = arrays
    result = {"data_loss": float(np.mean((field.predict(data["x_train"], data["t_train"]) - data["u_train"])**2)) if len(data["u_train"]) else None,
              "validation_data_loss": float(np.mean((field.predict(data["x_valid"], data["t_valid"]) - data["u_valid"])**2)) if len(data["u_valid"]) else None,
              "pde_residual_mse": None, "balance_rms": None, "parameter_relative_error": None}
    times = np.linspace(0, end, 41)
    zero, right = np.zeros_like(times), np.full_like(times, length)
    left_values, _, left_gradient, _ = field.derivatives(tensor(zero), tensor(times, gradient=True), False)
    right_values, _, right_gradient, _ = field.derivatives(tensor(right), tensor(times, gradient=True), False)
    result["boundary_loss"] = float(((left_values - right_values)**2 + (length * (left_gradient - right_gradient))**2).mean().detach())
    if equation is not None:
        residual = time_derivative - equation.rhs(values, gradient, laplacian)
        result["pde_residual_mse"] = float(np.mean(residual**2))
        sx = np.linspace(0, length, 96, endpoint=False)
        xx, tt = np.meshgrid(sx, times)
        v, dt, _, _ = field.derivatives(tensor(xx), tensor(tt, gradient=True), False)
        v, dt = [a.detach().numpy().reshape(xx.shape) for a in (v, dt)]
        reaction = np.zeros_like(v)
        for term, coefficient in zip(equation.terms, equation.coefficients):
            if term in ("u", "u2", "u3", "1"):
                reaction += coefficient * (np.ones_like(v) if term == "1" else v**{"u": 1, "u2": 2, "u3": 3}[term])
        result["balance_rms"] = float(np.sqrt(np.mean((length * np.mean(dt - reaction, axis=1))**2)))
        result["dimensional_consistency"] = equation.admissibility()[0]
        if true_equation is not None:
            learned = dict(zip(equation.terms, equation.coefficients))
            truth = dict(zip(true_equation.terms, true_equation.coefficients))
            vector = np.array([learned.get(term, 0) for term in TERMS])
            reference_vector = np.array([truth.get(term, 0) for term in TERMS])
            result["parameter_relative_error"] = relative_error(vector, reference_vector)
            result["support_exact"] = set(equation.terms) == set(true_equation.terms)
            result["coefficient_errors"] = {term: abs(learned.get(term, 0) - c) / max(abs(c), 1e-12) for term, c in truth.items()}
    if reference is not None:
        xx, tt = np.meshgrid(reference["x"], reference["t"])
        prediction = field.predict(xx, tt)
        in_domain = reference["t"] <= end + 1e-10
        out_domain = ~in_domain
        result["prediction_relative_error"] = relative_error(prediction[in_domain], reference["u"][in_domain])
        result["ood_relative_error"] = relative_error(prediction[out_domain], reference["u"][out_domain]) if out_domain.any() else None
        result["initial_loss"] = float(np.mean((prediction[0] - reference["u"][0])**2))
        result["minimum_prediction"] = float(prediction.min())
    return result


def calibration_quantile(scores, alpha=0.1):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Calibration requires finite scores and a valid error probability.")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError("Too few calibration observations for a finite interval at this coverage.")
    return float(np.sort(scores)[rank - 1])


def uncertainty(fields, equations, data, reference, alpha=0.1):
    if len(fields) < 3:
        raise ValueError("At least three fitted ensemble members are required.")
    xx, tt = np.meshgrid(reference["x"], reference["t"])
    predictions = np.stack([f.predict(xx, tt) for f in fields])
    cal = np.stack([f.predict(data["x_cal"], data["t_cal"]) for f in fields])
    cal_scale = np.sqrt(cal.var(axis=0, ddof=1) + float(data["sigma"])**2)
    scores = np.abs(cal.mean(axis=0) - data["u_cal"]) / np.maximum(cal_scale, 1e-12)
    n = len(scores)
    multiplier = calibration_quantile(scores, alpha)
    mean = predictions.mean(axis=0)
    scale = np.sqrt(predictions.var(axis=0, ddof=1) + float(data["sigma"])**2)
    lower, upper = mean - multiplier * scale, mean + multiplier * scale
    covered = (reference["u"] >= lower) & (reference["u"] <= upper)
    noisy_reference = reference["u"] + np.random.default_rng(419).normal(0, float(data["sigma"]), reference["u"].shape)
    noisy_covered = (noisy_reference >= lower) & (noisy_reference <= upper)
    id_mask = reference["t"] <= float(data["time_end"]) + 1e-10
    values = np.array([e.coefficients for e in equations])
    result = {"method": "bootstrap ensemble with calibration on reserved noisy sensors",
              "members": len(fields), "nominal_coverage": 1 - alpha, "calibration_size": n,
              "calibration_multiplier": multiplier,
              "clean_reference_coverage_id": float(covered[id_mask].mean()),
              "clean_reference_coverage_ood": float(covered[~id_mask].mean()) if (~id_mask).any() else None,
              "independent_noisy_coverage_id": float(noisy_covered[id_mask].mean()),
              "independent_noisy_coverage_ood": float(noisy_covered[~id_mask].mean()) if (~id_mask).any() else None,
              "calibration_rank": int(np.ceil((n + 1) * (1 - alpha))),
              "mean_interval_width": float((upper - lower).mean()),
              "coefficient_mean": values.mean(axis=0).tolist(), "coefficient_std": values.std(axis=0, ddof=1).tolist(),
              "coefficient_percentile_interval": np.quantile(values, [0.05, 0.95], axis=0).tolist(),
              "interpretation": "Approximate predictive intervals. Clean-reference coverage is empirical; no OOD coverage guarantee."}
    return result, {"mean": mean, "lower": lower, "upper": upper, "members": predictions}


def assess_case(case, evidence_ids=()):
    kind = case["kind"]
    payload = case["input"]
    if kind == "equation":
        try:
            equation = Equation.from_dict(payload)
            valid, reason = equation.admissibility(payload.get("units"), payload.get("forward", True))
            return {"status": "accept" if valid else "reject", "reason": reason}
        except (ValueError, KeyError, TypeError):
            return {"status": "reject", "reason": "invalid or unsupported expression"}
    if kind == "coordinates":
        valid = payload["data_unit"] == payload["model_unit"] or payload.get("conversion_applied", False)
        return {"status": "accept" if valid else "reject", "reason": "coordinate conversion checked"}
    if kind == "identifiability":
        matrix = np.asarray(payload["design"], dtype=float)
        rank = np.linalg.matrix_rank(matrix)
        return {"status": "accept" if rank == matrix.shape[1] else "insufficient",
                "reason": f"design rank {rank} for {matrix.shape[1]} unknowns"}
    if kind == "boundary":
        if payload.get("inferred_from_interior_only"):
            return {"status": "insufficient", "reason": "boundary information is unavailable"}
        mismatch = abs(payload["initial_corner"] - payload["boundary_corner"])
        return {"status": "reject" if mismatch > payload.get("tolerance", 1e-8) else "accept",
                "reason": "initial-boundary compatibility checked"}
    if kind == "balance":
        expected = payload["inward_flux"] + payload["integrated_reaction"]
        valid = abs(payload["claimed_mass_rate"] - expected) <= payload.get("tolerance", 1e-8)
        return {"status": "accept" if valid else "reject", "reason": f"balance predicts mass rate {expected:g}"}
    if kind == "positivity":
        negative = min(payload["values"]) < 0
        valid = not negative or payload["quantity"] != "concentration"
        return {"status": "accept" if valid else "reject", "reason": "positivity applied only to nonnegative quantities"}
    if kind == "resolution":
        valid = payload["sensors"] >= 2 * payload["spatial_modes"] + 1
        return {"status": "accept" if valid else "insufficient", "reason": "spatial sampling rank checked"}
    if kind == "noise":
        sigma = np.asarray(payload["sigma"])
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            return {"status": "reject", "reason": "invalid noise standard deviation"}
        return {"status": "qualify" if np.ptp(sigma) > 0 else "accept", "reason": "heterogeneous noise needs weighted fitting"}
    if kind == "citation":
        valid = set(payload["citations"]).issubset(set(evidence_ids))
        return {"status": "accept" if valid else "reject", "reason": "source IDs resolved against evidence"}
    if kind == "assumptions":
        contradictory = bool(set(payload["required"]) & set(payload["excluded"]))
        return {"status": "reject" if contradictory else "accept", "reason": "source assumptions checked"}
    if kind == "ood":
        status = "accept" if payload["independently_validated"] else "qualify"
        return {"status": status, "reason": "extrapolation requires independent validation"}
    raise ValueError(f"Unknown adversarial case kind: {kind}")
