import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from .equations import Equation, TERMS
from .simulation import solve


def identifiability(matrix, terms):
    design = matrix[:, [TERMS.index(term) for term in terms]]
    normalized = design / np.maximum(np.linalg.norm(design, axis=0), 1e-14)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-8, 1e-12)))
    condition = float(singular[0] / max(singular[-1], 1e-14))
    overlap = np.abs(normalized.T @ normalized)
    np.fill_diagonal(overlap, 0)
    return {"rank": rank, "unknowns": len(terms), "scaled_condition_number": condition,
            "maximum_term_overlap": float(overlap.max()), "singular_values": singular.tolist(),
            "status": "rank_deficient" if rank < len(terms) else ("ill_conditioned" if condition > 1e4 else "full_rank"),
            "interpretation": "Local weak-design diagnostic; full rank does not establish unique structure."}


def sample_trajectory(x, times, values, query_x, query_t, length):
    extended = np.column_stack((values, values[:, 0]))
    interpolator = RegularGridInterpolator((times, np.r_[x, length]), extended)
    return interpolator(np.column_stack((query_t, query_x)))


def independent_refit(data, equation, fit, max_evaluations=40):
    times = np.unique(np.r_[0, data["t_train"], data["t_valid"], fit.end])
    scale = max(float(np.sqrt(np.mean(data["u_train"]**2))), 1e-8)
    lower = np.full(len(equation.terms), -np.inf)
    if "u_xx" in equation.terms:
        lower[equation.terms.index("u_xx")] = 0

    def trajectory(coefficients):
        candidate = Equation(equation.terms, tuple(coefficients))
        return solve(candidate, lambda x: fit.predict(x, np.zeros_like(x)), times, points=64, length=fit.length)

    def residual(coefficients):
        try:
            x, values = trajectory(coefficients)
            prediction = sample_trajectory(x, times, values, data["x_train"], data["t_train"], fit.length)
            return (prediction - data["u_train"]) / scale
        except (ValueError, RuntimeError, FloatingPointError):
            return np.full(len(data["u_train"]), 1e6)

    result = least_squares(residual, equation.coefficients, bounds=(lower, np.full(len(lower), np.inf)),
                           x_scale=np.maximum(np.abs(equation.coefficients), 0.01), max_nfev=max_evaluations,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    x, values = trajectory(result.x)
    predicted = sample_trajectory(x, times, values, data["x_valid"], data["t_valid"], fit.length)
    return {"converged": bool(result.success), "evaluations": result.nfev,
            "equation": Equation(equation.terms, tuple(result.x)).as_dict(),
            "training_mse": float(np.mean(residual(result.x)**2) * scale**2),
            "validation_mse": float(np.mean((predicted - data["u_valid"])**2)),
            "method": "Independent finite-difference coefficient fit using training sensors and reconstructed initial values."}


def propose_measurements(data, field, trials, count=5):
    length, end = float(data["length"]), float(data["time_end"])
    times = np.linspace(0, 1.5 * end, 91)
    predictions, equations, failures = [], [], []
    for trial in trials:
        if trial["status"] not in ("selected", "competitive"):
            continue
        equation = Equation.from_dict(trial["equation"])
        try:
            x, values = solve(equation, lambda x: field.predict(x, np.zeros_like(x)), times, points=96, length=length)
            predictions.append(values)
            equations.append(equation.as_dict())
        except (RuntimeError, ValueError) as error:
            failures.append({"equation": equation.as_dict(), "reason": str(error)})
    if len(predictions) < 2:
        return {"status": "insufficient_alternatives", "probes": [], "models": equations, "failed_rollouts": failures}, None
    predictions = np.stack(predictions)
    spread = predictions.std(axis=0)
    available = np.broadcast_to(times[:, None] > end + 1e-10 * max(end, 1), spread.shape).copy()
    probes = []
    for _ in range(count):
        candidates = np.where(available, spread, -1)
        i, j = np.unravel_index(np.argmax(candidates), candidates.shape)
        if candidates[i, j] <= 1e-12:
            break
        probes.append({"x": float(x[j]), "t": float(times[i]), "prediction_std": float(spread[i, j]),
                       "model_predictions": predictions[:, i, j].tolist(),
                       "disagreement_to_noise": float(spread[i, j] / max(float(data["sigma"]), 1e-8))})
        dx = np.minimum(np.abs(x - x[j]), length - np.abs(x - x[j])) / length
        dt = np.abs(times - times[i]) / end
        available &= (dt[:, None] / 0.12)**2 + (dx[None, :] / 0.12)**2 > 1
    return {"status": "proposed", "probes": probes, "models": equations, "failed_rollouts": failures,
            "method": "Space-time separated maxima of disagreement between independently integrated competitive laws.",
            "interpretation": "Suggested measurements, not observed data or a guarantee of information gain."}, {
                "x": x, "t": times, "predictions": predictions, "disagreement": spread}
