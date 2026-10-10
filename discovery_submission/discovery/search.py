from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear

from .equations import Equation, TERMS, supports
from .simulation import solve


def spatial_features(x, modes, length=1.0, order=0):
    x = np.asarray(x).reshape(-1)
    columns = [np.ones_like(x) if order == 0 else np.zeros_like(x)]
    for k in range(1, modes + 1):
        frequency = 2 * np.pi * k / length
        angle = frequency * x + order * np.pi / 2
        columns.extend([frequency**order * np.cos(angle), frequency**order * np.sin(angle)])
    return np.column_stack(columns)


@dataclass
class ObservationFit:
    coefficients: np.ndarray
    modes: int
    degree: int
    end: float
    length: float

    def predict(self, x, t):
        x, t = np.broadcast_arrays(x, t)
        spatial = spatial_features(x.ravel(), self.modes, self.length)
        temporal = chebvander(2 * t.ravel() / self.end - 1, self.degree)
        return np.einsum("ni,ij,nj->n", spatial, self.coefficients, temporal).reshape(x.shape)


def validate_observations(data):
    required = {f"{q}_{s}" for q in ("x", "t", "u") for s in ("train", "valid")} | {"sigma", "length", "time_end"}
    if not required.issubset(data):
        raise ValueError("Observation file is missing required arrays or metadata.")
    calibration = {f"{q}_cal" for q in ("x", "t", "u")}
    if calibration & set(data) and not calibration.issubset(data):
        raise ValueError("Calibration arrays must be supplied together.")
    splits = ("train", "valid", "cal") if calibration.issubset(data) else ("train", "valid")
    for split in splits:
        arrays = [np.asarray(data[f"{q}_{split}"]) for q in ("x", "t", "u")]
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1 or not np.isfinite(np.concatenate(arrays)).all():
            raise ValueError("Observation arrays must be aligned finite one-dimensional arrays.")
    if len(data["u_train"]) < 30 or len(data["u_valid"]) < 8:
        raise ValueError("Insufficient training or selection observations.")
    if any(np.asarray(data[k]).ndim != 0 for k in ("sigma", "length", "time_end")):
        raise ValueError("Noise and domain metadata must be scalars.")
    metadata = np.array([float(data[k]) for k in ("sigma", "length", "time_end")])
    if not np.isfinite(metadata).all() or metadata[0] < 0 or np.any(metadata[1:] <= 0):
        raise ValueError("Invalid noise level or physical domain scales.")
    coordinates = []
    for split in splits:
        x, t = np.asarray(data[f"x_{split}"]), np.asarray(data[f"t_{split}"])
        if np.any(x < 0) or np.any(x >= metadata[1]) or np.any(t < 0) or np.any(t > metadata[2] + 1e-10):
            raise ValueError("Observation coordinates exceed the declared physical domain.")
        coordinates.append(set(zip(x, t)))
    if any(left & right for i, left in enumerate(coordinates) for right in coordinates[i + 1:]):
        raise ValueError("Training, selection and calibration coordinates must be disjoint.")
    if len(np.unique(data["t_train"])) < 4:
        raise ValueError("Training measurements must span at least four distinct times.")


def fit_observations(data, modes=4, degree=5, ridge=1e-5):
    validate_observations(data)
    x, t, u = [np.asarray(data[f"{k}_train"]) for k in ("x", "t", "u")]
    end, length = float(data["time_end"]), float(data["length"])
    if np.var(u) <= max(3 * float(data["sigma"])**2, 1e-12):
        raise ValueError("Observations do not excite identifiable dynamics above the noise level.")
    modes = min(modes, max((len(np.unique(x)) - 1) // 2, 0))
    spatial = spatial_features(x, modes, length)
    temporal = chebvander(2 * t / end - 1, degree)
    matrix = np.einsum("ni,nj->nij", spatial, temporal).reshape(len(u), -1)
    penalty = ridge * np.eye(matrix.shape[1])
    coefficients = lstsq(np.vstack([matrix, np.sqrt(penalty)]),
                         np.r_[u, np.zeros(matrix.shape[1])], lapack_driver="gelsy")[0]
    return ObservationFit(coefficients.reshape(spatial.shape[1], degree + 1), modes, degree, end, length)


def bump(z, order=0):
    inside = np.abs(z) < 1
    if order == 0:
        value = (1 - z**2)**4
    elif order == 1:
        value = -8 * z * (1 - z**2)**3
    elif order == 2:
        value = -8 * (1 - z**2)**3 + 48 * z**2 * (1 - z**2)**2
    else:
        raise ValueError("Bump derivative order must be zero, one or two.")
    return np.where(inside, value, 0)


def weak_system(fit, nx=100, nt=85):
    x = np.linspace(0, fit.length, nx)
    t = np.linspace(0, fit.end, nt)
    xx, tt = np.meshgrid(x, t)
    u = fit.predict(xx, tt)
    wx, wt = fit.length * 0.22, fit.end * 0.24
    matrix, target, blocks = [], [], []

    def integral(value):
        return np.trapezoid(np.trapezoid(value, x=x, axis=1), x=t)

    for bi, ct in enumerate(np.linspace(wt, fit.end - wt, 7)):
        for cx in np.linspace(wx, fit.length - wx, 9):
            zx, zt = (xx - cx) / wx, (tt - ct) / wt
            phi = bump(zx) * bump(zt)
            phit = bump(zx) * bump(zt, 1) / wt
            phix = bump(zx, 1) * bump(zt) / wx
            phixx = bump(zx, 2) * bump(zt) / wx**2
            columns = [integral(u * phixx), -integral(u * phix), integral(u * phi),
                       integral(u**2 * phi), integral(u**3 * phi),
                       -0.5 * integral(u**2 * phix), integral(phi)]
            matrix.append(columns)
            target.append(-integral(u * phit))
            blocks.append(bi)
    return np.asarray(matrix), np.asarray(target), np.asarray(blocks)


def fit_support(matrix, target, support, constrained=True):
    indices = [TERMS.index(t) for t in support]
    design = matrix[:, indices]
    scales = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    lower = np.full(len(indices), -np.inf)
    if constrained and "u_xx" in support:
        lower[support.index("u_xx")] = 0
    result = lsq_linear(design / scales, target, bounds=(lower, np.full(len(indices), np.inf)),
                        tol=1e-10, lsmr_tol=1e-10)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Constrained weak regression did not converge.")
    coefficients = result.x / scales
    singular = np.linalg.svd(design / scales, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return Equation(tuple(support), tuple(coefficients)), condition


def regression_uncertainty(matrix, target, support, repeats=24):
    rng = np.random.default_rng(83)
    indices = [TERMS.index(t) for t in support]
    predictions = []
    for _ in range(repeats):
        rows = rng.choice(len(target), len(target), replace=True)
        equation, _ = fit_support(matrix[rows], target[rows], support)
        predictions.append(matrix[:, indices] @ equation.coefficients)
    return float(np.mean(np.var(predictions, axis=0, ddof=1)) / max(np.mean(target**2), 1e-14))


def discover(data, proposals=(), maximum=4, complexity=None, use_constraints=True, terms=TERMS, modes=4):
    fit = fit_observations(data, modes=modes)
    matrix, target, blocks = weak_system(fit)
    target_energy = max(np.mean(target**2), 1e-14)
    if complexity is None:
        n = len(data["u_valid"])
        complexity = max(float(data["sigma"])**2 / max(np.mean(data["u_valid"]**2), 1e-12), 1e-8) * np.log(n) / n
    candidates = []
    proposed = {tuple(p["terms"]) for p in proposals}
    grammar = supports(terms=terms, maximum=maximum)
    if proposals:
        frontier = {tuple(t for t in TERMS if t in p["terms"]) for p in proposals}
        generated = set(frontier)
        for _ in range(2):
            next_frontier = set()
            for support in frontier:
                for term in terms:
                    changed = set(support) ^ {term}
                    if 1 <= len(changed) <= maximum:
                        next_frontier.add(tuple(t for t in TERMS if t in changed))
            generated.update(next_frontier)
            frontier = next_frontier
        grammar = [s for s in grammar if s in generated]
    for support in grammar:
        equation, condition = fit_support(matrix, target, support, use_constraints)
        if use_constraints and not equation.admissibility()[0]:
            continue
        fold_losses = []
        for fold in (0, 1, 2):
            valid = blocks % 3 == fold
            train_eq, _ = fit_support(matrix[~valid], target[~valid], support, use_constraints)
            prediction = matrix[valid][:, [TERMS.index(t) for t in support]] @ train_eq.coefficients
            fold_losses.append(float(np.mean((prediction - target[valid])**2) / target_energy))
        weak_loss = float(np.mean(fold_losses))
        candidates.append({**equation.as_dict(), "weak_cv": weak_loss, "condition": condition,
                           "complexity": len(support), "proposed_by_llm": support in proposed,
                           "screen_score": weak_loss + complexity * len(support)})
    candidates.sort(key=lambda c: c["screen_score"])
    shortlist = candidates[:16]
    for c in shortlist:
        equation = Equation.from_dict(c)
        try:
            x, trajectory = solve(equation, lambda x: fit.predict(x, np.zeros_like(x)),
                                  np.linspace(0, fit.end, 101), points=96, length=fit.length)
            from scipy.interpolate import RegularGridInterpolator
            extended_x = np.r_[x, fit.length]
            extended = np.column_stack([trajectory, trajectory[:, 0]])
            interpolator = RegularGridInterpolator((np.linspace(0, fit.end, 101), extended_x), extended)
            prediction = interpolator(np.column_stack([data["t_valid"], data["x_valid"]]))
            c["validation_mse"] = float(np.mean((prediction - data["u_valid"])**2))
            c["score"] = c["validation_mse"] / max(np.mean(data["u_valid"]**2), 1e-12) + complexity * c["complexity"]
            c["status"] = "shortlisted"
        except (RuntimeError, ValueError, FloatingPointError) as error:
            c["score"] = 1e6
            c["status"] = "rejected"
            c["reason"] = str(error)
    shortlist.sort(key=lambda c: c["score"])
    if not shortlist or shortlist[0]["score"] >= 1e6:
        raise RuntimeError("No candidate passed independent rollout validation.")
    return fit, shortlist, {"evaluated_supports": len(candidates), "weak_rows": len(target),
                            "search_terms": list(terms), "matrix": matrix, "target": target,
                            "generation": "two rounds of evidence-seeded term addition/removal" if proposals else "bounded grammar enumeration"}
