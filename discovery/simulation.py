import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

from .equations import Equation


def initial_profile(x, variant=0):
    if variant == 0:
        return 0.50 + 0.26 * np.cos(2 * np.pi * x) + 0.12 * np.sin(4 * np.pi * x)
    if variant == 1:
        return 0.38 + 0.17 * np.sin(2 * np.pi * x) - 0.13 * np.cos(4 * np.pi * x)
    raise ValueError("Unknown initial-profile variant.")


def solve(equation, initial, times, points=128, length=1.0, method="finite_difference"):
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Solver times must start at zero and increase strictly.")
    if points < 16 or length <= 0 or not equation.admissibility()[0]:
        raise ValueError("Unsupported or inadmissible forward problem.")
    x = np.arange(points) * length / points
    y0 = np.asarray(initial(x) if callable(initial) else initial, dtype=float)
    if y0.shape != (points,) or not np.isfinite(y0).all():
        raise ValueError("Initial profile must match the numerical grid.")
    if method == "spectral":
        k = 2 * np.pi * np.fft.fftfreq(points, d=length / points)

        def rhs(t, y):
            spectrum = np.fft.fft(y)
            ux = np.fft.ifft(1j * k * spectrum).real
            uxx = np.fft.ifft(-k**2 * spectrum).real
            return equation.rhs(y, ux, uxx)

        solution = solve_ivp(rhs, (0, times[-1]), y0, t_eval=times,
                             method="BDF", rtol=1e-9, atol=1e-11)
    elif method == "finite_difference":
        dx = length / points
        sparsity = diags([np.ones(points - 1), np.ones(points), np.ones(points - 1)], [-1, 0, 1]).tolil()
        sparsity[0, -1] = sparsity[-1, 0] = 1

        def rhs(t, y):
            ux = (np.roll(y, -1) - np.roll(y, 1)) / (2 * dx)
            uxx = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / dx**2
            return equation.rhs(y, ux, uxx)

        solution = solve_ivp(rhs, (0, times[-1]), y0, t_eval=times, method="BDF",
                             jac_sparsity=sparsity.tocsr(), rtol=1e-8, atol=1e-10)
    else:
        raise ValueError("Unknown solver method.")
    if not solution.success or not np.isfinite(solution.y).all() or np.max(np.abs(solution.y)) > 100:
        raise RuntimeError("Numerical integration failed or diverged.")
    return x, solution.y.T


def make_observations(x, times, values, sensors=16, snapshots=25, noise=0.01, seed=11, end=0.8):
    rng = np.random.default_rng(seed)
    xi = np.arange(sensors) * len(x) // sensors
    ti = np.unique(np.linspace(0, np.searchsorted(times, end, side="right") - 1, snapshots).astype(int))
    clean = values[np.ix_(ti, xi)]
    sigma = noise * np.sqrt(np.mean(clean**2))
    observed = clean + rng.normal(0, sigma, clean.shape)
    mask = np.ones(clean.shape, dtype=bool)
    mask[:, np.arange(sensors) % 4 == 3] = False
    tx, xx = np.meshgrid(times[ti], x[xi], indexing="ij")
    selection = (~mask) & (tx <= end * 0.75)
    calibration = (~mask) & (tx > end * 0.75)
    return {"x_train": xx[mask], "t_train": tx[mask], "u_train": observed[mask],
            "x_valid": xx[selection], "t_valid": tx[selection], "u_valid": observed[selection],
            "x_cal": xx[calibration], "t_cal": tx[calibration], "u_cal": observed[calibration],
            "sigma": np.array(sigma), "length": np.array(1.0), "time_end": np.array(end)}
