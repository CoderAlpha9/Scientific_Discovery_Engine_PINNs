from pathlib import Path

import numpy as np
import pytest
import torch

from discovery.equations import Equation, coefficient_dimension, canonical_support
from discovery.experiments import load_arrays
from discovery.pinn import Field, tensor, save_model, load_model
from discovery.search import bump, fit_support, weak_system, validate_observations
from discovery.simulation import solve


ROOT = Path(__file__).resolve().parents[1]


def test_units_and_forward_admissibility():
    diffusion = Equation(("u_xx",), (0.02,))
    assert coefficient_dimension("u_xx") == (0, 2, -1)
    assert coefficient_dimension("u2") == (-1, 0, -1)
    assert not diffusion.admissibility({"u_xx": (0, 1, -1)})[0]
    assert not Equation(("u_xx",), (-0.02,)).admissibility()[0]
    assert Equation(("u_x",), (-0.2,)).admissibility()[0]
    with pytest.raises(ValueError):
        canonical_support(["__import__('os')"])
    with pytest.raises(ValueError):
        Equation(("u", "u"), (1, 2))


def test_field_derivatives_against_independent_autograd():
    torch.manual_seed(14)
    model = Field(modes=3)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.1)
    x = tensor(np.linspace(0.1, 0.9, 12), gradient=True)
    t = tensor(np.linspace(0.05, 0.7, 12), gradient=True)
    u = model(x, t)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, retain_graph=True)[0]
    ut = torch.autograd.grad(u.sum(), t, retain_graph=True)[0]
    actual = model.derivatives(x, t)
    np.testing.assert_allclose(actual[1].detach(), ut.detach(), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[2].detach(), ux.detach(), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual[3].detach(), uxx.detach(), rtol=1e-11, atol=1e-11)


def test_compact_test_functions_have_zero_boundary_derivatives():
    for order in (0, 1, 2):
        np.testing.assert_array_equal(bump(np.array([-1, 1, -2, 2]), order), np.zeros(4))


def test_weak_integration_signs_on_manufactured_solution():
    class Manufactured:
        length = 1.0
        end = 0.8

        def predict(self, x, t):
            value = np.zeros_like(x)
            for k, amplitude in ((1, 0.6), (2, 0.2)):
                value += amplitude * np.exp((0.2 - 0.02 * (2 * np.pi * k)**2) * t) * np.cos(2 * np.pi * k * (x - 0.1 * t))
            return value

    matrix, target, _ = weak_system(Manufactured(), nx=180, nt=160)
    equation, _ = fit_support(matrix, target, ("u_xx", "u_x", "u"))
    np.testing.assert_allclose(equation.coefficients, [0.02, -0.1, 0.2], rtol=0.002, atol=1e-5)


def test_numerical_solver_against_analytical_heat_solution():
    equation = Equation(("u_xx",), (0.03,))
    times = np.linspace(0, 1, 25)
    x, u = solve(equation, lambda x: 0.4 + 0.2 * np.cos(2 * np.pi * x), times, points=128)
    expected = 0.4 + 0.2 * np.exp(-0.03 * (2 * np.pi)**2 * times[:, None]) * np.cos(2 * np.pi * x)
    assert np.max(np.abs(u - expected)) < 2e-5
    np.testing.assert_allclose(u.mean(axis=1), 0.4, atol=1e-10)


def test_observation_splits_are_disjoint_and_truth_is_private():
    data = load_arrays(ROOT / "data" / "observations.npz")
    sets = [set(zip(data[f"x_{split}"], data[f"t_{split}"])) for split in ("train", "valid", "cal")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert not {"terms", "coefficients", "reference", "truth"} & set(data)
    assert data["t_train"].max() <= data["time_end"]
    assert data["t_valid"].max() < data["t_cal"].min()


def test_checkpoint_roundtrip(tmp_path):
    model = Field(modes=2, time_end=0.8)
    equation = Equation(("u_xx", "u_x"), (0.03, -0.2))
    path = tmp_path / "model.pt"
    save_model(path, model, equation)
    recovered, recovered_equation = load_model(path)
    x, t = np.array([0.2, 0.6]), np.array([0.3, 0.7])
    np.testing.assert_array_equal(model.predict(x, t), recovered.predict(x, t))
    assert recovered_equation == equation


def test_invalid_observation_contract_fails_clearly():
    with pytest.raises(ValueError, match="missing required"):
        validate_observations({"u_train": np.ones(40)})
