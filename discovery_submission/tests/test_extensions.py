from pathlib import Path

import numpy as np
import pytest

from discovery.diagnostics import identifiability, independent_refit, propose_measurements
from discovery.equations import Equation
from discovery.evidence import Evidence, ModelCache
from discovery.experiments import load_arrays
from discovery.pinn import Coefficients
from discovery.search import discover, fit_observations, validate_observations
from discovery.simulation import make_observations, solve
from discovery.validation import calibration_quantile


ROOT = Path(__file__).resolve().parents[1]


def test_identification_respects_nonunit_domain():
    length = 2.5
    truth = Equation(("u_xx", "u_x"), (0.08, -0.25))
    x, values = solve(truth, lambda x: .5 + .2 * np.cos(2 * np.pi * x / length) + .1 * np.sin(4 * np.pi * x / length),
                      np.linspace(0, 1.2, 121), length=length, method="spectral")
    data = make_observations(x, np.linspace(0, 1.2, 121), values, noise=0)
    assert float(data["length"]) == length
    _, candidates, _ = discover(data, terms=("u_xx", "u_x"))
    assert candidates[0]["terms"] == ["u_xx", "u_x"]
    np.testing.assert_allclose(candidates[0]["coefficients"], truth.coefficients, rtol=.04)
    assert candidates[0]["validation_mse"] < 1e-6


def test_calibration_uses_the_exact_finite_sample_order_statistic():
    assert calibration_quantile(np.arange(1, 25), .1) == 23
    with pytest.raises(ValueError, match="Too few"):
        calibration_quantile([1, 2], .1)
    with pytest.raises(ValueError, match="finite scores"):
        calibration_quantile([1, np.nan], .5)


def test_identifiability_detects_indistinguishable_terms():
    matrix = np.ones((20, 7))
    diagnostic = identifiability(matrix, ("u", "u2"))
    assert diagnostic["rank"] == 1
    assert diagnostic["status"] == "rank_deficient"


def test_fixed_zero_diffusivity_is_preserved_and_large_values_are_stable():
    zero = Equation(("u_xx", "u_x"), (0.0, -.2))
    assert Coefficients(zero, trainable=False).equation() == zero
    large = Coefficients(Equation(("u_xx",), (1000.0,)))
    assert np.isfinite(large.equation().coefficients).all()
    assert large.equation().coefficients[0] == pytest.approx(1000)


@pytest.mark.parametrize("fault", ["overlap", "partial_calibration", "nonfinite_calibration", "nonscalar", "outside_domain"])
def test_invalid_observations_are_rejected_before_inference(fault):
    data = load_arrays(ROOT / "data" / "observations.npz")
    if fault == "overlap":
        data["x_valid"][0], data["t_valid"][0] = data["x_train"][0], data["t_train"][0]
    elif fault == "partial_calibration":
        del data["u_cal"]
    elif fault == "nonfinite_calibration":
        data["u_cal"][0] = np.nan
    elif fault == "nonscalar":
        data["length"] = np.array([1, 2])
    else:
        data["t_valid"][0] = 2 * data["time_end"]
    with pytest.raises(ValueError):
        validate_observations(data)


def test_offline_cache_never_starts_uncached_inference(tmp_path):
    cache = ModelCache(tmp_path / "cache.json", offline=True)
    with pytest.raises(RuntimeError, match="not cached"):
        cache.embed("uncached question")
    with pytest.raises(RuntimeError, match="not cached"):
        cache.generate([{"role": "user", "content": "uncached question"}])
    with pytest.raises(RuntimeError, match="uncached requests"):
        cache.generate_many([[{"role": "user", "content": "uncached question"}]])


def test_hypothesis_support_paths_resolve_to_graph_edges():
    evidence = Evidence(ROOT / "data" / "evidence.jsonl", ModelCache(ROOT / "data" / "model_cache.json", offline=True))
    ids, paths = evidence.ground_terms(("u_xx", "u_x", "u", "u2"))
    assert len({evidence.by_id[i]["source"] for i in ids}) >= 2
    edges = {(edge["from"], edge["to"]) for edge in evidence.graph["edges"]}
    for path in paths:
        anchor, concept, bridge = path["path"]
        assert (anchor, concept) in edges and (bridge, concept) in edges
        assert path["term"] in evidence.by_id[anchor]["concepts"]
        assert evidence.by_id[anchor]["source"] != evidence.by_id[bridge]["source"]


def test_independent_solver_rejects_mass_conserving_law_for_reaction_data():
    data = load_arrays(ROOT / "data" / "observations.npz")
    fit = fit_observations(data)
    check = independent_refit(data, Equation(("u_xx",), (.02,)), fit)
    assert check["converged"]
    assert check["validation_mse"] > 3 * float(data["sigma"])**2


def test_measurement_proposals_use_future_disagreement_without_reference():
    class Field:
        def predict(self, x, t):
            return .5 + .2 * np.cos(2 * np.pi * x / 2)

    data = {"length": np.array(2.0), "time_end": np.array(.8), "sigma": np.array(.01)}
    trials = [{"status": "competitive", "equation": Equation(("u_xx",), (value,)).as_dict()} for value in (.01, .1)]
    report, arrays = propose_measurements(data, Field(), trials)
    assert report["status"] == "proposed"
    assert len(report["probes"]) == 5
    assert np.isfinite(arrays["predictions"]).all()
    for probe in report["probes"]:
        assert .8 + 1e-10 < probe["t"] <= 1.2 + 1e-10
        assert 0 <= probe["x"] < 2
        assert probe["prediction_std"] > 0


def test_solver_rejects_invalid_time_grid():
    with pytest.raises(ValueError, match="increase strictly"):
        solve(Equation(("u_xx",), (.02,)), np.ones(16), [0, np.nan], points=16)
