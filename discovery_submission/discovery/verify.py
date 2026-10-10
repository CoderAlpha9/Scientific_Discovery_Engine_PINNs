from pathlib import Path

import numpy as np

from .common import digest, read_json, read_jsonl, relative_error
from .equations import Equation
from .experiments import coefficient_error, load_arrays
from .pinn import load_model
from .search import validate_observations


def verify_study(root):
    root = Path(root)
    results = root / "results"
    data = load_arrays(root / "data" / "observations.npz")
    validate_observations(data)
    if {"terms", "coefficients", "truth", "reference"} & set(data):
        raise RuntimeError("Ground truth leaked into discovery observations.")
    equation = Equation.from_dict(read_json(results / "equation.json"))
    commitment = read_json(results / "prediction_commitment.json")
    if digest(results / "frozen_prediction.npz") != commitment["prediction_sha256"]:
        raise RuntimeError("Prediction commitment does not match saved predictions.")
    if Equation.from_dict(commitment["equation"]) != equation:
        raise RuntimeError("Committed equation differs from the selected equation.")
    model, fitted = load_model(results / "models" / "inverse.pt")
    if fitted != equation:
        raise RuntimeError("Checkpoint coefficients differ from the selected equation.")
    prediction = load_arrays(results / "frozen_prediction.npz")
    xx, tt = np.meshgrid(prediction["x"], prediction["t"])
    difference = float(np.max(np.abs(model.predict(xx, tt) - prediction["u"])))
    if not np.isfinite(difference) or difference > 1e-10:
        raise RuntimeError("Reloaded model does not reproduce committed predictions.")
    reference = load_arrays(root / "data" / "evaluation" / "reference.npz")
    truth = Equation(tuple(reference["terms"]), tuple(reference["coefficients"]))
    metrics = read_json(results / "metrics.json")["inverse"]
    observed = prediction["t"] <= float(data["time_end"]) + 1e-12
    actual = {"prediction_relative_error": relative_error(prediction["u"][observed], reference["u"][observed]),
              "ood_relative_error": relative_error(prediction["u"][~observed], reference["u"][~observed]),
              "parameter_relative_error": coefficient_error(equation, truth)}
    for key, value in actual.items():
        if not np.isclose(value, metrics[key], rtol=1e-8, atol=1e-12):
            raise RuntimeError(f"Saved {key} disagrees with the committed result.")
    if set(equation.terms) != set(truth.terms) or actual["parameter_relative_error"] >= .1:
        raise RuntimeError("The demonstration did not accurately identify the governing law.")
    if max(actual["prediction_relative_error"], actual["ood_relative_error"]) >= .05:
        raise RuntimeError("The demonstration exceeds the declared field accuracy target.")
    candidates = read_json(results / "candidates.json")
    selected = [trial for trial in candidates["pinn_trials"] if trial["status"] == "selected"]
    if len(selected) != 1 or Equation.from_dict(selected[0]["equation"]) != equation:
        raise RuntimeError("Candidate selection is inconsistent with the saved law.")
    for trial in candidates["pinn_trials"]:
        if trial.get("checkpoint") and not (results / trial["checkpoint"]).is_file():
            raise RuntimeError("A candidate checkpoint is missing.")
        if trial["status"] == "rejected":
            check = trial.get("independent_confirmation")
            floor = max(float(data["sigma"])**2, 1e-10)
            if not check or not check["converged"] or min(check["validation_mse"], trial["metrics"]["validation_data_loss"]) <= 3 * floor:
                raise RuntimeError("A rejected law lacks independent numerical confirmation.")
    design = read_json(results / "experiment_design.json")
    for probe in design["probes"]:
        if not (0 <= probe["x"] < float(data["length"]) and float(data["time_end"]) < probe["t"] <= 1.5 * float(data["time_end"]) + 1e-12):
            raise RuntimeError("A proposed measurement falls outside the future experiment domain.")
    intervals = load_arrays(results / "uncertainty.npz")
    if not all(np.isfinite(intervals[key]).all() for key in ("lower", "mean", "upper")) or np.any(intervals["lower"] > intervals["upper"]):
        raise RuntimeError("Prediction intervals are invalid.")
    return {"model_reload_max_error": difference, "prediction_hash_verified": True,
            "metrics_recomputed": actual, "future_measurements": len(design["probes"])}


def verify_results(root):
    root = Path(root)
    required = ("metrics.json", "equation.json", "candidates.json", "hypotheses.json", "knowledge_graph.json",
                "benchmark.json", "ablation_chain.json", "sparse_study.json", "generalization.json",
                "prediction_commitment.json", "forward_commitment.json", "experiment_design.json")
    for filename in required:
        if not (root / "results" / filename).exists():
            raise RuntimeError(f"Missing research output: {filename}")
    main = verify_study(root)
    benchmark = read_json(root / "results" / "benchmark.json")
    cases = read_jsonl(root / "data" / "benchmark.jsonl")
    if len(benchmark["items"]) != len(cases) or {r["id"] for r in benchmark["items"]} != {q["id"] for q in cases}:
        raise RuntimeError("Benchmark is incomplete.")
    counts = {c: sum(q["category"] == c for q in cases) for c in ("scientific", "multi_hop", "adversarial")}
    if counts != {"scientific": 50, "multi_hop": 25, "adversarial": 20}:
        raise RuntimeError("Benchmark counts do not meet the specification.")
    metrics = read_json(root / "results" / "metrics.json")
    if not metrics["ode"]["support_exact"] or metrics["ode"]["parameter_relative_error"] >= .1:
        raise RuntimeError("The saved ODE demonstration did not identify the governing law.")
    if metrics["forward_new_initial_condition"]["prediction_relative_error"] >= .05:
        raise RuntimeError("Forward demonstration exceeds the declared accuracy target.")
    if not metrics["numerical_checks"]["grid_error_decreases"]:
        raise RuntimeError("Reference-solver convergence check failed.")
    hypotheses = read_json(root / "results" / "hypotheses.json")["hypotheses"]
    if len({tuple(h["terms"]) for h in hypotheses}) < 3:
        raise RuntimeError("Fewer than three distinct generated hypotheses.")
    graph = read_json(root / "results" / "knowledge_graph.json")
    edges = {(edge["from"], edge["to"]) for edge in graph["edges"]}
    evidence = {record["id"]: record for record in read_jsonl(root / "data" / "evidence.jsonl")}
    for hypothesis in hypotheses:
        if len({evidence[key]["source"] for key in hypothesis["evidence"]}) < 2:
            raise RuntimeError("A hypothesis lacks independent source support.")
        if not hypothesis["support_paths"]:
            raise RuntimeError("A hypothesis lacks recorded support paths.")
        for path in hypothesis["support_paths"]:
            anchor, concept, bridge = path["path"]
            if (anchor, concept) not in edges or (bridge, concept) not in edges:
                raise RuntimeError("Hypothesis support does not resolve to actual graph edges.")
    candidates = read_json(root / "results" / "candidates.json")
    rejected = sum(c["status"] == "rejected" for c in candidates["pinn_trials"])
    if not rejected:
        raise RuntimeError("No failed hypothesis was rejected.")
    commitment = read_json(root / "results" / "forward_commitment.json")
    prediction = load_arrays(root / "results" / "forward_prediction.npz")
    forward, equation = load_model(root / "results" / "models" / "forward.pt")
    xx, tt = np.meshgrid(prediction["x"], prediction["t"])
    if digest(root / "results" / "forward_prediction.npz") != commitment["sha256"]:
        raise RuntimeError("Forward prediction commitment is invalid.")
    if equation != Equation.from_dict(read_json(root / "results" / "equation.json")):
        raise RuntimeError("Forward solve did not preserve the discovered coefficients.")
    if not np.allclose(forward.predict(xx, tt), prediction["u"], rtol=0, atol=1e-10):
        raise RuntimeError("Forward checkpoint does not reproduce committed predictions.")
    suite = read_json(root / "results" / "generalization.json")["experiments"]
    if {record["example"] for record in suite} != {"transport", "burgers"} or any(record["state"] != "completed" for record in suite):
        raise RuntimeError("The cross-equation validation suite is incomplete.")
    generalization = {record["example"]: verify_study(root / "results" / "generalization" / record["example"]) for record in suite}
    return {"status": "passed", "benchmark_counts": counts, "generated_hypotheses": len(hypotheses),
            "rejected_candidates": rejected, **main, "generalization": generalization,
            "prediction_hashes_verified": True, "ground_truth_absent_from_discovery_file": True,
            "research_targets": {"inverse_and_future_field_error_below": .05, "parameter_error_below": .1, "forward_field_error_below": .05}}
