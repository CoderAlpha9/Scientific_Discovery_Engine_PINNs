from pathlib import Path

import numpy as np

from .common import digest, read_json, read_jsonl
from .experiments import load_arrays
from .pinn import load_model


def verify_results(root):
    root = Path(root)
    required = ("metrics.json", "equation.json", "candidates.json", "hypotheses.json", "knowledge_graph.json",
                "benchmark.json", "ablation_chain.json", "sparse_study.json", "prediction_commitment.json", "forward_commitment.json")
    for filename in required:
        if not (root / "results" / filename).exists():
            raise RuntimeError(f"Missing research output: {filename}")
    benchmark = read_json(root / "results" / "benchmark.json")
    cases = read_jsonl(root / "data" / "benchmark.jsonl")
    if len(benchmark["items"]) != len(cases) or set(r["id"] for r in benchmark["items"]) != set(q["id"] for q in cases):
        raise RuntimeError("Benchmark is incomplete.")
    counts = {c: sum(q["category"] == c for q in cases) for c in ("scientific", "multi_hop", "adversarial")}
    if counts != {"scientific": 50, "multi_hop": 25, "adversarial": 20}:
        raise RuntimeError("Benchmark counts do not meet the specification.")
    metrics = read_json(root / "results" / "metrics.json")
    if not metrics["inverse"]["support_exact"] or not metrics["ode"]["support_exact"]:
        raise RuntimeError("The saved demonstration did not identify the governing support.")
    if metrics["inverse"]["prediction_relative_error"] >= 0.05 or metrics["inverse"]["parameter_relative_error"] >= 0.1:
        raise RuntimeError("Inverse demonstration exceeds the declared accuracy target.")
    if metrics["forward_new_initial_condition"]["prediction_relative_error"] >= 0.05:
        raise RuntimeError("Forward demonstration exceeds the declared accuracy target.")
    if not metrics["numerical_checks"]["grid_error_decreases"]:
        raise RuntimeError("Reference-solver convergence check failed.")
    hypotheses = read_json(root / "results" / "hypotheses.json")["hypotheses"]
    if len({tuple(h["terms"]) for h in hypotheses}) < 3:
        raise RuntimeError("Fewer than three distinct generated hypotheses.")
    candidates = read_json(root / "results" / "candidates.json")
    if not any(c["status"] == "rejected" for c in candidates["pinn_trials"]):
        raise RuntimeError("No failed hypothesis was rejected.")
    for name, commitment_name, hash_key in (("frozen_prediction.npz", "prediction_commitment.json", "prediction_sha256"),
                                            ("forward_prediction.npz", "forward_commitment.json", "sha256")):
        commitment = read_json(root / "results" / commitment_name)
        if digest(root / "results" / name) != commitment[hash_key]:
            raise RuntimeError("Prediction commitment does not match saved predictions.")
    prediction = load_arrays(root / "results" / "frozen_prediction.npz")
    model, equation = load_model(root / "results" / "models" / "inverse.pt")
    xx, tt = np.meshgrid(prediction["x"], prediction["t"])
    difference = float(np.max(np.abs(model.predict(xx, tt) - prediction["u"])))
    if difference > 1e-10:
        raise RuntimeError("Reloaded model does not reproduce committed predictions.")
    data = load_arrays(root / "data" / "observations.npz")
    if {"terms", "coefficients", "truth", "reference"} & set(data):
        raise RuntimeError("Ground truth leaked into discovery observations.")
    return {"status": "passed", "benchmark_counts": counts, "generated_hypotheses": len(hypotheses),
            "rejected_candidates": sum(c["status"] == "rejected" for c in candidates["pinn_trials"]),
            "model_reload_max_error": difference, "prediction_hashes_verified": True,
            "ground_truth_absent_from_discovery_file": True,
            "research_targets": {"inverse_field_error_below": 0.05, "parameter_error_below": 0.1, "forward_field_error_below": 0.05}}
