from pathlib import Path

import numpy as np

from .common import write_json
from .equations import Equation
from .diagnostics import identifiability, independent_refit, propose_measurements
from .pinn import train, save_model
from .search import discover, fit_support, regression_uncertainty, validate_observations
from .validation import evaluate


QUESTION = ("What local evolution law explains sparse noisy scalar concentration observations in a one-dimensional "
            "periodic channel with possible drift, spreading, and concentration-dependent internal production?")


def run_discovery(question, data, evidence, output, steps=1600, lbfgs=600, seed=11):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_observations(data)
    write_json(output / "progress.json", {"state": "running", "stage": "evidence", "completed": 0, "total": 1})
    hypotheses = evidence.hypotheses(question)
    write_json(output / "hypotheses.json", hypotheses)
    write_json(output / "knowledge_graph.json", evidence.graph)
    write_json(output / "progress.json", {"state": "running", "stage": "equation search", "completed": 0, "total": 1})
    fit, candidates, search = discover(data, proposals=hypotheses["hypotheses"])
    write_json(output / "search.json", {k: v for k, v in search.items() if k not in ("matrix", "target")})
    np.savez_compressed(output / "weak_system.npz", matrix=search["matrix"], target=search["target"])
    trials, fields = [], []
    shortlist = [Equation.from_dict(c) for c in candidates[:4]]
    simple, _ = fit_support(search["matrix"], search["target"], ("u_xx",))
    if simple.terms not in {e.terms for e in shortlist}:
        shortlist.append(simple)
    for i, candidate in enumerate(shortlist):
        write_json(output / "progress.json", {"state": "running", "stage": "PINN validation", "completed": i, "total": len(shortlist)})
        try:
            field, equation, training = train(data, candidate, steps=steps, lbfgs=lbfgs, seed=seed)
            metrics = evaluate(field, equation, data)
            scale = max(float(np.mean(data["u_train"]**2)), 1e-12)
            noise_floor = max(float(data["sigma"])**2, 1e-10)
            complexity = noise_floor / scale * np.log(len(data["u_valid"])) / len(data["u_valid"])
            sensitivity = regression_uncertainty(search["matrix"], search["target"], equation.terms)
            components = {"data": metrics["validation_data_loss"] / scale,
                          "physics": 0.2 * metrics["pde_residual_mse"] * float(data["time_end"])**2 / scale,
                          "balance": 0.05 * metrics["balance_rms"]**2 * float(data["time_end"])**2 / (scale * float(data["length"])**2),
                          "complexity": complexity * len(equation.terms), "uncertainty": 0.001 * sensitivity}
            score = sum(components.values())
            confirmation, reason, status = None, None, "competitive"
            if metrics["validation_data_loss"] > 3 * noise_floor:
                confirmation = independent_refit(data, equation, fit)
                rejected = confirmation["converged"] and confirmation["validation_mse"] > 3 * noise_floor
                status = "rejected" if rejected else "optimization_failed"
                reason = ("Both PINN and independent numerical fitting disagree with withheld measurements at the assumed noise level."
                          if rejected else "The PINN fit is insufficient; independent fitting does not establish rejection.")
            trial = {"index": i, "equation": equation.as_dict(), "metrics": metrics, "training": training,
                     "selection_score": score, "score_components": components, "status": status,
                     "identifiability": identifiability(search["matrix"], equation.terms),
                     "uncertainty_score": sensitivity, "uncertainty_method": "weak-regression bootstrap prediction sensitivity",
                     "independent_confirmation": confirmation, "rejection_reason": reason,
                     "checkpoint": f"models/candidate_{i}.pt"}
            save_model(output / trial["checkpoint"], field, equation)
            print(f"candidate {i}: {equation.terms}, score={score:.6g}, {status}", flush=True)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            field = None
            trial = {"index": i, "equation": candidate.as_dict(), "status": "optimization_failed",
                     "selection_score": None, "rejection_reason": str(error), "checkpoint": None}
            print(f"candidate {i}: optimization failed: {error}", flush=True)
        trials.append(trial)
        fields.append(field)
        write_json(output / "candidates.json", {"screened": candidates, "pinn_trials": trials, "status": "running"})
    eligible = [i for i, r in enumerate(trials) if r["status"] == "competitive"]
    if not eligible:
        write_json(output / "candidates.json", {"screened": candidates, "pinn_trials": trials, "status": "inconclusive"})
        write_json(output / "progress.json", {"state": "inconclusive", "stage": "validation", "completed": len(trials), "total": len(trials)})
        raise RuntimeError("No candidate explains the held-out observations within the noise tolerance.")
    selected = min(eligible, key=lambda i: trials[i]["selection_score"])
    trials[selected]["status"] = "selected"
    equation = Equation.from_dict(trials[selected]["equation"])
    save_model(output / "models" / "inverse.pt", fields[selected], equation)
    (output / "models" / f"candidate_{selected}.pt").unlink()
    trials[selected]["checkpoint"] = "models/inverse.pt"
    write_json(output / "equation.json", equation.as_dict())
    write_json(output / "candidates.json", {"screened": candidates, "pinn_trials": trials,
                                           "selected_index": selected, "status": "selected"})
    design, rollouts = propose_measurements(data, fields[selected], trials)
    write_json(output / "experiment_design.json", design)
    if rollouts is not None:
        np.savez_compressed(output / "candidate_rollouts.npz", **rollouts)
    write_json(output / "progress.json", {"state": "completed", "stage": "discovery", "completed": len(trials), "total": len(trials)})
    return fields[selected], equation, trials, fit
