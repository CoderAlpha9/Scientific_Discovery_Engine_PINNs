from pathlib import Path

import numpy as np

from .common import write_json
from .equations import Equation
from .pinn import train, save_model
from .search import discover, fit_support, regression_uncertainty, validate_observations
from .validation import evaluate


QUESTION = ("What local evolution law explains sparse noisy scalar concentration observations in a one-dimensional "
            "periodic channel with possible drift, spreading, and concentration-dependent internal production?")


def run_discovery(question, data, evidence, output, steps=1600, lbfgs=600, seed=11):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_observations(data)
    hypotheses = evidence.hypotheses(question)
    write_json(output / "hypotheses.json", hypotheses)
    write_json(output / "knowledge_graph.json", evidence.graph)
    fit, candidates, search = discover(data, proposals=hypotheses["hypotheses"])
    write_json(output / "search.json", {k: v for k, v in search.items() if k not in ("matrix", "target")})
    np.savez_compressed(output / "weak_system.npz", matrix=search["matrix"], target=search["target"])
    trials, fields = [], []
    shortlist = [Equation.from_dict(c) for c in candidates[:4]]
    simple, _ = fit_support(search["matrix"], search["target"], ("u_xx",))
    if simple.terms not in {e.terms for e in shortlist}:
        shortlist.append(simple)
    for i, candidate in enumerate(shortlist):
        field, equation, training = train(data, candidate, steps=steps, lbfgs=lbfgs, seed=seed)
        metrics = evaluate(field, equation, data)
        scale = max(float(np.mean(data["u_train"]**2)), 1e-12)
        noise_floor = max(float(data["sigma"])**2, 1e-10)
        complexity = noise_floor / scale * np.log(len(data["u_valid"])) / len(data["u_valid"])
        sensitivity = regression_uncertainty(search["matrix"], search["target"], equation.terms)
        score = (metrics["validation_data_loss"] / scale + 0.2 * metrics["pde_residual_mse"] * float(data["time_end"])**2 / scale
                 + 0.05 * metrics["balance_rms"]**2 * float(data["time_end"])**2 / scale
                 + complexity * len(equation.terms) + 0.001 * sensitivity)
        rejected = metrics["validation_data_loss"] > 3 * noise_floor
        trial = {"index": i, "equation": equation.as_dict(), "metrics": metrics, "training": training,
                 "selection_score": score, "status": "rejected" if rejected else "competitive",
                 "uncertainty_score": sensitivity, "uncertainty_method": "weak-regression bootstrap prediction sensitivity",
                 "rejection_reason": "held-out observations are inconsistent with the assumed noise" if rejected else None}
        trials.append(trial)
        fields.append(field)
        save_model(output / "models" / f"candidate_{i}.pt", field, equation)
        print(f"candidate {i}: {equation.terms}, score={score:.6g}, {trial['status']}", flush=True)
    eligible = [i for i, r in enumerate(trials) if r["status"] != "rejected"]
    if not eligible:
        write_json(output / "candidates.json", {"screened": candidates, "pinn_trials": trials, "status": "inconclusive"})
        raise RuntimeError("No candidate explains the held-out observations within the noise tolerance.")
    selected = min(eligible, key=lambda i: trials[i]["selection_score"])
    trials[selected]["status"] = "selected"
    equation = Equation.from_dict(trials[selected]["equation"])
    save_model(output / "models" / "inverse.pt", fields[selected], equation)
    (output / "models" / f"candidate_{selected}.pt").unlink()
    for trial in trials:
        trial["checkpoint"] = "models/inverse.pt" if trial["index"] == selected else f"models/candidate_{trial['index']}.pt"
    write_json(output / "equation.json", equation.as_dict())
    write_json(output / "candidates.json", {"screened": candidates, "pinn_trials": trials,
                                           "selected_index": selected, "status": "selected"})
    return fields[selected], equation, trials, fit
