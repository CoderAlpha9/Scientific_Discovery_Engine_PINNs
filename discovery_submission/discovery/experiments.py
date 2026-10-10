from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch

from .common import digest, read_json, relative_error, write_json
from .engine import QUESTION, run_discovery
from .equations import Equation
from .pinn import train, save_model
from .search import discover, fit_observations, fit_support, weak_system
from .simulation import initial_profile, make_observations, solve
from .validation import evaluate, uncertainty


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def prepare_data(root):
    root = Path(root)
    data = root / "data"
    (data / "evaluation").mkdir(parents=True, exist_ok=True)
    truth = Equation(("u_xx", "u_x", "u", "u2"), (0.012, -0.16, 0.9, -0.9))
    times = np.linspace(0, 1.2, 121)
    x, reference = solve(truth, initial_profile, times, method="spectral")
    observations = make_observations(x, times, reference)
    np.savez_compressed(data / "observations.npz", **observations)
    np.savez_compressed(data / "evaluation" / "reference.npz", x=x, t=times, u=reference,
                        terms=np.array(truth.terms), coefficients=np.array(truth.coefficients))
    _, shifted = solve(truth, lambda x: initial_profile(x, 1), times, method="spectral")
    np.savez_compressed(data / "evaluation" / "new_initial_reference.npz", x=x, t=times, u=shifted)
    ode_times = np.linspace(0, 4, 121)
    rate, saturation = 0.85, 0.7
    ode = (rate / saturation) / (1 + ((rate / saturation) / 0.2 - 1) * np.exp(-rate * ode_times))
    ode_values = np.repeat(ode[:, None], len(x), axis=1)
    ode_data = make_observations(x, ode_times, ode_values, sensors=8, snapshots=35, end=3)
    np.savez_compressed(data / "ode_observations.npz", **ode_data)
    np.savez_compressed(data / "evaluation" / "ode_reference.npz", x=x, t=ode_times, u=ode_values,
                        terms=np.array(["u", "u2"]), coefficients=np.array([rate, -saturation]))
    write_json(data / "dataset.json", {"origin": "software-generated controlled observations",
               "reference_solver": "Fourier pseudospectral spatial derivatives with BDF time integration",
               "validation_solver": "independent second-order central finite differences with BDF integration",
               "units": {"u": "concentration U", "x": "length L", "t": "time T"},
               "boundary": "periodic values and first derivatives", "length": 1,
               "observation_end": 0.8, "prediction_end": 1.2, "noise_rms_fraction": 0.01,
               "noise_distribution": "independent Gaussian", "seed": 11,
               "training_points": len(observations["u_train"]), "selection_points": len(observations["u_valid"]),
               "calibration_points": len(observations["u_cal"]),
               "split": "Every fourth sensor withheld. Its early times select models; late times calibrate intervals.",
               "isolation": "Discovery receives observations.npz only. Dense references and true coefficients reside under data/evaluation."})


def numerical_checks(truth, times, reference):
    errors = []
    for points in (64, 128, 256):
        x, numerical = solve(truth, initial_profile, times, points=points)
        dense_x = reference["x"]
        expected = np.stack([np.interp(x, np.r_[dense_x, 1], np.r_[row, row[0]]) for row in reference["u"]])
        errors.append({"points": points, "relative_error": relative_error(numerical, expected)})
    heat = Equation(("u_xx",), (0.025,))
    x, numerical = solve(heat, lambda x: np.cos(2 * np.pi * x), np.linspace(0, 1, 31), points=128)
    analytical = np.exp(-0.025 * (2 * np.pi)**2 * np.linspace(0, 1, 31)[:, None]) * np.cos(2 * np.pi * x)
    return {"grid_convergence": errors, "heat_exact_relative_error": relative_error(numerical, analytical),
            "grid_error_decreases": errors[-1]["relative_error"] < errors[0]["relative_error"]}


def sparse_study(root, reference, truth, config):
    root = Path(root)
    directory = root / "data" / "sparse_study"
    directory.mkdir(exist_ok=True)
    results = []
    for sensors in (8, 16):
        for noise in (0.0, 0.01, 0.05):
            for seed in (11, 23, 37):
                data = make_observations(reference["x"], reference["t"], reference["u"],
                                         sensors=sensors, snapshots=25, noise=noise, seed=seed)
                name = f"s{sensors}_n{noise:.2f}_seed{seed}"
                np.savez_compressed(directory / f"{name}.npz", **data)
                _, candidates, _ = discover(data)
                weak_equation = Equation.from_dict(candidates[0])
                record = {"sensors": sensors, "noise_fraction": noise, "seed": seed,
                          "observations": len(data["u_train"]), "weak_equation": weak_equation.as_dict(),
                          "weak_support_exact": set(weak_equation.terms) == set(truth.terms),
                          "weak_parameter_error": coefficient_error(weak_equation, truth)}
                if seed == 11:
                    field, equation, training = train(data, weak_equation, steps=config["study_steps"], lbfgs=config["study_lbfgs"], seed=seed)
                    record.update({"pinn_equation": equation.as_dict(), "pinn_metrics": evaluate(field, equation, data, reference, truth),
                                   "training_seconds": training["seconds"]})
                results.append(record)
                write_json(root / "results" / "sparse_study.json", {"experiments": results})
                print(f"sparse study {len(results)}/18", flush=True)
    summary = []
    for sensors in (8, 16):
        for noise in (0.0, 0.01, 0.05):
            subset = [r for r in results if r["sensors"] == sensors and r["noise_fraction"] == noise]
            summary.append({"sensors": sensors, "noise_fraction": noise,
                            "support_recovery_fraction": float(np.mean([r["weak_support_exact"] for r in subset])),
                            "mean_parameter_error": float(np.mean([r["weak_parameter_error"] for r in subset]))})
    write_json(root / "results" / "sparse_study.json", {"summary": summary, "experiments": results})
    return summary


def coefficient_error(equation, truth):
    terms = tuple(sorted(set(equation.terms) | set(truth.terms)))
    values, expected = dict(zip(equation.terms, equation.coefficients)), dict(zip(truth.terms, truth.coefficients))
    return relative_error(np.array([values.get(t, 0) for t in terms]), np.array([expected.get(t, 0) for t in terms]))


def structure_uncertainty(data, selected, repeats=12):
    rng = np.random.default_rng(71)
    counts, records = {}, []
    for i in range(repeats):
        index = rng.choice(len(data["u_train"]), len(data["u_train"]), replace=True)
        sample = dict(data)
        for quantity in ("x", "t", "u"):
            sample[f"{quantity}_train"] = data[f"{quantity}_train"][index]
        _, candidates, _ = discover(sample)
        support = " + ".join(candidates[0]["terms"])
        counts[support] = counts.get(support, 0) + 1
        records.append({"sample": i, "support": candidates[0]["terms"], "coefficients": candidates[0]["coefficients"]})
        print(f"structure bootstrap {i + 1}/{repeats}", flush=True)
    return {"resamples": repeats, "selection_frequency": {k: v / repeats for k, v in sorted(counts.items())},
            "selected_support_frequency": counts.get(" + ".join(selected.terms), 0) / repeats,
            "method": "Observation bootstrap followed by weak identification and held-out numerical rollout selection.",
            "interpretation": "Empirical structural stability, not posterior model probabilities.", "samples": records}


def run_ablation_chain(root, evidence, config):
    root = Path(root)
    data = load_arrays(root / "data" / "observations.npz")
    fit = fit_observations(data)
    matrix, target, _ = weak_system(fit)
    reference = load_arrays(root / "data" / "evaluation" / "reference.npz")
    truth = Equation(tuple(reference["terms"].tolist()), tuple(reference["coefficients"].tolist()))
    results, best_graph_equation = {}, None
    for name, use_graph in (("llm_rag", False), ("rag_kg", True)):
        hypotheses = evidence.hypotheses(QUESTION, graph=use_graph)
        proposals = []
        for hypothesis in hypotheses["hypotheses"]:
            equation, _ = fit_support(matrix, target, tuple(hypothesis["terms"]))
            times = np.linspace(0, float(data["time_end"]), 101)
            x, trajectory = solve(equation, lambda x: fit.predict(x, np.zeros_like(x)), times, points=96, length=fit.length)
            from scipy.interpolate import RegularGridInterpolator
            interpolator = RegularGridInterpolator((times, np.r_[x, fit.length]), np.column_stack([trajectory, trajectory[:, 0]]))
            prediction = interpolator(np.column_stack([data["t_valid"], data["x_valid"]]))
            loss = float(np.mean((prediction - data["u_valid"])**2))
            proposals.append({"equation": equation.as_dict(), "validation_data_loss": loss})
        best = min(proposals, key=lambda p: p["validation_data_loss"])
        equation = Equation.from_dict(best["equation"])
        _, prediction = solve(equation, lambda x: fit.predict(x, np.zeros_like(x)), reference["t"], points=128, length=fit.length)
        results[name] = {"hypotheses": hypotheses, "evaluated_proposals": proposals, "equation": equation.as_dict(),
                         "validation_data_loss": best["validation_data_loss"],
                         "parameter_relative_error": coefficient_error(equation, truth),
                         "prediction_relative_error": relative_error(prediction[reference["t"] <= 0.8], reference["u"][reference["t"] <= 0.8]),
                         "ood_relative_error": relative_error(prediction[reference["t"] > 0.8], reference["u"][reference["t"] > 0.8]),
                         "support_exact": set(equation.terms) == set(truth.terms),
                         "evaluation_note": "Weak coefficient fitting and numerical rollout are evaluator tools; the retrieval arm supplies structure only."}
        if use_graph:
            best_graph_equation = equation
    predefined, predefined_equation, training = train(data, best_graph_equation, steps=config["steps"], lbfgs=config["lbfgs"])
    save_model(root / "results" / "models" / "predefined_pinn.pt", predefined, predefined_equation)
    results["predefined_pinn"] = {"equation": predefined_equation.as_dict(),
                                 **evaluate(predefined, predefined_equation, data, reference, truth), "training": training,
                                 "evaluation_note": "The best graph proposal is supplied as a fixed support. Coefficients are inferred; no structural search."}
    metrics = read_json(root / "results" / "metrics.json")
    results["generative_pinn_discovery"] = {"equation": read_json(root / "results" / "equation.json"), **metrics["inverse"]}
    results["comparability"] = "Same observation split and reference. RAG arms propose support; coefficients and rollouts are fitted by a common evaluator. The predefined PINN fixes the best graph support; the full engine searches support."
    write_json(root / "results" / "ablation_chain.json", results)
    return results


def run_physics(root, evidence, config):
    root = Path(root)
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    data = load_arrays(root / "data" / "observations.npz")
    field, equation, trials, observation_fit = run_discovery(QUESTION, data, evidence, result_dir,
                                                             config["steps"], config["lbfgs"])
    grid_x = np.arange(128) / 128
    grid_t = np.linspace(0, 1.2, 121)
    xx, tt = np.meshgrid(grid_x, grid_t)
    frozen_prediction = field.predict(xx, tt)
    np.savez_compressed(result_dir / "frozen_prediction.npz", x=grid_x, t=grid_t, u=frozen_prediction)
    write_json(result_dir / "prediction_commitment.json", {"prediction_sha256": digest(result_dir / "frozen_prediction.npz"),
               "equation": equation.as_dict(), "observed_time_end": float(data["time_end"]),
               "prediction_time_end": float(grid_t[-1]), "labels_used_for_extrapolation": False,
               "collocation_after_observation_window": False})
    reference = load_arrays(root / "data" / "evaluation" / "reference.npz")
    truth = Equation(tuple(reference["terms"].tolist()), tuple(reference["coefficients"].tolist()))
    main = evaluate(field, equation, data, reference, truth)
    members, equations, ensemble_training = [field], [equation], []
    for seed in (23, 37):
        member, member_equation, training = train(data, equation, bootstrap=True, seed=seed,
                                                 steps=config["steps"], lbfgs=config["lbfgs"])
        save_model(result_dir / "models" / f"ensemble_{seed}.pt", member, member_equation)
        members.append(member)
        equations.append(member_equation)
        ensemble_training.append(training)
    uq, arrays = uncertainty(members, equations, data, reference)
    np.savez_compressed(result_dir / "uncertainty.npz", x=reference["x"], t=reference["t"], **arrays)
    forward_data = {"length": np.array(1.0), "time_end": np.array(1.2),
                    **{f"{q}_{split}": np.array([]) for q in ("x", "t", "u") for split in ("train", "valid")}}
    forward, fixed, forward_training = train(forward_data, equation, initial=lambda x: initial_profile(x, 1),
                                             steps=config["steps"], lbfgs=config["lbfgs"], seed=11)
    save_model(result_dir / "models" / "forward.pt", forward, fixed)
    forward_prediction = forward.predict(xx, tt)
    np.savez_compressed(result_dir / "forward_prediction.npz", x=grid_x, t=grid_t, u=forward_prediction)
    write_json(result_dir / "forward_commitment.json", {"sha256": digest(result_dir / "forward_prediction.npz"),
               "initial_profile_variant": 1, "interior_labels_used": False, "coefficients_frozen": True,
               "prediction_end": 1.2, "physics_collocation_end": 1.2})
    new_reference = load_arrays(root / "data" / "evaluation" / "new_initial_reference.npz")
    forward_metrics = evaluate(forward, fixed, forward_data, new_reference, truth)
    ode_data = load_arrays(root / "data" / "ode_observations.npz")
    _, ode_candidates, _ = discover(ode_data, terms=("u", "u2", "u3", "1"), modes=0)
    ode_field, ode_equation, ode_training = train(ode_data, Equation.from_dict(ode_candidates[0]), modes=0,
                                                steps=config["study_steps"], lbfgs=config["lbfgs"])
    save_model(result_dir / "models" / "ode.pt", ode_field, ode_equation)
    ode_reference = load_arrays(root / "data" / "evaluation" / "ode_reference.npz")
    ode_truth = Equation(tuple(ode_reference["terms"].tolist()), tuple(ode_reference["coefficients"].tolist()))
    ode_metrics = evaluate(ode_field, ode_equation, ode_data, ode_reference, ode_truth)
    neural, _, neural_training = train(data, steps=config["steps"], lbfgs=config["lbfgs"])
    save_model(result_dir / "models" / "neural.pt", neural)
    neural_metrics = evaluate(neural, truth, data, reference)
    neural_metrics["residual_equation_source"] = "evaluator's true law; not supplied in neural training"
    weak = load_arrays(result_dir / "weak_system.npz")
    oracle_initial, _ = fit_support(weak["matrix"], weak["target"], truth.terms)
    oracle, oracle_equation, oracle_training = train(data, oracle_initial, steps=config["steps"], lbfgs=config["lbfgs"])
    save_model(result_dir / "models" / "oracle_pinn.pt", oracle, oracle_equation)
    oracle_metrics = evaluate(oracle, oracle_equation, data, reference, truth)
    screened = read_json(result_dir / "candidates.json")["screened"]
    selected_initial = Equation.from_dict(next(item for item in screened if tuple(item["terms"]) == equation.terms))
    uniform, uniform_equation, uniform_training = train(data, selected_initial, steps=config["steps"],
                                                        lbfgs=config["lbfgs"], adaptive=False)
    save_model(result_dir / "models" / "uniform_collocation.pt", uniform, uniform_equation)
    uniform_metrics = evaluate(uniform, uniform_equation, data, reference, truth)
    _, no_evidence_candidates, no_evidence_info = discover(data)
    weak_equation = Equation.from_dict(screened[0])
    xnum, unumerical = solve(equation, initial_profile, grid_t, points=128)
    _, weak_prediction = solve(weak_equation, lambda x: observation_fit.predict(x, np.zeros_like(x)), grid_t, points=128)
    ablations = {"conventional_neural_network": neural_metrics, "oracle_predefined_pinn": oracle_metrics,
                 "full_generative_discovery": main,
                 "uniform_collocation": {**uniform_metrics, "equation": uniform_equation.as_dict(),
                     "comparison": "Same selected support, starting coefficients, seed and optimizer budgets; fixed uniform collocation."},
                 "without_pinn_refinement": {"equation": weak_equation.as_dict(), "parameter_relative_error": coefficient_error(weak_equation, truth),
                     "prediction_relative_error": relative_error(weak_prediction[grid_t <= 0.8], reference["u"][grid_t <= 0.8]),
                     "ood_relative_error": relative_error(weak_prediction[grid_t > 0.8], reference["u"][grid_t > 0.8])},
                 "without_evidence_seeding": {"equation": no_evidence_candidates[0], "evaluated_supports": no_evidence_info["evaluated_supports"]},
                 "discovered_law_numerical_solver": {"prediction_relative_error": relative_error(unumerical, reference["u"]),
                    "initial_condition_source": "known experimental initial profile, for solver comparison only"}}
    checks = numerical_checks(truth, grid_t, reference)
    metrics = {"inverse": main, "forward_new_initial_condition": forward_metrics,
               "ode": {"equation": ode_equation.as_dict(), **ode_metrics}, "uncertainty": uq,
               "ablations": ablations, "numerical_checks": checks,
               "training": {"forward": forward_training, "ode": ode_training, "neural": neural_training,
                            "oracle": oracle_training, "uniform_collocation": uniform_training, "ensemble": ensemble_training},
               "initial_condition_inference": {"method": "Evaluate the learned field at t=0 without supplying the true initial profile to the inverse problem.",
                    "initial_mse": main["initial_loss"], "boundary_inference": "Periodic boundary type is known, not inferred."},
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "torch": torch.__version__, "device": "cpu", "threads": 2}, "physics_seconds": time.perf_counter() - started}
    write_json(result_dir / "metrics.json", metrics)
    np.savez_compressed(result_dir / "comparison.npz", x=grid_x, t=grid_t, truth=reference["u"],
                        discovered=field.predict(xx, tt), neural=neural.predict(xx, tt), oracle=oracle.predict(xx, tt),
                        numerical=unumerical, forward=forward_prediction, forward_truth=new_reference["u"])
    sparse = sparse_study(root, reference, truth, config)
    metrics["sparse_study"] = sparse
    metrics["structure_uncertainty"] = structure_uncertainty(data, equation)
    metrics["physics_seconds"] = time.perf_counter() - started
    write_json(result_dir / "metrics.json", metrics)
    return metrics
