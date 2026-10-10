from pathlib import Path
import time

import numpy as np

from .common import digest, read_json, write_json
from .engine import QUESTION, run_discovery
from .equations import Equation
from .experiments import load_arrays
from .pinn import train, save_model
from .simulation import initial_profile, make_observations, solve
from .validation import evaluate, uncertainty


EXAMPLES = {
    "transport_reaction": {"title": "Transport and reaction", "length": 1.0,
                           "terms": ("u_xx", "u_x", "u", "u2"), "coefficients": (0.012, -0.16, 0.9, -0.9)},
    "transport": {"title": "Transport on a longer domain", "length": 2.0,
                  "terms": ("u_xx", "u_x"), "coefficients": (0.048, -0.22)},
    "burgers": {"title": "Nonlinear transport", "length": 1.0,
                "terms": ("u_xx", "u*u_x"), "coefficients": (0.035, -0.4)},
}


def prepare_case(output, name="transport_reaction", sensors=16, noise=0.01, seed=11):
    if name not in EXAMPLES:
        raise ValueError("Unknown controlled experiment.")
    if (type(sensors) is not int or sensors not in (8, 16, 24, 32)
            or not np.isfinite(noise) or not 0 <= noise <= 0.1
            or type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("Choose 8, 16, 24 or 32 sensors, noise in [0, 0.1], and a valid nonnegative seed.")
    output = Path(output)
    definition = EXAMPLES[name]
    length = definition["length"]
    truth = Equation(definition["terms"], definition["coefficients"])
    times = np.linspace(0, 1.2, 121)
    initial = lambda x: initial_profile(x / length)
    x, reference = solve(truth, initial, times, length=length, method="spectral")
    data = make_observations(x, times, reference, sensors=sensors, noise=noise, seed=seed)
    (output / "data" / "evaluation").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "data" / "observations.npz", **data)
    np.savez_compressed(output / "data" / "evaluation" / "reference.npz", x=x, t=times, u=reference,
                        terms=np.array(truth.terms), coefficients=np.array(truth.coefficients))
    write_json(output / "data" / "dataset.json", {"example": name, "title": definition["title"], "seed": seed,
               "sensors": sensors, "noise_rms_fraction": noise, "length": length,
               "observation_end": 0.8, "prediction_end": 1.2, "boundary": "periodic",
               "training_points": len(data["u_train"]), "selection_points": len(data["u_valid"]),
               "calibration_points": len(data["u_cal"]), "origin": "software-generated controlled observations"})
    return data


def run_case(output, evidence, config, name="transport_reaction", sensors=16, noise=0.01, seed=11):
    output = Path(output)
    started = time.perf_counter()
    data = prepare_case(output, name, sensors, noise, seed)
    results = output / "results"
    field, equation, trials, fit = run_discovery(QUESTION, data, evidence, results,
                                                steps=config["steps"], lbfgs=config["lbfgs"], seed=seed)
    x = np.arange(128) * float(data["length"]) / 128
    times = np.linspace(0, 1.5 * float(data["time_end"]), 121)
    xx, tt = np.meshgrid(x, times)
    np.savez_compressed(results / "frozen_prediction.npz", x=x, t=times, u=field.predict(xx, tt))
    write_json(results / "prediction_commitment.json", {"prediction_sha256": digest(results / "frozen_prediction.npz"),
               "equation": equation.as_dict(), "observed_time_end": float(data["time_end"]),
               "prediction_time_end": float(times[-1]), "labels_used_for_extrapolation": False,
               "collocation_after_observation_window": False})
    members, equations, training_records = [field], [equation], []
    for offset in (12, 26):
        write_json(results / "progress.json", {"state": "running", "stage": "uncertainty ensemble", "completed": len(members), "total": 3})
        member, fitted, training = train(data, equation, bootstrap=True, seed=(seed + offset) % (2**32),
                                         steps=config["steps"], lbfgs=config["lbfgs"])
        save_model(results / "models" / f"ensemble_{offset}.pt", member, fitted)
        members.append(member)
        equations.append(fitted)
        training_records.append(training)
    reference = load_arrays(output / "data" / "evaluation" / "reference.npz")
    truth = Equation(tuple(reference["terms"]), tuple(reference["coefficients"]))
    metrics = evaluate(field, equation, data, reference, truth)
    uq, arrays = uncertainty(members, equations, data, reference)
    np.savez_compressed(results / "uncertainty.npz", x=x, t=times, **arrays)
    record = {"inverse": metrics, "uncertainty": uq, "training": {"ensemble": training_records},
              "experiment": read_json(output / "data" / "dataset.json"), "seconds": time.perf_counter() - started}
    write_json(results / "metrics.json", record)
    write_json(results / "progress.json", {"state": "completed", "stage": "experiment", "completed": 1, "total": 1})
    print(f"experiment {name}: completed", flush=True)
    return record


def run_suite(root, evidence, config):
    root = Path(root)
    records = []
    for name in ("transport", "burgers"):
        output = root / "results" / "generalization" / name
        try:
            record = run_case(output, evidence, config, name, sensors=24, noise=0.005, seed=23)
            records.append({"example": name, "state": "completed", "metrics": record["inverse"],
                            "equation": read_json(output / "results" / "equation.json"),
                            "seconds": record["seconds"]})
        except (RuntimeError, ValueError) as error:
            records.append({"example": name, "state": "inconclusive", "reason": str(error)})
        write_json(root / "results" / "generalization.json", {"experiments": records})
    return records
