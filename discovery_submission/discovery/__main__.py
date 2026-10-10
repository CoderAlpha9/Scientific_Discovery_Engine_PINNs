import argparse
from pathlib import Path
import sys

from .common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Evidence-guided scientific equation discovery")
    parser.add_argument("command", choices=("run", "physics", "benchmark", "verify", "report", "discover", "ask", "experiment", "suite"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--live", action="store_true", help="Regenerate model responses instead of replaying matching cached requests")
    parser.add_argument("--offline", action="store_true", help="Require cached pretrained requests; never download weights")
    parser.add_argument("--example", choices=("transport_reaction", "transport", "burgers"), default="transport_reaction")
    parser.add_argument("--sensors", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--graph", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    config = read_json(root / "config.json")
    if arguments.command == "verify":
        from .verify import verify_results
        result = verify_results(root)
        write_json(root / "results" / "verification.json", result)
        print(result)
        return
    if arguments.command == "report":
        from .report import make_report
        make_report(root)
        return
    from .evidence import Evidence, ModelCache
    evidence = Evidence(root / "data" / "evidence.jsonl",
                        ModelCache(root / "data" / "model_cache.json", arguments.cache_dir, arguments.live, arguments.offline))
    if arguments.command == "experiment":
        if arguments.output is None:
            parser.error("experiment requires --output")
        from .cases import run_case
        run_case(arguments.output, evidence, config, arguments.example, arguments.sensors, arguments.noise, arguments.seed)
        return
    if arguments.command == "ask":
        if not arguments.question:
            parser.error("ask requires --question")
        print(evidence.answer(arguments.question, arguments.graph))
        return
    if arguments.command == "discover":
        if arguments.observations is None or arguments.output is None:
            parser.error("discover requires --observations and --output")
        from .engine import QUESTION, run_discovery
        from .experiments import load_arrays
        run_discovery(arguments.question or QUESTION, load_arrays(arguments.observations), evidence,
                      arguments.output, config["steps"], config["lbfgs"])
        return
    if arguments.command in ("run", "physics"):
        from .experiments import prepare_data, run_physics
        prepare_data(root)
        run_physics(root, evidence, config)
    if arguments.command in ("run", "suite"):
        from .cases import run_suite
        run_suite(root, evidence, config)
    if arguments.command in ("run", "benchmark"):
        from .benchmark import run_benchmark
        from .experiments import run_ablation_chain
        run_benchmark(root / "data" / "benchmark.jsonl", evidence, root / "results" / "benchmark.json")
        run_ablation_chain(root, evidence, config)
    if arguments.command == "run":
        from .report import make_report
        from .verify import verify_results
        make_report(root)
        write_json(root / "results" / "verification.json", verify_results(root))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
