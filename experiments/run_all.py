"""Drive configs/experiments.yaml to reproduce Table 9/10: every dataset x every
baseline x every (beta, alpha) pair for both GRS3WDC-N and GRS3WDC-MC.

This is the container's default entrypoint. Full reproduction (six datasets,
20 runs each, including maximal-clique search on Phishing/Letter) can take a
long time -- the paper itself notes clique enumeration doesn't scale well to
large datasets. Use --datasets/--n-runs to run a smaller demo.

Example:
    python -m experiments.run_all --datasets banknote sonar --n-runs 5
"""

import argparse
import time
from pathlib import Path

from grs3wdc.data import DATASET_REGISTRY, load_dataset

from experiments._common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RESULTS_DIR,
    MODEL_DISPLAY_NAMES,
    append_result_row,
    evaluate,
    format_grs3wdc_name,
    load_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)

    datasets = args.datasets or config["datasets"]
    n_runs = args.n_runs or config["n_runs"]
    test_size = config["test_size"]
    seed_base = config["seed_base"]
    results_dir = Path(args.results_dir)

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        X, y = load_dataset(dataset)

        for model in config["baselines"]:
            t0 = time.time()
            result = evaluate(model, X, y, n_runs=n_runs, test_size=test_size, seed_base=seed_base)
            model_name = MODEL_DISPLAY_NAMES[model]
            append_result_row(dataset, model_name, result, results_dir=results_dir)
            print(f"  {model_name:15s} acc={result.accuracy:.4f} f1={result.f1:.4f} ({time.time() - t0:.1f}s)")

        for model_key in ("grs3wdc_n", "grs3wdc_mc"):
            model = "grs3wdc-n" if model_key == "grs3wdc_n" else "grs3wdc-mc"
            spec = config[model_key]
            for params in spec["params"]:
                t0 = time.time()
                result = evaluate(
                    model, X, y, n_runs=n_runs, test_size=test_size, seed_base=seed_base,
                    beta=params["beta"], alpha=params["alpha"],
                    metric=spec.get("metric", "euclidean"),
                    threshold_mode=spec.get("threshold_mode", "percentile"),
                )
                model_name = format_grs3wdc_name(model, params["beta"], params["alpha"])
                append_result_row(dataset, model_name, result, results_dir=results_dir)
                print(f"  {model_name:22s} acc={result.accuracy:.4f} f1={result.f1:.4f} "
                      f"coverage={result.coverage:.4f} ({time.time() - t0:.1f}s)")

    print(f"\nDone. Results written under {args.results_dir}/<dataset>_res.csv")


if __name__ == "__main__":
    main()
