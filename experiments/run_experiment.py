"""Run a single dataset/model/parameter configuration and print mean +/- std metrics.

Replaces the one-off archive/notebooks/validate_maximal.ipynb and
validate_neighborhood.ipynb templates with a reusable, parameterized CLI.

Examples:
    python -m experiments.run_experiment --dataset banknote --model grs3wdc-n --beta 1 --alpha 0.82
    python -m experiments.run_experiment --dataset breast --model svm --runs 20
"""

import argparse

from grs3wdc.data import DATASET_REGISTRY, load_dataset
from grs3wdc.evaluation import repeated_holdout

from experiments._common import (
    BASELINE_MODELS,
    GRS3WDC_MODELS,
    MODEL_DISPLAY_NAMES,
    append_result_row,
    format_grs3wdc_name,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--model", required=True, choices=sorted(set(GRS3WDC_MODELS) | BASELINE_MODELS))
    parser.add_argument("--beta", type=float, help="Neighborhood/clique distance threshold (percentile by default)")
    parser.add_argument("--alpha", type=float, help="Granule purity threshold")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "chebyshev"])
    parser.add_argument("--threshold-mode", default="percentile", choices=["percentile", "absolute"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-save", action="store_true", help="Don't append the result to results/<dataset>_res.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    X, y = load_dataset(args.dataset)

    if args.model in GRS3WDC_MODELS:
        if args.beta is None or args.alpha is None:
            raise SystemExit(f"--model {args.model} requires --beta and --alpha")
        module = GRS3WDC_MODELS[args.model]

        def fit_predict(X_train, y_train, X_test, y_test):
            return module.run_model(
                X_train, y_train, X_test, y_test,
                threshold=args.beta, alpha=args.alpha,
                metric=args.metric, threshold_mode=args.threshold_mode, n_jobs=args.n_jobs,
            )

        model_name = format_grs3wdc_name(args.model, args.beta, args.alpha)
    else:
        from grs3wdc import baselines

        def fit_predict(X_train, y_train, X_test, y_test):
            return baselines.run_model(X_train, y_train, X_test, y_test, model_name=args.model)

        model_name = MODEL_DISPLAY_NAMES[args.model]

    summary = repeated_holdout(fit_predict, X, y, n_runs=args.runs, test_size=args.test_size, seed_base=args.seed_base)

    print(f"{model_name} on {args.dataset} ({args.runs} runs):")
    for field in summary.mean._fields:
        mean_val = getattr(summary.mean, field)
        std_val = getattr(summary.std, field)
        print(f"  {field:10s}: {mean_val:.4f} +/- {std_val:.4f}")

    if not args.no_save:
        out_path = append_result_row(args.dataset, model_name, summary.mean)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
