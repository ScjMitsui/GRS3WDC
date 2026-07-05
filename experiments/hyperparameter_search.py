"""Grid-search (beta, alpha) for a GRS3WDC model across one or more datasets.

Replaces archive/notebooks/hypertunning.ipynb's ad hoc per-dataset grid search
with a reusable CLI. Ranks parameter pairs by mean F1 averaged across datasets.

Example:
    python -m experiments.hyperparameter_search --model grs3wdc-n \\
        --datasets banknote sonar image --beta-values 0.5 1 2 3 --alpha-values 0.80 0.82 0.85
"""

import argparse
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from grs3wdc.data import DATASET_REGISTRY, load_dataset

from experiments._common import GRS3WDC_MODELS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(GRS3WDC_MODELS))
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--beta-values", nargs="+", type=float, required=True)
    parser.add_argument("--alpha-values", nargs="+", type=float, required=True)
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "chebyshev"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    module = GRS3WDC_MODELS[args.model]

    splits = {}
    for name in args.datasets:
        X, y = load_dataset(name)
        splits[name] = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    param_grid = list(product(args.beta_values, args.alpha_values))

    def evaluate(beta: float, alpha: float) -> dict:
        f1_scores = []
        for name in args.datasets:
            X_train, X_test, y_train, y_test = splits[name]
            result = module.run_model(
                X_train, y_train, X_test, y_test,
                threshold=beta, alpha=alpha, metric=args.metric, threshold_mode="percentile", n_jobs=1,
            )
            f1_scores.append(result.f1)
        return {"beta": beta, "alpha": alpha, "avg_f1": sum(f1_scores) / len(f1_scores)}

    rows = Parallel(n_jobs=args.n_jobs)(delayed(evaluate)(beta, alpha) for beta, alpha in param_grid)
    results_df = pd.DataFrame(rows).sort_values("avg_f1", ascending=False).reset_index(drop=True)

    print(f"Top {args.top} (beta, alpha) pairs for {args.model} on {args.datasets}:")
    print(results_df.head(args.top).to_string(index=False))

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Saved full results to {args.output}")


if __name__ == "__main__":
    main()
