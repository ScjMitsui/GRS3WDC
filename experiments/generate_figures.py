"""Render Fig. 9-style bar charts (Accuracy/Precision/Recall/F1 by model) from
results/<dataset>_res.csv. Parameterized version of
archive/notebooks/visulization_Untitled.ipynb, which was hardcoded to one
dataset at a time.

Example:
    python -m experiments.generate_figures --datasets banknote sonar image
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grs3wdc.data import DATASET_REGISTRY

from experiments._common import DEFAULT_RESULTS_DIR

METRICS = ["Accuracy", "Precision", "Recall", "F1"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=None, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--format", default="png", choices=["png", "eps", "svg"])
    return parser


def plot_dataset(dataset: str, results_dir: Path, fmt: str) -> Path:
    data = pd.read_csv(results_dir / f"{dataset}_res.csv")
    data_long = data.melt(id_vars="Models", value_vars=METRICS, var_name="Metrics", value_name="Value")

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))
    sns.barplot(x="Metrics", y="Value", hue="Models", data=data_long, palette=sns.color_palette("pastel"))
    plt.legend(title="Models", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("")
    plt.ylabel("Value")
    plt.ylim(0, 1.02)
    plt.tight_layout()

    out_path = results_dir / "figures" / f"{dataset}.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format=fmt, dpi=300)
    plt.close()
    return out_path


def main() -> None:
    args = build_arg_parser().parse_args()
    results_dir = Path(args.results_dir)
    datasets = args.datasets or sorted(DATASET_REGISTRY)

    for dataset in datasets:
        res_file = results_dir / f"{dataset}_res.csv"
        if not res_file.exists():
            print(f"Skipping {dataset}: {res_file} not found (run experiments.run_all first)")
            continue
        out_path = plot_dataset(dataset, results_dir, args.format)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
