"""Shared helpers for the experiment CLIs."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from grs3wdc import baselines
from grs3wdc.evaluation import EvaluationResult, repeated_holdout
from grs3wdc.granules import maximal_clique, neighborhood

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

GRS3WDC_MODELS = {"grs3wdc-n": neighborhood, "grs3wdc-mc": maximal_clique}
BASELINE_MODELS = set(baselines.BASELINE_FACTORIES)

MODEL_DISPLAY_NAMES = {
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_grs3wdc_name(model: str, beta: float, alpha: float) -> str:
    prefix = "GRS3WDC-N" if model == "grs3wdc-n" else "GRS3WDC-MC"
    beta_str = f"{beta:g}"
    alpha_str = f"{alpha:g}"
    return f"{prefix}({beta_str},{alpha_str})"


def evaluate(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int,
    test_size: float,
    seed_base: int,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    metric: str = "euclidean",
    threshold_mode: str = "percentile",
    n_jobs: int = -1,
) -> EvaluationResult:
    """Run `model` over `n_runs` holdout splits and return the mean EvaluationResult."""
    if model in GRS3WDC_MODELS:
        if beta is None or alpha is None:
            raise ValueError(f"{model} requires --beta and --alpha")
        module = GRS3WDC_MODELS[model]

        def fit_predict(X_train, y_train, X_test, y_test):
            return module.run_model(
                X_train, y_train, X_test, y_test,
                threshold=beta, alpha=alpha,
                metric=metric, threshold_mode=threshold_mode, n_jobs=n_jobs,
            )

    elif model in BASELINE_MODELS:
        def fit_predict(X_train, y_train, X_test, y_test):
            return baselines.run_model(X_train, y_train, X_test, y_test, model_name=model)

    else:
        raise ValueError(f"Unknown model {model!r}, expected one of "
                          f"{sorted(set(GRS3WDC_MODELS) | BASELINE_MODELS)}")

    return repeated_holdout(fit_predict, X, y, n_runs=n_runs, test_size=test_size, seed_base=seed_base).mean


def append_result_row(
    dataset: str, model_name: str, result: EvaluationResult, results_dir: Path = DEFAULT_RESULTS_DIR
) -> Path:
    """Append one row to results/<dataset>_res.csv in the format expected by generate_figures.py."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}_res.csv"
    row = pd.DataFrame(
        [[model_name, result.accuracy, result.precision, result.recall, result.f1]],
        columns=["Models", "Accuracy", "Precision", "Recall", "F1"],
    )
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["Models"] != model_name]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(out_path, index=False)
    return out_path
