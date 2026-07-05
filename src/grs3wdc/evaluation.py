"""Shared evaluation harness: every notebook in the original project re-implemented
this train/test-split loop and mean/std aggregation by hand."""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split


class EvaluationResult(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    coverage: float = 1.0


FitPredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], EvaluationResult]


@dataclass
class SummaryStats:
    mean: EvaluationResult
    std: EvaluationResult
    raw: pd.DataFrame


def repeated_holdout(
    fit_predict_fn: FitPredictFn,
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = 20,
    test_size: float = 0.2,
    seed_base: int = 42,
) -> SummaryStats:
    """Run `fit_predict_fn` over `n_runs` random train/test splits and aggregate metrics."""
    rows = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed_base + run
        )
        result = fit_predict_fn(X_train, y_train, X_test, y_test)
        rows.append(result._asdict())

    raw = pd.DataFrame(rows)
    mean = EvaluationResult(**raw.mean().to_dict())
    std = EvaluationResult(**raw.std().to_dict())
    return SummaryStats(mean=mean, std=std, raw=raw)


def summarize_three_way(
    y_true_filtered: Sequence, y_pred_filtered: Sequence, n_test: int
) -> EvaluationResult:
    """Aggregate metrics for a three-way classifier that abstains (predicts None) on
    undecided instances -- shared by the neighborhood- and maximal-clique-induced models.

    Recall and F1 are computed relative to the full test set (accuracy * coverage),
    not just the covered subset, since abstained instances count against recall.
    """
    if not y_pred_filtered:
        return EvaluationResult(0.0, 0.0, 0.0, 0.0, 0.0)
    coverage = len(y_pred_filtered) / n_test
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)
    recall = accuracy * coverage
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return EvaluationResult(accuracy, precision, recall, f1, coverage)
