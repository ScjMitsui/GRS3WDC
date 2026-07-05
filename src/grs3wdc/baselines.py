"""Baseline classifiers compared against GRS3WDC in Table 9 (ported from archive/notebooks/baseline.ipynb)."""

from typing import Callable, Dict

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from grs3wdc.evaluation import EvaluationResult

BASELINE_FACTORIES: Dict[str, Callable] = {
    "svm": lambda: LinearSVC(),
    "decision_tree": lambda: DecisionTreeClassifier(criterion="gini"),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": lambda: GaussianNB(),
}


def run_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> EvaluationResult:
    if model_name not in BASELINE_FACTORIES:
        raise ValueError(f"Unknown baseline {model_name!r}, expected one of {sorted(BASELINE_FACTORIES)}")
    model = BASELINE_FACTORIES[model_name]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return EvaluationResult(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        coverage=1.0,
    )
