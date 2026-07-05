"""GRS3WDC-N: three-way decision classifier over a neighborhood-induced granular
space (paper Section 3.3, Definition 12-15).

Consolidates archive/legacy_modules/twdneighborhood.py, twdnp.py and twdnpchev.py,
which differed only in distance metric (Euclidean/Chebyshev) and whether the
distance cutoff was given as an absolute delta or a percentile (beta) of pairwise
training distances -- both now selected via the `metric` / `threshold_mode` args.
"""

from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from grs3wdc.distances import pairwise_distance_matrix, resolve_delta
from grs3wdc.evaluation import EvaluationResult, summarize_three_way


def _distance_to_point(X: np.ndarray, point: np.ndarray, metric: str) -> np.ndarray:
    if metric == "chebyshev":
        return np.max(np.abs(X - point), axis=1)
    return np.linalg.norm(X - point, axis=1)


def run_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    alpha: float,
    metric: str = "euclidean",
    threshold_mode: str = "percentile",
    n_jobs: int = -1,
) -> EvaluationResult:
    """Fit neighborhood granules on the training set and classify the test set.

    `threshold` is either an absolute distance cutoff (threshold_mode="absolute")
    or a percentile of pairwise training distances (threshold_mode="percentile"),
    matching the paper's theta parameterization via beta in Section 5.2.
    """
    n = X_train.shape[0]
    dis_arr = pairwise_distance_matrix(X_train, metric=metric)
    delta = resolve_delta(dis_arr, threshold, threshold_mode)

    def process_training_sample(i: int) -> Optional[int]:
        neighbors = np.where(dis_arr[i] < delta)[0]
        counts = pd.Series(y_train[neighbors]).value_counts()
        qualifying = counts[counts > alpha * len(neighbors)]
        return i if not qualifying.empty else None

    results = Parallel(n_jobs=n_jobs)(delayed(process_training_sample)(i) for i in range(n))
    granule_centers = np.array([i for i in results if i is not None])

    def process_test_sample(i: int, t: np.ndarray):
        # Prediction uses the granule center's own label, not the granule's
        # aggregated majority label -- this matches the original twdnp.py
        # implementation used to produce the paper's Table 9/10 results.
        if granule_centers.size == 0:
            return y_test[i], None
        dist_to_centers = _distance_to_point(X_train[granule_centers], t, metric)
        matched_labels = y_train[granule_centers[dist_to_centers < delta]].tolist()
        if matched_labels and all(label == matched_labels[0] for label in matched_labels):
            return y_test[i], matched_labels[0]
        return y_test[i], None

    test_results = Parallel(n_jobs=n_jobs)(
        delayed(process_test_sample)(i, t) for i, t in enumerate(X_test)
    )

    y_true_filtered = [true for true, pred in test_results if pred is not None]
    y_pred_filtered = [pred for true, pred in test_results if pred is not None]
    return summarize_three_way(y_true_filtered, y_pred_filtered, len(y_test))
