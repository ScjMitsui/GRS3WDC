"""GRS3WDC-MC: three-way decision classifier over a maximal-clique-induced
granular space (paper Section 3.4, Definition 17-20).

Consolidates archive/legacy_modules/twdmaximal.py, twdmp.py and twdmpchev.py,
which differed only in distance metric (Euclidean/Chebyshev) and whether the
distance cutoff was given as an absolute delta or a percentile (beta) of pairwise
training distances -- both now selected via the `metric` / `threshold_mode` args.
"""

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from grs3wdc.distances import pairwise_distance_matrix, resolve_delta
from grs3wdc.evaluation import EvaluationResult, summarize_three_way


def _distance_to_point(X: np.ndarray, point: np.ndarray, metric: str) -> np.ndarray:
    if metric == "chebyshev":
        return np.max(np.abs(X - point), axis=1)
    return np.linalg.norm(X - point, axis=1)


def find_maximal_cliques(adjacency: np.ndarray) -> list:
    graph = nx.Graph(adjacency)
    return list(nx.find_cliques(graph))


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
    """Fit maximal-clique granules on the training set and classify the test set.

    `threshold` is either an absolute distance cutoff (threshold_mode="absolute")
    or a percentile of pairwise training distances (threshold_mode="percentile"),
    matching the paper's theta parameterization via beta in Section 5.2.
    """
    dis_arr = pairwise_distance_matrix(X_train, metric=metric)
    delta = resolve_delta(dis_arr, threshold, threshold_mode)
    adjacency = (dis_arr < delta).astype(int)
    cliques = find_maximal_cliques(adjacency)

    def filter_clique(clique) -> Optional[tuple]:
        counts = pd.Series(y_train[clique]).value_counts()
        qualifying = counts[counts > alpha * len(clique)]
        return (clique, qualifying.index[0]) if not qualifying.empty else None

    filtered = [res for res in Parallel(n_jobs=n_jobs)(
        delayed(filter_clique)(clique) for clique in cliques
    ) if res is not None]
    filtered_cliques = [clique for clique, _ in filtered]
    clique_labels = [label for _, label in filtered]

    def process_test_sample(i: int, t: np.ndarray):
        # A test point satisfies a clique's granule description only if it is
        # within delta of every member of the clique (Definition 19).
        matched_labels = []
        for clique, label in zip(filtered_cliques, clique_labels):
            distances = _distance_to_point(X_train[clique], t, metric)
            if np.all(distances < delta):
                matched_labels.append(label)
        if matched_labels and all(label == matched_labels[0] for label in matched_labels):
            return y_test[i], matched_labels[0]
        return y_test[i], None

    test_results = Parallel(n_jobs=n_jobs)(
        delayed(process_test_sample)(i, t) for i, t in enumerate(X_test)
    )

    y_true_filtered = [true for true, pred in test_results if pred is not None]
    y_pred_filtered = [pred for true, pred in test_results if pred is not None]
    return summarize_three_way(y_true_filtered, y_pred_filtered, len(y_test))
