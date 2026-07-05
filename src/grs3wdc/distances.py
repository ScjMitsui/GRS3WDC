"""Distance helpers shared by the neighborhood- and maximal-clique-induced models."""

import numpy as np
from sklearn.metrics import pairwise_distances

SUPPORTED_METRICS = ("euclidean", "chebyshev")


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Minkowski distance of order p between two vectors, kept for reference (Eq. 3 in the paper)."""
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def pairwise_distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric {metric!r}, expected one of {SUPPORTED_METRICS}")
    return pairwise_distances(X, metric=metric)


def resolve_delta(dis_arr: np.ndarray, threshold: float, threshold_mode: str) -> float:
    """Convert a `threshold` into an absolute distance cutoff (delta, Eq. 31's theta).

    threshold_mode="percentile": `threshold` is a percentile (beta) of the pairwise
        training-set distances, as used for the paper's reported (theta, alpha) grid.
    threshold_mode="absolute": `threshold` is already an absolute distance cutoff.
    """
    if threshold_mode == "percentile":
        distances = dis_arr[np.triu_indices_from(dis_arr, k=1)]
        return np.percentile(distances, threshold)
    if threshold_mode == "absolute":
        return threshold
    raise ValueError(f"Unsupported threshold_mode {threshold_mode!r}, expected 'percentile' or 'absolute'")
