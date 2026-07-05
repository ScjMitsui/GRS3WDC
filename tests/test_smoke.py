"""Smoke tests: confirm both granule models and all baselines run end-to-end on
synthetic data without error. Used to sanity-check the Docker build; these do
not assert against the paper's reported numbers (see experiments/run_experiment.py
for that -- it can be spot-checked manually against Table 9)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from grs3wdc import baselines
from grs3wdc.evaluation import EvaluationResult
from grs3wdc.granules import maximal_clique, neighborhood


@pytest.fixture
def synthetic_data():
    X, y = make_classification(n_samples=60, n_features=5, n_informative=3, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def _assert_valid(result: EvaluationResult):
    for value in result:
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev"])
def test_neighborhood_model(synthetic_data, metric):
    X_train, X_test, y_train, y_test = synthetic_data
    result = neighborhood.run_model(
        X_train, y_train, X_test, y_test, threshold=50, alpha=0.6, metric=metric, n_jobs=1
    )
    _assert_valid(result)


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev"])
def test_maximal_clique_model(synthetic_data, metric):
    X_train, X_test, y_train, y_test = synthetic_data
    result = maximal_clique.run_model(
        X_train, y_train, X_test, y_test, threshold=50, alpha=0.6, metric=metric, n_jobs=1
    )
    _assert_valid(result)


@pytest.mark.parametrize("model_name", sorted(baselines.BASELINE_FACTORIES))
def test_baseline_models(synthetic_data, model_name):
    X_train, X_test, y_train, y_test = synthetic_data
    result = baselines.run_model(X_train, y_train, X_test, y_test, model_name=model_name)
    _assert_valid(result)
