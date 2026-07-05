"""Dataset loading for the six datasets reported in Table 8 of the paper.

Note: the local `image.csv` has 210 instances, not the 2310 reported in Table 8
(it matches only the small UCI Image Segmentation "test" partition, not the
combined train+test set used for the paper). Re-fetch the full UCI dataset if
you need to exactly reproduce that row of Table 9/10.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# name -> (csv filename, approximate (instances, features, classes) from Table 8)
DATASET_REGISTRY = {
    "sonar": ("sonar.csv", (208, 60, 2)),
    "breast": ("breast.csv", (569, 30, 2)),
    "banknote": ("banknote.csv", (1372, 4, 2)),
    "image": ("image.csv", (2310, 19, 7)),
    "phishing": ("phishing.csv", (11430, 87, 2)),
    "letter": ("letter.csv", (20000, 16, 26)),
}


class Dataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray


def load_dataset(name: str, data_dir: Path = DEFAULT_DATA_DIR, scale: bool = True) -> Dataset:
    """Load a registered dataset: last column is the label, all others are features."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}, expected one of {sorted(DATASET_REGISTRY)}")
    filename, _ = DATASET_REGISTRY[name]
    df = pd.read_csv(Path(data_dir) / filename)

    feature_cols = df.columns[:-1]
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    y = df.iloc[:, -1].to_numpy()
    X = df[feature_cols].to_numpy(dtype=float)
    if scale:
        X = MinMaxScaler().fit_transform(X)
    return Dataset(X=X, y=y)
