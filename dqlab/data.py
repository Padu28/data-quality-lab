"""Dataset loaders. Every loader returns (X, y, name) with X as float64 and y as int labels 0..k-1."""
from __future__ import annotations

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


def _encode(y):
    return LabelEncoder().fit_transform(y)


def load(name: str, n_synthetic: int = 100_000, seed: int = 0):
    """Load a dataset by name.

    breast_cancer, digits, wine: small real datasets bundled with scikit-learn (no download).
    synthetic: a large 10-class tabular problem from make_classification, used for scale tests.
    covtype: 581K-row Forest Covertype dataset (downloads ~11 MB on first use).
    """
    if name == "breast_cancer":
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif name == "digits":
        X, y = datasets.load_digits(return_X_y=True)
    elif name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
    elif name == "synthetic":
        X, y = datasets.make_classification(
            n_samples=n_synthetic, n_features=40, n_informative=20, n_redundant=10,
            n_classes=10, n_clusters_per_class=1, class_sep=1.5, random_state=seed,
        )
    elif name == "covtype":
        X, y = datasets.fetch_covtype(return_X_y=True)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return np.asarray(X, dtype=np.float64), _encode(y), name


ALL_DATASETS = ["breast_cancer", "digits", "wine", "synthetic"]
