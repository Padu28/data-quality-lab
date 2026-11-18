"""Repair strategies: what to do once a data problem is found.

Label issues  -> drop the flagged rows (prune) or keep everything (no-op baseline).
Missing cells -> one of several imputers, optionally with missing-indicator columns appended.
"""
from __future__ import annotations

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion

IMPUTERS = ["mean", "median", "knn", "iterative"]


def make_imputer(name: str, add_indicator: bool = False, seed: int = 0):
    """Build an imputer by name.

    Experiments standardise features before corrupting them, so KNN distances are not dominated by
    one large-scale column and imputation error is comparable across columns (units = std devs).
    """
    if name == "mean":
        imp = SimpleImputer(strategy="mean")
    elif name == "median":
        imp = SimpleImputer(strategy="median")
    elif name == "knn":
        imp = KNNImputer(n_neighbors=5)
    elif name == "iterative":
        imp = IterativeImputer(max_iter=10, random_state=seed)
    else:
        raise ValueError(f"unknown imputer: {name}")
    if add_indicator:
        return FeatureUnion([("impute", imp), ("indicator", MissingIndicator(features="all"))])
    return imp


def prune(X: np.ndarray, y: np.ndarray, flagged: np.ndarray):
    """Drop flagged rows. Never drops a whole class, so the retrained model can still predict it."""
    keep = ~flagged
    for c in np.unique(y):
        if not np.any(keep & (y == c)):
            keep[y == c] = True
    return X[keep], y[keep]
