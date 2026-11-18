"""Controlled data corruption. Each function returns the corrupted copy plus a ground-truth mask,
so detection and repair methods can be scored against what was actually broken."""
from __future__ import annotations

import numpy as np


def flip_labels(y: np.ndarray, rate: float, rng: np.random.Generator, n_classes: int | None = None):
    """Uniform label noise: a `rate` fraction of labels is moved to a different random class."""
    y = y.copy()
    k = n_classes or int(y.max()) + 1
    n_flip = int(round(rate * len(y)))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    offsets = rng.integers(1, k, size=n_flip)  # never maps a label to itself
    y[idx] = (y[idx] + offsets) % k
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return y, mask


def mask_values(X: np.ndarray, rate: float, rng: np.random.Generator, mechanism: str = "MCAR"):
    """Blank out a `rate` fraction of cells.

    MCAR: missing completely at random.
    MNAR: missing not at random -- the largest values of each column are more likely to vanish,
          the harder case where naive imputation is biased.
    """
    X = X.copy()
    n, d = X.shape
    if mechanism == "MCAR":
        mask = rng.random((n, d)) < rate
    elif mechanism == "MNAR":
        ranks = X.argsort(axis=0).argsort(axis=0) / max(n - 1, 1)  # 0 = smallest, 1 = largest
        weights = ranks ** 3
        probs = weights / weights.mean(axis=0, keepdims=True) * rate
        mask = rng.random((n, d)) < np.clip(probs, 0, 1)
    else:
        raise ValueError(mechanism)
    X[mask] = np.nan
    return X, mask


def leak_duplicates(X_train, y_train, X_test, y_test, rate: float, rng: np.random.Generator,
                    jitter: float = 0.0):
    """Copy a `rate` fraction of test rows into the training set (optionally with tiny noise),
    reproducing the train/test leakage that happens when a dataset is scraped or merged twice."""
    n_leak = int(round(rate * len(X_test)))
    idx = rng.choice(len(X_test), size=n_leak, replace=False)
    leaked = X_test[idx] + jitter * rng.standard_normal(X_test[idx].shape) * X_test.std(axis=0)
    X_aug = np.vstack([X_train, leaked])
    y_aug = np.concatenate([y_train, y_test[idx]])
    return X_aug, y_aug, idx
