"""Finding label errors without knowing which labels are wrong.

The approach follows the idea behind confident learning (Northcutt et al., 2021): train with
K-fold cross-validation so every sample gets an out-of-fold predicted probability, then flag a
sample when the probability of its *given* label falls below that class's average self-confidence.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def out_of_fold_proba(model, X, y, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    k = int(y.max()) + 1
    proba = np.zeros((len(y), k))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        proba[np.ix_(va, m.classes_)] = p
    return proba


def flag_label_issues(proba: np.ndarray, y: np.ndarray, method: str = "class_threshold") -> np.ndarray:
    """Return a boolean mask of samples whose label looks wrong.

    argmax: model's top prediction disagrees with the given label (simple baseline).
    class_threshold: confident learning. Each class j gets a threshold t_j = mean predicted
        probability of j over samples labelled j. A sample counts as "confidently" class j when
        p_j >= t_j. A sample is flagged when it is confidently some class other than its given
        label (an off-diagonal entry of the confident joint). Per-class thresholds make this robust
        to classes the model is systematically more or less confident about, which a single argmax
        rule ignores.
    """
    n, k = proba.shape
    if method == "argmax":
        return proba.argmax(axis=1) != y
    if method == "class_threshold":
        thresholds = class_thresholds(proba, y)
        confident = proba >= thresholds[None, :]           # n x k: confidently class j?
        # a sample confident in several classes is assigned to the most probable of them;
        # a sample confident in none is left alone
        assigned = np.where(confident, proba, -1.0).argmax(axis=1)
        return confident.any(axis=1) & (assigned != y)
    raise ValueError(method)


def class_thresholds(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """t_j = average self-confidence of class j (falls back to 1/k for an empty class)."""
    k = proba.shape[1]
    return np.array([proba[y == c, c].mean() if np.any(y == c) else 1.0 / k for c in range(k)])


def precision_recall(flagged: np.ndarray, truth: np.ndarray):
    tp = np.sum(flagged & truth)
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max(truth.sum(), 1)
    return float(precision), float(recall)
