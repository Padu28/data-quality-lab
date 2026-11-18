"""Small statistics helpers so every comparison in the report comes with uncertainty."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95):
    """Mean and t-distribution confidence half-width across seeds."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    half = stats.t.ppf((1 + confidence) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def paired_test(a, b):
    """Paired t-test p-value for 'method a differs from method b' over matched seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)
