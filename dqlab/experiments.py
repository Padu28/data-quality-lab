"""The three experiments behind the report.

Every run follows the same protocol:
  1. stratified train/test split; the test set is never corrupted for label experiments,
     so accuracy always measures performance against the true labels
  2. corrupt the training data in a controlled way and keep the ground-truth mask
  3. apply a detection / repair method, retrain, evaluate
  4. repeat over several seeds so every number has a confidence interval
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import corrupt, detect, repair


def make_model(name: str, seed: int = 0):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
    raise ValueError(f"unknown model: {name}")


def _split(X, y, seed):
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def _fit_score(model, X_tr, y_tr, X_te, y_te) -> float:
    m = clone(model).fit(X_tr, y_tr)
    return accuracy_score(y_te, m.predict(X_te))


def _prf(pred_mask, true_mask):
    if not true_mask.any():
        return np.nan, np.nan, np.nan
    p, r, f, _ = precision_recall_fscore_support(true_mask, pred_mask, average="binary", zero_division=0)
    return p, r, f


# ---------------------------------------------------------------- experiment 1: label noise
def label_noise(X, y, dataset, model_name, rates, seeds, n_splits=5):
    """Noisy baseline vs. two detectors (+ prune) vs. an oracle that knows the true noise."""
    rows = []
    k = int(y.max()) + 1
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_tr, X_te, y_tr, y_te = _split(X, y, seed)
        model = make_model(model_name, seed)
        clean_acc = _fit_score(model, X_tr, y_tr, X_te, y_te)
        for rate in rates:
            y_noisy, noise_mask = corrupt.flip_labels(y_tr, rate, rng, k)
            base = dict(dataset=dataset, model=model_name, seed=seed, noise_rate=rate, clean_acc=clean_acc)

            t0 = time.perf_counter()
            noisy_acc = _fit_score(model, X_tr, y_noisy, X_te, y_te)
            rows.append({**base, "method": "no_repair", "test_acc": noisy_acc, "flagged": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan,
                         "seconds": time.perf_counter() - t0})

            Xo, yo = repair.prune(X_tr, y_noisy, noise_mask)
            rows.append({**base, "method": "oracle_prune", "test_acc": _fit_score(model, Xo, yo, X_te, y_te),
                         "flagged": int(noise_mask.sum()), "precision": 1.0 if rate else np.nan,
                         "recall": 1.0 if rate else np.nan, "f1": 1.0 if rate else np.nan, "seconds": 0.0})

            t0 = time.perf_counter()
            proba = detect.out_of_fold_proba(model, X_tr, y_noisy, n_splits=n_splits, seed=seed)
            oof_time = time.perf_counter() - t0
            for method in ["argmax", "class_threshold"]:
                t0 = time.perf_counter()
                flagged = detect.flag_label_issues(proba, y_noisy, method)
                Xp, yp = repair.prune(X_tr, y_noisy, flagged)
                acc = _fit_score(model, Xp, yp, X_te, y_te)
                p, r, f = _prf(flagged, noise_mask)
                rows.append({**base, "method": f"{method}_prune", "test_acc": acc,
                             "flagged": int(flagged.sum()), "precision": p, "recall": r, "f1": f,
                             "seconds": oof_time + time.perf_counter() - t0})
    return rows


# ---------------------------------------------------------------- experiment 2: missing values
def missing_values(X, y, dataset, rates, seeds, mechanisms=("MCAR", "MNAR")):
    """Imputer x mechanism x rate, with and without missing-indicator features.

    Missingness is applied to train AND test, because in production the model meets incomplete
    rows at inference time too. Imputers are fit on train only.
    """
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_tr, X_te, y_tr, y_te = _split(X, y, seed)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        lr = make_model("logreg", seed)
        clean_acc = _fit_score(lr, X_tr, y_tr, X_te, y_te)
        for mech in mechanisms:
            for rate in rates:
                Xm_tr, m_tr = corrupt.mask_values(X_tr, rate, rng, mech)
                Xm_te, _ = corrupt.mask_values(X_te, rate, rng, mech)
                base = dict(dataset=dataset, seed=seed, mechanism=mech, missing_rate=rate, clean_acc=clean_acc)

                hgb = make_model("hgb", seed)  # handles NaN natively: a no-imputation reference
                rows.append({**base, "imputer": "none (HGB native NaN)", "indicator": False,
                             "test_acc": _fit_score(hgb, Xm_tr, y_tr, Xm_te, y_te), "impute_rmse": np.nan})

                for name in repair.IMPUTERS:
                    for ind in (False, True):
                        imp = repair.make_imputer(name, add_indicator=ind, seed=seed)
                        Z_tr = imp.fit_transform(Xm_tr)
                        Z_te = imp.transform(Xm_te)
                        rmse = np.nan
                        if not ind:
                            rmse = float(np.sqrt(np.mean((Z_tr[m_tr] - X_tr[m_tr]) ** 2))) if m_tr.any() else 0.0
                        rows.append({**base, "imputer": name, "indicator": ind,
                                     "test_acc": _fit_score(lr, Z_tr, y_tr, Z_te, y_te), "impute_rmse": rmse})
    return rows


# ---------------------------------------------------------------- experiment 3: ablation
def detector_ablation(X, y, dataset, model_name, rate, seeds, folds=(2, 3, 5, 10)):
    """How many cross-validation folds does confident learning need? More folds = each
    out-of-fold model sees more data, but costs proportionally more compute."""
    rows = []
    k = int(y.max()) + 1
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_tr, X_te, y_tr, y_te = _split(X, y, seed)
        y_noisy, noise_mask = corrupt.flip_labels(y_tr, rate, rng, k)
        model = make_model(model_name, seed)
        for n_splits in folds:
            t0 = time.perf_counter()
            proba = detect.out_of_fold_proba(model, X_tr, y_noisy, n_splits=n_splits, seed=seed)
            flagged = detect.flag_label_issues(proba, y_noisy, "class_threshold")
            secs = time.perf_counter() - t0
            p, r, f = _prf(flagged, noise_mask)
            Xp, yp = repair.prune(X_tr, y_noisy, flagged)
            rows.append(dict(dataset=dataset, model=model_name, seed=seed, noise_rate=rate, folds=n_splits,
                             precision=p, recall=r, f1=f, seconds=secs,
                             test_acc=_fit_score(model, Xp, yp, X_te, y_te)))
    return rows


# ---------------------------------------------------------------- error analysis
def error_analysis(X, y, dataset, model_name, rate, seed=0, n_splits=5):
    """Per-class view of what the detector misses, plus the confidence profile of its mistakes."""
    k = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    X_tr, _, y_tr, _ = _split(X, y, seed)
    y_noisy, noise_mask = corrupt.flip_labels(y_tr, rate, rng, k)
    proba = detect.out_of_fold_proba(make_model(model_name, seed), X_tr, y_noisy, n_splits=n_splits, seed=seed)
    flagged = detect.flag_label_issues(proba, y_noisy, "class_threshold")
    self_conf = proba[np.arange(len(y_noisy)), y_noisy]
    true_conf = proba[np.arange(len(y_tr)), y_tr]

    per_class = []
    for c in range(k):
        in_c = y_tr == c  # group by TRUE class: which real classes hide their noise best?
        noisy_c = noise_mask & in_c
        caught = flagged & noisy_c
        per_class.append(dict(dataset=dataset, true_class=c, noisy=int(noisy_c.sum()),
                              caught=int(caught.sum()),
                              recall=caught.sum() / noisy_c.sum() if noisy_c.any() else np.nan,
                              false_alarms=int((flagged & ~noise_mask & (y_noisy == c)).sum())))
    groups = {
        "caught (true positive)": flagged & noise_mask,
        "missed (false negative)": ~flagged & noise_mask,
        "false alarm (false positive)": flagged & ~noise_mask,
        "clean, not flagged": ~flagged & ~noise_mask,
    }
    confidence = [dict(dataset=dataset, group=g, count=int(m.sum()),
                       mean_given_label_conf=float(self_conf[m].mean()) if m.any() else np.nan,
                       mean_true_label_conf=float(true_conf[m].mean()) if m.any() else np.nan)
                  for g, m in groups.items()]
    return per_class, confidence
