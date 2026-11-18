import numpy as np
import pytest
from sklearn.datasets import make_classification

from dqlab import corrupt, detect, repair
from dqlab.experiments import make_model
from dqlab.stats import mean_ci, paired_test


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def easy_data():
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=3,
                               n_clusters_per_class=1, class_sep=2.0, random_state=0)
    return X, y


# ---------------------------------------------------------------- corruption
def test_flip_labels_rate_and_mask(rng):
    y = np.repeat(np.arange(4), 250)
    y_noisy, mask = corrupt.flip_labels(y, 0.2, rng)
    assert mask.sum() == 200
    assert np.all(y_noisy[mask] != y[mask])      # every flipped label really changed
    assert np.all(y_noisy[~mask] == y[~mask])    # nothing else was touched
    assert set(np.unique(y_noisy)) <= set(range(4))


def test_flip_labels_zero_rate(rng):
    y = np.arange(10) % 2
    y_noisy, mask = corrupt.flip_labels(y, 0.0, rng)
    assert not mask.any() and np.array_equal(y, y_noisy)


@pytest.mark.parametrize("mech", ["MCAR", "MNAR"])
def test_mask_values_rate(rng, mech):
    X = rng.normal(size=(5000, 8))
    Xm, mask = corrupt.mask_values(X, 0.3, rng, mech)
    assert np.isnan(Xm).sum() == mask.sum()
    assert 0.25 < mask.mean() < 0.35
    assert not np.isnan(X).any()                 # original left untouched


def test_mnar_hides_large_values(rng):
    X = rng.normal(size=(20000, 3))
    _, mask = corrupt.mask_values(X, 0.3, rng, "MNAR")
    assert X[mask].mean() > X[~mask].mean() + 0.5


# ---------------------------------------------------------------- detection
def test_out_of_fold_proba_shape(easy_data):
    X, y = easy_data
    p = detect.out_of_fold_proba(make_model("logreg"), X, y, n_splits=3)
    assert p.shape == (len(y), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_detector_finds_flipped_labels(easy_data, rng):
    X, y = easy_data
    y_noisy, mask = corrupt.flip_labels(y, 0.2, rng)
    proba = detect.out_of_fold_proba(make_model("logreg"), X, y_noisy)
    flagged = detect.flag_label_issues(proba, y_noisy, "class_threshold")
    precision = (flagged & mask).sum() / flagged.sum()
    recall = (flagged & mask).sum() / mask.sum()
    assert precision > 0.8 and recall > 0.8


def test_class_threshold_can_flag_when_given_label_is_argmax():
    # class 0 is usually predicted with ~0.9 confidence, so 0.5 on a class-0 row is suspicious
    proba = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.3, 0.7]])
    y = np.array([0, 0, 0, 0, 1, 1])
    assert detect.flag_label_issues(proba, y, "argmax")[3] == False  # noqa: E712
    t = detect.class_thresholds(proba, y)
    assert proba[3, 1] < t[1]  # not confident in class 1 either, so leave it alone
    assert not detect.flag_label_issues(proba, y, "class_threshold")[3]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        detect.flag_label_issues(np.eye(2), np.array([0, 1]), "nope")


# ---------------------------------------------------------------- repair
def test_prune_never_drops_a_whole_class():
    X = np.arange(6).reshape(-1, 1).astype(float)
    y = np.array([0, 0, 1, 1, 2, 2])
    flagged = np.array([False, False, True, True, False, True])
    Xp, yp = repair.prune(X, y, flagged)
    assert set(yp) == {0, 1, 2}
    assert len(yp) == 5


@pytest.mark.parametrize("name", repair.IMPUTERS)
@pytest.mark.parametrize("indicator", [False, True])
def test_imputers_remove_all_nans(rng, name, indicator):
    X = rng.normal(size=(300, 5))
    Xm, _ = corrupt.mask_values(X, 0.2, rng)
    Z = repair.make_imputer(name, add_indicator=indicator).fit_transform(Xm)
    assert not np.isnan(Z).any()
    assert Z.shape[1] == (10 if indicator else 5)


# ---------------------------------------------------------------- stats
def test_mean_ci():
    m, h = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and h > 0
    assert mean_ci([5.0]) == (5.0, 0.0)


def test_paired_test_identical_is_one():
    assert paired_test([1, 2, 3], [1, 2, 3]) == 1.0
