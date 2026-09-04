# -*- coding: utf-8 -*-
"""Tests for the vectorised prefix walk added in 0.1.6.

The change replaces the inner loop of every transform, so the tests that matter
are equivalence tests: the prefix form must agree with the value functions it
stands in for, the reference path must still reproduce 0.1.5 exactly, and
additivity must survive.

Numerical tolerances here are deliberately loose in absolute terms and tight
relative to what the estimator can resolve. A Shapley value from M permutations
carries Monte Carlo error of order 1/sqrt(M) -- about 1e-1 of a column's spread
at M=1024 -- so a 1e-9 disagreement is irrelevant, while a 1e-2 one is not.
"""
import numpy as np
import pytest

from shapley_behaviors import ShapleyBehaviors, prefix_values

FUNCTIONS = ("mean", "variance", "skewness", "kurtosis", "entropy")


def batch_prefix(sb, x, name):
    """What the value function itself says, on every prefix."""
    vf = sb.get_value_function(name)
    return np.array([vf(np.asarray(x)[:j + 1]) for j in range(len(x))])


@pytest.fixture
def sb():
    return ShapleyBehaviors(n_permutations=8, n_jobs=1, random_state=0)


# --------------------------------------------------------------- equivalence
@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("label", [
    "gaussian", "lognormal", "with_outlier", "with_nan", "constant",
    "descending", "single", "pair", "leading_nan",
])
def test_prefix_matches_value_function(sb, name, label):
    rng = np.random.RandomState(3)
    data = {
        "gaussian": rng.randn(60),
        "lognormal": np.exp(rng.randn(60)),
        "with_outlier": np.r_[rng.randn(59), 30.0],
        "with_nan": np.r_[rng.randn(45), np.full(15, np.nan)][rng.permutation(60)],
        "constant": np.full(20, 2.5),
        "descending": np.sort(rng.randn(40))[::-1].copy(),
        "single": np.array([4.0]),
        "pair": np.array([1.0, 3.0]),
        "leading_nan": np.r_[np.nan, np.nan, rng.randn(20)],
    }[label]
    got = prefix_values(data, name)
    ref = batch_prefix(sb, data, name)
    scale = max(float(np.abs(ref).max()), 1.0)
    assert got.shape == ref.shape
    assert np.all(np.isfinite(got))
    assert np.abs(got - ref).max() / scale < 1e-9


def test_all_missing_gives_the_empty_set_value():
    """A column with nothing observed must not produce NaN."""
    for name in FUNCTIONS:
        got = prefix_values(np.full(7, np.nan), name)
        assert np.array_equal(got, np.zeros(7))


def test_missing_entries_repeat_the_previous_value():
    """Adding an unobserved sample cannot change a statistic."""
    x = np.array([1.0, np.nan, 4.0, np.nan, np.nan, 9.0])
    got = prefix_values(x, "variance")
    assert got[1] == got[0]
    assert got[3] == got[2] == pytest.approx(np.var([1.0, 4.0]))
    assert got[4] == got[3]


def test_pair_kurtosis_is_exactly_minus_two():
    """Two distinct values have excess kurtosis -2 and skewness 0, always.

    This is the case that exposed the power-sum cancellation: two nearly-equal
    values far from the centring point gave -2.0105 before the repair.
    """
    for offset in (0.0, 1.0, 1e3, 1e6):
        for gap in (1.0, 1e-3, 1e-6):
            x = np.array([offset, offset + gap])
            assert prefix_values(x, "kurtosis")[1] == pytest.approx(-2.0, abs=1e-6)
            assert prefix_values(x, "skewness")[1] == pytest.approx(0.0, abs=1e-6)


def test_unknown_function_is_rejected():
    with pytest.raises(ValueError):
        prefix_values(np.arange(5.0), "median")


# ------------------------------------------------- badly scaled columns (fix)
def _exact_moment_statistic(values, name):
    """The statistic of these exact float64 values, at 60 significant digits.

    The reference cannot be the statistic of the pattern the values were built
    from: at 1e8 + 1e-6 * pattern, ulp(1e8) is 1.5e-8 against increments of
    1e-6, so the stored doubles genuinely no longer represent that pattern to
    better than a percent. Anchoring to Decimal on the values as stored tests
    the arithmetic rather than the data's resolution.
    """
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    vals = [Decimal(float(v)) for v in values]
    k = Decimal(len(vals))
    mu = sum(vals) / k
    d = [v - mu for v in vals]
    m2 = sum(x * x for x in d) / k
    if name == "variance":
        return float(m2)
    if name == "skewness":
        return float((sum(x ** 3 for x in d) / k) / (m2 * m2.sqrt()))
    return float((sum(x ** 4 for x in d) / k) / (m2 * m2) - 3)


@pytest.mark.parametrize("name", ["variance", "skewness", "kurtosis"])
@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (1e6, 1e-3), (1e8, 1e-6)])
def test_offset_column_is_accurate(name, offset, scale):
    """A narrow spread on a large offset used to lose most of its digits.

    Before 0.1.6 the value functions took deviations about a mean computed at
    the offset's magnitude, so the rounding in that mean was a large fraction of
    a deviation: 16% relative error on skewness at 1e8 +/- 1e-6.
    """
    pattern = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 2.5])
    x = offset + scale * pattern
    got = prefix_values(x, name)[-1]
    assert got == pytest.approx(_exact_moment_statistic(x, name), rel=1e-9)


# ---------------------------------------------------- integration: transform
@pytest.mark.parametrize("name", FUNCTIONS)
def test_incremental_agrees_with_reference_path(name):
    rng = np.random.RandomState(7)
    X = np.where(rng.rand(50, 4) < 0.1, np.nan, rng.randn(50, 4))
    kw = dict(n_permutations=32, n_jobs=1, random_state=5)
    ref = ShapleyBehaviors(incremental=False, **kw).transform(X, name, verbose=False)
    inc = ShapleyBehaviors(incremental=True, **kw).transform(X, name, verbose=False)
    spread = np.maximum(ref.max(axis=0) - ref.min(axis=0), 1e-12)
    assert np.max(np.abs(ref - inc) / spread) < 1e-8


@pytest.mark.parametrize("name", FUNCTIONS)
def test_additivity(name):
    """Shapley values must sum to the value of the full set."""
    rng = np.random.RandomState(2)
    X = rng.randn(40, 3)
    sb = ShapleyBehaviors(n_permutations=64, n_jobs=1, random_state=1)
    phi = sb.transform(X, name, verbose=False)
    vf = sb.get_value_function(name)
    total = np.array([vf(X[:, j]) for j in range(X.shape[1])])
    assert np.abs(phi.sum(axis=0) - total).max() / max(np.abs(total).max(), 1e-12) < 1e-10


def test_reference_path_is_reachable_from_convenience_functions():
    """behavioral_space_explorer only ever calls these, so the escape hatch
    has to exist here too."""
    from shapley_behaviors import compute_shapley_variance
    rng = np.random.RandomState(0)
    X = rng.randn(30, 2)
    a = compute_shapley_variance(X, n_permutations=8, n_jobs=1, incremental=False)
    b = compute_shapley_variance(X, n_permutations=8, n_jobs=1, incremental=True)
    assert a.shape == b.shape
    spread = max(np.ptp(a), 1e-12)
    assert np.abs(a - b).max() / spread < 1e-8


def test_custom_value_function_still_works():
    """A callable with no prefix form must fall back to the original walk."""
    sb = ShapleyBehaviors(n_permutations=8, n_jobs=1, random_state=0)
    x = np.random.RandomState(1).randn(20)
    out = sb._compute_shapley_column(x, sb.variance_function, 8, None)
    assert out.shape == (20,)
    assert np.isfinite(out).all()


def test_odd_permutation_count_still_normalises_correctly():
    """Guards the 0.1.3 antithetic normalisation fix against the new path."""
    rng = np.random.RandomState(4)
    X = rng.randn(30, 2)
    sb = ShapleyBehaviors(n_permutations=25, n_jobs=1, random_state=1)
    phi = sb.transform(X, "variance", verbose=False)
    total = np.array([sb.variance_function(X[:, j]) for j in range(2)])
    assert np.abs(phi.sum(axis=0) - total).max() < 1e-10
