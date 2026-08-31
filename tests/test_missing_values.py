"""Regression tests for the 0.1.5 missing-value fix.

The bug these exist to catch was **silent**: `entropy_function` returned a finite
0.0 for any subset containing a NaN, so nothing downstream could detect it. A
test is the only thing that would have caught it, which is why one ships now.
"""

import numpy as np
import pytest

from shapley_behaviors import ShapleyBehaviors

FUNCTIONS = ["variance", "mean", "skewness", "kurtosis", "entropy"]


@pytest.fixture
def sb():
    return ShapleyBehaviors(n_permutations=16, n_jobs=1, random_state=0)


def test_entropy_does_not_silently_return_zero(sb):
    """The 0.1.4 failure mode, pinned so it cannot come back unnoticed."""
    value = sb.entropy_function(np.array([1.0, 2.0, np.nan, 4.0]))
    assert value > 0.0, "entropy of a subset with one NaN collapsed to zero again"
    assert np.isclose(value, sb.entropy_function(np.array([1.0, 2.0, 4.0])))


@pytest.mark.parametrize("name", FUNCTIONS)
def test_value_functions_ignore_missing(sb, name):
    """Each function must equal the same computation on the observed values."""
    fn = sb.get_value_function(name)
    with_gap = np.array([1.0, 2.0, np.nan, 4.0, 7.0])
    without = np.array([1.0, 2.0, 4.0, 7.0])
    assert np.isclose(fn(with_gap), fn(without))
    assert np.isfinite(fn(with_gap))


@pytest.mark.parametrize("name", FUNCTIONS)
def test_all_missing_returns_zero(sb, name):
    """No observed values is the same case as an empty subset."""
    fn = sb.get_value_function(name)
    assert fn(np.array([np.nan, np.nan])) == 0.0


@pytest.mark.parametrize("name", FUNCTIONS)
def test_transform_finite_and_informative_with_gaps(sb, name):
    rng = np.random.RandomState(0)
    X = rng.rand(20, 8)
    X[rng.rand(*X.shape) < 0.2] = np.nan
    out = np.asarray(sb.transform(X, name, verbose=False))
    assert np.isfinite(out).all()
    assert np.abs(out).sum() > 0, "attribution collapsed to all zeros"


@pytest.mark.parametrize("name", FUNCTIONS)
def test_complete_data_is_unchanged(sb, name):
    """The fix must be additive: complete input takes the original path."""
    fn = sb.get_value_function(name)
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    expected = {
        "variance": np.var(x, ddof=0),
        "mean": np.mean(x),
    }
    if name in expected:
        assert np.isclose(fn(x), expected[name])
    assert np.isfinite(fn(x))
