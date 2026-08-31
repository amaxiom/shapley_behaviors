# Changelog

## 0.1.5

### Fixed

- **Missing values no longer corrupt the value functions.** All five
  (`variance`, `mean`, `skewness`, `kurtosis`, `entropy`) now reduce over
  observed values only, via a new `ShapleyBehaviors._observed` helper, and
  return `0.0` when a subset has none, matching the existing empty-input
  convention.

  Before this release, `variance`, `mean`, `skewness` and `kurtosis` propagated
  `NaN` through to the returned Shapley matrix.

  **`entropy_function` failed silently instead, which is worse.** For any subset
  containing a `NaN` it returned a finite `0.0`: `np.min` gave `NaN`, so every
  probability became `NaN`, and the `probs > 0` filter then removed all of them,
  leaving the empty-input branch. A propagated `NaN` is visible to the caller; a
  zero is not. **Entropy results computed with 0.1.4 or earlier on data
  containing missing values should be recomputed.**

  `entropy_function([1, 2, nan, 4])` now returns `0.8113`; it previously
  returned `0.0`.

- `entropy_function` also guards against a non-positive normalising total rather
  than dividing by it.
