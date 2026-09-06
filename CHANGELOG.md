# Changelog

## Unreleased

### Fixed

- **`behavioral_space_explorer.py` never displayed a figure.** Display was
  guarded on whether a file was being written:

      if save_path:
          plt.savefig(...); plt.close()
      else:
          plt.show()

  All three call sites (label plots, feature-concentration plots, outlier
  profile plots) build a `save_path`, so the `else` branch was dead. Every
  figure went to disk and none appeared in the notebook. Saving and showing are
  independent choices and are now independent statements, with a `SHOW_PLOTS`
  setting (default `True`) to opt out. Verified on a synthetic run: 15 figures
  saved and 15 shown, against 15 saved and 0 shown before; `SHOW_PLOTS=False`
  still saves all 15.

- `behavioral_region_explorer.py` showed six of its seven figures; the seventh,
  `all_regions_in_original_space`, had its `plt.show()` commented out. It now
  displays with the rest.

- **`README.md` (repo root) carried 596 stray backslashes** in runs of up to
  seven, from being round-tripped through an escaping converter more than once.
  Inside the fenced code blocks a backslash is literal, so none of the
  quick-start snippets could be copied and run; the badge lines were escaped in
  prose, where escaping does apply, so they rendered as text instead of images;
  and the closing horizontal rule rendered as literal dashes. Every one of the
  596 preceded `_`, `[`, `.` or `-`, so none was load-bearing. The PyPI
  description is generated from `pypi_staging/README.md`, which was never
  affected.

## 0.1.6

### Changed

- **Each permutation is now walked in one vectorised pass instead of n value
  function calls, making `transform` 27x faster at n=100 and 206x at n=900.**
  The old walk called the value function on `X_col[perm[:j+1]]` for every j: a
  fancy-index copy plus a full reduction at each step, so one permutation cost
  O(n^2) and a transform cost O(M n^2 d) rather than the O(M n d) a description
  of the algorithm would suggest.

  All five statistics are functions of a growing prefix, so the whole
  permutation is computed at once (`prefix_values`, now exported). The four
  moment functions use cumulative power sums; entropy keeps a loop over blocks
  of constant running minimum, which is O(n log n) because a random permutation
  has only ~ln n prefix minima.

  **This changes results in the last few digits.** Pass `incremental=False` to
  `ShapleyBehaviors` or to any `compute_shapley_*` function to reproduce 0.1.5
  and earlier **bit-identically** -- verified against the 0.1.5 module itself on
  seven datasets and all five value functions. Both paths draw the same
  permutations, so they differ only in floating-point detail:

  - well-scaled data: at most 5e-11 of a column's spread,
  - the estimator's own Monte Carlo error at n_permutations=1024: about 1e-1 of
    a column's spread.

  The difference is therefore some 10 orders of magnitude below the noise floor
  of the estimate it belongs to. Additivity holds to 2e-14.

### Fixed

- **Columns whose offset dwarfs their spread were being decomposed
  inaccurately.** The value functions take deviations about a mean computed at
  the offset's magnitude, so at 1e8 +/- 1e-6 the rounding in that mean is a
  large fraction of a deviation. Measured against a 60-digit reference, prefix
  skewness carried **16% relative error** and variance 9e-4. The new path
  centres each column once before accumulating, which removes it: the same
  cases come back at 4e-16.

  Affected any feature held in absolute units with a narrow range -- a lattice
  parameter, a temperature in kelvin, a near-constant concentration. `mean` was
  never affected, and `entropy` only mildly.

  Note that `transform`'s additivity check compares against the batch value
  function, so on such a column the check now reports the *reference* total's
  error rather than the estimate's. Its warning says so.

- The additivity warning suggested only "increase n_permutations". It now also
  names the badly-scaled-column cause, which more permutations cannot fix.

- **`behavioral_cluster_explorer.py` no longer destroys a previous space's
  results in silence.** Output names key on `DATASET_NAME` and not on `SPACE`,
  so clustering the variance space and then the kurtosis space of one dataset
  overwrote every CSV and figure from the first run without a word. The
  docstring listed this as something to remember.

  A run now records which space produced its outputs, and a run that would
  overwrite a different space's results says so, names the files at risk, and
  gives the fix. A new `OUTPUT_STEM` setting (default `DATASET_NAME`, so
  existing paths do not move) separates them automatically when set to
  `f"{DATASET_NAME}_{SPACE}"`.

  Default filenames are unchanged, so nothing downstream breaks.

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
