"""
Shapley Behavioral Transformations for Materials Data
Based on Liu & Barnard (2025) Machine Learning: Engineering

Version 0.1.8

Authors:
    Amanda S. Barnard - Lead Developer, Methodology
    Tommy Liu - Co-Developer, Implementation

License:
    MIT License - Copyright (c) 2024 Amanda S. Barnard and Tommy Liu
"""

import numpy as np
from typing import Callable, Tuple
from joblib import Parallel, delayed
import warnings


_PREFIX_FUNCTIONS = ('mean', 'variance', 'skewness', 'kurtosis', 'entropy')
_ENTROPY_EPS = 1e-10


def _prefix_entropy(y):
    """Prefix entropy of the observed, centred values `y`.

    Entropy is the one value function not expressible from running moments:
    u_i = x_i - min(prefix) + eps, so every earlier u shifts when a new minimum
    arrives and T = sum(u log2 u) must be rebuilt. Between minima it is a plain
    cumulative sum, and a random permutation has only H_n ~ ln n prefix minima,
    so the total cost is O(n log n) rather than O(n^2). Antithetic sampling does
    not defeat this: the reverse of a uniform permutation is uniform too.

    Written as H = log2(total) - T/total, which is the closed form of
    -sum(p log2 p) with p = u / total, since sum(u) == total.
    """
    k = y.size
    out = np.empty(k, dtype=float)
    s1 = np.cumsum(y)
    run_min = np.minimum.accumulate(y)
    starts = np.flatnonzero(np.r_[True, run_min[1:] < run_min[:-1]])
    edges = np.r_[starts, k]
    for b in range(starts.size):
        lo, hi = edges[b], edges[b + 1]
        m = run_min[lo]
        u = y[:hi] - m + _ENTROPY_EPS
        f = u * np.log2(u)
        # everything before the block re-summed at this minimum, then a running
        # sum inside it
        head = f[:lo].sum() if lo else 0.0
        t = head + np.cumsum(f[lo:hi])
        kk = np.arange(lo + 1, hi + 1, dtype=float)
        total = s1[lo:hi] - kk * m + kk * _ENTROPY_EPS
        with np.errstate(divide='ignore', invalid='ignore'):
            h = np.log2(total) - t / total
        out[lo:hi] = np.where(total > 0, h, 0.0)
    return out


# How much cancellation is tolerated in m2 = S2/k - mu^2 before a prefix is
# recomputed two-pass.
#
# The ratio r = (S2/k) / m2 measures it: r ~ 1 + mu^2/m2, so r is near 1 for an
# ordinary prefix and large for one whose values are tightly clustered a long
# way from the centring point. The resulting relative error is r*eps for
# variance, r**1.5 * eps for skewness and r**2 * eps for kurtosis, since each
# divides by a different power of m2. Kurtosis therefore sets the threshold: at
# r = 1e3 its error stays near 1e-10, which is far below the Monte Carlo error
# of any Shapley estimate.
#
# An earlier value of 1e8 was useless. The case that actually bites -- two
# nearly-equal values first in a permutation -- came in at r = 5e6, passed the
# test, and left kurtosis at -2.0105 where the exact answer for any two distinct
# values is -2.
_CANCELLATION_TOL = 1e3


def _repair_moments(raw, k, s2k, m2, m3=None, m4=None):
    """Recompute, two-pass, any prefix whose power-sum moments cancelled.

    `raw` is the column's observed values BEFORE centring, `k` the prefix
    lengths and `s2k` the mean square of each centred prefix -- the quantity m2
    is the small remainder of. A prefix is suspect when m2 came out
    non-positive, or when what was subtracted dwarfs what survived. Both tests
    are cheap and catch the case that actually hurts: two nearly-equal values
    early in a permutation, where m2 collapses and kurtosis magnifies it.

    Repairing from the uncentred values matters. Deviations taken from the
    centred array carry the centring's own cancellation (y ~ 1.2 with a spread
    of 5e-4 loses four digits), which left a repaired kurtosis 6e-10 away from
    the reference instead of on top of it.

    Returns (m2, m3, m4), suspect entries replaced, with m3 and m4 passed
    through untouched when not supplied. Modifies the arrays in place.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        suspect = ~(m2 > 0) | (s2k > _CANCELLATION_TOL * m2)
    suspect &= k >= 2
    for i in np.flatnonzero(suspect):
        seg = raw[:i + 1]
        d = seg - seg.mean()
        m2[i] = float(np.mean(d * d))
        if m3 is not None:
            m3[i] = float(np.mean(d * d * d))
        if m4 is not None:
            m4[i] = float(np.mean(d * d * d * d))
    return m2, m3, m4

def prefix_values(x, function_name):
    """Value of `function_name` on every prefix x[:j+1], for j = 0..n-1.

    This is the whole permutation walk in one vectorised pass. The Shapley
    marginal credited to the element at position j is the difference between
    consecutive entries, so a permutation costs O(n) instead of the O(n^2) of
    calling a value function on n growing slices.

    Prefix moments come from cumulative power sums:

        m2 = S2/k - mu^2
        m3 = S3/k - 3 mu S2/k + 2 mu^3
        m4 = S4/k - 4 mu S3/k + 6 mu^2 S2/k - 3 mu^4        (mu = S1/k)

    Those identities are the textbook unstable ones, but only because mu is
    normally large next to the spread. The column is therefore centred ONCE on
    the mean of its observed values, after which mu is the deviation of the
    prefix mean from the global mean, of order spread/sqrt(k), and the
    cancellation all but disappears. All four moment functions are invariant to
    that shift except `mean`, which is corrected by adding it back.

    Missing entries do not contribute: the value at j is the statistic of the
    finite values among x[:j+1], matching ShapleyBehaviors._observed. Positions
    holding a missing value repeat the previous prefix value, since adding an
    unobserved sample cannot change the statistic.
    """
    if function_name not in _PREFIX_FUNCTIONS:
        raise ValueError("No prefix form for {0}".format(function_name))

    a = np.asarray(x, dtype=float)
    n = a.size
    out = np.zeros(n, dtype=float)
    observed = np.isfinite(a)
    if not observed.any():
        return out

    obs_values = a[observed]
    centre = float(obs_values.mean())
    y = obs_values - centre
    k = np.arange(1, y.size + 1, dtype=float)

    if function_name == 'entropy':
        values = _prefix_entropy(y)
    else:
        s1 = np.cumsum(y)
        mu = s1 / k
        if function_name == 'mean':
            values = mu + centre
        else:
            s2 = np.cumsum(y * y)
            s2k = s2 / k
            m2 = s2k - mu * mu
            if function_name == 'variance':
                m2, _, _ = _repair_moments(obs_values, k, s2k, m2)
                values = np.maximum(m2, 0.0)
            else:
                s3 = np.cumsum(y ** 3)
                if function_name == 'skewness':
                    m3 = s3 / k - 3.0 * mu * s2k + 2.0 * mu ** 3
                    m2, m3, _ = _repair_moments(obs_values, k, s2k, m2, m3=m3)
                    degenerate = ~(m2 > 0)
                    safe = np.where(degenerate, 1.0, m2)
                    values = np.where(degenerate, 0.0, m3 / safe ** 1.5)
                else:
                    s4 = np.cumsum(y ** 4)
                    m4 = (s4 / k - 4.0 * mu * (s3 / k)
                          + 6.0 * mu * mu * s2k - 3.0 * mu ** 4)
                    m2, _, m4 = _repair_moments(obs_values, k, s2k, m2, m4=m4)
                    degenerate = ~(m2 > 0)
                    safe = np.where(degenerate, 1.0, m2)
                    values = np.where(degenerate, 0.0,
                                      m4 / (safe * safe) - 3.0)
                # matches the size < 2 guard in the value functions
                values = np.where(k < 2, 0.0, values)

    # scatter back, then carry each value forward across missing positions
    out[observed] = values
    carry = np.where(observed, np.arange(n), 0)
    np.maximum.accumulate(carry, out=carry)
    filled = out[carry]
    filled[:int(np.argmax(observed))] = 0.0
    return filled

class ShapleyBehaviors:
    """
    Compute Shapley value transformations of data to create behavioral spaces.
    
    Each data point's contribution to summary statistics (variance, skewness, etc.)
    is computed using Shapley values, creating interpretable behavioral vectors.
    """
    
    def __init__(self, n_permutations: int = 100, n_jobs: int = -1, random_state: int = 42,
                 incremental: bool = True):
        """
        Parameters
        ----------
        n_permutations : int
            Number of permutations for Monte Carlo estimation (paper uses ~100)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed for reproducibility
        incremental : bool
            Walk each permutation with the vectorised prefix form
            (``prefix_values``) instead of calling the value function on every
            growing slice. The default from 0.1.6, and 35x faster at n=100
            rising to ~400x at n=1600, because it removes a factor of n from
            the cost of a permutation.

            Set ``False`` to reproduce output from 0.1.5 and earlier exactly.
            Both paths draw identical permutations, so they differ only in
            floating-point detail: at most ~3e-11 of a column's spread on
            well-scaled data, which is orders of magnitude below the Monte
            Carlo error of the estimate itself (~1/sqrt(n_permutations)). On a
            column whose offset dwarfs its spread the incremental path is the
            more accurate of the two, by a wide margin.
        """
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.incremental = incremental
        self.rng = np.random.RandomState(random_state)
        
    @staticmethod
    def _observed(X_subset: np.ndarray) -> np.ndarray:
        """The finite values of a subset.

        Every value function reduces a column subset to one number, so a missing
        entry is a sample that did not contribute rather than something to
        impute. Dropping it here keeps all five functions consistent and keeps
        the statistic on the scale it would have had if that sample had never
        been collected.

        Added in 0.1.5. Before it, three functions propagated NaN and
        ``entropy_function`` did something worse: it returned a finite 0.0 for
        any subset containing one, because ``np.min`` gave NaN, every
        probability became NaN, and the ``probs > 0`` filter then removed them
        all. A silent zero cannot be detected downstream.
        """
        a = np.asarray(X_subset, dtype=float).ravel()
        return a[np.isfinite(a)]

    def variance_function(self, X_subset: np.ndarray) -> float:
        """Compute variance (centered second moment), over observed values."""
        a = self._observed(X_subset)
        if a.size == 0:
            return 0.0
        return float(np.var(a, ddof=0))
    
    def skewness_function(self, X_subset: np.ndarray) -> float:
        """Compute skewness (normalized third moment), over observed values."""
        a = self._observed(X_subset)
        if a.size < 2:
            return 0.0
        
        mean = np.mean(a)
        std = np.std(a, ddof=0)
        
        if std == 0:
            return 0.0
        
        n = a.size
        m3 = np.sum((a - mean) ** 3) / n
        return m3 / (std ** 3)
    
    def kurtosis_function(self, X_subset: np.ndarray) -> float:
        """Compute excess kurtosis (fourth moment - 3), over observed values."""
        a = self._observed(X_subset)
        if a.size < 2:
            return 0.0
        
        mean = np.mean(a)
        std = np.std(a, ddof=0)
        
        if std == 0:
            return 0.0
        
        n = a.size
        m4 = np.sum((a - mean) ** 4) / n
        m2 = std ** 2
        return (m4 / (m2 ** 2)) - 3.0
    
    def entropy_function(self, X_subset: np.ndarray) -> float:
        """
        Compute entropy (information content), over observed values.
        Note: Requires positive values. Uses normalization if needed.
        """
        a = self._observed(X_subset)
        if a.size == 0:
            return 0.0
        
        # Normalize to positive probability distribution
        X_pos = a - np.min(a) + 1e-10
        total = np.sum(X_pos)
        if not total > 0:
            return 0.0
        probs = X_pos / total
        
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        return -np.sum(probs * np.log2(probs))
    
    def mean_function(self, X_subset: np.ndarray) -> float:
        """Compute mean (first moment), over observed values."""
        a = self._observed(X_subset)
        if a.size == 0:
            return 0.0
        return float(np.mean(a))
    
    def get_value_function(self, function_name: str) -> Callable:
        """Get value function by name."""
        functions = {
            'mean': self.mean_function,
            'variance': self.variance_function,
            'skewness': self.skewness_function,
            'kurtosis': self.kurtosis_function,
            'entropy': self.entropy_function,
        }
        
        if function_name not in functions:
            raise ValueError(f"Unknown function: {function_name}. "
                           f"Choose from {list(functions.keys())}")
        
        return functions[function_name]
    
    def _compute_shapley_column(self, X_col: np.ndarray, value_func: Callable,
                                n_perm: int, function_name: str = None) -> np.ndarray:
        """
        Compute Shapley values for a single feature column using antithetic sampling.
        
        Parameters
        ----------
        X_col : np.ndarray
            Single feature column (n_samples,)
        value_func : Callable
            Value function to decompose
        n_perm : int
            Number of permutations
        function_name : str, optional
            Name of the value function. When given, and when ``self.incremental``
            is set, each permutation is walked with the vectorised prefix form
            instead of n calls to ``value_func``. A custom callable has no
            prefix form, so leaving this ``None`` keeps the original path.

        Returns
        -------
        shapley_values : np.ndarray
            Shapley value for each sample (n_samples,)
        """
        n = len(X_col)
        shapley_values = np.zeros(n)
        
        # Use antithetic sampling (permutation + reverse) for variance reduction
        n_pairs = n_perm // 2
        
        fast = (getattr(self, 'incremental', True)
                and function_name in _PREFIX_FUNCTIONS)

        for _ in range(n_pairs):
            # Generate random permutation. Drawn identically in both paths, so
            # the reference path still reproduces earlier releases exactly.
            perm = self.rng.permutation(n)
            perm_reverse = perm[::-1]

            if fast:
                # One vectorised pass per permutation. The marginal credited to
                # the element at position j is the step between consecutive
                # prefix values, and the empty set is worth 0.0 for all five
                # value functions, so prepend 0.0 rather than evaluating it.
                for order in (perm, perm_reverse):
                    prefix = prefix_values(X_col[order], function_name)
                    shapley_values[order] += np.diff(prefix, prepend=0.0)
            else:
                # Process forward permutation
                self._update_shapley_values(X_col, perm, value_func, shapley_values)

                # Process reverse (antithetic) permutation
                self._update_shapley_values(X_col, perm_reverse, value_func, shapley_values)
        
        # Average over the permutations actually run. Antithetic sampling
        # evaluates 2*(n_perm//2) permutations, so normalising by n_perm
        # would silently scale every value by (n_perm-1)/n_perm when
        # n_perm is odd.
        n_run = 2 * n_pairs
        if n_run == 0:
            raise ValueError(
                "n_permutations must be >= 2 for antithetic sampling")
        shapley_values /= n_run

        return shapley_values
    
    def _update_shapley_values(self, X_col: np.ndarray, perm: np.ndarray,
                               value_func: Callable, shapley_values: np.ndarray):
        """Update Shapley values for one permutation (in-place)."""
        n = len(X_col)
        
        # Compute initial value (empty set)
        prev_value = value_func(np.array([]))
        
        # Iterate through permutation
        for j in range(n):
            idx = perm[j]
            
            # Add current element to subset
            subset = X_col[perm[:j+1]]
            curr_value = value_func(subset)
            
            # Marginal contribution
            marginal = curr_value - prev_value
            shapley_values[idx] += marginal
            
            prev_value = curr_value
    
    def transform(self, X: np.ndarray, value_function: str = 'variance',
                  verbose: bool = True) -> np.ndarray:
        """
        Transform data into behavioral space using Shapley values.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features)
        value_function : str
            Which summary statistic to decompose:
            'mean', 'variance', 'skewness', 'kurtosis', 'entropy'
        verbose : bool
            Print progress
            
        Returns
        -------
        Phi : np.ndarray
            Behavioral vectors (n_samples, n_features)
            Each element Phi[i,j] = contribution of sample i to value_function of feature j
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"\nComputing Shapley behavioral transformation: {value_function}")
            print(f"  Data shape: {X.shape}")
            print(f"  Permutations: {self.n_permutations}")
            print(f"  Parallel jobs: {self.n_jobs if self.n_jobs != -1 else 'all cores'}")
        
        # Get value function
        value_func = self.get_value_function(value_function)
        
        # Process each feature column in parallel
        if verbose:
            print(f"  Processing {n_features} features...")
        
        shapley_columns = Parallel(n_jobs=self.n_jobs, verbose=1 if verbose else 0)(
            delayed(self._compute_shapley_column)(
                X[:, j], value_func, self.n_permutations, value_function
            )
            for j in range(n_features)
        )
        
        # Stack columns to form behavioral matrix
        Phi = np.column_stack(shapley_columns)
        
        if verbose:
            print(f"  Transformation complete. Output shape: {Phi.shape}")
            
            # Verify additivity property (sum of contributions ≈ total value)
            total_from_shapley = np.sum(Phi, axis=0)
            total_actual = np.array([value_func(X[:, j]) for j in range(n_features)])
            relative_error = np.mean(np.abs(total_from_shapley - total_actual) / 
                                    (np.abs(total_actual) + 1e-10))
            print(f"  Additivity check - mean relative error: {relative_error:.6f}")
            if relative_error > 0.01:
                warnings.warn(
                    f"High additivity error ({relative_error:.4f}). Usually this "
                    f"means n_permutations is too low. It can also mean a column "
                    f"whose offset dwarfs its spread (e.g. 1e8 +/- 1e-6), where "
                    f"the batch value function used for this check is itself "
                    f"imprecise; centring or rescaling such a column fixes it.")
        
        return Phi
    
    def transform_multiple(self, X: np.ndarray, 
                          value_functions: list = None,
                          verbose: bool = True) -> dict:
        """
        Transform data into multiple behavioral spaces at once.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features)
        value_functions : list
            List of value function names. If None, uses all standard functions.
        verbose : bool
            Print progress
            
        Returns
        -------
        behavioral_spaces : dict
            Dictionary mapping function name to transformed data
        """
        if value_functions is None:
            value_functions = ['variance', 'skewness', 'kurtosis', 'entropy']
        
        behavioral_spaces = {}
        
        for func_name in value_functions:
            behavioral_spaces[func_name] = self.transform(
                X, value_function=func_name, verbose=verbose
            )
        
        return behavioral_spaces


def identify_outliers(Phi: np.ndarray, threshold: float = 3.0,
                     method: str = 'zscore') -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outliers in behavioral space.
    
    Points with extreme contributions to distributional properties
    (especially skewness/kurtosis) are likely outliers.
    
    Parameters
    ----------
    Phi : np.ndarray
        Behavioral vectors (n_samples, n_features)
    threshold : float
        Threshold for outlier detection (default: 3.0 std deviations)
    method : str
        'zscore' or 'iqr' (interquartile range)
        
    Returns
    -------
    outlier_indices : np.ndarray
        Indices of outlier samples
    outlier_scores : np.ndarray
        Outlier score for each sample (higher = more outlying)
    """
    n_samples = Phi.shape[0]
    
    if method == 'zscore':
        # Compute magnitude of behavioral vector for each sample
        magnitudes = np.linalg.norm(Phi, axis=1)
        
        # Z-score based detection
        mean = np.mean(magnitudes)
        std = np.std(magnitudes)
        
        if std == 0:
            return np.array([]), np.zeros(n_samples)
        
        z_scores = np.abs((magnitudes - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0]
        outlier_scores = z_scores
        
    elif method == 'iqr':
        # Compute magnitude of behavioral vector for each sample
        magnitudes = np.linalg.norm(Phi, axis=1)
        
        # IQR based detection
        q1 = np.percentile(magnitudes, 25)
        q3 = np.percentile(magnitudes, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_mask = (magnitudes < lower_bound) | (magnitudes > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        
        # Score as distance from median in IQR units
        median = np.median(magnitudes)
        outlier_scores = np.abs(magnitudes - median) / (iqr + 1e-10)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outlier_indices, outlier_scores


# =====================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility)
# =====================================================================

def compute_shapley_variance(X, n_permutations=100, n_jobs=-1, random_state=42,
                             incremental=True):
    """Convenience function for variance behavioral space.

    Pass ``incremental=False`` to reproduce 0.1.5 and earlier exactly.
    """
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs,
                          random_state=random_state, incremental=incremental)
    return sb.transform(X, value_function='variance')


def compute_shapley_skewness(X, n_permutations=100, n_jobs=-1, random_state=42,
                             incremental=True):
    """Convenience function for skewness behavioral space.

    Pass ``incremental=False`` to reproduce 0.1.5 and earlier exactly.
    """
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs,
                          random_state=random_state, incremental=incremental)
    return sb.transform(X, value_function='skewness')


def compute_shapley_kurtosis(X, n_permutations=100, n_jobs=-1, random_state=42,
                             incremental=True):
    """Convenience function for kurtosis behavioral space.

    Pass ``incremental=False`` to reproduce 0.1.5 and earlier exactly.
    """
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs,
                          random_state=random_state, incremental=incremental)
    return sb.transform(X, value_function='kurtosis')


def compute_shapley_entropy(X, n_permutations=100, n_jobs=-1, random_state=42,
                             incremental=True):
    """Convenience function for entropy behavioral space.

    Pass ``incremental=False`` to reproduce 0.1.5 and earlier exactly.
    """
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs,
                          random_state=random_state, incremental=incremental)
    return sb.transform(X, value_function='entropy')


def find_break_zones(values, z_threshold=2.5, min_region_fraction=0.05,
                     max_straggler_fraction=0.02):
    """
    Find statistically significant sparse bands ("break zones") along one axis.

    Intended for the principal components of a behavioral space projection:
    the dense blocks between zones are natural candidate regions, and
    samples inside a zone are best left unassigned. This is the algorithmic
    core of the behavioral_break_finder.py script distributed with this
    package (see copy_scripts).

    A break zone is built in three steps:
      1. Sort the values and compute nearest-neighbor gaps. Gaps whose
         z-score (against the mean and standard deviation of all gaps)
         exceeds ``z_threshold`` are significant.
      2. Significant gaps that would isolate fewer than
         ``min_region_fraction * n`` samples on either side are rejected,
         so single extreme observations never define a boundary. Rejected
         gaps that are still strong (z >= 3) and separate a coherent group
         (>= 1% of samples on each side) are flagged as satellite
         candidates: the natural boundaries of small clusters split off
         from a dominant central population.
      3. Surviving gaps separated by at most
         ``max_straggler_fraction * n`` straggler points are merged into
         a single zone.

    Parameters
    ----------
    values : array-like
        One-dimensional values (e.g. one principal component).
    z_threshold : float
        Significance threshold for gap z-scores.
    min_region_fraction : float
        Minimum fraction of samples required on each side of a zone.
    max_straggler_fraction : float
        Gaps separated by at most this fraction of samples merge into
        one zone.

    Returns
    -------
    zones : list of dict
        Each with keys ``lower_edge``, ``upper_edge``, ``midpoint``,
        ``n_below``, ``n_inside``, ``n_above``, ``max_z``.
    rejected : list of dict
        Rejected significant gaps, each with keys ``position``,
        ``n_below``, ``n_above``, ``z``, ``satellite``.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n < 3:
        return [], []
    gaps = np.diff(v)
    gap_std = gaps.std()
    if gap_std == 0:
        return [], []
    z = (gaps - gaps.mean()) / gap_std

    sig = np.where(z > z_threshold)[0]
    min_side = int(np.ceil(min_region_fraction * n))
    max_stragglers = max(1, int(round(max_straggler_fraction * n)))
    min_satellite = max(2, int(np.ceil(0.01 * n)))

    rejected = []
    kept = []
    for i in sig:
        n_below, n_above = i + 1, n - i - 1
        if n_below < min_side or n_above < min_side:
            rejected.append({
                'position': 0.5 * (v[i] + v[i + 1]),
                'n_below': n_below, 'n_above': n_above, 'z': z[i],
                'satellite': (min(n_below, n_above) >= min_satellite
                              and z[i] >= 3.0),
            })
        else:
            kept.append(i)

    zones = []
    if kept:
        groups = [[kept[0]]]
        for i in kept[1:]:
            if i - groups[-1][-1] <= max_stragglers:
                groups[-1].append(i)
            else:
                groups.append([i])

        for g in groups:
            lo, hi = g[0], g[-1] + 1
            zones.append({
                'lower_edge': v[lo],
                'upper_edge': v[hi],
                'midpoint': 0.5 * (v[lo] + v[hi]),
                'n_below': lo + 1,
                'n_inside': hi - lo - 1,
                'n_above': n - hi,
                'max_z': z[g].max(),
            })

    return zones, rejected


if __name__ == "__main__":
    # Quick test
    print("Testing Shapley Behavioral Transformations...")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Add some outliers
    X[0, :] = 5.0  # Strong outlier
    
    # Transform
    sb = ShapleyBehaviors(n_permutations=50, n_jobs=-1)
    
    # Test variance transformation
    Phi_var = sb.transform(X, 'variance', verbose=True)
    print(f"\nVariance behavioral space shape: {Phi_var.shape}")
    
    # Test skewness transformation  
    Phi_skew = sb.transform(X, 'skewness', verbose=True)
    print(f"Skewness behavioral space shape: {Phi_skew.shape}")
    
    # Identify outliers in skewness space
    outlier_idx, outlier_scores = identify_outliers(Phi_skew, threshold=2.0)
    print(f"\nOutliers detected in skewness space: {outlier_idx}")
    print(f"Outlier scores: {outlier_scores[outlier_idx]}")
    
    print("\n✓ Shapley behavioral transformations working correctly!")
