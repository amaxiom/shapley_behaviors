"""
Shapley Behavioral Transformations for Materials Data
Based on Liu & Barnard (2025) Machine Learning: Engineering

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


class ShapleyBehaviors:
    """
    Compute Shapley value transformations of data to create behavioral spaces.
    
    Each data point's contribution to summary statistics (variance, skewness, etc.)
    is computed using Shapley values, creating interpretable behavioral vectors.
    """
    
    def __init__(self, n_permutations: int = 100, n_jobs: int = -1, random_state: int = 42):
        """
        Parameters
        ----------
        n_permutations : int
            Number of permutations for Monte Carlo estimation (paper uses ~100)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int
            Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def variance_function(self, X_subset: np.ndarray) -> float:
        """Compute variance (centered second moment)."""
        if len(X_subset) == 0:
            return 0.0
        return np.var(X_subset, ddof=0)
    
    def skewness_function(self, X_subset: np.ndarray) -> float:
        """Compute skewness (normalized third moment)."""
        if len(X_subset) < 2:
            return 0.0
        
        mean = np.mean(X_subset)
        std = np.std(X_subset, ddof=0)
        
        if std == 0:
            return 0.0
        
        n = len(X_subset)
        m3 = np.sum((X_subset - mean) ** 3) / n
        return m3 / (std ** 3)
    
    def kurtosis_function(self, X_subset: np.ndarray) -> float:
        """Compute excess kurtosis (normalized fourth moment - 3)."""
        if len(X_subset) < 2:
            return 0.0
        
        mean = np.mean(X_subset)
        std = np.std(X_subset, ddof=0)
        
        if std == 0:
            return 0.0
        
        n = len(X_subset)
        m4 = np.sum((X_subset - mean) ** 4) / n
        m2 = std ** 2
        return (m4 / (m2 ** 2)) - 3.0
    
    def entropy_function(self, X_subset: np.ndarray) -> float:
        """
        Compute entropy (information content).
        Note: Requires positive values. Uses normalization if needed.
        """
        if len(X_subset) == 0:
            return 0.0
        
        # Normalize to positive probability distribution
        X_pos = X_subset - np.min(X_subset) + 1e-10
        probs = X_pos / np.sum(X_pos)
        
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        return -np.sum(probs * np.log2(probs))
    
    def mean_function(self, X_subset: np.ndarray) -> float:
        """Compute mean (first moment) - trivial transformation."""
        if len(X_subset) == 0:
            return 0.0
        return np.mean(X_subset)
    
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
                                n_perm: int) -> np.ndarray:
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
            
        Returns
        -------
        shapley_values : np.ndarray
            Shapley value for each sample (n_samples,)
        """
        n = len(X_col)
        shapley_values = np.zeros(n)
        
        # Use antithetic sampling (permutation + reverse) for variance reduction
        n_pairs = n_perm // 2
        
        for _ in range(n_pairs):
            # Generate random permutation
            perm = self.rng.permutation(n)
            
            # Process forward permutation
            self._update_shapley_values(X_col, perm, value_func, shapley_values)
            
            # Process reverse (antithetic) permutation
            perm_reverse = perm[::-1]
            self._update_shapley_values(X_col, perm_reverse, value_func, shapley_values)
        
        # Average over all permutations
        shapley_values /= n_perm
        
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
                X[:, j], value_func, self.n_permutations
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
                warnings.warn(f"High additivity error ({relative_error:.4f}). "
                            f"Consider increasing n_permutations.")
        
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

def compute_shapley_variance(X, n_permutations=100, n_jobs=-1, random_state=42):
    """Convenience function for variance behavioral space."""
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs, random_state=random_state)
    return sb.transform(X, value_function='variance')


def compute_shapley_skewness(X, n_permutations=100, n_jobs=-1, random_state=42):
    """Convenience function for skewness behavioral space."""
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs, random_state=random_state)
    return sb.transform(X, value_function='skewness')


def compute_shapley_kurtosis(X, n_permutations=100, n_jobs=-1, random_state=42):
    """Convenience function for kurtosis behavioral space."""
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs, random_state=random_state)
    return sb.transform(X, value_function='kurtosis')


def compute_shapley_entropy(X, n_permutations=100, n_jobs=-1, random_state=42):
    """Convenience function for entropy behavioral space."""
    sb = ShapleyBehaviors(n_permutations=n_permutations, n_jobs=n_jobs, random_state=random_state)
    return sb.transform(X, value_function='entropy')


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
