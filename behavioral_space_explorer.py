"""
Behavioral Space Explorer 
============================================
Analyzes Shapley behavioral transformations (variance, skewness, kurtosis, entropy)
to reveal interpretable clustering patterns in any dataset.

USAGE IN JUPYTER:
-----------------
# First, define your configuration variables:
SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "ABC"  # Used in file naming
DATA_FILE = "Your_data.csv"
ID_COLUMN = "ID"  # or any unique identifier
DROP_COLUMNS = ["A", "B", "C"] # as many as you define
LABEL_COLUMNS = ['a', 'b', 'c']
OUTPUT_DIR = 'behavioral_exploration'

# Select specific features and instances to visualize (max 3 recommended)
SELECTED_FEATURES = None        # Change these to any 3 features (rerun again if >3 needed)
CREATE_OUTLIER_PROFILES = True  # Set to False to skip automatic outlier profiles
MAX_OUTLIERS_PER_SPACE = 3      # Number of top outliers to profile per space

# Then run:
%run -i behavioral_space_explorer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sys

# Add current directory to path for imports
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

sys.path.insert(0, current_dir)

# Import shapley_behaviors module
try:
    from shapley_behaviors import compute_shapley_variance, compute_shapley_skewness
    from shapley_behaviors import compute_shapley_kurtosis, compute_shapley_entropy
except ImportError:
    print("\nError: shapley_behaviors.py not found!")
    print("Please ensure shapley_behaviors.py is in the same directory.")
    sys.exit(1)


# =====================================================================
# CHECK REQUIRED VARIABLES
# =====================================================================

# CHECK REQUIRED VARIABLES
required_vars = ['SEED', 'N_PERMUTATIONS', 'N_JOBS', 'DATASET_NAME', 
                'DATA_FILE', 'ID_COLUMN', 'DROP_COLUMNS', 
                'LABEL_COLUMNS', 'OUTPUT_DIR']

missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print("\n" + "="*70)
    print("ERROR: Missing required configuration variables!")
    print("="*70)
    print(f"Missing: {', '.join(missing_vars)}")
    print("\nPlease define these variables BEFORE running this script.")
    print("See the docstring at the top of this file for an example.")
    print("="*70 + "\n")
    raise SystemExit("Configuration variables not defined")

# Optional variables with defaults
if 'SELECTED_FEATURES' not in globals():
    SELECTED_FEATURES = None

# NEW: Optional outlier profile settings
if 'CREATE_OUTLIER_PROFILES' not in globals():
    CREATE_OUTLIER_PROFILES = True  # Default: create profiles

if 'MAX_OUTLIERS_PER_SPACE' not in globals():
    MAX_OUTLIERS_PER_SPACE = 5  # Default: top 5 outliers per space

np.random.seed(SEED)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def identify_outliers(Phi, threshold=2.5):
    """Identify outliers in behavioral space using z-score method."""
    magnitudes = np.linalg.norm(Phi, axis=1)
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    z_scores = (magnitudes - mean_mag) / std_mag if std_mag > 0 else np.zeros_like(magnitudes)
    outlier_idx = np.where(np.abs(z_scores) > threshold)[0]
    return outlier_idx, np.abs(z_scores)


def compute_clustering_statistics(X, random_state=42):
    """Compute statistics that characterize clustering tendency and structure."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(10, X.shape[1]), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    var_exp = pca.explained_variance_ratio_
    var_pc1_pc2 = var_exp[0] + var_exp[1] if len(var_exp) > 1 else var_exp[0]
    
    pairwise_dist = pdist(X_scaled, metric='euclidean')
    mean_dist = np.mean(pairwise_dist)
    std_dist = np.std(pairwise_dist)
    cv_dist = std_dist / mean_dist if mean_dist > 0 else 0
    
    stats = {
        'variance_pc1_pc2': var_pc1_pc2,
        'cv_pairwise_distance': cv_dist,
        'mean_pairwise_distance': mean_dist,
        'std_pairwise_distance': std_dist
    }
    
    return stats


def hopkins_statistic(X, n_samples=None, random_state=42):
    """Compute Hopkins statistic for clustering tendency."""
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    
    if n_samples is None:
        n_samples = min(int(0.1 * n), 100)
    
    n_samples = min(n_samples, n - 1)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_indices = rng.choice(n, size=n_samples, replace=False)
    X_sample = X_scaled[sample_indices]
    
    min_vals = X_scaled.min(axis=0)
    max_vals = X_scaled.max(axis=0)
    X_random = rng.uniform(min_vals, max_vals, size=(n_samples, X_scaled.shape[1]))
    
    from scipy.spatial.distance import cdist
    
    # Distances for real samples
    dist_real = cdist(X_sample, X_scaled, metric='euclidean')
    np.fill_diagonal(dist_real[:, sample_indices], np.inf)
    u = np.min(dist_real, axis=1)
    
    # Distances for random samples
    dist_random = cdist(X_random, X_scaled, metric='euclidean')
    w = np.min(dist_random, axis=1)
    
    H = np.sum(w) / (np.sum(u) + np.sum(w))
    
    return H


def hopkins_statistic_with_pvalue(X, n_permutations=100, random_state=42):
    """Compute Hopkins statistic with p-value via permutation test."""
    H_observed = hopkins_statistic(X, random_state=random_state)
    
    rng = np.random.RandomState(random_state)
    n_samples = min(int(0.1 * X.shape[0]), 100)
    
    H_null = []
    for i in range(n_permutations):
        X_shuffled = X.copy()
        for col in range(X.shape[1]):
            rng.shuffle(X_shuffled[:, col])
        
        H_perm = hopkins_statistic(X_shuffled, n_samples=n_samples, 
                                   random_state=random_state + i + 1)
        H_null.append(H_perm)
    
    H_null = np.array(H_null)
    p_value = np.mean(H_null >= H_observed)
    
    return H_observed, p_value

def is_categorical(labels):
    """Determine if labels are categorical or continuous."""
    # Convert to pandas Series if it's a numpy array for easier handling
    if isinstance(labels, np.ndarray):
        labels = pd.Series(labels)
    
    # Remove NaN for type checking
    valid_labels = labels[~pd.isna(labels)]
    
    if len(valid_labels) == 0:
        return False
    
    # Check if dtype is object or string
    if pd.api.types.is_object_dtype(valid_labels) or pd.api.types.is_string_dtype(valid_labels):
        return True
    
    # For numeric data, check uniqueness
    n_unique = len(np.unique(valid_labels))
    if n_unique <= 20 and n_unique < len(valid_labels) * 0.05:  # Less than 5% unique
        return True
    
    return False


def plot_pca_space(X, labels, label_names, title, is_continuous=False,
                   has_missing=False, is_categorical=False,
                   is_feature_concentration=False,
                   random_state=42, save_path=None):
    """Plot a single PCA projection with appropriate coloring."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    var_exp = pca.explained_variance_ratio_
    
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    if is_feature_concentration:
        # For feature concentration: Red (high) to Grey (low)
        colors_list = ['grey', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('feature_concentration', colors_list, N=n_bins)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                            cmap=cmap, s=30, alpha=0.7,
                            edgecolors='k', linewidth=0.2,
                            vmin=0, vmax=labels.max())
        cbar = plt.colorbar(scatter, pad=0.02)
        cbar.set_label(title.split('by ')[1] if 'by ' in title else '', fontsize=12)

    elif is_categorical:
        # For categorical labels: viridis discrete colors + different markers
        missing_mask = pd.isna(labels)
        present_mask = ~missing_mask
        
        if present_mask.any():
            # Get unique categories (excluding NaN)
            unique_cats = np.unique(labels[present_mask])
            n_cats = len(unique_cats)
            
            # Use viridis colormap with discrete colors
            colors = plt.cm.viridis(np.linspace(0, 1, n_cats))
            
            # Define markers (cycle through if more than categories)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            
            # Plot each category
            for i, cat in enumerate(unique_cats):
                cat_mask = (labels == cat)
                marker = markers[i % len(markers)]
                
                plt.scatter(X_pca[cat_mask, 0], X_pca[cat_mask, 1],
                           c=[colors[i]], s=50, alpha=0.7, marker=marker,
                           edgecolors='k', linewidth=0.2,
                           label=str(cat))
            
            # Add legend for categories
            plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                      fontsize=10, framealpha=0.9)
        
        # Plot missing values in RED on top
        if missing_mask.any():
            plt.scatter(X_pca[missing_mask, 0], X_pca[missing_mask, 1],
                       c='red', s=30, alpha=0.7, marker='x',
                       edgecolors='k', linewidth=0.2,
                       label='Missing', zorder=10)
            
            # Update legend to include missing
            if present_mask.any():
                handles, labels_list = ax.get_legend_handles_labels()
                ax.legend(handles, labels_list, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
                
    elif is_continuous and has_missing:
        # For continuous label with missing values: viridis for data, RED for missing
        missing_mask = np.isnan(labels)
        present_mask = ~missing_mask
        
        # Plot samples WITH data in viridis
        if present_mask.any():
            scatter = plt.scatter(X_pca[present_mask, 0], X_pca[present_mask, 1],
                                c=labels[present_mask],
                                cmap='viridis', s=30, alpha=0.7,
                                edgecolors='k', linewidth=0.2)
            cbar = plt.colorbar(scatter, pad=0.02)
            cbar.set_label(title.split('by ')[1] if 'by ' in title else '', fontsize=12)
        
        # Plot samples with MISSING data in RED (on top)
        if missing_mask.any():
            plt.scatter(X_pca[missing_mask, 0], X_pca[missing_mask, 1],
                       c='red', s=30, alpha=0.7,
                       edgecolors='k', linewidth=0.2,
                       zorder=5)
            
            # Rotated "Missing" label
            ax.text(1.25, 0.45, 'Missing', 
                   transform=ax.transAxes,
                   fontsize=12,
                   rotation=90,
                   verticalalignment='bottom',
                   horizontalalignment='center')
            
            # Red circle marker below text
            ax.plot([1.25], [0.4], 'o', 
                   color='red', 
                   markersize=8,
                   markeredgecolor='k',
                   markeredgewidth=0.2,
                   transform=ax.transAxes,
                   clip_on=False,
                   alpha=0.7)
        
    elif is_continuous:
        # For continuous label without missing values
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                            cmap='viridis', s=30, alpha=0.7,
                            edgecolors='k', linewidth=0.2)
        cbar = plt.colorbar(scatter, pad=0.02)
        cbar.set_label(title.split('by ')[1] if 'by ' in title else '', fontsize=12)
    
    plt.xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=14)
    plt.ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_behavioral_spaces(X, n_permutations=100, n_jobs=-1, random_state=42):
    """Compute all four behavioral spaces using Shapley values."""
    print("\n" + "="*70)
    print("COMPUTING BEHAVIORAL SPACES")
    print("="*70)
    print(f"Using {n_permutations} permutations per space")
    print(f"Parallel jobs: {n_jobs}")
    
    behavioral_spaces = {}
    
    # Variance space
    print("\n[1/4] Computing VARIANCE space...")
    Phi_var = compute_shapley_variance(X, n_permutations=n_permutations,
                                       n_jobs=n_jobs, random_state=random_state)
    behavioral_spaces['variance'] = Phi_var
    print(f"  ✓ Variance space: {Phi_var.shape}")
    
    # Skewness space
    print("\n[2/4] Computing SKEWNESS space...")
    Phi_skew = compute_shapley_skewness(X, n_permutations=n_permutations,
                                        n_jobs=n_jobs, random_state=random_state)
    behavioral_spaces['skewness'] = Phi_skew
    print(f"  ✓ Skewness space: {Phi_skew.shape}")
    
    # Kurtosis space
    print("\n[3/4] Computing KURTOSIS space...")
    Phi_kurt = compute_shapley_kurtosis(X, n_permutations=n_permutations,
                                        n_jobs=n_jobs, random_state=random_state)
    behavioral_spaces['kurtosis'] = Phi_kurt
    print(f"  ✓ Kurtosis space: {Phi_kurt.shape}")
    
    # Entropy space
    print("\n[4/4] Computing ENTROPY space...")
    Phi_ent = compute_shapley_entropy(X, n_permutations=n_permutations,
                                      n_jobs=n_jobs, random_state=random_state)
    behavioral_spaces['entropy'] = Phi_ent
    print(f"  ✓ Entropy space: {Phi_ent.shape}")
    
    return behavioral_spaces


def save_behavioral_spaces(behavioral_spaces, output_dir, dataset_name):
    """Save behavioral spaces to disk."""
    filename = f'{dataset_name}_behavioral_spaces.npy'
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, behavioral_spaces, allow_pickle=True)
    print(f"\n✓ Saved behavioral spaces to: {filename}")
    return filepath


def save_statistics(hopkins_scores, hopkins_pvalues, clustering_stats, 
                   outliers_data, output_dir, dataset_name):
    """Save statistical results to CSV files."""
    
    # Hopkins statistics
    hopkins_df = pd.DataFrame({
        'Space': list(hopkins_scores.keys()),
        'Hopkins_Statistic': list(hopkins_scores.values()),
        'P_Value': [hopkins_pvalues[k] for k in hopkins_scores.keys()]
    })
    filename = f'{dataset_name}_hopkins_statistics.csv'
    filepath = os.path.join(output_dir, filename)
    hopkins_df.to_csv(filepath, index=False)
    print(f"✓ Saved Hopkins statistics: {filename}")
    
    # Clustering statistics
    clustering_df = pd.DataFrame(clustering_stats).T
    filename = f'{dataset_name}_clustering_statistics.csv'
    filepath = os.path.join(output_dir, filename)
    clustering_df.to_csv(filepath)
    print(f"✓ Saved clustering statistics: {filename}")
    
    # Outliers for each space
    for space_name, outlier_info in outliers_data.items():
        outlier_df = pd.DataFrame(outlier_info)
        filename = f'{dataset_name}_outliers_{space_name}.csv'
        filepath = os.path.join(output_dir, filename)
        outlier_df.to_csv(filepath, index=False)
    
    print(f"✓ Saved outlier reports: {len(outliers_data)} files")


def analyze_and_visualize(X, behavioral_spaces, labels_dict, 
                         feature_concentrations, sample_ids,
                         output_dir, dataset_name, random_state=42):
    """Analyze behavioral spaces and generate all visualizations."""
    print("\n" + "="*70)
    print("ANALYZING BEHAVIORAL SPACES")
    print("="*70)
    
    all_spaces = {'original': X}
    all_spaces.update(behavioral_spaces)
    
    hopkins_scores = {}
    hopkins_pvalues = {}
    clustering_stats = {}
    outliers_data = {}
    
    # Identify continuous labels for outlier reporting
    continuous_labels = []
    for label_name, label_values in labels_dict.items():
        if not is_categorical(label_values):
            continuous_labels.append(label_name)
    
    # Compute Hopkins and clustering statistics for each space
    for name, space_data in all_spaces.items():
        print(f"\n--- {name.upper()} Space ---")
        
        # Hopkins statistic with p-value
        H, p_val = hopkins_statistic_with_pvalue(space_data, n_permutations=100,
                                                  random_state=random_state)
        hopkins_scores[name] = H
        hopkins_pvalues[name] = p_val
        
        clustering_tendency = "CLUSTERED" if H > 0.7 else "INTERMEDIATE" if H > 0.5 else "RANDOM"
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"Hopkins statistic: {H:.4f} ({clustering_tendency}) p={p_val:.4f} {significance}")
        
        # Clustering statistics
        stats = compute_clustering_statistics(space_data, random_state=random_state)
        clustering_stats[name] = stats
        
        print(f"Variance concentration (PC1+PC2): {stats['variance_pc1_pc2']:.1%}")
        print(f"Clustering structure (CV): {stats['cv_pairwise_distance']:.3f}")
        
        # Identify outliers (skip for original space)
        if name != 'original':
            outlier_idx, z_scores = identify_outliers(space_data)
            magnitudes = np.linalg.norm(space_data, axis=1)
            
            outlier_info = {
                'Sample_Index': outlier_idx,
                'Sample_ID': [sample_ids[i] for i in outlier_idx],
                'Outlier_Score': z_scores[outlier_idx],
                'Behavior_Magnitude': magnitudes[outlier_idx]
            }
            outliers_data[name] = outlier_info
            
            print(f"Outliers detected: {len(outlier_idx)}")
            
            # Print outlier details if any found
            if len(outlier_idx) > 0:
                print(f"\nOutlier Details ({name} space):")
                print("-" * 80)
                
                # Build header with continuous labels
                header = f"{'Sample_ID':<20} {'Z-Score':<12} {'Magnitude':<12}"
                for label in continuous_labels:
                    header += f" {label:<12}"
                print(header)
                print("-" * 80)
                
                # Print each outlier
                for i, idx in enumerate(outlier_idx):
                    sample_id = sample_ids[idx]
                    outlier_score = z_scores[idx]
                    magnitude = magnitudes[idx]
                    
                    # Start row
                    row = f"{str(sample_id):<20} {outlier_score:<12.4f} {magnitude:<12.4f}"
                    
                    # Add continuous label values
                    for label in continuous_labels:
                        label_value = labels_dict[label][idx]
                        if pd.isna(label_value):
                            row += f" {'Missing':<12}"
                        else:
                            row += f" {label_value:<12.3f}"
                    
                    print(row)
                
                print("-" * 80)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Count total plots
    n_label_plots = len(labels_dict)
    n_feature_plots = len(feature_concentrations) if feature_concentrations else 0
    total_plots = (n_label_plots + n_feature_plots) * len(all_spaces)
    
    print(f"Creating PCA plots: {total_plots} plots total")
    print(f"  - {n_label_plots} labels × {len(all_spaces)} spaces = {n_label_plots * len(all_spaces)} plots")
    if n_feature_plots > 0:
        print(f"  - {n_feature_plots} features × {len(all_spaces)} spaces = {n_feature_plots * len(all_spaces)} plots")
    
    plot_count = 0
    
    for space_name, space_data in all_spaces.items():
        # Plot labels
        for label_name, label_values in labels_dict.items():
            filename = f'{dataset_name}_behave_{space_name}_{label_name.lower()}.png'
            save_path = os.path.join(output_dir, filename)
            title = f'{space_name.capitalize()} Space ({label_name})'
            
            # Determine if categorical or continuous
            is_cat = is_categorical(label_values)
            has_missing = pd.isna(label_values).any()
            
            plot_pca_space(space_data, label_values, None, title, 
                          is_continuous=not is_cat, 
                          is_categorical=is_cat,
                          has_missing=has_missing,
                          random_state=random_state, save_path=save_path)
            plot_count += 1
        
        # Plot feature concentrations if provided
        if feature_concentrations:
            for feat_name, feat_values in feature_concentrations.items():
                filename = f'{dataset_name}_behave_{space_name}_{feat_name.lower()}.png'
                save_path = os.path.join(output_dir, filename)
                title = f'{space_name.capitalize()} Space ({feat_name} Concentration)'
                
                plot_pca_space(space_data, feat_values, None, title,
                              is_feature_concentration=True,
                              random_state=random_state, save_path=save_path)
                plot_count += 1
        
        if plot_count % 5 == 0:
            print(f"  Generated {plot_count}/{total_plots} plots...")
    
    print(f"\n✓ Generated all {plot_count} PCA plots")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("BEHAVIORAL SPACE COMPARISON")
    print("="*70)
    
    best_hopkins = max(hopkins_scores.items(), key=lambda x: x[1])
    print(f"\nBest clustering tendency: {best_hopkins[0]} (H={best_hopkins[1]:.4f})")
    
    best_var = max(clustering_stats.items(),
                   key=lambda x: x[1]['variance_pc1_pc2'])
    print(f"Best variance concentration: {best_var[0]} "
          f"(PC1+PC2={best_var[1]['variance_pc1_pc2']:.1%})")
    
    best_cv = max(clustering_stats.items(),
                  key=lambda x: x[1]['cv_pairwise_distance'])
    print(f"Most clustered structure: {best_cv[0]} "
          f"(CV={best_cv[1]['cv_pairwise_distance']:.3f})")
    
    # Print total outliers across all spaces
    total_outliers = sum(len(data['Sample_Index']) for data in outliers_data.values())
    print(f"\nTotal outliers across all behavioral spaces: {total_outliers}")
    
    return hopkins_scores, hopkins_pvalues, clustering_stats, outliers_data

def compute_feature_importance(Phi):
    """
    Compute global feature importance from behavioral space.
    
    Feature importance is the mean absolute Shapley value across all samples.
    
    Args:
        Phi: Behavioral space matrix (samples × features)
    
    Returns:
        importance: Array of importance scores per feature
        sorted_indices: Feature indices sorted by importance (descending)
    """
    importance = np.mean(np.abs(Phi), axis=0)
    sorted_indices = np.argsort(importance)[::-1]  # Descending order
    return importance, sorted_indices

    
def create_profile_plot(Phi, X, feature_names, sample_idx, space_name, 
                        sample_id=None, figsize=(14, 6), fontsize=12, 
                        save_path=None):
    """
    Create a profile plot for a single sample showing Shapley values.
    
    Profile plots show how each feature contributes to the
    sample's behavioral signature. Features are ordered by global importance
    (left to right), with bar heights showing the sample's Shapley values and
    colors showing normalized feature values.
    
    Args:
        Phi: Behavioral space matrix (samples × features)
        X: Original feature matrix (samples × features)
        feature_names: List of feature names
        sample_idx: Index of sample to profile
        space_name: Name of behavioral space (e.g., 'variance')
        sample_id: Optional sample identifier for title
        figsize: Figure size (width, height)
        fontsize: Base font size for labels
        save_path: Optional path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Get feature importance and sort order
    importance, sorted_indices = compute_feature_importance(Phi)
    
    # Get sample's Shapley values and feature values
    shapley_values = Phi[sample_idx, sorted_indices]
    feature_values = X[sample_idx, sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    
    # Normalize feature values RELATIVE TO DATASET (not within sample)
    # This shows where this sample's features sit relative to all samples
    X_sorted = X[:, sorted_indices]
    feature_min = X_sorted.min(axis=0)
    feature_max = X_sorted.max(axis=0)
    feature_range = feature_max - feature_min
    
    # Avoid division by zero for constant features
    feature_range[feature_range == 0] = 1.0
    
    # Normalize each feature value by its range across the dataset
    normalized_values = (feature_values - feature_min) / feature_range
    normalized_values = np.clip(normalized_values, 0, 1)  # Ensure [0, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get viridis colormap
    cmap = plt.cm.viridis
    
    # Map each normalized value to color
    colors = cmap(normalized_values)
    
    # Create bar plot
    x_pos = np.arange(len(sorted_feature_names))
    bars = ax.bar(x_pos, shapley_values, color=colors, edgecolor='black', 
                  linewidth=0.5, alpha=0.9)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.3)
    
    # Customize axes
    ax.set_xlabel('Features (Ordered by Global Importance)', fontsize=fontsize + 2)
    ax.set_ylabel(f'Shapley Value ({space_name.capitalize()} Space)', 
                  fontsize=fontsize + 2)
    
    # Title
    title_text = f'Behavioral Profile: '
    if sample_id is not None:
        title_text += f'Sample: {sample_id}'
    else:
        title_text += f'Sample Index: {sample_idx}'
    ax.set_title(title_text, fontsize=fontsize + 2, pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_feature_names, rotation=45, ha='right', 
                       fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    
    # Add colorbar for feature values
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Normalized Feature Value', 
                   fontsize=fontsize+2, rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=fontsize)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add text box with statistics
    magnitude = np.linalg.norm(Phi[sample_idx])
    mean_mag = np.mean(np.linalg.norm(Phi, axis=1))
    std_mag = np.std(np.linalg.norm(Phi, axis=1))
    z_score = (magnitude - mean_mag) / std_mag if std_mag > 0 else 0
    
    # stats_text = f'Behavioral Magnitude: {magnitude:.3f}\n'
    # stats_text += f'Z-Score: {z_score:.3f}\n'
    # stats_text += f'Features: {len(feature_names)}'
    
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         fontsize=fontsize - 1, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax


def create_profile_plots_for_outliers(behavioral_spaces, X, feature_names, 
                                      sample_ids, outliers_data, output_dir, 
                                      dataset_name, max_outliers_per_space=5):
    """
    Create profile plots for top outliers in each behavioral space.
    
    Args:
        behavioral_spaces: Dictionary of behavioral space matrices
        X: Original feature matrix
        feature_names: List of feature names
        sample_ids: Array of sample identifiers
        outliers_data: Dictionary of outlier information per space
        output_dir: Directory to save plots
        dataset_name: Dataset identifier for filenames
        max_outliers_per_space: Maximum number of outlier profiles per space
    """
    print("\n" + "="*70)
    print("GENERATING OUTLIER PROFILE PLOTS")
    print("="*70)
    
    total_plots = 0
    
    for space_name, space_data in behavioral_spaces.items():
        if space_name not in outliers_data:
            continue
        
        outlier_info = outliers_data[space_name]
        n_outliers = len(outlier_info['Sample_Index'])
        
        if n_outliers == 0:
            continue
        
        # Sort outliers by z-score (descending)
        sorted_idx = np.argsort(outlier_info['Outlier_Score'])[::-1]
        n_to_plot = min(n_outliers, max_outliers_per_space)
        
        print(f"\n{space_name.capitalize()} space: {n_to_plot} outlier profiles")
        
        for i in range(n_to_plot):
            outlier_rank = sorted_idx[i]
            sample_idx = outlier_info['Sample_Index'][outlier_rank]
            sample_id = outlier_info['Sample_ID'][outlier_rank]
            z_score = outlier_info['Outlier_Score'][outlier_rank]
            
            # Create filename
            filename = f'{dataset_name}_profile_{space_name}_outlier_{i+1}_{sample_id}.png'
            save_path = os.path.join(output_dir, filename)
            
            # Create profile plot
            create_profile_plot(space_data, X, feature_names, sample_idx, 
                              space_name, sample_id=sample_id, 
                              save_path=save_path)
            
            total_plots += 1
            print(f"  ✓ Outlier {i+1}: {sample_id} (z={z_score:.2f})")
    
    print(f"\n✓ Generated {total_plots} outlier profile plots")


def create_custom_profile(sample_idx_or_id, space_name='variance', 
                         figsize=(14, 6), fontsize=12, save_path=None):
    """
    Convenience function to create a profile plot after running main analysis.
    
    Args:
        sample_idx_or_id: Either integer index or sample ID string
        space_name: 'variance', 'skewness', 'kurtosis', or 'entropy'
        figsize: Figure size tuple
        fontsize: Base font size
        save_path: Optional path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axes
    
    Example:
        # After running main()
        results = main()
        
        # Create profile by index
        fig, ax = create_custom_profile(42, 'variance')
        
        # Create profile by sample ID
        fig, ax = create_custom_profile('Sample_1247', 'variance')
    """
    if 'results' not in globals():
        raise RuntimeError("Please run main() first to generate results")
    
    behavioral_spaces = results['behavioral_spaces']
    X = results['X']
    feature_names = results['feature_names']
    sample_ids = results['sample_ids']
    
    # Convert sample ID to index if needed
    if isinstance(sample_idx_or_id, str):
        try:
            sample_idx = list(sample_ids).index(sample_idx_or_id)
            sample_id = sample_idx_or_id
        except ValueError:
            raise ValueError(f"Sample ID '{sample_idx_or_id}' not found in dataset")
    else:
        sample_idx = sample_idx_or_id
        sample_id = sample_ids[sample_idx]
    
    # Validate space name
    if space_name not in behavioral_spaces:
        raise ValueError(f"Space '{space_name}' not found. Choose from: {list(behavioral_spaces.keys())}")
    
    # Create profile
    return create_profile_plot(
        Phi=behavioral_spaces[space_name],
        X=X,
        feature_names=feature_names,
        sample_idx=sample_idx,
        space_name=space_name,
        sample_id=sample_id,
        figsize=figsize,
        fontsize=fontsize,
        save_path=save_path
    )


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(f"BEHAVIORAL SPACE EXPLORER - {DATASET_NAME.upper()}")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATA_FILE}")
    print(f"  Permutations: {N_PERMUTATIONS}")
    print(f"  Random seed: {SEED}")
    print(f"  Parallel jobs: {N_JOBS}")
    if SELECTED_FEATURES:
        print(f"  Selected features: {SELECTED_FEATURES}")
    print(f"  Outlier profiles: {'Enabled' if CREATE_OUTLIER_PROFILES else 'Disabled'}")
    if CREATE_OUTLIER_PROFILES:
        print(f"  Max outliers per space: {MAX_OUTLIERS_PER_SPACE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(DATA_FILE)
    
    # Set ID as index
    if ID_COLUMN in df.columns:
        df = df.set_index(ID_COLUMN)
    else:
        print(f"Warning: {ID_COLUMN} not found, using default index")
    
    # Drop non-feature columns
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Extract labels
    labels_dict = {}
    for label in LABEL_COLUMNS:
        if label in df.columns:
            label_values = df[label].values
            labels_dict[label] = label_values
            
            # Determine type
            if is_categorical(label_values):
                n_valid = pd.notna(label_values).sum()
                n_missing = pd.isna(label_values).sum()
                unique_vals = np.unique(label_values[pd.notna(label_values)])
                print(f"Label '{label}' (categorical): {n_valid} valid, {n_missing} missing, {len(unique_vals)} categories")
            else:
                n_valid = pd.notna(label_values).sum()
                n_missing = pd.isna(label_values).sum()
                print(f"Label '{label}' (continuous): {n_valid} valid, {n_missing} missing")
        else:
            print(f"Warning: Label '{label}' not found in data")
    
    # Drop label columns from features
    df_features = df.drop(columns=LABEL_COLUMNS, errors='ignore')
    
    # CRITICAL: Keep only numeric columns for feature matrix
    numeric_columns = df_features.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df_features.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_columns) > 0:
        print(f"\nWarning: Dropping {len(non_numeric_columns)} non-numeric columns from features:")
        print(f"  {list(non_numeric_columns)}")
        df_features = df_features[numeric_columns]
    
    print(f"\nFeature columns: {len(df_features.columns)}")
    print(f"Features: {list(df_features.columns)}")
    
    # Get feature matrix (now guaranteed to be all numeric)
    X = df_features.values
    feature_names = df_features.columns.tolist()
    sample_ids = df.index.values
    
    # Extract selected feature concentrations if specified
    feature_concentrations = None
    if SELECTED_FEATURES:
        feature_concentrations = {}
        for feat in SELECTED_FEATURES:
            if feat in df_features.columns:
                feature_concentrations[feat] = df_features[feat].values
                print(f"Selected feature '{feat}' for concentration plots")
            else:
                print(f"Warning: Selected feature '{feat}' not found in data")
    
    # Compute behavioral spaces
    behavioral_spaces = compute_behavioral_spaces(X, n_permutations=N_PERMUTATIONS,
                                                  n_jobs=N_JOBS, random_state=SEED)
    
    # Save behavioral spaces
    save_behavioral_spaces(behavioral_spaces, OUTPUT_DIR, DATASET_NAME)
    
    # Analyze and visualize
    hopkins_scores, hopkins_pvalues, clustering_stats, outliers_data = analyze_and_visualize(
        X, behavioral_spaces, labels_dict, feature_concentrations,
        sample_ids, OUTPUT_DIR, DATASET_NAME, random_state=SEED
    )
    
    # Save statistics
    save_statistics(hopkins_scores, hopkins_pvalues, clustering_stats,
                   outliers_data, OUTPUT_DIR, DATASET_NAME)

    if CREATE_OUTLIER_PROFILES:
        create_profile_plots_for_outliers(
            behavioral_spaces, X, feature_names, sample_ids, 
            outliers_data, OUTPUT_DIR, DATASET_NAME,
            max_outliers_per_space=MAX_OUTLIERS_PER_SPACE
        )
    else:
        print("\n" + "="*70)
        print("OUTLIER PROFILE PLOTS: DISABLED")
        print("="*70)
        print("Set CREATE_OUTLIER_PROFILES = True to enable automatic profiles")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"  - Behavioral spaces: {DATASET_NAME}_behavioral_spaces_all.npy")
    print(f"  - Statistics: {DATASET_NAME}_hopkins_statistics.csv")
    print(f"  - Statistics: {DATASET_NAME}_clustering_statistics.csv")
    print(f"  - Outliers: {len(outliers_data)} CSV files")
    print(f"  - Plots: {len(labels_dict) * 5 + (len(feature_concentrations) * 5 if feature_concentrations else 0)} PNG files")
    if CREATE_OUTLIER_PROFILES:
        n_profile_plots = sum(min(len(data['Sample_Index']), MAX_OUTLIERS_PER_SPACE) 
                             for data in outliers_data.values())
        print(f"  - Outlier profiles: {n_profile_plots} PNG files")
    print("="*70 + "\n")

    return {
        'behavioral_spaces': behavioral_spaces,
        'X': X,
        'feature_names': feature_names,
        'sample_ids': sample_ids,
        'labels_dict': labels_dict,
        'outliers_data': outliers_data,
        'hopkins_scores': hopkins_scores,
        'clustering_stats': clustering_stats
    }

# =====================================================================
# RUN MAIN
# =====================================================================

if __name__ == "__main__":
    results = main()
else:
    # When run with %run -i, execute main and store results
    results = main()
