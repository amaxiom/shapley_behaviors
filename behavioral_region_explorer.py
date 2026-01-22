"""
Flexible Multi-Region Analysis Tool for Behavioral Spaces
==========================================================
Users define their own regions, visualize them together or separately,
and get comprehensive composition and label statistics in tables.

USAGE IN JUPYTER:
-----------------
# First, define your configuration variables:
DATASET_NAME = "ABC"  # Used in file naming
DATA_FILE = "Your_data.csv"
ID_COLUMN = "ID"  # or any unique identifier
DROP_COLUMNS = ["A", "B", "C"] # as many as you define
LABEL_COLUMNS = ['a', 'b', 'c']
OUTPUT_DIR = 'behavioral_exploration'
BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/ABC_behavioral_spaces.npy'
PLOT_MODE = 'combined'

USER_REGIONS = {
    'region_name': {
        'space': 'variance',
        'pc1_range': (0.25, 0.50),
        'pc2_range': (-0.3, 0.3),
        'description': 'Description',
        'color': 'red'
    }
}

# Then run:
%run -i behavioral_region_explorer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# =====================================================================
# CHECK REQUIRED VARIABLES
# =====================================================================

required_vars = ['DATASET_NAME', 'DATA_FILE', 'ID_COLUMN', 'DROP_COLUMNS', 
                'LABEL_COLUMNS', 'BEHAVIORAL_SPACES_FILE', 'OUTPUT_DIR', 
                'PLOT_MODE', 'USER_REGIONS']

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


# =====================================================================
# MAIN ANALYSIS FUNCTIONS
# =====================================================================

def load_data_and_spaces():
    """Load dataset and behavioral spaces."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df = df.set_index(ID_COLUMN)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    
    print(f"Loaded {len(df)} samples from {DATA_FILE}")
    print(f"Index: {ID_COLUMN}")
    
    # Load behavioral spaces
    behavioral_spaces = np.load(BEHAVIORAL_SPACES_FILE, allow_pickle=True).item()
    
    print(f"\nLoaded behavioral spaces:")
    for space_name, space_data in behavioral_spaces.items():
        print(f"  - {space_name}: {space_data.shape}")
    
    # Identify feature columns (everything except labels)
    feature_columns = [col for col in df.columns if col not in LABEL_COLUMNS]
    
    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Label columns ({len(LABEL_COLUMNS)}): {LABEL_COLUMNS}")
    
    return df, behavioral_spaces, feature_columns


def extract_region_samples(behavioral_space, sample_ids, pc1_range, pc2_range):
    """Extract samples from a defined region."""
    # Scale and PCA
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(behavioral_space)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Define region mask
    pc1_min, pc1_max = pc1_range
    pc2_min, pc2_max = pc2_range
    
    region_mask = (
        (X_pca[:, 0] >= pc1_min) & (X_pca[:, 0] <= pc1_max) &
        (X_pca[:, 1] >= pc2_min) & (X_pca[:, 1] <= pc2_max)
    )
    
    # Extract samples
    selected_indices = np.where(region_mask)[0]
    selected_samples = pd.DataFrame({
        'Sample_ID': [sample_ids[i] for i in selected_indices],
        'Array_Index': selected_indices,
        'PC1': X_pca[selected_indices, 0],
        'PC2': X_pca[selected_indices, 1]
    })
    
    return selected_samples, X_pca, pca


def visualize_regions_combined(behavioral_spaces, df, regions_by_space):
    """Visualize all regions for each behavioral space on combined plots."""
    print("\n" + "="*70)
    print("GENERATING COMBINED REGION VISUALIZATIONS")
    print("="*70)
    
    for space_name, regions in regions_by_space.items():
        print(f"\nPlotting {space_name} space with {len(regions)} regions...")
        
        # Get space data and perform PCA
        space_data = behavioral_spaces[space_name]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(space_data)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        var_exp = pca.explained_variance_ratio_
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot all samples in light grey
        ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                  c='grey', s=40, alpha=0.5,
                  edgecolors='none', label='All samples')
        
        # Plot each region
        for region_name, region_data in regions.items():
            config = USER_REGIONS[region_name]
            samples = region_data['samples']
            
            # Highlight samples in this region
            region_indices = samples['Array_Index'].values
            ax.scatter(X_pca[region_indices, 0], X_pca[region_indices, 1],
                      c=config['color'], s=50, alpha=0.7,
                      edgecolors='k', linewidth=0.5,
                      label=f"{region_name} (n={len(samples)})")
            
            # Draw rectangle boundary
            pc1_min, pc1_max = config['pc1_range']
            pc2_min, pc2_max = config['pc2_range']
            
            rect = Rectangle((pc1_min, pc2_min), 
                           pc1_max - pc1_min, pc2_max - pc2_min,
                           linewidth=2, edgecolor=config['color'], 
                           facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=14)
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=14)
        ax.set_title(f'{space_name.capitalize()} Space - Multi-Region Analysis', fontsize=14)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save
        filename = f'{DATASET_NAME}_regions_{space_name}_combined.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Saved: {filename}")


def visualize_regions_separate(behavioral_spaces, df, all_regions):
    """Visualize each region separately."""
    print("\n" + "="*70)
    print("GENERATING SEPARATE REGION VISUALIZATIONS")
    print("="*70)
    
    for region_name, region_data in all_regions.items():
        config = USER_REGIONS[region_name]
        space_name = config['space']
        samples = region_data['samples']
        
        print(f"\nPlotting {region_name}...")
        
        # Get space data and perform PCA
        space_data = behavioral_spaces[space_name]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(space_data)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        var_exp = pca.explained_variance_ratio_
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot all samples in light grey
        ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                  c='grey', s=20, alpha=0.4,
                  edgecolors='none', label='All samples')
        
        # Highlight samples in this region
        region_indices = samples['Array_Index'].values
        ax.scatter(X_pca[region_indices, 0], X_pca[region_indices, 1],
                  c=config['color'], s=60, alpha=0.8,
                  edgecolors='k', linewidth=0.5,
                  label=f"Selected (n={len(samples)})")
        
        # Draw rectangle boundary
        pc1_min, pc1_max = config['pc1_range']
        pc2_min, pc2_max = config['pc2_range']
        
        rect = Rectangle((pc1_min, pc2_min), 
                       pc1_max - pc1_min, pc2_max - pc2_min,
                       linewidth=3, edgecolor=config['color'], 
                       facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(rect)
        
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=14)
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=14)
        ax.set_title(f'{region_name}: {config["description"]}\n{space_name.capitalize()} Space', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save
        filename = f'{DATASET_NAME}_region_{region_name}.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Saved: {filename}")


def analyze_region_composition(df, region_samples, feature_columns, region_name):
    """Analyze composition statistics for a region - ALL ELEMENTS IN TABLE."""
    sample_ids = region_samples['Sample_ID'].values
    df_region = df.loc[sample_ids]
    
    # Composition data
    composition = df_region[feature_columns]
    
    print(f"\n{'='*70}")
    print(f"COMPOSITION ANALYSIS: {region_name.upper()}")
    print(f"{'='*70}")
    print(f"Samples: {len(df_region)}")
    
    # Create comprehensive composition table
    comp_table_data = []
    
    for elem in feature_columns:
        elem_data = composition[elem]
        elem_data_all = df[elem]  # All data for comparison
        
        # Check if element is present
        n_present = (elem_data > 0).sum()
        pct_present = 100 * n_present / len(elem_data)
        
        # Region statistics
        region_mean = elem_data.mean()
        region_median = elem_data.median()
        
        # All data statistics
        all_mean = elem_data_all.mean()
        all_median = elem_data_all.median()
        
        # Percent change from all data
        pct_change_mean = 100 * (region_mean - all_mean) / all_mean if all_mean != 0 else 0
        pct_change_median = 100 * (region_median - all_median) / all_median if all_median != 0 else 0
        
        comp_table_data.append({
            'Element': elem,
            'N_Present': n_present,
            'Pct_Present': f'{pct_present:.1f}%',
            'Mean': region_mean,
            'Std': elem_data.std(),
            'Min': elem_data.min(),
            'Max': elem_data.max(),
            'Median': region_median,
            'All_Mean': all_mean,
            'All_Median': all_median,
            'Change_Mean_%': pct_change_mean,
            'Change_Median_%': pct_change_median
        })
    
    comp_table = pd.DataFrame(comp_table_data)
    
    # Sort by mean concentration (descending)
    comp_table = comp_table.sort_values('Mean', ascending=False)
    
    print(f"\nFull Elemental Composition Table:")
    print(comp_table.to_string(index=False))
    
    # Save composition table
    filename = f'{DATASET_NAME}_region_{region_name}_composition_table.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    comp_table.to_csv(filepath, index=False)
    print(f"\n✓ Saved composition table: {filename}")
    
    # Also save full descriptive statistics
    comp_stats = composition.describe()
    filename = f'{DATASET_NAME}_region_{region_name}_composition_stats.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    comp_stats.to_csv(filepath)
    print(f"✓ Saved composition statistics: {filename}")
    
    return composition, comp_table


def analyze_region_labels(df, region_samples, label_columns, region_name):
    """Analyze label statistics for a region - ALL LABELS IN TABLE."""
    sample_ids = region_samples['Sample_ID'].values
    df_region = df.loc[sample_ids]
    
    # Label data
    labels = df_region[label_columns]
    
    print(f"\n{'='*70}")
    print(f"LABEL ANALYSIS: {region_name.upper()}")
    print(f"{'='*70}")
    
    # Create comprehensive label table
    label_table_data = []
    
    for label in label_columns:
        if label in labels.columns:
            label_data = labels[label]
            label_data_all = df[label]  # All data for comparison
            
            n_valid = label_data.notna().sum()
            n_missing = label_data.isna().sum()
            pct_valid = 100 * n_valid / len(label_data)
            
            # Determine if categorical or continuous
            is_cat = is_categorical(label_data.values)
            
            if is_cat:
                # CATEGORICAL LABEL - count categories
                if n_valid > 0:
                    unique_cats = label_data.dropna().unique()
                    n_categories = len(unique_cats)
                    most_common = label_data.value_counts().index[0] if n_valid > 0 else None
                    most_common_count = label_data.value_counts().iloc[0] if n_valid > 0 else 0
                    most_common_pct = 100 * most_common_count / n_valid if n_valid > 0 else 0
                    
                    # For "all data" stats
                    all_unique_cats = label_data_all.dropna().unique()
                    all_n_categories = len(all_unique_cats)
                    all_most_common = label_data_all.value_counts().index[0] if label_data_all.notna().sum() > 0 else None
                    
                    label_table_data.append({
                        'Label': label,
                        'Type': 'Categorical',
                        'N_Valid': n_valid,
                        'N_Missing': n_missing,
                        'Pct_Valid': f'{pct_valid:.1f}%',
                        'N_Categories': n_categories,
                        'Most_Common': str(most_common),
                        'Most_Common_Count': most_common_count,
                        'Most_Common_Pct': f'{most_common_pct:.1f}%',
                        'All_N_Categories': all_n_categories,
                        'All_Most_Common': str(all_most_common),
                        'Mean': 'N/A',
                        'Std': 'N/A',
                        'Min': 'N/A',
                        'Max': 'N/A',
                        'Median': 'N/A',
                        'All_Mean': 'N/A',
                        'All_Median': 'N/A',
                        'Change_Mean_%': 'N/A',
                        'Change_Median_%': 'N/A'
                    })
                else:
                    label_table_data.append({
                        'Label': label,
                        'Type': 'Categorical',
                        'N_Valid': 0,
                        'N_Missing': n_missing,
                        'Pct_Valid': '0.0%',
                        'N_Categories': 0,
                        'Most_Common': 'N/A',
                        'Most_Common_Count': 0,
                        'Most_Common_Pct': '0.0%',
                        'All_N_Categories': len(label_data_all.dropna().unique()),
                        'All_Most_Common': str(label_data_all.value_counts().index[0]) if label_data_all.notna().sum() > 0 else 'N/A',
                        'Mean': 'N/A',
                        'Std': 'N/A',
                        'Min': 'N/A',
                        'Max': 'N/A',
                        'Median': 'N/A',
                        'All_Mean': 'N/A',
                        'All_Median': 'N/A',
                        'Change_Mean_%': 'N/A',
                        'Change_Median_%': 'N/A'
                    })
            else:
                # CONTINUOUS LABEL - compute statistics
                if n_valid > 0:
                    # Region statistics
                    region_mean = label_data.mean()
                    region_median = label_data.median()
                    
                    # All data statistics
                    all_mean = label_data_all.mean()
                    all_median = label_data_all.median()
                    
                    # Percent change from all data
                    pct_change_mean = 100 * (region_mean - all_mean) / all_mean if all_mean != 0 else 0
                    pct_change_median = 100 * (region_median - all_median) / all_median if all_median != 0 else 0
                    
                    label_table_data.append({
                        'Label': label,
                        'Type': 'Continuous',
                        'N_Valid': n_valid,
                        'N_Missing': n_missing,
                        'Pct_Valid': f'{pct_valid:.1f}%',
                        'N_Categories': 'N/A',
                        'Most_Common': 'N/A',
                        'Most_Common_Count': 'N/A',
                        'Most_Common_Pct': 'N/A',
                        'All_N_Categories': 'N/A',
                        'All_Most_Common': 'N/A',
                        'Mean': region_mean,
                        'Std': label_data.std(),
                        'Min': label_data.min(),
                        'Max': label_data.max(),
                        'Median': region_median,
                        'All_Mean': all_mean,
                        'All_Median': all_median,
                        'Change_Mean_%': pct_change_mean,
                        'Change_Median_%': pct_change_median
                    })
                else:
                    label_table_data.append({
                        'Label': label,
                        'Type': 'Continuous',
                        'N_Valid': 0,
                        'N_Missing': n_missing,
                        'Pct_Valid': '0.0%',
                        'N_Categories': 'N/A',
                        'Most_Common': 'N/A',
                        'Most_Common_Count': 'N/A',
                        'Most_Common_Pct': 'N/A',
                        'All_N_Categories': 'N/A',
                        'All_Most_Common': 'N/A',
                        'Mean': np.nan,
                        'Std': np.nan,
                        'Min': np.nan,
                        'Max': np.nan,
                        'Median': np.nan,
                        'All_Mean': label_data_all.mean(),
                        'All_Median': label_data_all.median(),
                        'Change_Mean_%': np.nan,
                        'Change_Median_%': np.nan
                    })
    
    label_table = pd.DataFrame(label_table_data)
    
    print(f"\nFull Label Table:")
    print(label_table.to_string(index=False))
    
    # Save label table
    filename = f'{DATASET_NAME}_region_{region_name}_label_table.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    label_table.to_csv(filepath, index=False)
    print(f"\n✓ Saved label table: {filename}")
    
    # Also save full descriptive statistics (only for continuous)
    continuous_labels = [col for col in label_columns if col in labels.columns and not is_categorical(labels[col].values)]
    if continuous_labels:
        label_stats = labels[continuous_labels].describe()
        filename = f'{DATASET_NAME}_region_{region_name}_label_stats.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        label_stats.to_csv(filepath)
        print(f"✓ Saved continuous label statistics: {filename}")
    
    return labels, label_table


def create_comparison_summary(all_regions, df, label_columns):
    """Create comparison table across all regions."""
    print("\n" + "="*70)
    print("CROSS-REGION COMPARISON SUMMARY")
    print("="*70)
    
    summary_data = []
    
    for region_name, region_data in all_regions.items():
        samples = region_data['samples']
        sample_ids = samples['Sample_ID'].values
        df_region = df.loc[sample_ids]
        
        row = {
            'Region': region_name,
            'Space': USER_REGIONS[region_name]['space'],
            'N_Samples': len(samples),
            'Pct_Total': f"{100*len(samples)/len(df):.1f}%"
        }
        
        # Add label statistics
        for label in label_columns:
            if label in df_region.columns:
                label_data = df_region[label]
                
                # Determine if categorical or continuous
                is_cat = is_categorical(label_data.values)
                
                if is_cat:
                    # CATEGORICAL: Report most common category and count
                    n_valid = label_data.notna().sum()
                    n_missing = label_data.isna().sum()
                    
                    if n_valid > 0:
                        most_common = label_data.value_counts().index[0]
                        most_common_count = label_data.value_counts().iloc[0]
                        most_common_pct = 100 * most_common_count / n_valid
                        
                        row[f'{label}_type'] = 'Categorical'
                        row[f'{label}_most_common'] = str(most_common)
                        row[f'{label}_most_common_count'] = most_common_count
                        row[f'{label}_most_common_pct'] = f'{most_common_pct:.1f}%'
                        row[f'{label}_n_categories'] = len(label_data.dropna().unique())
                        row[f'{label}_missing'] = n_missing
                    else:
                        row[f'{label}_type'] = 'Categorical'
                        row[f'{label}_most_common'] = 'N/A'
                        row[f'{label}_most_common_count'] = 0
                        row[f'{label}_most_common_pct'] = '0.0%'
                        row[f'{label}_n_categories'] = 0
                        row[f'{label}_missing'] = n_missing
                else:
                    # CONTINUOUS: Report mean, std, min, max
                    label_data_numeric = label_data.dropna()
                    
                    if len(label_data_numeric) > 0:
                        row[f'{label}_type'] = 'Continuous'
                        row[f'{label}_mean'] = label_data_numeric.mean()
                        row[f'{label}_std'] = label_data_numeric.std()
                        row[f'{label}_min'] = label_data_numeric.min()
                        row[f'{label}_max'] = label_data_numeric.max()
                        row[f'{label}_missing'] = label_data.isna().sum()
                    else:
                        row[f'{label}_type'] = 'Continuous'
                        row[f'{label}_mean'] = np.nan
                        row[f'{label}_std'] = np.nan
                        row[f'{label}_min'] = np.nan
                        row[f'{label}_max'] = np.nan
                        row[f'{label}_missing'] = len(df_region)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    filename = f'{DATASET_NAME}_regions_comparison_summary.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    summary_df.to_csv(filepath, index=False)
    print(f"\n✓ Saved comparison summary: {filename}")
    
    return summary_df


def create_composition_comparison(all_regions, df, feature_columns):
    """Create composition comparison table across all regions."""
    print("\n" + "="*70)
    print("CROSS-REGION COMPOSITION COMPARISON")
    print("="*70)
    
    # For each element, show mean concentration across all regions
    comp_comparison_data = []
    
    for elem in feature_columns:
        row = {'Element': elem}
        
        for region_name, region_data in all_regions.items():
            samples = region_data['samples']
            sample_ids = samples['Sample_ID'].values
            df_region = df.loc[sample_ids]
            
            if elem in df_region.columns:
                row[f'{region_name}_mean'] = df_region[elem].mean()
                row[f'{region_name}_std'] = df_region[elem].std()
            else:
                row[f'{region_name}_mean'] = 0.0
                row[f'{region_name}_std'] = 0.0
        
        comp_comparison_data.append(row)
    
    comp_comparison_df = pd.DataFrame(comp_comparison_data)
    
    # Sort by overall mean across all regions
    mean_cols = [col for col in comp_comparison_df.columns if col.endswith('_mean')]
    comp_comparison_df['Overall_Mean'] = comp_comparison_df[mean_cols].mean(axis=1)
    comp_comparison_df = comp_comparison_df.sort_values('Overall_Mean', ascending=False)
    comp_comparison_df = comp_comparison_df.drop('Overall_Mean', axis=1)
    
    print("\nComposition Comparison (Mean ± Std) - Top 15 elements:")
    print(comp_comparison_df.head(15).to_string(index=False))
    
    # Save full comparison
    filename = f'{DATASET_NAME}_regions_composition_comparison.csv'
    filepath = os.path.join(OUTPUT_DIR, filename)
    comp_comparison_df.to_csv(filepath, index=False)
    print(f"\n✓ Saved full composition comparison: {filename}")
    
    return comp_comparison_df


def create_label_boxplots(all_regions, df, label_columns):
    """Create separate box plots for each CONTINUOUS label comparing across all regions."""
    print("\n" + "="*70)
    print("GENERATING LABEL COMPARISON BOX PLOTS")
    print("="*70)
    
    region_names = list(all_regions.keys())
    colors = [USER_REGIONS[name]['color'] for name in region_names]
    
    # Filter to only continuous labels
    continuous_labels = []
    for label in label_columns:
        if label in df.columns:
            if not is_categorical(df[label].values):
                continuous_labels.append(label)
            else:
                print(f"\nSkipping categorical label '{label}' (box plots only for continuous)")
    
    if len(continuous_labels) == 0:
        print("\nNo continuous labels found - skipping box plots")
        return
    
    for label in continuous_labels:
        print(f"\nGenerating box plot for {label}...")
        
        # Collect data for each region
        data_to_plot = []
        labels_list = []
        box_colors = []
        
        for region_name in region_names:
            samples = all_regions[region_name]['samples']
            sample_ids = samples['Sample_ID'].values
            df_region = df.loc[sample_ids]
            
            if label in df_region.columns:
                label_data = df_region[label].dropna()
                
                if len(label_data) > 0:
                    data_to_plot.append(label_data)
                    labels_list.append(region_name)
                    box_colors.append(USER_REGIONS[region_name]['color'])
        
        # Create individual plot for this label
        if len(data_to_plot) > 0:
            fig, ax = plt.subplots(figsize=(6,5))
            
            bp = ax.boxplot(data_to_plot, labels=labels_list, patch_artist=True,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='black', 
                                        markeredgecolor='black', markersize=8),
                           medianprops=dict(color='black', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Color each box
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel(label, fontsize=14)
            ax.set_xlabel('Region', fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=11)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_title(f'{label} Distribution Across Regions', fontsize=14)
            
            plt.tight_layout()
            
            # Save individual plot
            filename = f'{DATASET_NAME}_label_{label}_boxplot.png'
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"  ✓ Saved: {filename}")
    
    print(f"\n✓ Generated {len(continuous_labels)} label box plots")


def create_sample_distribution_histogram(all_regions, df):
    """Create histogram showing sample distribution across all regions."""
    print("\n" + "="*70)
    print("GENERATING SAMPLE DISTRIBUTION HISTOGRAM")
    print("="*70)
    
    region_names = list(all_regions.keys())
    sample_counts = [len(all_regions[r]['samples']) for r in region_names]
    colors = [USER_REGIONS[name]['color'] for name in region_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6,5))
    
    # Create bar chart
    bars = ax.bar(range(len(region_names)), sample_counts, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Region', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title('Sample Distribution Across Behavioral Regions', fontsize=14)
    ax.set_xticks(range(len(region_names)))
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    filename = f'{DATASET_NAME}_sample_distribution.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved: {filename}")

def visualize_regions_in_original_space(behavioral_spaces, all_regions, df, feature_columns):
    """
    Visualize regions in ORIGINAL (non-behavioral) space to show they don't cluster there.
    """
    print("\n" + "="*70)
    print("GENERATING ORIGINAL SPACE COMPARISONS")
    print("="*70)
    
    # Create original space from feature data (not in behavioral_spaces)
    print("Creating original space from feature data...")
    original_data = df[feature_columns].values
    print(f"Original data shape: {original_data.shape}")
    
    # Scale and PCA for original space
    scaler_orig = MinMaxScaler()
    X_scaled_orig = scaler_orig.fit_transform(original_data)
    pca_orig = PCA(n_components=2, random_state=42)
    X_pca_orig = pca_orig.fit_transform(X_scaled_orig)
    var_exp_orig = pca_orig.explained_variance_ratio_
    
    print(f"Original space PCA: PC1={var_exp_orig[0]:.1%}, PC2={var_exp_orig[1]:.1%}")
    
    # Get unique behavioral spaces that have regions defined
    spaces_with_regions = set()
    for region_name, region_data in all_regions.items():
        space_name = USER_REGIONS[region_name]['space']
        spaces_with_regions.add(space_name)
    
    # For each behavioral space with regions, create comparison plot
    for space_name in sorted(spaces_with_regions):
        print(f"\nGenerating comparison for {space_name} space...")
        
        # Get behavioral space PCA
        space_data = behavioral_spaces[space_name]
        scaler_behav = MinMaxScaler()
        X_scaled_behav = scaler_behav.fit_transform(space_data)
        pca_behav = PCA(n_components=2, random_state=42)
        X_pca_behav = pca_behav.fit_transform(X_scaled_behav)
        var_exp_behav = pca_behav.explained_variance_ratio_
        
        # Create figure with two subplots side by side
        # Adjust to make room for legend on the left
        fig = plt.figure(figsize=(14, 5))
        
        # Create grid: [left plot | right plot | legend ]
        gs = fig.add_gridspec(1, 3, width_ratios=[6.5, 6.5, 0], wspace=0.3)
        ax_orig = fig.add_subplot(gs[0])
        ax_behav = fig.add_subplot(gs[1])
        ax_legend = fig.add_subplot(gs[2])
        
        # Get regions for this space
        regions_this_space = {name: data for name, data in all_regions.items() 
                             if USER_REGIONS[name]['space'] == space_name}
        
        # Create color mapping for all samples
        n_samples = len(df)
        sample_colors_orig = ['grey'] * n_samples
        sample_colors_behav = ['grey'] * n_samples
        sample_sizes_orig = [20] * n_samples
        sample_sizes_behav = [20] * n_samples
        sample_alphas_orig = [0.3] * n_samples
        sample_alphas_behav = [0.3] * n_samples
        sample_zorder_orig = [1] * n_samples
        sample_zorder_behav = [1] * n_samples
        
        # Assign colors based on region membership
        for region_name, region_data in regions_this_space.items():
            color = USER_REGIONS[region_name]['color']
            indices = region_data['samples']['Array_Index'].values
            
            for idx in indices:
                sample_colors_orig[idx] = color
                sample_colors_behav[idx] = color
                sample_sizes_orig[idx] = 50
                sample_sizes_behav[idx] = 50
                sample_alphas_orig[idx] = 0.7
                sample_alphas_behav[idx] = 0.7
                sample_zorder_orig[idx] = 5
                sample_zorder_behav[idx] = 5
        
        # LEFT PLOT: Original space
        # Plot in order (grey first, colored on top)
        plot_order = sorted(range(n_samples), key=lambda i: sample_zorder_orig[i])
        for i in plot_order:
            ax_orig.scatter(X_pca_orig[i, 0], X_pca_orig[i, 1],
                          c=sample_colors_orig[i], s=sample_sizes_orig[i], 
                          alpha=sample_alphas_orig[i],
                          edgecolors='k' if sample_colors_orig[i] != 'grey' else 'none',
                          linewidth=0.5,
                          zorder=sample_zorder_orig[i])
        
        ax_orig.set_xlabel(f'PC1 ({var_exp_orig[0]:.1%})', fontsize=14)
        ax_orig.set_ylabel(f'PC2 ({var_exp_orig[1]:.1%})', fontsize=14)
        ax_orig.set_title('Original Space', fontsize=14)
        ax_orig.grid(True, alpha=0.3)
        ax_orig.tick_params(labelsize=12)
        
        # RIGHT PLOT: Behavioral space with region boundaries
        # Plot in order (grey first, colored on top)
        plot_order = sorted(range(n_samples), key=lambda i: sample_zorder_behav[i])
        for i in plot_order:
            ax_behav.scatter(X_pca_behav[i, 0], X_pca_behav[i, 1],
                           c=sample_colors_behav[i], s=sample_sizes_behav[i], 
                           alpha=sample_alphas_behav[i],
                           edgecolors='k' if sample_colors_behav[i] != 'grey' else 'none',
                           linewidth=0.5,
                           zorder=sample_zorder_behav[i])
        
        # Draw region boundaries
        for region_name, region_data in regions_this_space.items():
            config = USER_REGIONS[region_name]
            pc1_min, pc1_max = config['pc1_range']
            pc2_min, pc2_max = config['pc2_range']
            
            rect = Rectangle((pc1_min, pc2_min), 
                           pc1_max - pc1_min, pc2_max - pc2_min,
                           linewidth=2.5, edgecolor=config['color'], 
                           facecolor='none', linestyle='--', alpha=0.8)
            ax_behav.add_patch(rect)
        
        ax_behav.set_xlabel(f'PC1 ({var_exp_behav[0]:.1%})', fontsize=14)
        ax_behav.set_ylabel(f'PC2 ({var_exp_behav[1]:.1%})', fontsize=14)
        ax_behav.set_title(f'{space_name.upper()} Benhavioral Space', fontsize=14)
        ax_behav.grid(True, alpha=0.3)
        ax_behav.tick_params(labelsize=12)
        
        # Create legend in the RIGHT panel
        ax_legend.axis('off')  # Hide axes
        
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor='grey', markersize=8, alpha=0.5,
                                         label='Not in any region'))
        
        for region_name in sorted(regions_this_space.keys()):
            config = USER_REGIONS[region_name]
            n_samples_region = len(regions_this_space[region_name]['samples'])
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=config['color'], 
                          markersize=10, markeredgecolor='k', markeredgewidth=0.5,
                          label=f"{region_name}\n(n={n_samples_region})")
            )
        
        # Place legend in the legend panel, centered vertically
        ax_legend.legend(handles=legend_elements, loc='center',fontsize=10)
        
        # Save
        filename = f'{DATASET_NAME}_original_vs_{space_name}_comparison.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Saved: {filename}")
    
    print(f"\n✓ Generated {len(spaces_with_regions)} original vs behavioral space comparisons")


def create_all_regions_original_space_overlay(behavioral_spaces, all_regions, df, feature_columns):
    """
    Create a single plot showing ALL regions overlaid on original space.
    """
    print("\n" + "="*70)
    print("GENERATING ALL-REGIONS ORIGINAL SPACE OVERLAY")
    print("="*70)
    
    # Create original space from feature data (not in behavioral_spaces)
    print("Creating original space from feature data...")
    original_data = df[feature_columns].values
    print(f"Original data shape: {original_data.shape}")
    
    # Scale and PCA for original space
    scaler_orig = MinMaxScaler()
    X_scaled_orig = scaler_orig.fit_transform(original_data)
    pca_orig = PCA(n_components=2, random_state=42)
    X_pca_orig = pca_orig.fit_transform(X_scaled_orig)
    var_exp_orig = pca_orig.explained_variance_ratio_
    
    print(f"Original space PCA: PC1={var_exp_orig[0]:.1%}, PC2={var_exp_orig[1]:.1%}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8,5))
    
    # Create color mapping for all samples
    n_samples = len(df)
    print(f"Total samples: {n_samples}")
    
    sample_colors = ['grey'] * n_samples
    sample_sizes = [20] * n_samples
    sample_alphas = [0.3] * n_samples
    sample_zorder = [1] * n_samples
    
    # Assign colors based on region membership
    total_colored = 0
    for region_name, region_data in all_regions.items():
        color = USER_REGIONS[region_name]['color']
        indices = region_data['samples']['Array_Index'].values
        
        print(f"Region {region_name}: {len(indices)} samples in {color}")
        
        for idx in indices:
            if idx < n_samples:  # Safety check
                sample_colors[idx] = color
                sample_sizes[idx] = 60
                sample_alphas[idx] = 0.8
                sample_zorder[idx] = 5
                total_colored += 1
    
    print(f"Total colored samples: {total_colored}")
    print(f"Grey samples: {n_samples - total_colored}")
    
    # Plot all points with assigned colors
    # Sort by zorder so colored points appear on top
    plot_order = sorted(range(n_samples), key=lambda i: sample_zorder[i])
    
    for i in plot_order:
        ax.scatter(X_pca_orig[i, 0], X_pca_orig[i, 1],
                  c=sample_colors[i], s=sample_sizes[i], 
                  alpha=sample_alphas[i],
                  edgecolors='k' if sample_colors[i] != 'grey' else 'none',
                  linewidth=0.5,
                  zorder=sample_zorder[i])
    
    ax.set_xlabel(f'PC1 ({var_exp_orig[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'PC2 ({var_exp_orig[1]:.1%})', fontsize=14)
    ax.set_title('Original Space', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    # Create legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='grey', 
                                     markersize=8, alpha=0.5,
                                     label='Not in any region'))
    
    for region_name in sorted(all_regions.keys()):
        config = USER_REGIONS[region_name]
        n_samples_region = len(all_regions[region_name]['samples'])
        space_name = config['space']
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=config['color'], 
                      markersize=10, markeredgecolor='k', markeredgewidth=0.5,
                      label=f"{region_name} ({space_name}, n={n_samples_region})")
        )
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    plt.tight_layout()
    
    # Save
    filename = f'{DATASET_NAME}_all_regions_in_original_space.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    #plt.show()

def main():
    """Main analysis workflow."""
    print("\n" + "="*70)
    print(f"MULTI-REGION BEHAVIORAL ANALYSIS: {DATASET_NAME.upper()}")
    print("="*70)
    print(f"Dataset: {DATA_FILE}")
    print(f"Regions defined: {len(USER_REGIONS)}")
    print(f"Visualization mode: {PLOT_MODE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df, behavioral_spaces, feature_columns = load_data_and_spaces()
    
    # Extract samples from all regions
    print("\n" + "="*70)
    print("EXTRACTING SAMPLES FROM REGIONS")
    print("="*70)
    
    all_regions = {}
    regions_by_space = {}  # Group regions by behavioral space
    
    for region_name, config in USER_REGIONS.items():
        space_name = config['space']
        space_data = behavioral_spaces[space_name]
        
        print(f"\n{region_name}:")
        print(f"  Space: {space_name}")
        print(f"  PC1 range: {config['pc1_range']}")
        print(f"  PC2 range: {config['pc2_range']}")
        
        # Extract samples
        samples, X_pca, pca = extract_region_samples(
            space_data, 
            df.index, 
            config['pc1_range'], 
            config['pc2_range']
        )
        
        print(f"  Samples selected: {len(samples)} ({100*len(samples)/len(df):.1f}%)")
        
        # Store results
        all_regions[region_name] = {
            'samples': samples,
            'X_pca': X_pca,
            'pca': pca
        }
        
        # Group by space for combined plotting
        if space_name not in regions_by_space:
            regions_by_space[space_name] = {}
        regions_by_space[space_name][region_name] = all_regions[region_name]
        
        # Save sample list
        filename = f'{DATASET_NAME}_region_{region_name}_samples.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        samples.to_csv(filepath, index=False)
        print(f"  ✓ Saved: {filename}")
    
    # Visualize regions
    if PLOT_MODE == 'combined':
        visualize_regions_combined(behavioral_spaces, df, regions_by_space)
    else:
        visualize_regions_separate(behavioral_spaces, df, all_regions)
    
    # Analyze each region
    print("\n" + "="*70)
    print("DETAILED REGION ANALYSIS")
    print("="*70)
    
    for region_name in USER_REGIONS.keys():
        samples = all_regions[region_name]['samples']
        
        # Composition analysis (ALL ELEMENTS IN TABLE)
        composition, comp_table = analyze_region_composition(
            df, samples, feature_columns, region_name
        )
        
        # Label analysis (ALL PROPERTIES IN TABLE)
        labels, prop_table = analyze_region_labels(
            df, samples, LABEL_COLUMNS, region_name
        )
        
        # Save full data
        sample_ids = samples['Sample_ID'].values
        df_full = df.loc[sample_ids]
        filename = f'{DATASET_NAME}_region_{region_name}_full_data.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_full.to_csv(filepath)
        print(f"\n✓ Saved full data: {filename}")
    
    # Create comparison summaries
    summary_df = create_comparison_summary(all_regions, df, LABEL_COLUMNS)
    comp_comparison_df = create_composition_comparison(all_regions, df, feature_columns)
    
    # Box plots for label comparisons
    create_label_boxplots(all_regions, df, LABEL_COLUMNS)
    
    # Sample distribution histogram
    create_sample_distribution_histogram(all_regions, df)
    
    # ORIGINAL VS BEHAVIORAL SPACE COMPARISONS
    visualize_regions_in_original_space(behavioral_spaces, all_regions, df, feature_columns)    

    # ALL REGIONS IN ORIGINAL SPACE
    create_all_regions_original_space_overlay(behavioral_spaces, all_regions, df, feature_columns)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(USER_REGIONS)} regions")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nGenerated files per region:")
    print(f"  - Sample list CSV")
    print(f"  - Composition table CSV (all elements)")
    print(f"  - Composition stats CSV")
    print(f"  - Label table CSV (all labels)")
    print(f"  - Label stats CSV")
    print(f"  - Full data CSV")
    print(f"\nGenerated comparison files:")
    print(f"  - Label comparison summary CSV")
    print(f"  - Composition comparison CSV (all elements)")
    print(f"\nGenerated visualizations:")
    print(f"  - Region plots: {len(regions_by_space) if PLOT_MODE=='combined' else len(USER_REGIONS)} plot(s)")
    print(f"  - Label comparison box plots: 1 figure")
    print(f"  - Sample distribution histogram: 1 plot")
    print(f"  - Composition heatmap: 1 plot + data CSV")
    
    # Count original vs behavioral comparisons
    spaces_with_regions = set(USER_REGIONS[name]['space'] for name in all_regions.keys())
    print(f"  - Original vs behavioral comparisons: {len(spaces_with_regions)} plot(s)")
    print(f"  - All regions in original space: 1 plot")
    print("="*70 + "\n")


# =====================================================================
# RUN MAIN
# =====================================================================

if __name__ == "__main__":
    main()
else:
    # When run with %run -i, execute main directly
    main()
