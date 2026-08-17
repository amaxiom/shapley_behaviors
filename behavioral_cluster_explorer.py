"""
K-means Cluster Analysis Tool for Behavioral Spaces
====================================================

A cluster-driven companion to behavioral_region_explorer.py. Instead of asking
the user to hand-draw rectangular PC1-PC2 regions, this tool runs K-means on the
2D PCA projection of a behavioral space and analyses each cluster directly.

Every sample belongs to exactly one cluster, so the groups are mutually
exclusive by construction (one cluster per group, no shared samples). Each
cluster is then summarised for feature composition and target properties, with
per-cluster CSV exports and comparison plots.

USAGE IN JUPYTER:
-----------------
# 1. Define the data configuration (same variables the space explorer uses):
DATASET_NAME = "ABC"                 # used in output file names
DATA_FILE = "Your_data.csv"          # raw feature/label table
ID_COLUMN = "ID"                     # unique-identifier column
DROP_COLUMNS = ["A", "B"]            # non-feature, non-label columns to drop
LABEL_COLUMNS = ['target1', 'target2']   # properties to summarise per cluster
OUTPUT_DIR = 'behavioral_exploration'
BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/ABC_behavioral_spaces.npy'

# 2. (Optional) tune the clustering knobs. Sensible defaults are used otherwise:
SPACE = 'skewness'                   # which behavioral space to cluster
K = 6                                # number of clusters
SEED = 42                            # reproducibility
DATASET_LABEL = 'ABC'                # prefix for cluster names / plot titles
COLORS = ['blue', 'red', 'orange', 'cyan', 'green', 'purple', 'yellow']

# 3. Run:
%run -i behavioral_cluster_explorer.py

# The results (labels, cluster table, summaries) are also returned by main()
# and left in the notebook namespace for further work.

NOTES:
------
- Row alignment between DATA_FILE and the behavioral space is positional: row i
  of the CSV (after set_index / drop) must correspond to row i of the space.
  This matches how the behavioral spaces are generated. A guard raises a clear
  error if the counts disagree.
- Clusters are ordered left-to-right by their PC1 centroid and labelled A, B,
  C, ... so the letters read naturally across the PCA plot.
- LABEL_COLUMNS may be continuous or categorical; each label is classified with
  the same is_categorical rule the region explorer uses. Continuous labels are
  summarised as mean/std/min/median/max and drawn as boxplots. Categorical
  labels are summarised as dominant category and share, with a full
  cluster-by-category cross-tab CSV, and drawn as stacked composition bars.
- Output file names are keyed on DATASET_NAME only, not SPACE. To cluster more
  than one behavioral space of the same dataset, vary DATASET_NAME per run
  (e.g. DATASET_NAME = "ABC_variance") or the second run overwrites the first.
- MAX_COMPOSITION_ROWS caps the PRINTED feature-composition table (default None
  = print every feature). Useful for wide feature sets; the CSV is unaffected
  and always holds every feature.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# =====================================================================
# CONFIGURATION
# =====================================================================

# Required data variables (must be defined before running this script).
_REQUIRED_VARS = ['DATASET_NAME', 'DATA_FILE', 'ID_COLUMN', 'DROP_COLUMNS',
                  'LABEL_COLUMNS', 'OUTPUT_DIR', 'BEHAVIORAL_SPACES_FILE']

_missing = [v for v in _REQUIRED_VARS if v not in globals()]
if _missing:
    print("\n" + "=" * 70)
    print("ERROR: Missing required configuration variables!")
    print("=" * 70)
    print(f"Missing: {', '.join(_missing)}")
    print("\nPlease define these variables BEFORE running this script.")
    print("See the docstring at the top of this file for an example.")
    print("=" * 70 + "\n")
    raise SystemExit("Configuration variables not defined")

# Optional clustering knobs: use the notebook value if present, else a default.
SPACE = globals().get('SPACE', 'skewness')
K = int(globals().get('K', 6))
SEED = int(globals().get('SEED', 42))
DATASET_LABEL = globals().get('DATASET_LABEL', DATASET_NAME)
COLORS = globals().get(
    'COLORS', ['blue', 'red', 'orange', 'cyan', 'green', 'purple', 'yellow',
               'brown', 'pink', 'olive'])
# Cap the printed feature-composition table (the CSV always holds every
# feature). None prints all of them, which is unreadable for wide datasets.
MAX_COMPOSITION_ROWS = globals().get('MAX_COMPOSITION_ROWS', None)


# =====================================================================
# CORE FUNCTIONS
# =====================================================================

def is_categorical(labels):
    """Determine if labels are categorical or continuous.

    Kept identical to the implementation in behavioral_region_explorer.py:
    the two tools must classify a label the same way.
    """
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


def load_data_and_space(space=SPACE):
    """Load the sample table and one behavioral space (positionally aligned)."""
    df = pd.read_csv(DATA_FILE).set_index(ID_COLUMN)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    behavioral_spaces = np.load(BEHAVIORAL_SPACES_FILE, allow_pickle=True).item()
    if space not in behavioral_spaces:
        raise KeyError(
            f"Space '{space}' not found. Available: {list(behavioral_spaces)}")

    space_data = behavioral_spaces[space]
    if len(df) != len(space_data):
        raise ValueError(
            f"Row mismatch: {DATA_FILE} has {len(df)} rows but behavioral "
            f"space '{space}' has {len(space_data)}. They must align positionally.")

    feature_columns = [c for c in df.columns if c not in LABEL_COLUMNS]
    return df, space_data, feature_columns


def compute_clusters(space_data, k=K, seed=SEED):
    """PCA to 2D, then K-means. Returns labels, PC coords and cluster ordering.

    Clusters are ordered left-to-right by PC1 centroid so that the assigned
    letters (A, B, C, ...) read naturally across the PCA plot.
    """
    X_pca = PCA(n_components=2, random_state=seed).fit_transform(
        MinMaxScaler().fit_transform(space_data))
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(np.column_stack([pc1, pc2]))
    centroids = km.cluster_centers_

    ordered_ids = sorted(range(k), key=lambda c: centroids[c, 0])
    letters = [chr(65 + j) for j in range(k)]
    return labels, pc1, pc2, centroids, ordered_ids, letters


def build_cluster_table(labels, ordered_ids, letters):
    """Build the exclusive cluster definition dict: name -> metadata."""
    sizes = np.bincount(labels, minlength=len(ordered_ids))
    clusters = {}
    for j, cid in enumerate(ordered_ids):
        name = f'{DATASET_LABEL}: {letters[j]}'
        clusters[name] = {
            'cluster_id': int(cid),
            'letter': letters[j],
            'color': COLORS[j % len(COLORS)],
            'description': f'cluster {letters[j]}',
            'n': int(sizes[cid]),
        }
    return clusters


def export_cluster_members(df, labels, pc1, pc2, clusters):
    """Write a sample list and full-data CSV for each cluster."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_ids = df.index.to_numpy()
    print("=" * 70)
    print(f"K-MEANS CLUSTER ANALYSIS: {DATASET_NAME.upper()}  (space={SPACE})")
    print("=" * 70)
    for name, meta in clusters.items():
        mask = labels == meta['cluster_id']
        members = pd.DataFrame({
            'Sample_ID': sample_ids[mask],
            'Array_Index': np.where(mask)[0],
            'PC1': pc1[mask],
            'PC2': pc2[mask],
        })
        letter = meta['letter']
        members.to_csv(
            os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_{letter}_samples.csv'),
            index=False)
        # Positional indexing: ID labels may repeat (e.g. duplicate alloy names)
        df.iloc[members['Array_Index'].values].to_csv(
            os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_{letter}_full_data.csv'))
        print(f"  {name}: n={int(mask.sum())} "
              f"({100 * mask.sum() / len(labels):.1f}%)  "
              f"-> cluster_{letter}_samples.csv / _full_data.csv")


def target_property_summary(df, labels, clusters):
    """Per-cluster summary of each LABEL_COLUMN.

    Continuous labels get mean/std/min/median/max. Categorical labels get the
    dominant category, its share, and the number of categories present; the
    full cluster-by-category breakdown is written to a separate cross-tab CSV
    per categorical label (percentages within each cluster).
    """
    categorical = {lab: is_categorical(df[lab].values) for lab in LABEL_COLUMNS}

    rows = []
    for meta in clusters.values():
        # Positional indexing: ID labels may repeat
        sub = df.iloc[np.where(labels == meta['cluster_id'])[0]]
        row = {'Cluster': meta['letter'], 'Description': meta['description'],
               'n': meta['n']}
        for lab in LABEL_COLUMNS:
            values = sub[lab]
            n_valid = int(values.notna().sum())
            if categorical[lab]:
                counts = values.value_counts()
                if n_valid > 0:
                    row[f'{lab}_dominant'] = str(counts.index[0])
                    row[f'{lab}_dominant_pct'] = 100.0 * counts.iloc[0] / n_valid
                    row[f'{lab}_n_categories'] = int((counts > 0).sum())
                else:
                    row[f'{lab}_dominant'] = 'N/A'
                    row[f'{lab}_dominant_pct'] = np.nan
                    row[f'{lab}_n_categories'] = 0
                row[f'{lab}_n_valid'] = n_valid
            else:
                row[f'{lab}_mean'] = values.mean()
                row[f'{lab}_std'] = values.std()
                row[f'{lab}_min'] = values.min()
                row[f'{lab}_median'] = values.median()
                row[f'{lab}_max'] = values.max()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index('Cluster')
    summary.to_csv(
        os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_property_summary.csv'))
    print("\n" + "=" * 70)
    print("TARGET PROPERTY SUMMARY PER CLUSTER")
    print("=" * 70)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary.round(4))

    # Full breakdown for each categorical label: counts and row percentages
    letter_of = {meta['cluster_id']: meta['letter'] for meta in clusters.values()}
    order = [meta['letter'] for meta in clusters.values()]
    for lab, is_cat in categorical.items():
        if not is_cat:
            continue
        cross = pd.crosstab(pd.Series([letter_of[c] for c in labels], name='Cluster'),
                            df[lab].values)
        cross = cross.reindex(order)
        pct = 100.0 * cross.div(cross.sum(axis=1), axis=0)
        pct.to_csv(os.path.join(
            OUTPUT_DIR, f'{DATASET_NAME}_cluster_{lab}_composition.csv'))
        print(f"\n{lab} composition per cluster (% of cluster):")
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(pct.round(1))
        print(f"  -> {DATASET_NAME}_cluster_{lab}_composition.csv")

    return summary


def feature_composition_summary(df, labels, feature_columns, clusters):
    """Per-cluster mean feature value, with an OVERALL reference column."""
    comp = pd.DataFrame(
        {meta['letter']: df.loc[labels == meta['cluster_id'], feature_columns].mean()
         for meta in clusters.values()})
    comp.index.name = 'Feature'
    comp['OVERALL'] = df[feature_columns].mean()
    comp.to_csv(
        os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_composition_summary.csv'))
    print("\n" + "=" * 70)
    print("FEATURE COMPOSITION SUMMARY PER CLUSTER (mean value)")
    print("=" * 70)

    shown = comp
    if MAX_COMPOSITION_ROWS is not None and len(comp) > MAX_COMPOSITION_ROWS:
        # Rank by how far any cluster mean departs from the overall mean,
        # scaled by the overall mean so features of different units compare.
        cluster_cols = [c for c in comp.columns if c != 'OVERALL']
        scale = comp['OVERALL'].abs().replace(0, np.nan)
        spread = comp[cluster_cols].sub(comp['OVERALL'], axis=0).abs().max(axis=1) / scale
        shown = comp.loc[spread.sort_values(ascending=False).index[:MAX_COMPOSITION_ROWS]]
        print(f"(showing the {MAX_COMPOSITION_ROWS} features that differ most between "
              f"clusters, of {len(comp)}; all are in the CSV)")

    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(shown.round(4))
    return comp


def plot_clusters_pc_space(labels, pc1, pc2, clusters):
    """Scatter of the exclusive clusters in the 2D PCA space."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for meta in clusters.values():
        mask = labels == meta['cluster_id']
        ax.scatter(pc1[mask], pc2[mask], s=12, alpha=0.6, color=meta['color'],
                   label=f"{meta['letter']} (n={int(mask.sum())})")
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{DATASET_LABEL} - exclusive k-means clusters ({SPACE})')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_pc_space.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_property_boxplots(df, labels, clusters):
    """One panel per target property.

    Continuous labels get a boxplot per cluster; categorical labels get a
    stacked bar of the category composition of each cluster (a boxplot is
    undefined for categories).
    """
    names = list(clusters.values())
    colors = [m['color'] for m in names]
    fig, axes = plt.subplots(1, len(LABEL_COLUMNS),
                             figsize=(6 * len(LABEL_COLUMNS), 5), squeeze=False)
    for k, lab in enumerate(LABEL_COLUMNS):
        axk = axes[0][k]
        if is_categorical(df[lab].values):
            categories = pd.Series(df[lab].dropna().unique()).sort_values().tolist()
            bottoms = np.zeros(len(names))
            cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(categories), 1)))
            for ci, cat in enumerate(categories):
                heights = []
                for m in names:
                    sub = df.iloc[np.where(labels == m['cluster_id'])[0]][lab]
                    n_valid = sub.notna().sum()
                    heights.append(100.0 * (sub == cat).sum() / n_valid if n_valid else 0.0)
                heights = np.array(heights)
                axk.bar(range(len(names)), heights, bottom=bottoms,
                        color=cmap[ci], edgecolor='white', linewidth=0.5,
                        label=str(cat))
                bottoms += heights
            axk.set_xticks(range(len(names)))
            axk.set_xticklabels([m['letter'] for m in names])
            axk.set_ylabel(f'{lab} (% of cluster)', fontsize=12)
            axk.set_ylim(0, 100)
            axk.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5),
                       title=lab, title_fontsize=9)
        else:
            data = [df.iloc[np.where(labels == m['cluster_id'])[0]][lab].dropna().values
                    for m in names]
            bp = axk.boxplot(data, patch_artist=True)
            axk.set_xticks(range(1, len(names) + 1))
            axk.set_xticklabels([m['letter'] for m in names])
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axk.set_ylabel(lab, fontsize=12)
        axk.set_xlabel('Cluster', fontsize=12)
        axk.set_title(f'{lab} by cluster', fontsize=13)
        axk.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_cluster_property_boxplots.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================================
# MAIN WORKFLOW
# =====================================================================

def main():
    """Run the full cluster analysis and return the key results."""
    df, space_data, feature_columns = load_data_and_space()
    labels, pc1, pc2, centroids, ordered_ids, letters = compute_clusters(space_data)
    clusters = build_cluster_table(labels, ordered_ids, letters)

    export_cluster_members(df, labels, pc1, pc2, clusters)
    property_summary = target_property_summary(df, labels, clusters)
    composition_summary = feature_composition_summary(df, labels, feature_columns, clusters)
    plot_clusters_pc_space(labels, pc1, pc2, clusters)
    plot_property_boxplots(df, labels, clusters)

    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}/")
    print("Per cluster : *_cluster_<L>_samples.csv, *_cluster_<L>_full_data.csv")
    print("Summaries   : *_cluster_property_summary.csv, "
          "*_cluster_composition_summary.csv")
    print("Categorical : *_cluster_<label>_composition.csv (one per categorical label)")
    print("Figures     : *_cluster_pc_space.png, *_cluster_property_boxplots.png")

    return {
        'df': df,
        'labels': labels,
        'pc1': pc1,
        'pc2': pc2,
        'clusters': clusters,
        'property_summary': property_summary,
        'composition_summary': composition_summary,
    }


# =====================================================================
# RUN
# =====================================================================

# Expose the results in the caller namespace (handy with %run -i).
results = main()
labels = results['labels']
clusters = results['clusters']
property_summary = results['property_summary']
composition_summary = results['composition_summary']
