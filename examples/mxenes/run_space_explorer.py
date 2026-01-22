"""
MXenes - Behavioral Space Explorer
==================================
Analyzes electrochemical properties and categorical labels (M, X, T, Z)
in relation to elemental composition using Shapley behavioral transformations.

Demonstrates mixed label types: continuous (Voltage, Capacity) + categorical (M, X, T, Z)
"""

import sys
sys.path.insert(0, '../..')  # Add parent directory to path

# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42
N_PERMUTATIONS = 200  # Increase to 500-1000 for publication
N_JOBS = -1  # Use all CPU cores

DATASET_NAME = "mxene"
DATA_FILE = "mxene_mendeleev.csv"  # Place your data file here
ID_COLUMN = "Formula"

# Non-feature, non-label columns to drop
DROP_COLUMNS = ["ID", "In-plane_lattice", "Intercalated_lattice"]
# Or: DROP_COLUMNS = None (if no columns to drop)

# Target labels (MIXED: continuous + categorical)
LABEL_COLUMNS = [
    # Continuous (will use viridis colormap)
    'Voltage', 
    'Capacity', 
    'Charge',
    # Categorical (will use markers + legend)
    'M',  # Metal element(s)
    'X',  # Carbon or Nitrogen
    'T',  # Termination group(s)
    'Z'   # Intercalated ion(s)
]

OUTPUT_DIR = 'behavioral_exploration'

# Select 3 elements to visualize with concentration gradients
# Transition metals common in MXenes:
SELECTED_FEATURES = ['Ti', 'V', 'Cr']

# Alternative selections:
# SELECTED_FEATURES = ['Nb', 'Mo', 'Ta']  # Heavier transition metals
# SELECTED_FEATURES = ['Sc', 'Y', 'La']   # Rare earth elements
# SELECTED_FEATURES = None                # Skip feature concentration plots

# Visualization of specific MXenes, outliers and others
CREATE_OUTLIER_PROFILES = True  # Set to False to skip automatic outlier profiles
MAX_OUTLIERS_PER_SPACE = 3      # Number of top outliers to profile per space

# =====================================================================
# RUN ANALYSIS
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("MXENES - BEHAVIORAL SPACE ANALYSIS (MIXED LABELS)")
    print("="*70)
    print(f"\nDataset: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"\nLabel types:")
    print("  Continuous: Voltage, Capacity, Charge")
    print("  Categorical: M, X, T, Z")
    print(f"\nSelected features: {SELECTED_FEATURES}")
    print("\nStarting analysis...")
    print("="*70 + "\n")
    
    # Import and run the main explorer
    exec(open('../../behavioral_space_explorer.py').read())
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print("1. Inspect plots in {OUTPUT_DIR}/")
    print("   - Continuous labels: viridis colormap")
    print("   - Categorical labels: markers + legend")
    print("2. Identify regions in behavioral spaces")
    print("3. Define USER_REGIONS in run_region_explorer.py")
    print("4. Run: python run_region_explorer.py")
    print("\nTips:")
    print("  - Entropy space often separates by termination type (T)")
    print("  - Variance space often separates by metal composition (M)")
    print("  - Check which categorical values cluster together")
    print("="*70 + "\n")
