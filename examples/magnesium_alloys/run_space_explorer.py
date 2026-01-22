"""
Magnesium Alloys - Behavioral Space Explorer
============================================
Analyzes mechanical properties (Yield Strength, Tensile Strength, Ductility)
in relation to elemental composition using Shapley behavioral transformations.
"""

import sys
sys.path.insert(0, '../..')  # Add parent directory to path

# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42
N_PERMUTATIONS = 200  # Increase to 500-1000 for publication
N_JOBS = -1  # Use all CPU cores

DATASET_NAME = "Mg"
DATA_FILE = "mg_data.csv"  # Place your data file here
ID_COLUMN = "ID"

# Non-feature, non-label columns to drop
DROP_COLUMNS = ["Condition", "Process-Heat-treatment", "Process", "Cast", "Extruded", "Wrought", "DOI"]
# Or: DROP_COLUMNS = None (if no columns to drop)

# Target labels (mechanical properties)
LABEL_COLUMNS = ['Yield_Strength', 'Tensile_Strength', 'Ductility']

OUTPUT_DIR = 'behavioral_exploration'

# Select 3 elements to visualize with concentration gradients
# Common alloying elements:
SELECTED_FEATURES = ['Al', 'Zn', 'Y']

# Alternative selections:
# SELECTED_FEATURES = ['Al', 'Zn', 'Mn']  # AZ/AM series focus
# SELECTED_FEATURES = ['Y', 'Gd', 'Nd']   # Rare earth (WE series)
# SELECTED_FEATURES = ['Zn', 'Zr', 'Ca']  # ZK series + grain refinement
# SELECTED_FEATURES = None  # Skip feature concentration plots

# Visualization of specific Alloys, outliers and others
CREATE_OUTLIER_PROFILES = True  # Set to False to skip automatic outlier profiles
MAX_OUTLIERS_PER_SPACE = 3      # Number of top outliers to profile per space

# =====================================================================
# RUN ANALYSIS
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("MAGNESIUM ALLOYS - BEHAVIORAL SPACE ANALYSIS")
    print("="*70)
    print(f"\nDataset: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"Selected features: {SELECTED_FEATURES}")
    print("\nStarting analysis...")
    print("="*70 + "\n")
    
    # Import and run the main explorer
    exec(open('../../behavioral_space_explorer.py').read())
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print("1. Inspect plots in {OUTPUT_DIR}/")
    print("2. Identify interesting regions in behavioral spaces")
    print("   - Look for AZ series (high Al+Zn)")
    print("   - Look for WE series (high Y+Gd+Nd)")
    print("   - Look for pure Mg (low alloying)")
    print("3. Define USER_REGIONS in run_region_explorer.py")
    print("4. Run: python run_region_explorer.py")
    print("="*70 + "\n")
