"""
MXenes - Behavioral Region Explorer
===================================
Analyzes user-defined regions in behavioral spaces to understand
compositional and termination signatures of different electrochemical regimes.

Handles mixed labels: continuous (Voltage, Capacity) + categorical (M, X, T, Z)
"""

import sys
sys.path.insert(0, '../..')  # Add parent directory to path

# =====================================================================
# CONFIGURATION (Same as space explorer)
# =====================================================================

SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "mxene"
DATA_FILE = "mxene_mendeleev.csv"
ID_COLUMN = "Formula"
DROP_COLUMNS = ["ID", "In-plane_lattice", "Intercalated_lattice"]
# Or: DROP_COLUMNS = None (if no columns to drop)

# Mixed label types
LABEL_COLUMNS = [
    'Voltage', 'Capacity', 'Charge',  # Continuous
    'M', 'X', 'T', 'Z'                 # Categorical
]

OUTPUT_DIR = 'behavioral_exploration'

# Path to behavioral spaces from space explorer
BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/mxene_behavioral_spaces_all.npy'

# Visualization mode: 'combined' or 'separate'
PLOT_MODE = 'combined'

# =====================================================================
# USER-DEFINED REGIONS
# =====================================================================
# Define regions based on visual inspection of behavioral space plots
# PC1/PC2 ranges are approximate - adjust based on your plots

USER_REGIONS = {
    'F_terminated': {
        'space': 'entropy',  # Entropy space often separates by termination
        'pc1_range': (0.30, 0.70),
        'pc2_range': (-0.30, 0.30),
        'description': 'F-terminated MXenes (high voltage)',
        'color': 'red'
    },
    
    'O_terminated': {
        'space': 'entropy',
        'pc1_range': (-0.50, -0.20),
        'pc2_range': (-0.30, 0.30),
        'description': 'O-terminated MXenes (lower voltage)',
        'color': 'blue'
    },
    
    'Ti_rich': {
        'space': 'variance',  # Variance space often separates by composition
        'pc1_range': (0.20, 0.60),
        'pc2_range': (-0.25, 0.25),
        'description': 'Ti-rich compositions',
        'color': 'green'
    },
    
    'Mixed_metals': {
        'space': 'variance',
        'pc1_range': (-0.40, 0.00),
        'pc2_range': (-0.30, 0.30),
        'description': 'Mixed metal compositions',
        'color': 'purple'
    },
    
    # Add more regions as needed:
    # 'high_capacity': {
    #     'space': 'kurtosis',
    #     'pc1_range': (0.0, 0.5),
    #     'pc2_range': (-0.3, 0.3),
    #     'description': 'High capacity region',
    #     'color': 'orange'
    # }
}

# =====================================================================
# RUN ANALYSIS
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("MXENES - REGION ANALYSIS (MIXED LABELS)")
    print("="*70)
    print(f"\nDataset: {DATA_FILE}")
    print(f"Behavioral spaces: {BEHAVIORAL_SPACES_FILE}")
    print(f"Number of regions: {len(USER_REGIONS)}")
    print(f"Mode: {PLOT_MODE}")
    print("\nRegions defined:")
    for name, config in USER_REGIONS.items():
        print(f"  - {name}: {config['space']} space, color={config['color']}")
    print("\nStarting analysis...")
    print("="*70 + "\n")
    
    # Import and run the region explorer
    exec(open('../../behavioral_region_explorer.py').read())
    
    print("\n" + "="*70)
    print("REGION ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nKey outputs in {OUTPUT_DIR}/:")
    print("  - Composition tables: Elemental enrichment/depletion per region")
    print("  - Label tables: MIXED statistics per region")
    print("    * Continuous (Voltage, Capacity): Mean ± Std, Change_%")
    print("    * Categorical (M, X, T, Z): Most common, counts, percentages")
    print("  - Box plots: Only for continuous labels")
    print("  - Validation: Original vs behavioral space comparisons")
    print("\nInterpretation tips:")
    print("  - Continuous labels: Check Change_% and box plot differences")
    print("  - Categorical labels: Check Most_Common and category counts")
    print("  - Example: 'T' might show 100% 'F' in F_terminated region")
    print("  - Validate: Regions should NOT cluster in original space")
    print("="*70 + "\n")
