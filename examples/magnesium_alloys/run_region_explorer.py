"""
Magnesium Alloys - Behavioral Region Explorer
=============================================
Analyzes user-defined regions in behavioral spaces to understand
compositional signatures of different alloy families and property regimes.
"""

import sys
sys.path.insert(0, '../..')  # Add parent directory to path

# =====================================================================
# CONFIGURATION (Same as space explorer)
# =====================================================================

SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "Mg"
DATA_FILE = "mg_data.csv"
ID_COLUMN = "ID"
DROP_COLUMNS = ["Condition", "Process-Heat-treatment", "Process", "Cast", "Extruded", "Wrought", "DOI"]
# Or: DROP_COLUMNS = None (if no columns to drop)
LABEL_COLUMNS = ['Yield_Strength', 'Tensile_Strength', 'Ductility']
OUTPUT_DIR = 'behavioral_exploration'

# Path to behavioral spaces from space explorer
BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/Mg_behavioral_spaces_all.npy'

# Visualization mode: 'combined' or 'separate'
PLOT_MODE = 'combined'

# =====================================================================
# USER-DEFINED REGIONS
# =====================================================================
# Define regions based on visual inspection of behavioral space plots
# PC1/PC2 ranges are approximate - adjust based on your plots

USER_REGIONS = {
    'AZ_series': {
        'space': 'variance',  # Which behavioral space
        'pc1_range': (0.30, 0.60),  # PC1 boundaries
        'pc2_range': (-0.30, 0.30),  # PC2 boundaries
        'description': 'Al-Zn strengthened alloys (AZ series)',
        'color': 'red'  # Color for visualization
    },
    
    'WE_series': {
        'space': 'entropy',
        'pc1_range': (-0.40, 0.00),
        'pc2_range': (0.20, 0.50),
        'description': 'Rare earth containing alloys (WE, Elektron series)',
        'color': 'blue'
    },
    
    'pure_mg': {
        'space': 'variance',
        'pc1_range': (-0.50, -0.20),
        'pc2_range': (-0.35, 0.35),
        'description': 'Pure and minimally alloyed Mg (high ductility)',
        'color': 'green'
    },
    
    'ZK_series': {
        'space': 'variance',
        'pc1_range': (-0.10, 0.25),
        'pc2_range': (-0.20, 0.30),
        'description': 'Zn-Zr alloys (ZK series)',
        'color': 'purple'
    },
    
    # Add more regions as needed:
    # 'AM_series': {
    #     'space': 'skewness',
    #     'pc1_range': (0.0, 0.4),
    #     'pc2_range': (-0.3, 0.2),
    #     'description': 'Al-Mn alloys (AM series)',
    #     'color': 'orange'
    # }
}

# =====================================================================
# RUN ANALYSIS
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("MAGNESIUM ALLOYS - REGION ANALYSIS")
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
    print("  - Composition tables: Element enrichment/depletion per region")
    print("  - Label tables: Property statistics per region")
    print("  - Box plots: Property comparisons across regions")
    print("  - Validation: Original vs behavioral space comparisons")
    print("\nInterpretation tips:")
    print("  - AZ series: Expect Al+Zn enrichment, high strength")
    print("  - WE series: Expect Y+Gd+Nd enrichment, creep resistance")
    print("  - Pure Mg: Expect all elements depleted, high ductility")
    print("  - Check Change_% columns for quantitative differences")
    print("  - Validate: Regions should cluster in behavioral but NOT original space")
    print("="*70 + "\n")
