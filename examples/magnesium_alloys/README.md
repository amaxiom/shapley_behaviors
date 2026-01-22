# Magnesium Alloys Example

This example demonstrates behavioral space analysis on a magnesium alloys dataset with mechanical properties.

## Dataset

**File:** `mg_data.csv` (not included - use your own data)

**Structure:**
- **ID column:** Unique identifier for each alloy
- **Features:** Elemental compositions (31 elements total)
  - Major alloying: Al, Zn, Mn, Zr, Ca, Y, Gd, Nd
  - Rare earth: Ce, La, Ho, Yb, Th, Pr, Tb, Dy, Er, Sc
  - Minor elements: Sn, Ag, Cu, Si, Li, Sb, Ga, Be, Fe, Ni, Sr, Bi
  - Base: Mg (typically 0.85-0.97 mass fraction)
- **Labels:** Mechanical properties
  - Yield_Strength (MPa)
  - Tensile_Strength (MPa)
  - Ductility (% elongation)
- **Drop columns:** Condition, Process-Heat-treatment, Process, Cast, Extruded, Wrought, DOI

**Samples:** 916 magnesium alloys
**Features:** 31 elements
**Processing info:** Mix of cast, extruded, and wrought alloys
**Missing data:** Some property measurements missing

## Running the Analysis

### Step 1: Behavioral Space Exploration

```bash
python run_space_explorer.py
```

**What it does:**
1. Loads mg_data.csv
2. Computes 4 behavioral spaces (variance, skewness, kurtosis, entropy)
3. Generates ~30 plots showing:
   - Each property in each space (with missing data in red)
   - Al, Zn, Y concentrations in each space (grey→red gradients)
4. Saves Hopkins statistics and clustering metrics
5. **Creates profile plots for top outliers** (feature-level breakdown)

**Expected output:**
- Variance space shows strongest clustering (Hopkins ~0.75-0.85)
- High-strength alloys cluster in variance space
- Different alloy families (AZ, WE, ZK series) may separate

### Step 2: Region Analysis

After inspecting the plots:

```bash
python run_region_explorer.py
```

**What it does:**
1. Loads behavioral spaces from Step 1
2. Extracts samples from user-defined regions
3. Analyzes compositional signatures:
   - High-strength region: Al+Zn enriched (AZ series)
   - RE-containing region: Y+Gd+Nd enriched (WE series)
   - High-ductility region: Pure Mg with minimal alloying
4. Validates regions aren't artifacts

**Expected output:**
- Composition tables showing enrichment/depletion percentages
- Box plots showing significant property differences
- Original vs behavioral comparison proving clustering is real

## Key Findings

### High Strength Region (Variance Space)
- **Typical location:** PC1: 0.3-0.6, PC2: -0.2-0.2
- **Composition:** Al +200%, Zn +150%, Mn +120% (vs dataset average)
- **Properties:** Yield ~300-500 MPa, Tensile ~400-650 MPa, Ductility ~1-5%
- **Interpretation:** AZ and AM series alloys (Al-Zn strengthening via precipitation)
- **Examples:** AZ80, AZ91, AM60

### Rare Earth Region (Entropy Space)
- **Typical location:** PC1: -0.4 to 0.0, PC2: 0.2-0.5
- **Composition:** Y +300%, Gd +400%, Nd +350%, Al -80%, Zn -70%
- **Properties:** Yield ~250-350 MPa, Tensile ~300-400 MPa, Ductility ~10-20%
- **Interpretation:** WE series and other RE-containing alloys (creep resistance)
- **Examples:** WE43, WE54, Mg-Gd-Y-Zr

### High Ductility Region (Variance Space)
- **Typical location:** PC1: -0.5 to -0.2, PC2: -0.3-0.3
- **Composition:** All alloying elements depleted (near-pure Mg)
- **Properties:** Yield ~150-250 MPa, Tensile ~200-300 MPa, Ductility ~15-25%
- **Interpretation:** Pure Mg and minimally alloyed grades (high formability)
- **Examples:** Commercially pure Mg, AZ31 (low Al content)

### Validation
- Regions cluster tightly in behavioral spaces (within defined boundaries)
- Same samples scattered randomly in original compositional space
- **Conclusion:** Behavioral transformation reveals genuine alloy family structure

## Common Magnesium Alloy Families

**By composition clustering:**
- **AZ Series:** Al-Zn (most common, good castability)
- **AM Series:** Al-Mn (good weldability)
- **ZK Series:** Zn-Zr (good elevated temperature properties)
- **WE Series:** Y-RE (rare earth, excellent creep resistance)
- **Elektron Series:** Various RE combinations
- **Pure Mg:** Minimal alloying (high ductility, low strength)

## Customization

### Select Different Features
Edit `run_space_explorer.py`:
```python
SELECTED_FEATURES = ['Al', 'Zn', 'Mn']  # AZ/AM series focus
# or
SELECTED_FEATURES = ['Y', 'Gd', 'Nd']   # Rare earth (WE series)
# or
SELECTED_FEATURES = ['Zn', 'Zr', 'Ca']  # ZK series + grain refinement
```

### Define Different Regions
Edit `run_region_explorer.py`:
```python
USER_REGIONS = {
    'AZ_series': {
        'space': 'variance',
        'pc1_range': (0.3, 0.6),
        'pc2_range': (-0.3, 0.3),
        'description': 'Al-Zn strengthened alloys',
        'color': 'red'
    },
    'WE_series': {
        'space': 'entropy',
        'pc1_range': (-0.4, 0.0),
        'pc2_range': (0.2, 0.5),
        'description': 'Rare earth containing alloys',
        'color': 'blue'
    },
    'pure_mg': {
        'space': 'variance',
        'pc1_range': (-0.5, -0.2),
        'pc2_range': (-0.3, 0.3),
        'description': 'Pure and minimally alloyed',
        'color': 'green'
    }
}
```

## Tips

1. **Start with N_PERMUTATIONS=100** for quick exploration, then increase to 500-1000 for publication
2. **Check Hopkins p-values** - p<0.05 means clustering is statistically significant
3. **Compare all 4 spaces** - different spaces may separate different alloy families:
   - Variance: Often separates by total alloying content
   - Entropy: Often separates by element diversity (RE vs non-RE)
   - Skewness: May separate asymmetric compositions
   - Kurtosis: May highlight alloys with extreme element concentrations
4. **Use Change_%** columns in composition tables to quantify enrichment
5. **Validate with domain knowledge** - do the patterns make metallurgical sense?
6. **Consider processing** - cast vs extruded may show different behavioral patterns

## Files Generated

After running both scripts:

```
behavioral_exploration/
├── Mg_behavioral_spaces_all.npy          # Behavioral spaces (for region explorer)
├── Mg_hopkins_statistics.csv             # Clustering metrics
├── Mg_clustering_statistics.csv          # PCA variance, CV, etc.
├── Mg_outliers_*.csv                     # 4 files, one per space
├── Mg_behave_*_*.png                     # ~30 visualization plots
├── Mg_profile_*_outlier_*.png            # Profile plots for outliers (optional)
├── Mg_region_*_samples.csv               # Sample lists per region
├── Mg_region_*_composition_table.csv     # Feature statistics per region
├── Mg_region_*_label_table.csv           # Property statistics per region
├── Mg_regions_comparison_summary.csv     # Cross-region comparison
├── Mg_label_*_boxplot.png                # 3 plots (one per property)
├── Mg_composition_heatmap.png            # Feature heatmap
├── Mg_original_vs_*_comparison.png       # Validation plots
└── Mg_all_regions_in_original_space.png  # All regions overlay
```

## Troubleshooting

**Problem:** "Missing required configuration variables"
**Solution:** Make sure all variables are defined before running scripts

**Problem:** Plots show no clustering
**Solution:** Try different behavioral spaces, increase N_PERMUTATIONS

**Problem:** All outliers detected
**Solution:** Data may need cleaning - check for extreme values or measurement errors

**Problem:** Hopkins statistic ~0.5 (random)
**Solution:** Dataset may not have natural clusters, or features need better selection

**Problem:** Many missing values for rare elements
**Solution:** This is expected - most alloys only contain 2-4 alloying elements

## Metallurgical Interpretation Notes

**Strength mechanisms in Mg alloys:**
- **Solid solution:** Al, Zn in Mg matrix
- **Precipitation:** Mg17Al12, MgZn2 phases
- **Grain refinement:** Zr, Ca additions
- **Texture modification:** RE elements

**Ductility considerations:**
- Pure Mg: High ductility but low strength
- Alloying reduces ductility (strength-ductility tradeoff)
- RE elements can improve ductility at elevated temperatures
- Processing (extrusion) can enhance both strength and ductility

**Expected behavioral patterns:**
- High Al+Zn → High strength, low ductility (variance space)
- RE elements → Different behavioral signature (entropy space)
- Minimal alloying → High ductility, low strength (variance space, opposite end)
