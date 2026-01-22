# MXenes Example

This example demonstrates behavioral space analysis on MXenes with mixed continuous and categorical labels.

## Dataset

**File:** `mxene_mendeleev.csv` (not included - use your own data)

**Structure:**
- **ID column:** Formula (e.g., "Ti2C", "Hf2N")
- **Features:** Elemental composition (integers: Ti, V, Cr, Nb, Mo, Ta, etc.)
- **Labels (continuous):**
  - Voltage (V)
  - Capacity (mAh/g)
  - Charge (unitless)
- **Labels (categorical):**
  - M: Metal element(s) (e.g., "Ti", "TiTiTiTiHfHfHfHf")
  - X: Carbon/Nitrogen (e.g., "C", "N")
  - T: Termination group(s) (e.g., "F", "O", "OH")
  - Z: Intercalated ion(s) (e.g., "Li", "Na", "K", "Mg")
- **Drop columns:** ID, In-plane_lattice, Intercalated_lattice

**Samples:** ~360 MXene compositions
**Features:** ~20 elements
**Mixed labels:** Voltage/Capacity (continuous) + M/X/T/Z (categorical)

## Key Features

This example showcases:
1. **Mixed label types:** Continuous (voltage, capacity) and categorical (M, X, T, Z) in the same analysis
2. **Categorical visualization:** Different marker shapes and colors per category
3. **Missing data handling:** Both continuous (red circles) and categorical (red X markers)
4. **Compositional analysis:** Integer counts rather than concentrations

## Running the Analysis

### Step 1: Behavioral Space Exploration

```bash
python run_space_explorer.py
```

**What it does:**
1. Loads mxene_mendeleev.csv
2. Auto-detects label types:
   - Voltage, Capacity, Charge → continuous (viridis colormap)
   - M, X, T, Z → categorical (different markers + colors)
3. Generates ~40 plots (7 labels × 5 spaces + feature concentrations)
4. Computes Hopkins statistics for each space
5. **Creates profile plots for top outliers in each behavioral space**

**Expected output:**
- Entropy space shows strong clustering by termination type (T)
- Variance space separates by metal composition (M)
- Categorical labels shown with legend (different markers per category)

### Step 2: Region Analysis

```bash
python run_region_explorer.py
```

**What it does:**
1. Extracts samples from user-defined regions
2. Analyzes:
   - **Continuous labels:** Mean ± std, change % vs dataset
   - **Categorical labels:** Most common category, counts, percentages
3. Validates regions in original vs behavioral space

**Expected output:**
- Composition tables showing elemental patterns (e.g., Ti-rich vs Hf-rich)
- Label tables with mixed statistics:
  - Voltage: 1.2 ± 0.3 V
  - M: Most common = "Ti" (75%), 3 categories
- Validation showing clear separation in behavioral but not original space

## Key Findings

### F-Terminated Region (Entropy Space)
- **Location:** PC1: 0.30-0.70, PC2: -0.30-0.30
- **Composition:** F enriched +400%, Ti +350%, Hf +350%
- **Labels:**
  - **T (categorical):** 100% "F" termination
  - **Voltage:** 1.5 ± 0.2 V (higher than average)
  - **M:** Primarily Ti and Hf
- **Interpretation:** F-terminated Ti/Hf MXenes, high voltage

### O-Terminated Region (Entropy Space)
- **Location:** PC1: -0.50 to -0.20, PC2: -0.30-0.30
- **Composition:** O enriched, H enriched, F depleted
- **Labels:**
  - **T (categorical):** 100% "O" termination
  - **Voltage:** 0.8 ± 0.3 V (lower than average)
  - **M:** More diverse metal compositions
- **Interpretation:** O-terminated MXenes, lower voltage, broader metal range

## Label Type Handling

### Continuous Labels (Voltage, Capacity, Charge)
**Visualization:**
- Viridis colormap
- Missing values: Red circles with "Missing" annotation

**Statistics:**
- Mean, Std, Min, Max, Median
- Change_% vs dataset average
- Box plots for cross-region comparison

### Categorical Labels (M, X, T, Z)
**Visualization:**
- Discrete viridis colors (one per category)
- Different marker shapes: o, s, ^, D, v, <, >, p, *, h
- Legend showing all categories
- Missing values: Red X markers labeled "Missing"

**Statistics:**
- N_Categories: Number of unique values in region
- Most_Common: Dominant category
- Most_Common_Count: Frequency
- Most_Common_Pct: Percentage
- NO mean/std (not applicable to categorical data)

## Customization

### Select Different Elements
Edit `run_space_explorer.py`:
```python
SELECTED_FEATURES = ['Ti', 'V', 'Cr']  # Early transition metals
# or
SELECTED_FEATURES = ['Sc', 'Y', 'La']  # Rare earth elements
# or
SELECTED_FEATURES = None  # Skip feature plots
```

### Include More Categorical Labels
Edit configuration:
```python
LABEL_COLUMNS = [
    'Voltage', 'Capacity', 'Charge',  # Continuous
    'M', 'X', 'T', 'Z',                # Categorical
    'Structure_Type'                   # Another categorical
]
```

## Tips for Mixed Labels

1. **Auto-detection works well** - toolkit automatically identifies categorical vs continuous
2. **Categorical with ≤20 unique values** - detected as categorical
3. **String columns** - always treated as categorical
4. **Box plots** - only generated for continuous labels (categorical skipped automatically)
5. **Missing data** - handled for both types (red markers in plots)

## Files Generated

After running both scripts:

```
behavioral_exploration/
├── mxene_behavioral_spaces_all.npy       # For region explorer
├── mxene_hopkins_statistics.csv
├── mxene_clustering_statistics.csv
├── mxene_outliers_*.csv                  # 4 files
├── mxene_behave_*_*.png                  # ~40 plots
│   ├── *_voltage.png                     # Continuous: viridis + red for missing
│   ├── *_capacity.png
│   ├── *_m.png                           # Categorical: markers + legend
│   ├── *_x.png
│   ├── *_t.png
│   └── *_z.png
├── mxene_profile_*_outlier_*.png         # Profile plots for outliers (optional)
├── mxene_region_*_label_table.csv        # Mixed statistics
├── mxene_label_voltage_boxplot.png       # Only continuous labels
├── mxene_label_capacity_boxplot.png
├── mxene_original_vs_*_comparison.png    # Validation
└── ...
```

## Troubleshooting

**Problem:** Categorical label treated as continuous
**Solution:** Reduce unique values or force string type: `df['label'] = df['label'].astype(str)`

**Problem:** String representations too long (e.g., "TiTiTiTiHfHfHfHf")
**Solution:** This is normal for compositional formulas - statistics still work correctly

**Problem:** Box plots fail on categorical
**Solution:** Toolkit automatically skips categorical labels for box plots

**Problem:** Too many categories in legend
**Solution:** Legend shows all categories - if >10, consider grouping similar categories
