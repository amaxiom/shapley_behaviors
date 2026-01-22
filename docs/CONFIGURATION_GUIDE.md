# Configuration Guide

This guide explains all configuration variables for the behavioral space explorer scripts.

## Required Variables

### SEED
**Type:** Integer  
**Example:** `SEED = 42`  
**Purpose:** Random seed for reproducibility. All random operations (permutations, PCA) will be deterministic.

### N_PERMUTATIONS
**Type:** Integer  
**Range:** 100-1000 recommended  
**Example:** `N_PERMUTATIONS = 200`  
**Purpose:** Number of permutations for Shapley value estimation. Higher = more accurate but slower.
- **100:** Quick exploration (5-10 min for ~1000 samples)
- **200:** Good balance (10-20 min)
- **500-1000:** Publication quality (30-60 min)

### N_JOBS
**Type:** Integer  
**Example:** `N_JOBS = -1`  
**Purpose:** Number of parallel workers for computation.
- **-1:** Use all CPU cores (recommended)
- **1:** Single core (slow but uses less memory)
- **N:** Use N cores

### DATASET_NAME
**Type:** String  
**Example:** `DATASET_NAME = "Al"`  
**Purpose:** Short identifier used in all output filenames.
- Keep it short (2-10 characters)
- No spaces or special characters
- Examples: "Al", "mxene", "steel", "polymer"

### DATA_FILE
**Type:** String (path)  
**Example:** `DATA_FILE = "al_data.csv"`  
**Purpose:** Path to your CSV file.
- Can be relative: `"data.csv"` or `"data/alloys.csv"`
- Or absolute: `"/home/user/data/alloys.csv"`
- Must be a CSV file

### ID_COLUMN
**Type:** String  
**Example:** `ID_COLUMN = "ID"`  
**Purpose:** Column name to use as sample identifier/index.
- Must be unique for each sample
- Common names: "ID", "Formula", "Sample_ID", "Name"
- Will be used as row index in all output tables

### DROP_COLUMNS
**Type:** List of strings OR None  
**Examples:**
```python
DROP_COLUMNS = ["Processing", "Age", "Date"]  # Drop specific columns
DROP_COLUMNS = []                              # Don't drop any
DROP_COLUMNS = None                            # Don't drop any (alternative)
```
**Purpose:** Columns to exclude from analysis (not features, not labels).
- Drop metadata: dates, processing notes, researcher names
- Drop identifiers: batch numbers, project codes
- **Use `None` or `[]` if all columns are features or labels**
- Script uses `errors='ignore'` so listing non-existent columns is safe

### LABEL_COLUMNS
**Type:** List of strings  
**Example:** `LABEL_COLUMNS = ['Strength', 'Ductility', 'Type']`  
**Purpose:** Target/response variables to visualize and analyze.
- Can be continuous (Strength, Voltage) or categorical (Type, Phase)
- Auto-detected: Toolkit determines if continuous or categorical
- Can be empty list `[]` if you only want behavioral spaces without label coloring
- Missing data handled automatically (shown in red)

### OUTPUT_DIR
**Type:** String (path)  
**Example:** `OUTPUT_DIR = 'behavioral_exploration'`  
**Purpose:** Directory where all outputs will be saved.
- Created automatically if doesn't exist
- Can be relative or absolute path
- All CSVs, plots, and .npy files saved here

## Optional Variables

### SELECTED_FEATURES
**Type:** List of strings OR None  
**Examples:**
```python
SELECTED_FEATURES = ['Si', 'Fe', 'Mn']  # Visualize 3 features
SELECTED_FEATURES = None                # Skip feature plots
```
**Purpose:** Specific features to visualize with concentration gradients (grey→red).
- **Recommended:** 3 features (more creates too many plots)
- **None:** Skip feature concentration plots entirely
- Useful for highlighting key alloying elements or important components

### CREATE_OUTLIER_PROFILES
**Type:** Boolean  
**Default:** `True`  
**Examples:**
```python
CREATE_OUTLIER_PROFILES = True   # Generate automatic outlier profiles
CREATE_OUTLIER_PROFILES = False  # Skip outlier profiles
```
**Purpose:** Controls whether profile plots are automatically generated for outliers.
- **True:** Creates detailed Shapley value histograms for top outliers in each space
- **False:** Skips automatic generation (saves time, you can still create manually)
- Profile plots show feature-by-feature contributions ordered by importance
- Helps understand WHY a sample is an outlier
- Set to `False` for large datasets or quick exploration

### MAX_OUTLIERS_PER_SPACE
**Type:** Integer  
**Default:** `5`  
**Range:** 1-20 recommended  
**Examples:**
```python
MAX_OUTLIERS_PER_SPACE = 3   # Quick overview
MAX_OUTLIERS_PER_SPACE = 5   # Good balance (default)
MAX_OUTLIERS_PER_SPACE = 10  # Comprehensive analysis
```
**Purpose:** Number of top outliers to profile per behavioral space.
- Controls how many profile plots are created per space
- Only used if `CREATE_OUTLIER_PROFILES = True`
- Top outliers selected by z-score (highest first)
- **1:** Just the most extreme outlier
- **3:** Quick overview
- **5:** Balanced (default)
- **10+:** Deep dive into outlier characteristics

### BEHAVIORAL_SPACES_FILE (region explorer only)
**Type:** String (path)  
**Example:** `BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/Al_behavioral_spaces_all.npy'`  
**Purpose:** Path to .npy file created by behavioral_space_explorer.py
- Must run space explorer first to create this file
- Path typically: `{OUTPUT_DIR}/{DATASET_NAME}_behavioral_spaces_all.npy`

### PLOT_MODE (region explorer only)
**Type:** String  
**Options:** `'combined'` or `'separate'`  
**Example:** `PLOT_MODE = 'combined'`  
**Purpose:** How to visualize multiple regions.
- **'combined':** All regions for each space on one plot (recommended)
- **'separate':** Individual plot for each region (many files)

### USER_REGIONS (region explorer only)
**Type:** Dictionary  
**Example:**
```python
USER_REGIONS = {
    'high_strength': {
        'space': 'variance',
        'pc1_range': (0.3, 0.6),
        'pc2_range': (-0.2, 0.2),
        'description': 'High strength region',
        'color': 'red'
    }
}
```
**Purpose:** Define regions to extract and analyze.

**Sub-parameters:**
- **space:** Which behavioral space ('variance', 'skewness', 'kurtosis', 'entropy')
- **pc1_range:** (min, max) for PC1 axis
- **pc2_range:** (min, max) for PC2 axis  
- **description:** Human-readable description
- **color:** Matplotlib color for visualization ('red', 'blue', '#FF0000', etc.)

---

## Configuration Templates

### Minimal Configuration (All Defaults)

```python
SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "mydata"
DATA_FILE = "data.csv"
ID_COLUMN = "ID"
DROP_COLUMNS = None              # No columns to drop
LABEL_COLUMNS = ['Target']       # Single target variable
OUTPUT_DIR = 'output'

SELECTED_FEATURES = None         # No feature plots
```

### Alloys Configuration

```python
SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "Al"
DATA_FILE = "aluminium_alloys.csv"
ID_COLUMN = "Alloy_ID"
DROP_COLUMNS = ["Date", "Researcher", "Notes", "Batch"]
LABEL_COLUMNS = ['Tensile_Strength', 'Elongation', 'Hardness']
OUTPUT_DIR = 'behavioral_exploration'

SELECTED_FEATURES = ['Cu', 'Mg', 'Zn']  # Key strengthening elements

# Outlier analysis
CREATE_OUTLIER_PROFILES = True
MAX_OUTLIERS_PER_SPACE = 5
```

### MXenes Configuration (Mixed Labels)

```python
SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "mxene"
DATA_FILE = "mxenes.csv"
ID_COLUMN = "Formula"
DROP_COLUMNS = ["Reference", "DOI", "Year"]
LABEL_COLUMNS = [
    'Voltage', 'Capacity',        # Continuous
    'M', 'X', 'T', 'Z'            # Categorical
]
OUTPUT_DIR = 'behavioral_exploration'

SELECTED_FEATURES = ['Ti', 'Nb', 'Mo']  # Common metals

# Outlier analysis
CREATE_OUTLIER_PROFILES = True
MAX_OUTLIERS_PER_SPACE = 5
```

### No Labels Configuration (Clustering Only)

```python
SEED = 42
N_PERMUTATIONS = 200
N_JOBS = -1

DATASET_NAME = "explore"
DATA_FILE = "compositions.csv"
ID_COLUMN = "Sample"
DROP_COLUMNS = None
LABEL_COLUMNS = []               # No labels - just find structure
OUTPUT_DIR = 'clustering_results'

SELECTED_FEATURES = None
```

---

## Common Mistakes

### ❌ Wrong: Including label in DROP_COLUMNS
```python
DROP_COLUMNS = ["Date", "Strength"]  # Don't drop labels!
LABEL_COLUMNS = ['Strength']
```
**Fix:** Remove 'Strength' from DROP_COLUMNS

### ❌ Wrong: Using non-existent columns
```python
LABEL_COLUMNS = ['Voltage', 'Capcity']  # Typo: "Capcity"
```
**Fix:** Check column names in your CSV (case-sensitive)

### ❌ Wrong: ID_COLUMN not unique
```python
ID_COLUMN = "Type"  # Multiple samples have same type
```
**Fix:** Use a truly unique identifier column

### ❌ Wrong: Too many SELECTED_FEATURES
```python
SELECTED_FEATURES = ['Fe', 'Cr', 'Ni', 'Mo', 'Cu', 'Si', 'Mn', ...]  # 20+ features
```
**Fix:** Choose top 3 most important features, or use `None`

### ❌ Wrong: Path doesn't exist
```python
DATA_FILE = "data/alloys.csv"  # But 'data' folder doesn't exist
```
**Fix:** Use correct path or move file

---

## Advanced Tips

### Handling Large Datasets
```python
N_PERMUTATIONS = 100    # Start with fewer permutations
N_JOBS = 8              # Limit cores if memory is tight
```

### Publication Quality
```python
N_PERMUTATIONS = 1000   # Maximum accuracy
SEED = 42               # Always use same seed for reproducibility
CREATE_OUTLIER_PROFILES = True
MAX_OUTLIERS_PER_SPACE = 10  # Comprehensive outlier analysis
```

### Deep Outlier Analysis
```python
CREATE_OUTLIER_PROFILES = True
MAX_OUTLIERS_PER_SPACE = 10  # Analyze top 10 outliers per space

# Then manually create additional profiles for specific samples:
# fig, ax = create_profile_plot(
#     Phi=behavioral_spaces['variance'],
#     X=X, 
#     feature_names=feature_names,
#     sample_idx=42,
#     space_name='variance',
#     sample_id='Sample_1247'
# )
```

### Quick Exploration
```python
N_PERMUTATIONS = 50     # Very fast, less accurate
N_JOBS = -1             # Use all cores
SELECTED_FEATURES = None  # Skip feature plots
CREATE_OUTLIER_PROFILES = False  # Skip profile plots for speed
```

### Multiple Datasets
```python
# Keep OUTPUT_DIR same but change DATASET_NAME
OUTPUT_DIR = 'all_results'
DATASET_NAME = "Al_6xxx"   # First dataset
# ... run ...
DATASET_NAME = "Al_7xxx"   # Second dataset  
# ... run ...
```

All outputs will be in same folder with different prefixes.

---

## Validation Checklist

Before running, verify:

- [ ] CSV file exists and is readable
- [ ] ID_COLUMN exists and is unique
- [ ] LABEL_COLUMNS exist in the CSV
- [ ] DROP_COLUMNS don't include labels
- [ ] OUTPUT_DIR is writable
- [ ] N_PERMUTATIONS > 0
- [ ] If using SELECTED_FEATURES, they exist in CSV
- [ ] DATASET_NAME contains no spaces or special characters

---

*This guide is part of the Shapley Behavioral Analysis Toolkit*  
*Authors: Amanda S. Barnard and Tommy Liu*
