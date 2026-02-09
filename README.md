# Shapley Behavioral Analysis Toolkit

**Reveal hidden patterns in data using Shapley value-based behavioral transformations.**

This toolkit provides two complementary Python tools for analyzing high-dimensional datasets by transforming raw features into interpretable behavioral spaces that expose clustering patterns invisible in the original data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/shapley_behaviors.svg)](https://pypi.org/project/shapley_behaviors/)

## What Does This Do?

Traditional analysis of data often misses important patterns because features interact in complex, nonlinear ways. This toolkit:

1. **Transforms** compositional features using Shapley values to create "behavioral spaces" that quantify how each feature contributes to statistical properties (variance, skewness, kurtosis, entropy)
2. **Reveals** clustering patterns that are invisible in the original compositional space
3. **Validates** that discovered patterns are real, not artifacts of dimensionality reduction
4. **Analyzes** user-defined regions to understand what compositional signatures drive different material properties

## Scientific Foundation

**Shapley values** (from cooperative game theory) fairly distribute a "coalition value" among players. We apply this to materials: features are players, and coalition values are statistical properties of feature subsets. This reveals which features most strongly influence the distributional characteristics of materials.

**Key insight:** Samples that cluster tightly in behavioral space (similar feature interaction patterns) do NOT cluster in original compositional space. This proves the transformation reveals genuine structure rather than just re-projecting existing patterns.

## Installation

### Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- joblib

### Setup

```bash
git clone https://github.com/amaxiom/shapley_behaviors.git
cd shapley_behaviors
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Behavioral Spaces

```python
import numpy as np
import pandas as pd

# Configuration
SEED = 42
N_PERMUTATIONS = 200  # Use 100-1000 depending on dataset size
N_JOBS = -1  # Use all CPU cores

DATASET_NAME = "Mg"
DATA_FILE = "mg_data.csv"
ID_COLUMN = "ID"
DROP_COLUMNS = ["Condition", "Process", "DOI"]  # Non-feature columns, or None
LABEL_COLUMNS = ['Yield_Strength', 'Tensile_Strength', 'Ductility']
OUTPUT_DIR = 'behavioral_exploration'

SELECTED_FEATURES = ['Al', 'Zn', 'Y']  # Optional: features to visualize, or None

# Run analysis
%run -i behavioral_space_explorer.py
```

**Output:**
- 4 behavioral spaces (variance, skewness, kurtosis, entropy)
- Hopkins statistics measuring clustering tendency
- PCA visualizations colored by labels and feature concentrations
- Outlier detection for each space
- **Profile plots showing feature-level breakdown for top outliers**

### Step 2: Explore Regions

After visually inspecting the behavioral space plots, define regions of interest:

```python
# Use same configuration as Step 1, plus:

USER_REGIONS = {
    'high_strength': {
        'space': 'variance',
        'pc1_range': (0.3, 0.6),
        'pc2_range': (-0.2, 0.2),
        'description': 'High tensile strength region',
        'color': 'red'
    },
    'high_ductility': {
        'space': 'variance',
        'pc1_range': (-0.5, -0.2),
        'pc2_range': (-0.3, 0.3),
        'description': 'High elongation region',
        'color': 'blue'
    }
}

PLOT_MODE = 'combined'  # or 'separate'
BEHAVIORAL_SPACES_FILE = 'behavioral_exploration/Al_behavioral_spaces_all.npy'

# Run region analysis
%run -i behavioral_region_explorer.py
```

**Output:**
- Composition tables showing enrichment/depletion of each feature per region
- Label statistics comparing regions
- Box plots, heatmaps, and distribution visualizations
- **Critical validation:** Original vs behavioral space comparisons proving regions are meaningful

## Example Results

### Magnesium Alloys

**Finding:** High-strength AZ-series alloys (Al-Zn) cluster in variance behavioral space but are randomly distributed in original compositional space.

**Interpretation:** The clustering is driven by specific patterns of Al+Zn interactions, not simple concentrations.

### MXenes

**Finding:** Different termination groups (F, O, OH) separate clearly in entropy space.

**Interpretation:** Termination chemistry creates distinct behavioral signatures in how elements contribute to compositional diversity.

## Tools Overview

### `behavioral_space_explorer.py`

**Purpose:** Generate and analyze behavioral transformations

**Key Features:**
- Computes 4 Shapley behavioral spaces (variance, skewness, kurtosis, entropy)
- Auto-detects continuous vs categorical labels
- Handles missing data (shown in red)
- Parallel processing for speed
- Comprehensive statistical validation (Hopkins, PCA variance, outliers)
- **Profile plots for understanding outliers** (SHAP-like feature breakdowns)

**When to use:** Starting point for any new dataset

### `behavioral_region_explorer.py`

**Purpose:** Analyze user-defined regions in behavioral spaces

**Key Features:**
- Extract samples from specified PC1/PC2 ranges
- Quantify compositional enrichment/depletion (Change_% vs dataset average)
- Compare labels across regions (box plots, statistics)
- **Validate** regions by comparing original vs behavioral space
- Works with continuous and categorical labels

**When to use:** After identifying interesting patterns in behavioral spaces

### `shapley_behaviors.py`

**Purpose:** Core Shapley value computation engine

**Key Features:**
- Monte Carlo permutation sampling
- Parallelized across features
- Four value functions: variance, skewness, kurtosis, entropy
- Optimized for large datasets

## Repository Structure

```
shapley_behaviors/
├── behavioral_space_explorer.py    # Main analysis tool
├── behavioral_region_explorer.py   # Region extraction and analysis
├── shapley_behaviors.py            # Core Shapley computations
├── README.md
├── LICENSE
├── requirements.txt
├── examples/
│   ├── magnesium_alloys/
│   │   ├── README.md
│   │   ├── run_space_explorer.py
│   │   └── run_region_explorer.py
│   └── mxenes/
│       ├── README.md
│       ├── mxene_meldeleev.csv
│       ├── run_space_explorer.py
│       └── run_region_explorer.py
└── docs/
    ├── toolkit_description.txt
    ├── CONFIGURATION_GUIDE.md
    └── INTERPRETATION_GUIDE.md
```

## How It Works

### 1. Shapley Behavioral Transformation

For each sample and each feature, compute:

**Φ(feature) = Average contribution of feature to coalition value across all permutations**

Where coalition value is a statistical property (variance, skewness, etc.) of the feature subset.

### 2. Behavioral Spaces

Each transformation creates a new space where:
- **Variance space:** Features weighted by contribution to compositional variance
- **Skewness space:** Features weighted by contribution to distributional asymmetry
- **Kurtosis space:** Features weighted by contribution to tail heaviness
- **Entropy space:** Features weighted by contribution to uncertainty/diversity

### 3. Pattern Discovery

Apply PCA to behavioral spaces to find 2D projections where samples cluster by similar behavioral signatures.

### 4. Validation

**Critical test:** Do regions that cluster in behavioral space also cluster in original space?
- **NO** → Transformation revealed new structure (genuine discovery)
- **YES** → Just found existing clusters (not interesting)

This toolkit ensures you find the first case, not the second.

## Documentation

- **[Comprehensive Description](docs/toolkit_description.txt)** - Complete technical details
- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - How to set up for study
- **[Interpretation Guide](docs/INTERPRETTION_GUIDE.md)** - How to interpret results
- **[Examples](examples/)** - Step-by-step tutorials

## Supported Data Types

### Labels (Target Dependent Variables)
- **Continuous:** Mechanical properties, voltages, capacities (auto-detected)
- **Categorical:** Material types, processing routes, phase labels (auto-detected)
- **Mixed:** Both in the same dataset

### Features (Independent Variables)
- **Must be numeric:** e.g. Elemental concentrations, structural parameters, biomarkers
- **Any dimensionality:** Works with 5-500+ features
- **Sparse OK:** Features can be zero for many samples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{shapley_behaviors,
  author = {Barnard, Amanda S. and Liu, Tommy},
  title = {Shapley Behavioral Analysis Toolkit},
  year = {2026},
  url = {https://github.com/amaxiom/shapley_behaviors},
  version = {0.1.1}
}
```
```bibtex
@article{Liu2025UnderstandingIP,
  title={Understanding Interpretable Patterns of Shapley Behaviours in Materials Data},
  author={Tommy Liu and Amanda S. Barnard},
  journal={Machine Learning: Engineering},
  year={2025},
  volume={1},
  issue = {1},
  pages = {015004},
  doi = {10.1088/3049-4761/adaaf6}
}
```

## Authors

- **Amanda S. Barnard** - *Lead Developer, Methodology* - [amaxiom](https://github.com/amaxiom)
  - Senior Professor and Computational Science Lead, ANU School of Computing
  - Member of the Order of Australia
  
- **Tommy Liu** - *Co-Developer, Implementation* - [uilymmot](https://github.com/uilymmot)
  - Contributed to core algorithm development and validation methodology

## Acknowledgments

- Shapley values concept from cooperative game theory (Lloyd Shapley, 1953)
- Hopkins statistic implementation adapted from scikit-learn
- Inspired by materials informatics and interpretable machine learning communities

## Contact

- **Primary Contact:** Amanda S. Barnard
- **Issues:** [GitHub Issues](https://github.com/amaxiom/shapley_behaviors/issues)

## 🔗 Related Resources

- [Materials Project](https://materialsproject.org/) - Materials database
- [SHAP](https://github.com/slundberg/shap) - General Shapley value ML explainability
- [Matminer](https://hackingmaterials.lbl.gov/matminer/) - Materials data mining tools

---
