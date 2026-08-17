# Shapley Behavioral Analysis Toolkit

**Reveal hidden compositional patterns in data using Shapley value-based behavioral transformations.**

This toolkit provides a family of complementary Python tools for analyzing high-dimensional datasets by transforming raw features into interpretable behavioral spaces that expose clustering patterns invisible in the original data: a space explorer to generate and validate the behavioral transformations, a region explorer with integrated automatic break detection to define and analyze regions of the projections, and a k-means cluster explorer for exclusive cluster-based analysis.

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

* Python 3.8+
* numpy
* pandas
* matplotlib
* scikit-learn
* scipy
* joblib

### Setup

```bash
git clone https://github.com/amaxiom/shapley\_behaviors.git
cd shapley\_behaviors
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Behavioral Spaces

```python
import numpy as np
import pandas as pd

# Configuration
SEED = 42
N\_PERMUTATIONS = 200  # Use 100-1000 depending on dataset size
N\_JOBS = -1  # Use all CPU cores

DATASET\_NAME = "Mg"
DATA\_FILE = "mg\_data.csv"
ID\_COLUMN = "ID"
DROP\_COLUMNS = \["Condition", "Process", "DOI"]  # Non-feature columns, or None
LABEL\_COLUMNS = \['Yield\_Strength', 'Tensile\_Strength', 'Ductility']
OUTPUT\_DIR = 'behavioral\_exploration'

SELECTED\_FEATURES = \['Al', 'Zn', 'Y']  # Optional: features to visualize, or None

# Run analysis
%run -i behavioral\_space\_explorer.py
```

**Output:**

* 4 behavioral spaces (variance, skewness, kurtosis, entropy)
* Hopkins statistics measuring clustering tendency
* PCA visualizations colored by labels and feature concentrations
* Outlier detection for each space
* **Profile plots showing feature-level breakdown for top outliers**

### Step 2: Detect Break Zones (optional but recommended)

Region boundaries do not need to be guessed. Run the region explorer
without `USER\_REGIONS` and it reports statistically significant break
zones and satellite candidate gaps along PC1 and PC2 of each requested
space, saves diagnostic figures, and leaves the detected boundaries in
the notebook namespace:

```python
# Use same configuration as Step 1, plus:
BREAK\_SPACES = \['variance']    # default: all spaces in the file
BREAK\_Z\_THRESHOLD = 2.5        # gap significance (z-score)
MIN\_REGION\_FRACTION = 0.05     # min fraction of samples on each side
MAX\_STRAGGLER\_FRACTION = 0.02  # merge gaps separated by <= this fraction

USER\_REGIONS = None            # break detection only
%run -i behavioral\_region\_explorer.py
# -> break\_zones, pc1\_zones, pc2\_zones available in the namespace
```

Gap midpoints make robust rectangle bounds because the gap interior
contains no samples.

### Step 3: Explore Regions

Define regions of interest (typically from the detected boundaries)
and run the region explorer again:

```python
USER\_REGIONS = {
    'high\_strength': {
        'space': 'variance',
        'pc1\_range': (0.3, 0.6),
        'pc2\_range': (-0.2, 0.2),
        'description': 'High tensile strength region',
        'color': 'red'
    },
    'high\_ductility': {
        'space': 'variance',
        'pc1\_range': (-0.5, -0.2),
        'pc2\_range': (-0.3, 0.3),
        'description': 'High elongation region',
        'color': 'blue'
    }
}

PLOT\_MODE = 'combined'  # or 'separate'
BEHAVIORAL\_SPACES\_FILE = 'behavioral\_exploration/Mg\_behavioral\_spaces.npy'

# Run region analysis
%run -i behavioral\_region\_explorer.py
```

**Output:**

* Composition tables showing enrichment/depletion of each feature per region
* Label statistics comparing regions
* Box plots, heatmaps, and distribution visualizations
* **Critical validation:** Original vs behavioral space comparisons proving regions are meaningful

## Example Results

### Magnesium Alloys

**Finding:** High-strength AZ-series alloys (Al-Zn) cluster in variance behavioral space but are randomly distributed in original compositional space.

**Interpretation:** The clustering is driven by specific patterns of Al+Zn interactions, not simple concentrations.

### MXenes

**Finding:** Different termination groups (F, O, OH) separate clearly in entropy space.

**Interpretation:** Termination chemistry creates distinct behavioral signatures in how elements contribute to compositional diversity.

## Tools Overview

### `behavioral\_space\_explorer.py`

**Purpose:** Generate and analyze behavioral transformations

**Key Features:**

* Computes 4 Shapley behavioral spaces (variance, skewness, kurtosis, entropy)
* Auto-detects continuous vs categorical labels
* Handles missing data (shown in red)
* Parallel processing for speed
* Comprehensive statistical validation (Hopkins, PCA variance, outliers)
* **Profile plots for understanding outliers** (SHAP-like feature breakdowns)

**When to use:** Starting point for any new dataset

### `behavioral\_region\_explorer.py`

**Purpose:** Detect region boundaries automatically, then analyze user-defined regions in behavioral spaces

**Key Features:**

* **Integrated automatic break detection:** statistically significant nearest-neighbor gaps merge into break zones; strong sub-threshold gaps are reported as satellite candidates (run without `USER\_REGIONS` for a break-detection-only pass)
* Extract samples from specified PC1/PC2 ranges
* Quantify compositional enrichment/depletion (Change\_% vs dataset average)
* Compare labels across regions (box plots, statistics)
* **Validate** regions by comparing original vs behavioral space
* Works with continuous and categorical labels
* Safe with non-unique sample IDs (all statistics use positional indexing)

**When to use:** After generating behavioral spaces, to find and characterise their internal structure

### `behavioral\_cluster\_explorer.py`

**Purpose:** K-means cluster analysis of behavioral spaces, as an alternative to rectangular regions

**Key Features:**

* Clusters the 2D PCA projection of any behavioral space; every sample belongs to exactly one cluster
* Clusters labelled A, B, C, ... left-to-right by PC1 centroid
* Handles mixed continuous and categorical labels: continuous targets get boxplots and mean/std/min/median/max, categorical targets get a dominant-category summary, a cluster-by-category cross-tab CSV, and a stacked composition bar
* Per-cluster sample lists, full-data exports, property and composition summaries
* Cluster scatter plot and per-property box plots

**When to use:** When exclusive, algorithmically-assigned groups are preferred over hand-defined rectangles, or to cross-validate region definitions

### `shapley\_behaviors.py`

**Purpose:** Core Shapley value computation engine

**Key Features:**

* Monte Carlo permutation sampling
* Parallelized across features
* Four value functions: variance, skewness, kurtosis, entropy
* Optimized for large datasets

## Repository Structure

```
shapley-behavioral-analysis/
├── behavioral\_space\_explorer.py    # Main analysis tool
├── behavioral\_region\_explorer.py   # Break detection + region analysis
├── behavioral\_cluster\_explorer.py  # K-means cluster analysis
├── shapley\_behaviors.py            # Core Shapley computations
├── README.md
├── LICENSE
├── requirements.txt
├── examples/
│   ├── gmm\_region\_finder.ipynb     # GMM-based boundary detection (alternative)
│   ├── GMM\_REGION\_FINDER.md
│   ├── magnesium\_alloys/
│   │   ├── README.md
│   │   ├── run\_space\_explorer.py
│   │   └── run\_region\_explorer.py
│   └── mxenes/
│       ├── README.md
│       ├── run\_space\_explorer.py
│       └── run\_region\_explorer.py
└── docs/
    ├── toolkit\_description.txt
    ├── CONFIGURATION\_GUIDE.md
    └── INTERPRETATION\_GUIDE.md
```

## How It Works

### 1\. Shapley Behavioral Transformation

For each sample and each feature, compute:

**Φ(feature) = Average contribution of feature to coalition value across all permutations**

Where coalition value is a statistical property (variance, skewness, etc.) of the feature subset.

### 2\. Behavioral Spaces

Each transformation creates a new space where:

* **Variance space:** Features weighted by contribution to compositional variance
* **Skewness space:** Features weighted by contribution to distributional asymmetry
* **Kurtosis space:** Features weighted by contribution to tail heaviness
* **Entropy space:** Features weighted by contribution to uncertainty/diversity

### 3\. Pattern Discovery

Apply PCA to behavioral spaces to find 2D projections where samples cluster by similar behavioral signatures.

### 4\. Validation

**Critical test:** Do regions that cluster in behavioral space also cluster in original space?

* **NO** → Transformation revealed new structure (genuine discovery)
* **YES** → Just found existing clusters (not interesting)

This toolkit ensures you find the first case, not the second.

## Documentation

* [**Comprehensive Description**](docs/toolkit_description.txt) - Complete technical details
* [**Configuration Guide**](docs/CONFIGURATION_GUIDE.md) - How to set up for study
* [**Interpretation Guide**](docs/INTERPRETTION_GUIDE.md) - How to interpret results
* [**Examples**](examples/) - Step-by-step tutorials

## Supported Data Types

### Labels (Target Dependent Variables)

* **Continuous:** Mechanical properties, voltages, capacities (auto-detected)
* **Categorical:** Material types, processing routes, phase labels (auto-detected)
* **Mixed:** Both in the same dataset

### Features (Independent Variables)

* **Must be numeric:** e.g. Elemental concentrations, structural parameters, biomarkers
* **Any dimensionality:** Works with 5-500+ features
* **Sparse OK:** Features can be zero for many samples

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{shapley\_behavioral\_analysis,
  author = {Barnard, Amanda S. and Liu, Tommy},
  title = {Shapley Behavioral Analysis Toolkit},
  year = {2026},
  url = {https://github.com/amaxiom/shapley_behaviors}
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

* **Amanda S. Barnard** - *Lead Developer, Methodology* - [amaxiom](https://github.com/amaxiom)

  * Senior Professor and Computational Science Lead, ANU School of Computing
  * Member of the Order of Australia
  * Prime Minister's Prize for Physical Scientist of the Year
* **Tommy Liu** - *Co-Developer, Implementation* - [uilymmot](https://github.com/uilymmot)

  * Contributed to core algorithm development and validation methodology

## Acknowledgments

* Shapley values concept from cooperative game theory (Lloyd Shapley, 1953)
* Hopkins statistic implementation adapted from scikit-learn
* Inspired by materials informatics and interpretable machine learning communities

## Contact

* **Primary Contact:** Amanda S. Barnard
* **Email:** amanda.s.barnard@anu.edu.au
* **Issues:** [GitHub Issues](https://github.com/amaxiom/shapley-behaviors/issues)

## 🔗 Related Resources

* [Materials Project](https://materialsproject.org/) - Materials database
* [SHAP](https://github.com/slundberg/shap) - General Shapley value ML explainability
* [Matminer](https://hackingmaterials.lbl.gov/matminer/) - Materials data mining tools

\---

