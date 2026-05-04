# GMM Region Finder

`gmm_region_finder.ipynb` automatically detects region boundaries in a
PCA projection of any Shapley behavioral space using Gaussian Mixture Models
(GMM) with a BIC elbow criterion, then generates a `USER_REGIONS` dictionary
ready to pass to `behavioral_region_explorer.py`.

---

## Prerequisites

Run `behavioral_space_explorer.py` first to compute the behavioral spaces for
your dataset.  It saves a `.npy` file (e.g. `behavioral_exploration/mxene_behavioral_spaces.npy`)
that this notebook loads.

**Python packages required** (all standard in a scientific Python environment):

```
numpy  pandas  matplotlib  scikit-learn  scipy
```

---

## Quickstart

1. Open `gmm_region_finder.ipynb` in Jupyter.
2. Edit the **Configuration** cell at the top:

   | Variable | Description | Default |
   |---|---|---|
   | `BEHAVIORAL_SPACES_FILE` | Path to the `.npy` file from `behavioral_space_explorer.py` | `behavioral_exploration/mxene_behavioral_spaces.npy` |
   | `SPACE_NAME` | Which space to analyse: `variance`, `skewness`, `kurtosis`, or `entropy` | `skewness` |
   | `DATASET_NAME` | Used in output filenames | `mxene` |
   | `OUTPUT_DIR` | Where to save figures | `behavioral_exploration` |

3. Run all cells.
4. Copy the printed `USER_REGIONS` dict (Step 6 output) into your
   `behavioral_region_explorer.py` session, or use Step 8 to run it directly.

---

## GMM tuning parameters

These control how aggressively boundaries are detected.
The defaults work well for most behavioral spaces; adjust only if the
diagnostic plots show unwanted splits or missed gaps.

| Parameter | What it does | Default | Adjust if... |
|---|---|---|---|
| `K_MAX` | Maximum number of Gaussian components to try | `8` | Your distribution has more than 8 distinct modes |
| `MIN_WEIGHT` | Ignore components whose mixing weight is below this fraction | `0.05` | Small sub-populations are being ignored (lower) or noise components are included (raise) |
| `ELBOW_FRACTION` | Stop adding components when the BIC gain drops below this fraction of the largest single-step gain | `0.10` | Too many components selected (raise to e.g. 0.20); too few (lower to e.g. 0.05) |
| `MAX_VALLEY_DENSITY` | Accept a boundary only if the GMM density at the crossing point is below this fraction of the global peak | `0.10` | Boundaries are being placed inside dense regions (lower); genuine gaps are being missed (raise to e.g. 0.20) |

---

## How boundary detection works

For each PC axis independently:

1. GMMs with k = 1 to `K_MAX` components are fitted to the marginal distribution.
2. The optimal k is chosen by a **BIC elbow criterion**: components are added
   only while each additional component provides at least `ELBOW_FRACTION` of
   the largest single-step BIC improvement seen so far.
3. Components with mixing weight below `MIN_WEIGHT` are discarded as noise.
4. For each pair of adjacent components (sorted by mean), the crossing point
   of the two weighted Gaussians is found.
5. A boundary is **kept** only if the total GMM density at that crossing falls
   below `MAX_VALLEY_DENSITY × peak density`, ensuring it sits in a genuine
   low-density gap rather than a shallow saddle within a dense region.

Regions are then defined as rectangular coordinate partitions of the PC1–PC2
plane separated by the accepted boundary positions.  This is deterministic and
fully reproducible.

---

## Reading the diagnostic plots

Each run produces a 2×2 figure saved as
`{OUTPUT_DIR}/{DATASET_NAME}_{SPACE_NAME}_gmm_boundaries.png`.

```
┌─────────────────────┬─────────────────────┐
│  PC1 histogram      │  PC1 BIC curve      │
│  (left column)      │  (right column)     │
├─────────────────────┼─────────────────────┤
│  PC2 histogram      │  PC2 BIC curve      │
└─────────────────────┴─────────────────────┘
```

**Histogram panels:**
- Blue bars: data distribution
- Coloured dashed curves: individual GMM components (solid = weight ≥ `MIN_WEIGHT`; dotted = below threshold)
- Black solid curve: total GMM mixture
- Grey dotted line: `MAX_VALLEY_DENSITY × peak` cutoff
- Red dashed vertical lines: accepted boundaries

A well-calibrated result shows red boundaries sitting clearly below the grey
cutoff line, in visually obvious gaps between modes.

**BIC panels:**
- Black curve: BIC score for each k
- Red dashed line: selected elbow k

A clear elbow followed by a flat plateau confirms the choice is well-determined.
If the BIC keeps decreasing monotonically, try raising `K_MAX`.

---

## Overriding boundaries

If the automatic result needs adjustment, edit the override cell (Step 4):

```python
# Force a single PC1 split at 0.35
pc1_boundaries = np.array([0.35])

# Force two PC2 splits
pc2_boundaries = np.array([-0.15, 0.20])

# Remove all boundaries on PC2 (one region across the full axis)
pc2_boundaries = np.array([])
```

Re-run Steps 5–7 after any override to regenerate `USER_REGIONS` and the preview plot.

---

## Output files

| File | Contents |
|---|---|
| `{OUTPUT_DIR}/{DATASET_NAME}_{SPACE_NAME}_gmm_boundaries.png` | Diagnostic 2×2 plot |
| `{OUTPUT_DIR}/{DATASET_NAME}_{SPACE_NAME}_gmm_regions_preview.png` | 2-D scatter with region grid overlaid |

The `USER_REGIONS` dict is printed to the notebook output (Step 6) and is
also available as the Python variable `USER_REGIONS` for direct use in Step 8.

---

## Connecting to behavioral_region_explorer.py

**Option A — copy and paste:**  
Copy the Step 6 output into any notebook or script that calls
`behavioral_region_explorer.py` via `%run -i`.

**Option B — run directly from this notebook:**  
Fill in the dataset variables in Step 8 and uncomment `%run`:

```python
DATA_FILE     = 'my_data.csv'
ID_COLUMN     = 'ID'
DROP_COLUMNS  = ['col_to_drop']
LABEL_COLUMNS = ['label1', 'label2']
PLOT_MODE     = 'combined'

%run -i behavioral_region_explorer.py
```

All variables set in Steps 1–6 (`USER_REGIONS`, `BEHAVIORAL_SPACES_FILE`,
`DATASET_NAME`, `OUTPUT_DIR`, `PLOT_MODE`) are already in scope, so
`behavioral_region_explorer.py` picks them up without any duplication.

---

## Region naming

Auto-generated region names follow the pattern `Region_PC1_{i}_PC2_{j}`,
where `i` = 1 is the lowest PC1 band and `j` = 1 is the lowest PC2 band.

For example, with 1 PC1 boundary and 2 PC2 boundaries you get a 2×3 grid:

```
PC2 band 3 (high) │ Region_PC1_1_PC2_3 │ Region_PC1_2_PC2_3
PC2 band 2 (mid)  │ Region_PC1_1_PC2_2 │ Region_PC1_2_PC2_2
PC2 band 1 (low)  │ Region_PC1_1_PC2_1 │ Region_PC1_2_PC2_1
                        PC1 band 1           PC1 band 2
```

Rename the keys in the printed dict before pasting if you prefer descriptive
labels (e.g. `C_Region_I`, `N_Region_III`).

---

---

# Part 2 — K-means Assisted Boundary Finder

## When to use this method

Use Part 2 when the PCA projection shows a **large central bulk with satellite arms**:
a small subset of samples has extreme feature values that radiate outward along one
axis only. The k-means finder identifies these arms as separate clusters and places
rectangular boundaries around them.

Use Part 1 (GMM) when the distribution has **parallel bands** separated cleanly
along a single PC axis (e.g. a chemical-family split spanning the full PC1 range).

The two methods are independent; run whichever suits your data, or both, and choose
the result that better matches the visual structure of your scatter plot.

---

## Quickstart (Part 2)

1. Set `KM_SPACE_NAME`, `K`, `KM_COLORS`, and `KM_DATASET_LABEL` in the
   **K-means Configuration** cell.
2. Run **Step K1** to cluster the data and compute boundaries.
3. Check the **Step K2** diagnostic plots.
4. Copy the **Step K3** output (`USER_REGIONS_KM`) into `behavioral_region_explorer.py`,
   or use **Step K4** to run it directly.

---

## K-means tuning parameters

| Parameter | What it does | Default |
|---|---|---|
| `K` | Total number of clusters, including the bulk | `4` |
| `KM_SPACE_NAME` | Which behavioral space to project | same as `SPACE_NAME` in Part 1 |
| `KM_COLORS` | Colours for clusters A, B, C, … | matplotlib tab10 subset |
| `KM_DATASET_LABEL` | Prefix for region dict keys | `DATASET_NAME` |

---

## How boundary detection works

1. KMeans is run on the 2-D PC1/PC2 projection with `n_init=20` for stability.
2. The **largest cluster** is labelled the bulk.
3. Each remaining cluster is classified as a **PC1 arm** (its centroid deviates
   more from the bulk in PC1) or a **PC2 arm** (deviates more in PC2).
4. **PC1 splits** are computed globally: the boundary between adjacent clusters
   along PC1 is the midpoint between `lo_cluster.max(PC1)` and
   `hi_cluster.min(PC1)`. If the edges overlap, the boundary is placed just past
   the lower cluster's maximum and a warning is printed.
5. **PC2 floors** are computed per PC1 band: within each vertical strip, the
   boundary between the bulk and any PC2-arm cluster is the midpoint of their
   data edges along PC2. PC1-arm regions use the actual data minimum in their
   strip as their PC2 floor.
6. Regions are named `{KM_DATASET_LABEL}_A`, `_B`, `_C`, … ordered from lowest
   to highest PC1 band, then descending along the PC2 arm.

---

## Reading the diagnostic plots

The Step K2 figure has two panels.

**Left panel — cluster assignments:**
- Each colour corresponds to one cluster.
- Red dashed vertical lines: PC1 splits.
- Red dashed horizontal segments: PC2 floor for each PC1 band.
- Black crosses: cluster centroids.

A good result shows each cluster occupying a visually distinct region, with
boundary lines running through gaps rather than through dense groups.

**Right panel — rectangular regions:**
- Each coloured rectangle is one entry in `USER_REGIONS_KM`.
- Region letters (A, B, C, …) are printed at the centroid of each rectangle.

If a rectangle is unexpectedly large or tiny, re-run Step K1 with a different `K`.

---

## Region naming

Regions are labelled `{KM_DATASET_LABEL}_A`, `_B`, `_C`, …

- A, B, … up to the bulk: ordered left to right along PC1 (lowest PC1 first).
- Bulk region: the largest cluster, assigned the letter immediately after the
  PC1-arm clusters.
- Letters after the bulk: PC2-arm clusters, ordered top to bottom along PC2.

Example with `K=4`, one PC1-arm cluster, one PC2-arm cluster:

```
            PC1 band 1 (arm)    PC1 band 2 (bulk+arm)
PC2 high    mxene_A             mxene_B  (bulk, tall)
PC2 low                         mxene_C  (PC2 arm)
```

Rename keys in the printed dict before pasting if you prefer descriptive labels.

---

## Connecting to behavioral_region_explorer.py

Same as Part 1. Either copy the Step K3 output, or in Step K4 set
`USER_REGIONS = USER_REGIONS_KM` and uncomment `%run -i behavioral_region_explorer.py`.

---

## Citation

If you use this tool in published work, please cite:

> Liu, T. & Barnard, A. S. (2025). Explainable Distributional Structure of
> MXene Compositions Revealed by Shapley Analysis. *Journal of Chemical Physics*.
