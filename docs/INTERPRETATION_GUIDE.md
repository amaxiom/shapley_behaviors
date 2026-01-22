# Behavioral Space Analysis - Interpretation Guide

## Introduction

This guide helps you interpret the results from the Shapley Behavioral Analysis Toolkit. It covers statistical metrics, visualizations, and how to draw meaningful conclusions from your analysis.

---

## Table of Contents

1. [Hopkins Statistic](#hopkins-statistic)
2. [Clustering Statistics](#clustering-statistics)
3. [Behavioral Space Plots](#behavioral-space-plots)
4. [Original vs Behavioral Comparison](#original-vs-behavioral-comparison)
5. [Region Analysis Tables](#region-analysis-tables)
6. [Box Plots and Heatmaps](#box-plots-and-heatmaps)
7. [Common Patterns](#common-patterns)
8. [Profile Plots](#profile-plots)
9. [Troubleshooting](#troubleshooting)

---

## Hopkins Statistic

### What It Measures
The Hopkins statistic (H) quantifies clustering tendency in your data.

### Interpretation Scale
```
H < 0.3   → Regularly spaced (ordered structure)
H ≈ 0.5   → Random distribution (no clustering)
H > 0.7   → Strong clustering tendency
H > 0.9   → Very strong, tight clusters
```

### P-Value
- **p < 0.05**: Clustering is statistically significant (not random)
- **p < 0.001**: Very strong evidence of clustering (marked as ***)
- **p > 0.05**: Cannot reject random distribution (ns)

### Example Output
```
Hopkins statistic: 0.8234 (CLUSTERED) p=0.0012 ***
```

**Interpretation:** The data has strong clustering tendency with very high confidence (not due to chance).

### What To Look For
- Compare Hopkins across all 5 spaces (original + 4 behavioral)
- **Higher Hopkins in behavioral space** → Transformation revealed structure
- **Similar Hopkins everywhere** → No strong behavioral patterns
- **Original space high Hopkins** → Clusters exist in raw composition (behavioral may not add much)

---

## Clustering Statistics

### Variance Concentration (PC1 + PC2)

**What it measures:** Percentage of total variance captured by the first two principal components.

```
< 40%  → Poor 2D projection (use caution)
40-60% → Moderate 2D representation
60-80% → Good 2D projection (most information preserved)
> 80%  → Excellent 2D projection (publication-ready)
```

**Why it matters:** Higher values mean your 2D plots accurately represent the full behavioral space.

### Coefficient of Variation (CV) of Pairwise Distances

**What it measures:** Heterogeneity in sample-to-sample distances (higher = more separated clusters).

```
< 0.2  → Homogeneous (no clear clusters)
0.2-0.4 → Moderate structure
> 0.4  → Strong cluster separation
```

**Example:**
```
Original space:  CV = 0.23 (moderate)
Variance space:  CV = 0.58 (strong)
```

**Interpretation:** Variance space separates samples much more clearly than original composition.

---

## Behavioral Space Plots

### Understanding the Visualizations

Each plot shows:
- **Grey points:** All samples in background
- **Colored points:** Samples colored by label value
- **PC1/PC2:** Principal components with variance explained (e.g., "PC1 (45.2%)")

### Color Schemes

#### Continuous Labels (Viridis Colormap)
- **Dark purple:** Low values
- **Green/Yellow:** Medium values
- **Bright yellow:** High values
- **Red circles:** Missing data

**Example interpretation:**
```
Variance Space (Tensile Strength)
- Upper right cluster: Bright yellow (high strength)
- Lower left cluster: Dark purple (low strength)
```
→ Variance space separates samples by strength

#### Categorical Labels (Markers + Colors)
- **Different markers:** o, s, ^, D, v, <, >, p, *, h
- **Different colors:** One per category (from viridis palette)
- **Red X:** Missing data

**Example interpretation:**
```
Entropy Space (Termination Type)
- Green circles (F): Upper right
- Blue squares (O): Lower left
- Purple triangles (OH): Center
```
→ Entropy space separates by termination chemistry

#### Feature Concentrations (Grey→Red Gradient)
- **Grey:** Low/zero concentration
- **Orange:** Medium concentration
- **Bright red:** High concentration

**Example interpretation:**
```
Variance Space (Cu Concentration)
- Red points clustered upper right
- Grey points scattered lower left
```
→ High Cu samples have distinct behavioral signature

### Which Space to Use?

**Variance Space:**
- Best for: Compositional heterogeneity patterns
- Highlights: Samples with unusual compositional spread
- Use when: Looking for alloy diversity effects

**Skewness Space:**
- Best for: Asymmetric compositional distributions
- Highlights: Samples with biased/skewed compositions
- Use when: Looking for one-sided enrichment patterns

**Kurtosis Space:**
- Best for: Extreme compositional values
- Highlights: Samples with heavy-tailed distributions
- Use when: Looking for outlier-driven behaviors

**Entropy Space:**
- Best for: Compositional diversity/uncertainty
- Highlights: Samples with balanced vs concentrated compositions
- Use when: Looking for mixing/ordering effects

---

## Original vs Behavioral Comparison

### The Critical Validation Test

This is the **most important** visualization for validating your findings.

### What You Should See

**LEFT PLOT (Original Space):**
- Colored regions scattered randomly
- No clear separation between colors
- Overlapping regions

**RIGHT PLOT (Behavioral Space):**
- Colored regions well-separated
- Clear clustering visible
- Rectangular boundaries contain most colored points

### Example Interpretation

```
✓ GOOD: Regions cluster in behavioral but scatter in original
   → Behavioral transformation revealed genuine structure
   → Patterns are NOT just from original composition

✗ BAD: Regions cluster in both spaces similarly
   → Behavioral transformation didn't add information
   → Just re-projecting existing compositional clusters
```

### Why This Matters

If regions cluster in **both** original and behavioral spaces, you've just found compositional clusters (which you could find with regular PCA). The power of behavioral analysis is finding structure that's **invisible** in raw composition but **clear** in behavioral space.

---

## Region Analysis Tables

### Composition Tables

**Key Columns:**
- **N_Present / Pct_Present:** How many samples contain this element
- **Mean / Median:** Central tendency in the region
- **All_Mean / All_Median:** Dataset-wide values for comparison
- **Change_Mean_%:** Enrichment/depletion percentage

#### Interpreting Change_%

```
+50%   → Region has 50% MORE than average (enriched)
0%     → Same as average (neutral)
-50%   → Region has 50% LESS than average (depleted)
-100%  → Completely absent from region
+400%  → Five times higher than average (strongly enriched)
```

**Example:**
```
Element  Mean   All_Mean  Change_Mean_%
Cu       4.2    2.8       +50%          → Enriched in Cu
Si       0.3    1.2       -75%          → Depleted in Si
Fe       0.0    0.8       -100%         → No Fe present
```

**Interpretation:** This region represents high-Cu, low-Si, Fe-free alloys (likely 2xxx series).

### Label Tables

#### For Continuous Labels

**Key Columns:**
- **Mean ± Std:** Average and spread
- **Min / Max:** Range
- **Change_Mean_%:** Difference from dataset average
- **N_Missing:** How many samples lack this measurement

**Example:**
```
Label              Mean   Std   All_Mean  Change_Mean_%
Tensile_Strength   420    35    280       +50%
Elongation         8      2     18        -56%
```

**Interpretation:** High-strength, low-ductility region (strength-ductility tradeoff).

#### For Categorical Labels

**Key Columns:**
- **N_Categories:** Number of unique values in region
- **Most_Common:** Dominant category
- **Most_Common_Count / Pct:** Frequency of dominant category

**Example:**
```
Label  N_Categories  Most_Common  Most_Common_Pct
M      2             Ti           75%
X      1             C            100%
T      1             F            100%
```

**Interpretation:** Mostly Ti-based carbide MXenes, exclusively F-terminated.

---

## Box Plots and Heatmaps

### Label Box Plots (Continuous Only)

**Components:**
- **Box:** Interquartile range (25th-75th percentile)
- **Black line inside:** Median
- **Red diamond:** Mean
- **Whiskers:** 1.5× IQR or min/max
- **n=X:** Sample count

**How to interpret:**
```
Wide box    → High variability within region
Narrow box  → Consistent property values
Outliers    → Unusual samples worth investigating
```

**Comparing regions:**
- **Non-overlapping boxes** → Statistically different (likely significant)
- **Large overlap** → Regions may not differ meaningfully
- **Different medians** → Central tendency differs

**Example:**
```
Region A: Median = 400 MPa, narrow box
Region B: Median = 200 MPa, wide box
```
→ Region A has consistently higher strength than Region B

### Composition Heatmap

**Color scale (YlOrRd):**
- **White/Pale yellow:** Low concentration
- **Orange:** Medium concentration  
- **Dark red:** High concentration

**How to read:**
- **Rows:** Top 20 elements by overall concentration
- **Columns:** Your defined regions
- **Values in cells:** Mean concentration

**Patterns to look for:**
- **Vertical stripes:** Element present across all regions
- **Horizontal stripes:** Region has many enriched elements
- **Dark spots:** Strong enrichment of specific element in specific region
- **White columns:** Region has low concentrations overall

---

## Common Patterns

### 1. Strength-Ductility Tradeoff (Alloys)

**Signature:**
- Two regions separate in variance/skewness space
- Region A: High strength, low elongation, Cu/Mg enriched
- Region B: Low strength, high elongation, pure Al (low alloying)

**Interpretation:** Classic metallurgical tradeoff - solid solution strengthening reduces ductility.

### 2. Termination Chemistry (MXenes)

**Signature:**
- Regions separate in entropy space
- Each region dominated by one termination type (F, O, OH)
- F-terminated: Higher voltage
- O-terminated: Lower voltage, more diverse metals

**Interpretation:** Termination group controls electrochemical properties.

### 3. Missing Data Patterns

**Signature:**
- Red points cluster together in behavioral space
- Missing data NOT randomly distributed

**Interpretation:** 
- Samples share compositional patterns that prevented measurement
- May indicate experimental challenges with specific alloy types
- Or systematic exclusion of certain compositions

### 4. Outliers

**Signature:**
- Samples with very high behavioral magnitude (z-score > 2.5)
- Often at edges of behavioral space plots

**Interpretation:**
- Unusual feature interaction patterns
- May be measurement errors OR genuinely novel compositions
- Check outlier CSV files for sample IDs and investigate

---

## Profile Plots

### What Are Profile Plots?

Profile plots are **feature-level breakdowns** of a single sample's behavioral signature, similar to SHAP force plots. They show exactly which features contribute most (positively or negatively) to that sample's position in behavioral space.

### Anatomy of a Profile Plot

**Components:**
1. **X-axis:** Features ordered by global importance (left = most important overall)
2. **Y-axis:** Shapley value for this specific sample
3. **Bar height:** Magnitude of contribution (positive or negative)
4. **Bar color (viridis):** Normalized feature value (0=min, 1=max in dataset)
5. **Zero line:** Reference for positive vs negative contributions
6. **Statistics box:** Behavioral magnitude, z-score, feature count
7. **Colorbar:** Legend for feature value interpretation

### How to Read a Profile Plot

**Bar Height (Shapley Value):**
- **Tall positive bar:** Feature strongly increases behavioral property
- **Tall negative bar:** Feature strongly decreases behavioral property
- **Short bar:** Feature has minimal effect
- **Zero height:** Feature not contributing

**Bar Color:**
- **Dark purple (low value):** Feature concentration near dataset minimum
- **Yellow (high value):** Feature concentration near dataset maximum
- **Green (medium value):** Feature concentration near dataset median

**Feature Order:**
- Features are sorted left-to-right by **global importance** (mean absolute Shapley across all samples)
- This lets you compare: "Is this sample's important feature the same as the global important feature?"

### Interpretation Examples

#### Example 1: High-Strength Outlier
```
Top features (left to right): Cu, Mg, Zn, Si, Fe

Cu:  Large positive bar (yellow)    → High Cu, strongly increases variance
Mg:  Large positive bar (yellow)    → High Mg, strongly increases variance  
Zn:  Small positive bar (purple)    → Low Zn, minor contribution
Si:  Large negative bar (purple)    → Low Si, reduces variance (expected)
Fe:  Medium negative bar (purple)   → Low Fe, reduces variance
```

**Interpretation:** This outlier has unusually high Cu+Mg (strengthening elements) and very low Si+Fe (typical impurities). The high variance comes from the extreme enrichment in key alloying elements.

#### Example 2: Compositionally Complex Sample
```
Top features: Ti, V, Cr, Nb, Mo

Ti:  Medium positive bar (green)    → Medium Ti, moderate contribution
V:   Large positive bar (yellow)    → High V, strong contribution
Cr:  Large positive bar (yellow)    → High Cr, strong contribution
Nb:  Small positive bar (purple)    → Low Nb, minor contribution
Mo:  Large positive bar (yellow)    → High Mo, strong contribution
```

**Interpretation:** This outlier has high concentrations of multiple transition metals (V, Cr, Mo) all contributing positively to entropy. The sample exhibits high compositional diversity, hence high entropy space magnitude.

### When to Use Profile Plots

**Use profile plots to:**
1. **Understand outliers:** Why is this sample flagged as unusual?
2. **Compare samples:** How do two high-magnitude samples differ in feature contributions?
3. **Validate findings:** Do the important features make physical sense?
4. **Identify errors:** Does an outlier have unrealistic feature values?
5. **Guide experiments:** Which features should be adjusted to achieve desired behavior?

**Don't rely on profile plots for:**
- Statistical significance (use box plots, t-tests instead)
- Global patterns (use regular behavioral space plots)
- Categorical label analysis (profile plots are for features)

### Automatic vs Manual Profiles

**Automatic (default):**
```python
CREATE_OUTLIER_PROFILES = True
MAX_OUTLIERS_PER_SPACE = 5  # Top 5 outliers per space
```
- Generates profiles for top outliers automatically
- Saved as: `{dataset}_profile_{space}_outlier_{rank}_{sample_id}.png`
- Good for initial exploration

**Manual (targeted):**
```python
# After running main analysis
fig, ax = create_profile_plot(
    Phi=behavioral_spaces['variance'],
    X=X,
    feature_names=feature_names,
    sample_idx=42,  # Your specific sample
    space_name='variance',
    sample_id='Sample_1247',
    figsize=(16, 7),
    fontsize=14,
    save_path='custom_profile.png'
)
```
- Full control over which samples to profile
- Useful for follow-up after identifying interesting samples
- Can profile non-outliers too

### Common Patterns in Profile Plots

#### Pattern 1: Single Dominant Feature
```
Cu: ████████████████ (very tall bar)
Mg: ██ (small bar)
Zn: █ (tiny bar)
...all others near zero
```
**Interpretation:** Sample's behavior driven almost entirely by one feature (Cu). Simple, interpretable pattern.

#### Pattern 2: Balanced Contributions
```
Cu: ██████
Mg: █████
Zn: ████
Si: ████
Fe: ███
```
**Interpretation:** Multiple features contribute similarly. Complex, multifactorial behavior.

#### Pattern 3: Opposing Forces
```
Cu:  +████████ (tall positive)
Si:  -████████ (tall negative)
Mg:  +█████ (positive)
Fe:  -████ (negative)
```
**Interpretation:** Features "fighting" each other. Net behavior is the balance.

#### Pattern 4: Unexpected Importance
```
Global order: Cu > Mg > Zn > Si > Fe
This sample:  Zn (huge) > Cu (medium) > Fe (medium) > Mg (small) > Si (zero)
```
**Interpretation:** Sample is unusual because a globally less-important feature (Zn) dominates. This is WHY it's an outlier.

### Profile Plot Checklist

When analyzing a profile plot, ask:

- [ ] Which features have the largest absolute Shapley values?
- [ ] Are those features also globally important? (left side of plot)
- [ ] Do high Shapley values correspond to high or low feature values? (color)
- [ ] Are there unexpected features contributing strongly?
- [ ] Do positive and negative contributions balance or reinforce?
- [ ] Does the pattern make physical/chemical sense?
- [ ] Is the behavioral magnitude reasonable? (statistics box)
- [ ] Is this sample actually an outlier? (z-score in statistics box)

### Comparing Multiple Profiles

To understand differences between samples:

1. **Generate profiles for both samples** (same space)
2. **Compare bar heights:** Which features differ most?
3. **Compare colors:** Are differences due to concentration or contribution pattern?
4. **Check feature order:** Do they respond to different features?

**Example comparison:**
```
Sample A (high strength):
  Cu: +8.2 (yellow), Mg: +6.5 (yellow), Si: -3.1 (purple)

Sample B (low strength):
  Cu: +2.1 (purple), Mg: +1.8 (purple), Si: +0.5 (green)
```

**Interpretation:** Sample A has much higher Cu and Mg concentrations (yellow vs purple) AND those features contribute more strongly (8.2 vs 2.1). Difference is both compositional and behavioral.

### Saving and Sharing Profiles

Profile plots are automatically saved as high-resolution PNGs (300 DPI). They are:
- **Publication-ready:** Clear labels, appropriate fonts
- **Self-contained:** Statistics box provides context
- **Standardized:** Same format across all samples
- **Interpretable:** Colorbar and labels explain everything

Include them in:
- Supplementary information for papers
- Presentations (especially for explaining outliers)
- Reports to collaborators
- Follow-up experimental design discussions

---

## Troubleshooting

### Problem: Hopkins ≈ 0.5 everywhere

**Possible causes:**
- No natural clustering in your dataset
- Features need preprocessing (normalization, log transform)
- Too few samples or too many features

**Solutions:**
- Check for data quality issues
- Try feature selection (remove uninformative features)
- Increase sample size if possible
- Consider if clustering is expected for your system

### Problem: Hopkins high in original space

**Interpretation:** 
- Strong compositional clusters already exist
- Behavioral spaces may not add much new information
- BUT: Still check if behavioral spaces reveal **different** clusters

**Next steps:**
- Compare which samples cluster together in each space
- Different clusterings = different aspects of composition matter

### Problem: Low PC1+PC2 variance (< 40%)

**Possible causes:**
- High-dimensional behavioral space not well-represented in 2D
- Need more components for full picture

**Solutions:**
- Examine PC3, PC4 (use 3D plots if needed)
- Use clustering metrics (Hopkins, CV) not just visual inspection
- Consider that structure may be genuinely high-dimensional

### Problem: All regions look the same

**Possible causes:**
- Regions defined too broadly
- Selected wrong behavioral space
- Insufficient compositional variation

**Solutions:**
- Refine PC1/PC2 ranges to tighter clusters
- Try different behavioral spaces
- Check if regions span multiple actual clusters (subdivide)

### Problem: Change_% values all near zero

**Interpretation:**
- Regions don't have distinctive compositions
- May have selected regions based on noise

**Solutions:**
- Redefine regions based on clearer clusters
- Check if labels (not composition) differ between regions
- Consider that behavioral patterns may be subtle

### Problem: Box plots completely overlap

**Interpretation:**
- Regions don't differ statistically in this label
- Behavioral separation may be driven by OTHER labels

**Solutions:**
- Check other label box plots
- Examine composition tables for differences
- Consider that behavioral clustering doesn't always correlate with labels

---

## Statistical Validation (Next Steps)

After exploratory analysis with this toolkit:

1. **Test label differences:**
   - T-tests (two regions) or ANOVA (multiple regions)
   - Effect sizes (Cohen's d)
   - Multiple testing correction (Bonferroni/FDR)

2. **Test composition differences:**
   - Permutation tests on mean enrichment
   - Multivariate tests (MANOVA)

3. **Cross-validation:**
   - Define regions on training set
   - Validate on held-out test set
   - Check if patterns replicate

4. **Domain validation:**
   - Do patterns make physical/chemical sense?
   - Literature support for compositional effects?
   - Confirm with experimental collaborators

---

## Quick Reference

### Good Signs ✓
- Hopkins > 0.7 in behavioral, < 0.6 in original
- PC1+PC2 variance > 60%
- Clear visual separation in behavioral plots
- Regions scatter in original, cluster in behavioral
- Change_% values > ±20% for key elements
- Non-overlapping box plots

### Warning Signs ⚠
- Hopkins ≈ 0.5 everywhere (no clustering)
- Hopkins similar in original and behavioral (no new info)
- PC1+PC2 variance < 40% (poor 2D representation)
- All Change_% values near zero (no compositional signature)
- Box plots completely overlap (no label differences)

### Red Flags 🚩
- Regions cluster in original space too (not discovering new structure)
- All samples are outliers (data quality issue)
- Missing data is >80% (insufficient information)
- Patterns contradict domain knowledge (check for errors)

---

## Example Interpretation Workflow

1. **Check Hopkins statistics:**
   - Which space has highest Hopkins?
   - Is it significantly higher than original?

2. **Examine behavioral plots:**
   - Do labels separate clearly?
   - Which labels correlate with which spaces?

3. **Define regions** based on visual clusters

4. **Validate regions:**
   - Check original vs behavioral comparison
   - Regions should NOT cluster in original

5. **Analyze composition:**
   - Which elements enriched/depleted?
   - Does it make sense for the material system?

6. **Compare labels:**
   - Are property differences significant?
   - Do they align with composition changes?

7. **Interpret scientifically:**
   - What physical mechanisms explain patterns?
   - Can you predict properties from composition?

8. **Validate statistically:**
   - Formal hypothesis tests
   - Cross-validation
   - Domain expert review

---

## Further Reading

- **Shapley values:** Shapley, L.S. (1953). "A value for n-person games." *Contributions to the Theory of Games*
- **Hopkins statistic:** Hopkins, B. & Skellam, J.G. (1954). "A new method for determining the type of distribution of plant individuals." *Annals of Botany*
- **PCA interpretation:** Jolliffe, I.T. (2002). *Principal Component Analysis*
- **Materials informatics:** Liu, T. & Barnard, A.S. (2025). "Shapley-Based Feature Engineering for Materials Science." *Machine Learning: Engineering*

---

*This guide is part of the Shapley Behavioral Analysis Toolkit*  
*Authors: Amanda S. Barnard and Tommy Liu*  
*License: MIT*
