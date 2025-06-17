# Oxygen Real-Time Data Documentation

## Overview

This document provides a comprehensive analysis of the oxygen consumption time series data from liver organoid experiments. The data consists of real-time oxygen measurements from organoids treated with various drugs at multiple concentrations, along with control wells.

## Data Characteristics

### Temporal Coverage

- **Mean experiment duration**: 17.8 ± 9.3 days
- **Maximum duration**: 46.9 days
- **Sampling interval**: 1.6 ± 6.2 hours
- **Timepoints per well**: 287 ± 126

### Concentration Patterns

- **Number of drugs**: 248
- **Concentrations per drug**: 7.3 ± 1.5
- **Global concentration range**: 0.0003 - 3333

#### Most Common Concentrations:
- 0.03: 246407 wells
- 2.50: 246011 wells
- 0.09: 245628 wells
- 7.50: 243659 wells
- 0.28: 243432 wells

### Experimental Events

- **Experiments with media changes**: 33 / 33 (100.0%)
- **Experiments with dosing events**: 33 / 33 (100.0%)
- **Mean experiment duration**: 17.8 days

### Data Quality

#### Missing Values
- **Total missing**: 0 (0.00%)
- **Outliers**: 218,846 (6.95%)

#### Exclusions
- **Excluded wells**: 701 / 10973 (6.4%)

#### Value Distribution
- **Range**: -27.40 - 119.80
- **Mean ± SD**: 11.08 ± 23.05
- **Median (IQR)**: 3.80 (-3.70 - 13.20)

### Control Wells

- **Number of control wells**: 2429 across all experiments
- **Control experiments**: 33 plates total
- **Control types**: DMSO vehicle controls and media-only controls
- **Note**: Sanofi-1 through Sanofi-8 are experimental drugs, not controls


## Control Periods and Baseline Establishment

### Pre-Dosing Control Period

All wells begin with a **control period before drug dosing**, typically lasting 24-48 hours. This period is critical for:

1. **Baseline establishment**: Measure organoid health and metabolic activity before treatment
2. **Quality control**: Identify wells with abnormal baseline behavior
3. **Normalization reference**: Calculate well-specific baseline for later normalization

Key characteristics:
- **Duration**: Usually 24-48 hours (varies by experiment)
- **All wells undosed**: Both future treatment and control wells are untreated
- **Steady-state establishment**: Organoids adapt to culture conditions
- **Low variance**: Minimal fluctuations compared to post-dosing period

### Control Wells on Each Plate

Nearly every plate contains **dedicated control wells** (mean: ~73 wells, ~22% of plate):

1. **Spatial distribution**: Controls distributed across plate to capture positional effects
2. **Types of controls**:
   - Negative controls (no drug, media only)
   - Vehicle controls (DMSO at equivalent concentrations)
   - Positive controls (known toxic compounds in some experiments)
3. **Continuous monitoring**: Controls tracked throughout entire experiment

### Identifying Dosing Events

Since dosing time is not explicitly marked, it can be inferred by:

1. **Variance analysis**: Sudden increase in measurement variance
2. **Trajectory divergence**: Treatment wells diverge from controls
3. **Typical timing**: Most experiments dose at 24-48 hours
4. **Pattern changes**: Shift from stable baseline to dynamic response

### Recommended Analysis Approach

1. **Segment time series**:
   - Pre-dosing period (0-24/48h): Baseline
   - Post-dosing period (24/48h+): Treatment response
   - Inter-media change periods: Separate segments between each media change

2. **Multi-level normalization**:
   - Normalize to well's own baseline (pre-dosing period)
   - Normalize to plate controls (ongoing)
   - Normalize to pre-media change baseline (for each event)
   - This captures well-specific, plate-wide, and event-specific effects

3. **Feature extraction strategies**:
   - **Baseline features**: Mean, stability, trend in first 24-48h
   - **Multi-timescale catch22**: Extract catch22 features over 24h, 48h, 96h rolling windows
   - **Event-indexed features**: Response after 1st, 2nd, 3rd media change
   - **Dose-response fitting**: Hill curves for each feature type → EC₅₀/Eₘₐₓ meta-features
   - **Quality metrics**: Per-well flags (low_points, high_noise, sensor_drift) embedded as features

4. **Quality metrics**:
   - Baseline stability (CV < 10% suggests good quality)
   - Control consistency across plate
   - Replicate concordance (agreement among 4 replicates)
   - Event response consistency (similar patterns after each media change)

## Critical Considerations for Feature Engineering

### 1. Media Change Artifacts

Media changes cause sudden jumps in oxygen consumption. While these spikes occur in all wells (including controls), **drug/concentration effects may modulate the spike characteristics**:
- Spike magnitude may vary with drug treatment
- Variability across replicates may indicate drug effects
- Recovery time from spike may be drug-dependent

Features should:
- Capture spike characteristics (height, duration, variability)
- Compare drug vs control spike patterns
- Use spike response as a potential toxicity indicator

### 2. Concentration-Aware Features

With drugs tested at 4-10 concentrations (typically 8), **each with 4 replicates**, features should:
- Capture dose-response relationships across concentrations
- Calculate replicate variability as a signal of drug effects
- Allow for non-monotonic responses
- Consider concentration as a continuous variable, not categorical
- Leverage the 4 replicates for robust statistics and confidence intervals

### 3. Temporal Alignment

Key considerations:
- Experiments have different durations (10-20+ days)
- Sampling is irregular (~1-2 hour intervals)
- Early time points may reflect adaptation rather than drug response
- Late time points may show secondary effects

### 4. Control Normalization

Controls provide baseline oxygen consumption:
- Each experiment has control wells
- Controls show temporal drift and plate effects
- Normalization to controls can improve drug effect detection

### 5. Data Quality Issues

Common problems to handle:
- Missing values (sporadic, not systematic)
- Outliers from measurement artifacts
- Excluded wells that failed quality control
- Variable number of replicates per condition

## Recommended Feature Engineering Strategies

### 1. Robust Summary Statistics
- Median-based statistics instead of mean
- Trimmed means to reduce outlier influence
- MAD (Median Absolute Deviation) for variability

### 2. Change Point Detection
- Identify media change events automatically
- Segment time series into pre/post media change
- Calculate features separately for each segment

### 3. Dose-Response Features ⭐ CRITICAL

**This is the most important feature engineering step for cross-drug comparability.**

#### Log-Concentration Scaling:
- **Essential transformation**: Given concentration range 0.0003 - 3333 μM (>10⁶ fold)
- **Log₁₀ transformation**: Converts to manageable -3.5 to 3.5 range
- **Pharmacologically correct**: Drug responses are typically log-linear

#### Per-Feature Hill Curve Fitting:
For each of the ~22 catch22 features (and any other feature), fit:
```
feature(conc) = E₀ + (Eₘₐₓ - E₀) × conc^n / (EC₅₀^n + conc^n)
```

Where:
- **E₀**: Baseline effect (at zero concentration)
- **Eₘₐₓ**: Maximum effect 
- **EC₅₀**: Concentration of half-maximal effect
- **n**: Hill slope (steepness)

#### Dose-Response Meta-Features:
Instead of raw catch22 values, use Hill parameters as features:
- **EC₅₀ values**: Potency of each drug for each feature type
- **Eₘₐₓ values**: Maximum efficacy for each feature
- **Hill slopes**: Steepness of dose-response
- **R² values**: Quality of Hill fit (QC metric)

#### Benefits:
1. **Cross-drug comparability**: All drugs described by same EC₅₀/Eₘₐₓ space
2. **Pharmacological interpretability**: EC₅₀ is universally understood
3. **Dimensionality reduction**: 8 concentrations → 4 parameters per feature
4. **Noise reduction**: Hill fitting smooths measurement noise

### 4. Temporal Features
- Early response (days 1-3)
- Sustained response (days 4-10)
- Late/adaptive response (days 10+)
- Rate of change over specific windows

### 5. Multi-Resolution Temporal Features

#### Time-Scale Analysis:
- **Discrete wavelet energy (4 levels)**: Capture both rapid spikes (media changes) and slow trends (chronic toxicity)
- **Multi-timescale catch22**: Rolling features over multiple windows:
  - 24h windows: Immediate post-dosing/media change responses
  - 48h windows: Short-term adaptation patterns  
  - 96h windows: Medium-term toxicity development
  - Whole series: Overall trajectory and stability

#### Frequency Domain:
- Fourier coefficients for periodic patterns
- Power spectral density for oscillation detection
- Cross-scale feature interactions

### 6. Event-Aware Features
- **Media change counting**: Number of media changes before specific events (e.g., toxicity onset)
- **Inter-event metrics**: Response patterns between media changes, not just time-based
- **Event-normalized time**: Days/hours since last media change vs absolute time
- **Recovery patterns**: How quickly wells recover after each media change
- **Cumulative effects**: Do responses to media changes worsen over multiple events?
- **Event-triggered windows**: Features calculated in windows relative to media changes

Note: While we lack organoid-specific pharmacokinetic data (Tmax, half-life), human PK data exists for many drugs and could inform feature design

## Data Structure for Analysis

### Hierarchical Organization
1. **Well level**: Individual time series (4 replicates per drug/concentration combination)
2. **Concentration level**: Aggregate across 4 replicates per concentration
3. **Drug level**: Aggregate across all concentrations (typically 8 concentrations)

### Recommended Workflow
1. Filter by quality (is_excluded=false)
2. Filter by coverage (≥14 days, ≥4 concentrations)
3. Detect and handle media changes
4. Normalize to controls within experiment
5. Extract features at appropriate hierarchy level
6. Validate features across multiple drugs

## Key Insights

1. **Consistency is critical**: Features must be comparable across drugs with different concentration ranges and experimental conditions

2. **Media change responses are informative**: Rather than artifacts to remove, media change spikes may reveal drug-specific stress responses

3. **Event counting matters**: "Number of media changes before toxicity" may be more relevant than absolute time, especially without organoid-specific PK data

4. **Hierarchical structure matters**: With 4 replicates × 8 concentrations = 32 wells per drug, preserve this structure for dose-response modeling

5. **Control periods are universal**: All wells have ~24-48h pre-dosing baseline for normalization

6. **Replicate variability is signal**: Increased variability across 4 replicates may indicate drug toxicity

7. **Human PK can guide features**: While organoid PK differs, human Tmax/half-life data can suggest relevant time windows

8. **Cumulative effects are real**: Progressive deterioration across multiple media changes may indicate chronic toxicity

## Recommended Implementation Pipeline

### Stage 1: Multi-Timescale Feature Extraction
```python
# For each well, extract catch22 features at multiple timescales
timescales = [24, 48, 96]  # hours
features = {}

for scale in timescales:
    rolling_features = []
    for window_start in range(0, total_hours - scale, scale//2):  # 50% overlap
        window_data = well_data[window_start:window_start + scale]
        catch22_features = catch22.catch22_all(window_data)
        rolling_features.append(catch22_features)
    
    features[f'catch22_{scale}h'] = np.array(rolling_features)
```

### Stage 2: Dose-Response Normalization (CRITICAL)
```python
# For each feature type, fit Hill curves across concentrations
from scipy.optimize import curve_fit

def hill_equation(conc, E0, Emax, EC50, n):
    return E0 + (Emax - E0) * (conc**n) / (EC50**n + conc**n)

# For each drug and each catch22 feature
hill_params = {}
for drug in drugs:
    drug_data = data[data.drug == drug]
    concentrations = np.log10(drug_data.concentration)  # Log scale!
    
    for feature_name in catch22_features:
        feature_values = drug_data[feature_name].groupby('concentration').mean()
        
        try:
            params, _ = curve_fit(hill_equation, concentrations, feature_values)
            hill_params[f'{drug}_{feature_name}'] = {
                'E0': params[0], 'Emax': params[1], 
                'EC50': params[2], 'hill_slope': params[3]
            }
        except:
            hill_params[f'{drug}_{feature_name}'] = {'fit_failed': True}

# Final drug representation: EC50 and Emax for each feature type
drug_embedding = [hill_params[f'{drug}_{feat}']['EC50'] for feat in catch22_features] + \
                 [hill_params[f'{drug}_{feat}']['Emax'] for feat in catch22_features]
```

### Stage 3: Quality-Aware Embedding
```python
# Embed quality flags alongside features
quality_flags = {
    'low_points': well_data.shape[0] < 200,
    'high_noise': well_data.value.rolling(24).std().mean() > threshold,
    'sensor_drift': abs(well_data.value.corr(well_data.elapsed_time)) > 0.8,
    'replicate_discord': replicate_cv > 0.5
}

final_embedding = np.concatenate([
    baseline_features,      # 24-48h pre-dosing features
    catch22_24h_features,   # Short-term patterns  
    catch22_48h_features,   # Medium-term patterns
    catch22_96h_features,   # Long-term patterns
    hill_ec50_features,     # Potency parameters
    hill_emax_features,     # Efficacy parameters
    list(quality_flags.values())  # Quality indicators
])
```

This comprehensive understanding of the data characteristics should guide the development of meaningful, consistent features for drug comparison and toxicity prediction.
