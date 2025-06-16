# Drug Preprocessing and Embedding Strategy

## Overview
Our pipeline processes ~3.15M oxygen measurements from organoid experiments to create comprehensive drug signatures that capture dose-response relationships and temporal dynamics.

## Data Structure
```
Drug → Multiple Concentrations → Multiple Replicates → Time Series (168+ hours)
```

## Preprocessing Pipeline

### 1. **Data Loading & Quality Control**
```python
# From well_map_data table
- Filter out controls and media-only wells
- Remove excluded wells (is_excluded = false)
- Require minimum 3 concentrations per drug
- Require minimum 20 wells total per drug
```

### 2. **Time Series Preprocessing**
Each well's oxygen time series undergoes:

#### a) **Temporal Alignment**
- Group by hour using `DATE_TRUNC('hour', timestamp)`
- Handle irregular sampling with averaging
- Result: Regular hourly time series

#### b) **Missing Value Handling**
```python
# Current approach (in demo_drug_embedding.py):
- Forward fill then backward fill
- Require minimum 20 time points per well
- Optional: Interpolation for gaps < 3 hours
```

#### c) **Outlier Detection** (planned)
```python
# From outlier_detection.py:
- Statistical methods: Z-score, IQR
- Model-based: Isolation Forest, LOF
- Domain-specific: Physiological bounds (0-100% O2)
```

#### d) **Normalization** (per concentration)
```python
# Options from normalizer.py:
1. Z-score normalization (default)
2. Min-Max scaling
3. Control-based normalization
4. Baseline correction (first 24h average)
```

### 3. **Feature Extraction**
Using `CustomFeaturesEmbedder` (46 features):

#### a) **Baseline Features**
- Initial oxygen level (first 6 hours mean)
- Baseline variability (first 6 hours std)
- Pre-treatment slope

#### b) **Response Features**
- Acute response (0-24h change)
- Sustained response (24-72h mean)
- Late response (>72h mean)
- Max drop from baseline
- Time to minimum
- Recovery rate

#### c) **Dynamics Features**
- Number of peaks/oscillations
- Dominant frequency (FFT)
- Trend components
- Smoothness metrics

#### d) **Stability Features**
- Coefficient of variation
- Autocorrelation at lag 24h
- Detrended fluctuation analysis

## Hierarchical Embedding Strategy

### Level 1: Well Embeddings
```python
# For each well:
time_series → preprocessing → feature_extraction → well_embedding (46 dims)
```

### Level 2: Concentration Aggregation
```python
# For each concentration:
well_embeddings = [emb1, emb2, emb3, emb4]  # 4 replicates
concentration_embedding = concat([
    mean(well_embeddings),      # Central tendency
    std(well_embeddings),       # Variability
    median(well_embeddings),    # Robust center
    iqr(well_embeddings)        # Robust spread
])
# Result: 184 dimensions (46 × 4)
```

### Level 3: Drug Signature
```python
# For each drug:
drug_signature = concat([
    dose_response_features,     # ~30 dims
    concentration_embeddings,   # Variable (184 × n_concs)
    global_statistics,         # ~20 dims
    temporal_evolution         # ~20 dims
])
```

#### Dose-Response Features:
- EC50/IC50 estimation (if applicable)
- Hill coefficient (slope at EC50)
- Maximum effect (Emax)
- Minimum effect (Emin)
- Area under dose-response curve
- Potency score
- Efficacy score

#### Global Statistics:
- Cross-concentration correlation matrix
- Response consistency score
- Temporal stability across concentrations
- Concentration-dependent variance

## Implementation Example

```python
class DrugEmbeddingPipeline:
    def process_drug(self, drug_name: str):
        # 1. Load all data for drug
        drug_data = self.load_drug_data(drug_name)
        # Structure: {conc1: {well1: ts1, well2: ts2}, conc2: {...}}
        
        # 2. Preprocess each time series
        for conc in drug_data:
            for well in drug_data[conc]:
                ts = drug_data[conc][well]
                ts = self.fill_missing(ts)
                ts = self.remove_outliers(ts)
                ts = self.normalize(ts, method='zscore')
                drug_data[conc][well] = ts
        
        # 3. Extract well embeddings
        well_embeddings = {}
        for conc in drug_data:
            embeddings = []
            for well, ts in drug_data[conc].items():
                emb = self.feature_extractor.extract(ts)
                embeddings.append(emb)
            well_embeddings[conc] = embeddings
        
        # 4. Aggregate by concentration
        conc_embeddings = {}
        for conc, embeddings in well_embeddings.items():
            conc_embeddings[conc] = self.aggregate_embeddings(embeddings)
        
        # 5. Create drug signature
        drug_signature = self.create_drug_signature(conc_embeddings)
        
        return drug_signature
```

## Quality Control Metrics

### 1. **Data Quality Scores**
- Completeness: % of non-missing values
- Consistency: Inter-replicate correlation
- Signal-to-noise ratio
- Temporal coverage

### 2. **Embedding Quality**
- Reconstruction error (if using autoencoders)
- Clustering metrics (silhouette score)
- Neighborhood preservation
- Discriminative power

## Scaling Considerations

### For Large-Scale Processing:
1. **Batch Processing**: Process drugs in batches of 10-20
2. **Parallel Well Processing**: Use multiprocessing for well embeddings
3. **Incremental Updates**: Store intermediate results
4. **Memory Management**: Process time series in chunks

### Storage Strategy:
```python
# Hierarchical storage
results/
├── well_embeddings/
│   └── {drug_name}/
│       └── {concentration}/
│           └── {well_id}.npy
├── concentration_embeddings/
│   └── {drug_name}/
│       └── {concentration}.npy
└── drug_signatures/
    └── {drug_name}.npy
```

## Next Steps

1. **Implement full preprocessing pipeline**
   - Add outlier detection
   - Add control-based normalization
   - Add event correction for media changes

2. **Enhance dose-response modeling**
   - Fit 4-parameter logistic curves
   - Handle non-monotonic responses
   - Identify hormetic effects

3. **Add temporal alignment**
   - Align all drugs to treatment start
   - Handle variable experiment durations
   - Account for circadian effects

4. **Validation framework**
   - Compare to known drug classes
   - Validate against toxicity data
   - Cross-validate with different sites/batches

## Key Parameters

```yaml
preprocessing:
  min_time_points: 20
  outlier_method: "iqr"
  outlier_threshold: 3.0
  normalization: "zscore"
  interpolation_limit: 3  # hours
  
feature_extraction:
  baseline_window: 6  # hours
  response_window: 24  # hours
  smoothing_window: 5  # time points
  
aggregation:
  min_replicates: 2
  aggregation_method: "robust"  # median + iqr
  
dose_response:
  min_concentrations: 3
  fit_method: "4pl"  # 4-parameter logistic
  weighting: "by_replicate_count"
```

This strategy ensures we capture both the immediate drug effects and long-term organoid responses while maintaining robustness to experimental variation.