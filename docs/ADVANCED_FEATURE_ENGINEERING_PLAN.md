# Advanced Feature Engineering Implementation Plan

## Overview

This document outlines our comprehensive plan to implement and experiment with advanced, pharmacologically-grounded features for liver organoid oxygen consumption data. The goal is to create robust, interpretable features that enable cross-drug comparison and toxicity prediction.

## Core Principles

1. **Pharmacological grounding**: Use dose-response normalization (Hill curves) as the foundation
2. **Multi-resolution analysis**: Capture patterns at multiple timescales (24h, 48h, 96h) and symbolic resolutions
3. **Event-aware features**: Leverage media changes as biological landmarks
4. **Quality transparency**: Embed quality flags rather than filtering out problematic data
5. **Hierarchical structure**: Preserve well → concentration → drug relationships
6. **Complementary feature types**: Combine catch22 (statistical) with SAX (symbolic) for comprehensive representation

## Why Hierarchical SAX?

SAX (Symbolic Aggregate approXimation) provides unique value complementary to catch22:

### **Symbolic Pattern Recognition**
- **Shape-based classification**: SAX captures qualitative trajectory patterns (steady decline, oscillation, recovery)
- **Pattern motifs**: Identify recurring symbolic patterns across drugs and conditions
- **Transition analysis**: Track how patterns evolve over time (e.g., "steady → spike → decline")

### **Hierarchical Multi-Resolution**
- **Coarse level** (4-8 symbols): Major trajectory classes (increasing, decreasing, stable, volatile)
- **Medium level** (8-16 symbols): Intermediate patterns (gradual vs sharp changes, recovery patterns)
- **Fine level** (16-32 symbols): Detailed shape characteristics specific to individual drugs

### **Robustness to Noise**
- **Discretization**: SAX is inherently robust to measurement noise that might affect catch22
- **Pattern-based**: Focuses on shape rather than exact numerical values
- **Media change resilience**: Symbolic representation can capture consistent patterns despite absolute value shifts

### **Complementarity with catch22**
- **catch22**: Statistical properties (autocorrelation, entropy, distribution moments)
- **SAX**: Symbolic shape properties (patterns, transitions, motifs)
- **Combined**: Statistical + symbolic = comprehensive time series characterization

### **Biological Relevance**
- **Toxicity patterns**: Different toxicity mechanisms may have characteristic shape signatures
- **Dose-response shapes**: SAX can capture non-monotonic dose relationships
- **Recovery patterns**: Post-media change recovery may follow recognizable symbolic patterns

## Why Deep Learning Autoencoders?

Autoencoders provide a powerful complement to hand-crafted features (catch22, SAX):

### **Automatic Pattern Discovery**
- **No feature engineering bias**: Learns patterns directly from data without human assumptions
- **Nonlinear relationships**: Captures complex interactions that linear methods miss
- **Hierarchical representations**: Learns both local (6h) and global (full series) patterns automatically

### **Robust Handling of Data Issues**
- **Missing data aware**: Mask channel explicitly handles gaps in measurements
- **Event integration**: Media change flags become part of learned representation
- **Noise robustness**: Reconstruction objective naturally denoises signals
- **Quality assessment**: Poor wells automatically flagged by high reconstruction error

### **End-to-End Optimization**
- **Task-specific tuning**: Can fine-tune representations for specific prediction tasks
- **Dose-response learning**: Can incorporate concentration information directly
- **Temporal modeling**: Naturally handles irregular sampling and variable lengths

### **Complementarity with Traditional Features**
- **catch22**: Hand-crafted statistical features (interpretable)
- **SAX**: Symbolic pattern features (robust, discrete)
- **Autoencoder**: Learned continuous features (flexible, powerful)
- **Combined**: Best of interpretable + learned representations

### **Scalability and Efficiency**
- **Compact representation**: Entire time series → 24 numbers (16 + 8 dims)
- **Fast inference**: Single forward pass for new wells
- **Transfer learning**: Pre-trained models can adapt to new experiments
- **Hierarchical training**: Can learn multi-scale patterns efficiently

## Implementation Steps

### Step 1: Data Pipeline Foundation
**Goal**: Create robust, standardized data loading and quality assessment

#### 1.1 Enhanced Data Loader
- [x] Create `DataLoader` class that handles DuckDB queries with proper timestamp parsing
- [x] Add automatic well_id generation (plate_id + well_number)
- [x] Implement elapsed time calculation in hours and days from timestamps
- [x] Add control well identification (DMSO, media-only, concentration=0)
- [x] Build plate-level statistics (duration, n_wells, control percentage)

#### 1.2 Quality Flag Implementation
- [x] **low_points**: Count timepoints per well, flag if < 200 measurements
- [x] **high_noise**: Calculate rolling CV over 24h windows, flag if mean CV > 0.3
- [x] **sensor_drift**: Compute correlation between O2 value and elapsed time, flag if |r| > 0.8
- [x] **replicate_discord**: For each drug/concentration, calculate CV across 4 replicates, flag if > 0.5
- [x] **baseline_unstable**: Calculate CV in first 48h, flag if > 0.1
- [x] **media_change_outlier**: Compare spike magnitude to control wells, flag extreme outliers

#### 1.3 Control Period Detection
- [x] For each well, identify first 24-48h as baseline period
- [x] Calculate baseline statistics (mean, std, trend) for this period
- [x] Flag wells with insufficient baseline data (< 20 timepoints in first 48h)
- [x] Store baseline values for later normalization

### Step 2: Multi-Timescale Feature Extraction
**Goal**: Extract catch22 and SAX features at multiple temporal resolutions

#### 2.1 Rolling Window Implementation
- [x] Create function to extract features over sliding windows
- [x] Implement for 24h windows (≈15 timepoints per window)
- [x] Implement for 48h windows (≈30 timepoints per window)  
- [x] Implement for 96h windows (≈60 timepoints per window)
- [x] Add 50% overlap between windows to smooth temporal transitions

#### 2.2 catch22 Pipeline
- [x] Install and test pycatch22 library
- [x] Handle missing values and short windows gracefully
- [x] Add error handling for failed catch22 computations
- [x] Store results in structured format: `well_id | window_start | window_size | feature_name | feature_value`
- [x] Add metadata: window_number, hours_since_baseline, media_changes_in_window

#### 2.3 Hierarchical SAX Features
- [x] Implement multi-level SAX transformation using existing `sax.py` code
- [x] Extract SAX features at multiple resolutions:
  - **Coarse level**: 4 symbols, 3 alphabet size (capture major trends)
  - **Medium level**: 8 symbols, 4 alphabet size (capture medium patterns)
  - **Fine level**: 16 symbols, 6 alphabet size (capture detailed patterns)
- [x] Calculate SAX pattern frequencies and transition probabilities
- [x] Extract SAX complexity measures (entropy, Lempel-Ziv, transitions)
- [x] Build pattern-based features (trend analysis, symbol dominance)

#### 2.4 Baseline-Specific Features
- [ ] Extract catch22 features from baseline period only (0-48h)
- [ ] Extract hierarchical SAX features from baseline period
- [ ] Calculate summary statistics (mean, std, cv) for baseline
- [ ] Store baseline features separately for normalization reference

### Step 3: Media Change Event Detection
**Goal**: Identify media change events and their characteristics

#### 3.1 Event Detection Algorithm
- [x] Use control wells to identify typical media change timing
- [x] Calculate rolling variance in 6h windows
- [x] Detect sudden increases in variance (>2x baseline)
- [x] Validate detected events by checking spike characteristics (height, duration)
- [x] Create event timeline for each plate

#### 3.2 Spike Characterization
- [x] For each detected media change, measure:
  - Pre-spike baseline (6h before)
  - Peak height (maximum deviation)
  - Time to peak (from event start)
  - Recovery time (return to 90% of baseline)
  - Post-spike new baseline (6h after recovery)
- [x] Compare drug wells vs control wells for each metric
- [x] Calculate variability across 4 replicates

#### 3.3 Event-Indexed Features
- [x] Extract catch22 features in periods between media changes
- [x] Number each media change event (1st, 2nd, 3rd, etc.)
- [x] Calculate cumulative response metrics
- [x] Build "media changes survived" counter for each well

### Step 4: Dose-Response Normalization ⭐ CRITICAL
**Goal**: Implement Hill curve fitting for cross-drug comparability

#### 4.1 Hill Equation Implementation
- [ ] Implement 4-parameter Hill equation: `f(c) = E0 + (Emax-E0) * c^n / (EC50^n + c^n)`
- [ ] Add log10 concentration transformation
- [ ] Use scipy.optimize.curve_fit with robust initial parameter guesses
- [ ] Add parameter bounds: EC50 > 0, reasonable Emax ranges
- [ ] Implement error handling for convergence failures

#### 4.2 Fitting Quality Assessment
- [ ] Calculate R² for each fit
- [ ] Compute residuals and check for systematic patterns
- [ ] Calculate parameter confidence intervals using bootstrap
- [ ] Flag poor fits (R² < 0.5) for manual inspection
- [ ] Implement fallback fitting (linear, 3-parameter Hill)

#### 4.3 Per-Feature Hill Curves
- [ ] For each feature type across multiple timescales:
  - **catch22 features** (22 features × 3 timescales = 66 features)
  - **SAX pattern frequencies** (coarse, medium, fine resolution)
  - **SAX transition features** (pattern change rates)
  - **Combined catch22+SAX** meta-features
- [ ] For each feature:
  - Aggregate across 4 replicates (mean, median options)
  - Fit Hill curve across concentration series
  - Extract EC50, Emax, Hill slope, baseline
  - Store fit quality metrics
- [ ] Build dose-response meta-feature matrix: `drug_id | feature_type | timescale | EC50 | Emax | hill_slope | R2`

#### 4.4 Cross-Drug Validation
- [ ] Compare EC50 values for drugs within same mechanism class
- [ ] Verify EC50 ordering matches known potency rankings
- [ ] Test that similar drugs have similar Hill parameters
- [ ] Create dose-response visualization for manual validation

### Step 5: Event-Aware Feature Engineering
**Goal**: Build features that leverage temporal structure and media changes

#### 5.1 Inter-Event Period Features
- [ ] For each period between media changes:
  - Extract catch22 features
  - Extract SAX patterns and transitions
  - Calculate stability metrics (CV, trend)
  - Measure response magnitude compared to baseline
  - Count duration in hours and as fraction of total
- [ ] Track how both catch22 and SAX features change across sequential media changes
- [ ] Build SAX pattern evolution maps showing how symbolic patterns shift over time

#### 5.2 Progressive Effect Analysis
- [ ] Calculate response to 1st media change, 2nd media change, etc.
- [ ] Measure if responses get stronger/weaker over time
- [ ] Build adaptation vs deterioration indicators
- [ ] Create acute vs chronic toxicity classifiers

#### 5.3 Event-Normalized Time Features
- [ ] Replace absolute time with "time since last media change"
- [ ] Extract features in windows relative to media changes
- [ ] Build "recovery trajectory" features post-media change
- [ ] Compare pre-event vs post-event feature distributions

### Step 6: Hierarchical Feature Architecture
**Goal**: Combine all features into structured, interpretable embeddings

#### 6.1 Feature Vector Construction
- [ ] Build structured feature vector for each drug:
  ```
  [baseline_catch22 | baseline_SAX | 
   catch22_24h_summary | catch22_48h_summary | catch22_96h_summary | 
   SAX_coarse | SAX_medium | SAX_fine |
   autoencoder_full_16dim | autoencoder_6h_8dim |
   hill_EC50s_catch22 | hill_Emaxs_catch22 | hill_slopes_catch22 |
   hill_EC50s_SAX | hill_Emaxs_SAX | hill_slopes_SAX |
   hill_EC50s_autoencoder | hill_Emaxs_autoencoder |
   event_responses | SAX_pattern_evolution | quality_flags]
  ```
- [ ] Add feature names and metadata for interpretability
- [ ] Implement feature scaling (StandardScaler, RobustScaler options)
- [ ] Handle missing values (Hill fit failures) with imputation
- [ ] Balance catch22 vs SAX vs autoencoder feature contributions through weighting

#### 6.2 Quality-Aware Embedding
- [ ] Embed quality flags as binary features rather than filters
- [ ] Weight features by quality scores
- [ ] Create embedding confidence metrics
- [ ] Test that poor-quality wells cluster separately

#### 6.3 Dimensionality Reduction
- [ ] Apply PCA to reduce dimensionality while preserving interpretability
- [ ] Use UMAP for nonlinear embedding preservation
- [ ] Create embedding stability tests (bootstrap, cross-validation)

### Step 6.5: Deep Learning Autoencoder Embeddings
**Goal**: Learn compact representations through reconstruction with hierarchical time scales

#### 6.5.1 Data Preprocessing for Autoencoder
- [ ] **Resampling pipeline**: Convert irregular ~1.6h sampling to standardized 1-hour grid
- [ ] **3-channel input construction**:
  - Channel 1: O₂ values (interpolated, control-normalized)
  - Channel 2: Mask (1=real data, 0=missing/interpolated)
  - Channel 3: Event flag (1=media change timepoint, 0=normal)
- [ ] **Sequence standardization**: Pad/truncate to fixed length (e.g., 400 hours)
- [ ] **Control normalization**: Normalize each well by plate control mean/std

#### 6.5.2 Full-Series Autoencoder Architecture
- [ ] **Encoder design**:
  - Block 1: Conv1D(64 filters, kernel=7) → ReLU → BatchNorm → Dropout(0.1)
  - Block 2: Conv1D(128 filters, kernel=5, stride=2) → ReLU → BatchNorm → Dropout(0.1)
  - Block 3: Conv1D(256 filters, kernel=3, stride=2) → ReLU → BatchNorm → Dropout(0.1)
  - Global average pooling → Dense(128) → ReLU → Dense(16) [latent space]
- [ ] **Alternative LSTM encoder**: Bi-LSTM(128) → Dense(64) → Dense(16)
- [ ] **Decoder design**: Mirror encoder with transposed convolutions and upsampling
- [ ] **Mask-aware output**: Final layer multiplied by mask to ignore missing data

#### 6.5.3 Short-Term Event Autoencoder
- [ ] **6-hour window extraction**: Extract 0-6h post-media change windows
- [ ] **Smaller architecture**: Similar to full-series but with 8-dim latent space
- [ ] **Event-specific patterns**: Focus on immediate post-media change responses
- [ ] **Multiple events**: Train on all detected media change events per well

#### 6.5.4 Training Configuration
- [ ] **Loss function**: 
  ```python
  loss = MSE(real_data, reconstruction) * mask + 1e-4 * L2(latent)
  ```
- [ ] **Training setup**: 50 epochs, batch_size=256 wells, Adam optimizer (lr=1e-3)
- [ ] **Early stopping**: Monitor validation loss with patience=10
- [ ] **Data splits**: 70/15/15 train/val/test by experiment (not by well)

#### 6.5.5 Quality Assessment via Reconstruction
- [ ] **Reconstruction RMSE**: Calculate per-well reconstruction error
- [ ] **Quality flagging**: Flag wells with RMSE > 3 × MAD as noisy
- [ ] **Validation plots**: Original vs reconstructed time series for manual inspection
- [ ] **Latent space analysis**: Check for outliers and clustering patterns

#### 6.5.6 Hierarchical Embedding Combination
- [ ] **Full-series embedding**: 16-dimensional latent vector per well
- [ ] **Short-term embedding**: 8-dimensional latent vector per media change event
- [ ] **Event aggregation**: Average/max short-term embeddings across events per well
- [ ] **Combined representation**: Concatenate [16-dim full + 8-dim short] = 24-dim total
- [ ] **Replicate aggregation**: Average across 4 replicates → concentration-level embedding

#### 6.5.7 Advanced Extensions
- [ ] **Triplet loss enhancement**:
  ```python
  triplet_loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
  # anchor=replicate, positive=same drug/dose, negative=different drug
  ```
- [ ] **Time2Vec integration**: Add learnable time embeddings for irregular sampling
- [ ] **β-VAE variant**: Add KL divergence term for disentangled representations
- [ ] **Dose conditioning**: Concatenate log-concentration at each timestep
- [ ] **Attention mechanism**: Replace final conv with transformer encoder
- [ ] **Contrastive pre-training**: SimCLR on augmented time series views

#### 6.5.8 Model Management
- [ ] **Model checkpointing**: Save best model based on validation loss
- [ ] **Inference pipeline**: Single `.encode(series)` call for new data
- [ ] **Hyperparameter optimization**: Optuna search for architecture parameters
- [ ] **Model interpretability**: Analyze which latent dimensions capture specific patterns

### Step 7: Validation and Experiments
**Goal**: Test feature performance and biological relevance

#### 7.1 Drug Clustering Validation
- [ ] Cluster drugs using new embeddings
- [ ] Compare clusters to known drug families/mechanisms
- [ ] Calculate clustering metrics (silhouette, ARI, NMI)
- [ ] Identify novel drug relationships for follow-up

#### 7.2 Toxicity Prediction
- [ ] Build DILI risk prediction models (binary classification)
- [ ] Test hepatotoxicity severity prediction (regression)
- [ ] Implement time-to-toxicity survival analysis
- [ ] Use cross-validation with held-out drugs to test generalization

#### 7.3 Feature Importance Analysis
- [ ] Identify most predictive feature types across tasks
- [ ] Compare EC50 vs Emax vs raw feature importance
- [ ] Analyze which timescales (24h/48h/96h) contribute most
- [ ] Test impact of event-aware vs time-based features
- [ ] Compare catch22 vs SAX vs combined feature importance
- [ ] Analyze SAX resolution contribution (coarse vs medium vs fine)

#### 7.4 Ablation Studies
- [ ] **Experiment A**: Raw features vs Hill-normalized features
- [ ] **Experiment B**: Single timescale vs multi-timescale features
- [ ] **Experiment C**: Quality filtering vs quality embedding
- [ ] **Experiment D**: Time-based vs event-based features
- [ ] **Experiment E**: Individual replicates vs aggregated features
- [ ] **Experiment F**: catch22-only vs SAX-only vs autoencoder-only vs combined features
- [ ] **Experiment G**: SAX resolution impact (single vs multi-level SAX)
- [ ] **Experiment H**: Autoencoder architecture comparison (CNN vs LSTM vs Transformer)
- [ ] **Experiment I**: Autoencoder extensions (triplet loss, β-VAE, contrastive pre-training)

### Step 8: Documentation and Reproducibility
**Goal**: Create comprehensive documentation and validation

#### 8.1 Feature Reference Sheet
- [ ] Document every feature with formula, window, units, prerequisites
- [ ] Add biological interpretation for each feature type
- [ ] Include quality thresholds and failure modes
- [ ] Create feature extraction flowchart

#### 8.2 Validation Reports
- [ ] Generate Hill curve fit quality report for all drugs
- [ ] Create embedding stability and interpretability analysis
- [ ] Build drug clustering validation against known mechanisms
- [ ] Produce prediction performance benchmarks

#### 8.3 Code Organization and Testing
- [ ] Organize code into modular, testable functions
- [ ] Add unit tests for each feature extraction step
- [ ] Create integration tests for full pipeline
- [ ] Add performance profiling and optimization
- [ ] Build command-line interface for feature extraction

## Experimental Design

### Core Experiments

#### Experiment 1: Timescale Sensitivity
**Question**: Which timescales (24h, 48h, 96h) are most informative for toxicity prediction?

**Design**:
- Build embeddings using only 24h features, only 48h, only 96h, and combined
- Test prediction performance on DILI classification
- Analyze which drugs benefit from which timescales

#### Experiment 2: Dose-Response vs Raw Features
**Question**: How much does Hill curve normalization improve cross-drug comparability?

**Design**:
- Compare embeddings using raw catch22 vs Hill-normalized catch22
- Measure drug clustering quality (silhouette score, known drug families)
- Test prediction performance with/without dose-response normalization

#### Experiment 3: Event-Awareness Impact
**Question**: Do event-aware features improve toxicity prediction over time-based features?

**Design**:
- Compare models using absolute time vs media-change-relative time
- Test "media changes survived" vs "days survived" as endpoints
- Analyze acute vs chronic toxicity prediction performance

#### Experiment 4: Quality Flag Utility
**Question**: Is it better to embed quality flags or filter out poor-quality data?

**Design**:
- Build embeddings with quality flags embedded vs pre-filtered data
- Test clustering and prediction performance
- Analyze whether poor-quality wells provide useful negative signal

### Success Metrics

#### Technical Metrics
- **Hill curve fit quality**: >80% of drug/feature combinations have R² > 0.7
- **Embedding stability**: Bootstrapped embeddings correlate >0.9
- **Feature interpretability**: EC₅₀ values correlate with known drug potencies
- **Quality separation**: Poor-quality wells cluster distinctly (silhouette > 0.3)

#### Biological Validation
- **Known drug relationships**: Similar drugs cluster together (ARI > 0.6)
- **Mechanism clustering**: Drugs cluster by mechanism of action
- **Dose-response consistency**: Similar EC₅₀ for drugs in same class
- **Literature validation**: Findings consistent with published organoid studies

#### Prediction Performance
- **DILI prediction**: AUC > 0.75 on held-out drugs
- **Cross-drug generalization**: Performance maintained across chemical classes
- **Feature importance**: Dose-response features among top predictors
- **Temporal resolution**: Multi-timescale features outperform single timescale

## Implementation Guidelines

### Code Organization
```
scripts/
├── features/
│   ├── multi_timescale_catch22.py      # Rolling catch22 extraction
│   ├── hierarchical_sax.py             # Multi-resolution SAX features
│   ├── dose_response_fitting.py        # Hill curve fitting
│   ├── event_aware_features.py         # Media change features
│   ├── quality_assessment.py           # Quality flag detection
│   └── baseline_extraction.py          # Pre-dosing features
├── embeddings/
│   ├── hierarchical_features.py        # Feature concatenation
│   ├── quality_aware_embedding.py      # Quality-embedded features
│   ├── autoencoder_embedding.py        # Deep autoencoder models
│   ├── autoencoder_training.py         # Training pipeline and utilities
│   └── autoencoder_inference.py        # Model inference and embedding extraction
├── analysis/
│   ├── validate_hill_curves.py         # Dose-response validation
│   ├── embedding_validation.py         # Embedding quality tests
│   ├── drug_clustering.py              # Drug relationship analysis
│   └── feature_importance.py           # Feature contribution analysis
└── prediction/
    ├── toxicity_models.py              # DILI/hepatotox prediction
    └── survival_analysis.py            # Time-to-toxicity models
```

### Development Standards
- **Reproducibility**: All random seeds fixed, environment locked
- **Documentation**: Each function documented with examples
- **Testing**: Unit tests for all feature extraction functions
- **Validation**: Sanity checks on every output (ranges, distributions)
- **Modularity**: Each feature type can be computed independently

### Reference Implementation
Create comprehensive feature reference sheet:

| Feature Category | Feature Name | Formula | Window | Units | Media-Change-Aware | Prerequisites |
|-----------------|--------------|---------|---------|-------|-------------------|---------------|
| Baseline | baseline_mean | mean(O2[0:48h]) | 0-48h | O2 units | No | baseline_stable=True |
| Catch22-24h | DN_HistogramMode_5 | mode(rolling_24h) | 24h windows | unitless | Partial | n_points >= 15/window |
| Dose-Response | EC50_catch22_1 | Hill fit parameter | All concentrations | log(μM) | No | n_conc >= 4, R² > 0.5 |
| Event-Aware | mc1_recovery_time | time(O2 = 0.9×baseline) | Post media change 1 | hours | Yes | media_change_detected |
| Quality | replicate_discord | CV(4_replicates) | Full series | CV | No | n_replicates = 4 |

## Risk Mitigation

### Technical Risks
- **Hill curve fitting failures**: Implement robust fallbacks (linear, polynomial)
- **Insufficient data**: Minimum thresholds for feature extraction
- **Computational complexity**: Optimize for large-scale processing
- **Memory requirements**: Stream processing for large datasets

### Biological Risks
- **Overfitting to artifacts**: Validate features on multiple datasets
- **Batch effects**: Include experiment/plate as covariates
- **Selection bias**: Test on diverse drug mechanisms
- **Translation gap**: Validate against human hepatotoxicity data

### Timeline Risks
- **Scope creep**: Focus on core features first, extensions later
- **Data quality issues**: Allocate extra time for data cleaning
- **Computational bottlenecks**: Profile and optimize early
- **Validation delays**: Start with simple validation metrics

## Expected Outcomes

### Short-term (Phase 1-2)
- Robust, scalable feature extraction pipeline
- Hill curve-normalized features for all drugs
- Basic embedding and clustering results
- Validation that dose-response normalization improves comparability

### Medium-term (Phase 3-4)
- Event-aware features that outperform time-based features
- High-quality embeddings with interpretable structure
- Drug clustering that reflects known pharmacology
- Quality metrics that enable confident predictions

### Long-term (Phase 5+)
- State-of-the-art toxicity prediction performance
- Novel insights into drug mechanisms from organoid data
- Validated feature engineering framework for other organoid assays
- Publication-ready results demonstrating organoid utility for drug safety

This implementation plan balances ambition with practicality, ensuring we build robust, interpretable features while maintaining scientific rigor and reproducibility.