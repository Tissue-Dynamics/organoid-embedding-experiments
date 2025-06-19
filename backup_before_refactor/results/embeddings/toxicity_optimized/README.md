# Toxicity-Optimized Drug Embedding System

## Overview
This is a compact, interpretable embedding specifically optimized for drug toxicity prediction.
It combines the most predictive components from multiple embedding methods.

## Files
- `toxicity_embedding.csv`: The embedding matrix (240 drugs × 15 features)
- `toxicity_features_metadata.csv`: Metadata about selected features and their correlations
- `toxicity_embedding_system.joblib`: Complete system for loading and using the embedding
- `toxicity_embedding_analysis.png`: Comprehensive visualization

## Performance Summary

## Feature Composition
- sax: 7 features
- custom: 4 features
- fourier: 3 features
- catch22: 1 features

## Usage Example
```python
import joblib
import pandas as pd

# Load the system
system = joblib.load('toxicity_embedding_system.joblib')
embedding = system['embedding']
metadata = system['features_metadata']

# Use for prediction
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# ... train model with embedding features
```

## Citation
Generated from organoid oxygen consumption time series using hierarchical embedding analysis.
