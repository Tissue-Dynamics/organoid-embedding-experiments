#!/usr/bin/env python3
"""
Save All Drug Embeddings

Saves individual method embeddings in multiple formats for research use.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def save_all_drug_embeddings():
    """Save all drug embeddings in multiple formats."""
    print("Saving all drug embeddings...")
    
    # Load hierarchical results
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    results = joblib.load(results_file)
    
    # Create output directories
    embeddings_dir = project_root / "results" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    individual_dir = embeddings_dir / "individual_methods"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    # Get drug metadata
    drug_metadata = results['drug_metadata']
    drug_names = drug_metadata['drug'].tolist()
    drug_embeddings = results['drug_embeddings']
    
    print(f"  Processing {len(drug_names)} drugs with {len(drug_embeddings)} methods")
    
    # Save each method individually
    for method in ['fourier', 'sax', 'catch22', 'tsfresh', 'custom']:
        if method in drug_embeddings:
            embeddings_array = drug_embeddings[method]
            
            # Create DataFrame
            embeddings_df = pd.DataFrame(
                embeddings_array,
                index=drug_names,
                columns=[f'{method}_PC{i+1}' for i in range(embeddings_array.shape[1])]
            )
            
            # Save in multiple formats
            method_dir = individual_dir / method
            method_dir.mkdir(exist_ok=True)
            
            # CSV format
            csv_path = method_dir / f'{method}_drug_embeddings.csv'
            embeddings_df.to_csv(csv_path)
            
            # NumPy format
            npy_path = method_dir / f'{method}_drug_embeddings.npy'
            np.save(npy_path, embeddings_array)
            
            # Drug names
            names_path = method_dir / f'{method}_drug_names.txt'
            with open(names_path, 'w') as f:
                for name in drug_names:
                    f.write(f"{name}\n")
            
            # Joblib format (includes metadata)
            joblib_path = method_dir / f'{method}_embedding_data.joblib'
            joblib.dump({
                'embeddings': embeddings_df,
                'drug_names': drug_names,
                'drug_metadata': drug_metadata,
                'method': method,
                'shape': embeddings_array.shape
            }, joblib_path)
            
            print(f"    {method}: {embeddings_array.shape} -> {method_dir}")
    
    # Save combined embeddings
    combined_dir = embeddings_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    
    # Create master embedding matrix (all methods concatenated)
    all_embeddings = []
    all_column_names = []
    
    for method in ['fourier', 'sax', 'catch22', 'tsfresh', 'custom']:
        if method in drug_embeddings:
            embeddings_array = drug_embeddings[method]
            all_embeddings.append(embeddings_array)
            all_column_names.extend([f'{method}_PC{i+1}' for i in range(embeddings_array.shape[1])])
    
    # Concatenate all methods
    combined_matrix = np.hstack(all_embeddings)
    combined_df = pd.DataFrame(
        combined_matrix,
        index=drug_names,
        columns=all_column_names
    )
    
    # Save combined
    combined_csv_path = combined_dir / 'all_methods_combined.csv'
    combined_df.to_csv(combined_csv_path)
    
    combined_joblib_path = combined_dir / 'all_methods_combined.joblib'
    joblib.dump({
        'embeddings': combined_df,
        'drug_names': drug_names,
        'drug_metadata': drug_metadata,
        'methods': list(drug_embeddings.keys()),
        'method_dimensions': {method: drug_embeddings[method].shape[1] for method in drug_embeddings.keys()}
    }, combined_joblib_path)
    
    print(f"    Combined: {combined_matrix.shape} -> {combined_dir}")
    
    # Create comprehensive README
    readme_content = f"""# Drug Embedding Collection

This directory contains drug embeddings generated from liver organoid oxygen consumption time series.

## Structure

### Individual Methods (`individual_methods/`)
Each method has its own directory with multiple formats:
- `method_drug_embeddings.csv`: Embeddings as CSV with drug names as index
- `method_drug_embeddings.npy`: Raw NumPy array 
- `method_drug_names.txt`: List of drug names (corresponds to array rows)
- `method_embedding_data.joblib`: Complete data package with metadata

### Combined Methods (`combined/`)
- `all_methods_combined.csv`: All methods concatenated horizontally
- `all_methods_combined.joblib`: Complete combined dataset

### Toxicity-Optimized (`toxicity_optimized/`)
- Compact 15-dimensional embedding optimized for toxicity prediction
- See toxicity_optimized/README.md for details

## Method Overview

"""
    
    for method in drug_embeddings.keys():
        shape = drug_embeddings[method].shape
        readme_content += f"- **{method}**: {shape[1]} dimensions\n"
        
        if method == 'fourier':
            readme_content += "  - Frequency domain features from time series\n"
        elif method == 'sax':
            readme_content += "  - Symbolic Aggregate approXimation\n"
        elif method == 'catch22':
            readme_content += "  - 22 canonical time series features\n"
        elif method == 'tsfresh':
            readme_content += "  - Comprehensive statistical time series features\n"
        elif method == 'custom':
            readme_content += "  - Organoid-specific toxicity patterns\n"
    
    readme_content += f"""
## Dataset Details

- **Drugs**: {len(drug_names)}
- **Total dimensions**: {combined_matrix.shape[1]}
- **Source**: Liver organoid oxygen consumption time series
- **Hierarchy**: Well-level → Concentration-level → Drug-level aggregation

## Usage Examples

### Load Individual Method
```python
import pandas as pd
import joblib

# CSV format
fourier_df = pd.read_csv('individual_methods/fourier/fourier_drug_embeddings.csv', index_col=0)

# Complete data package
fourier_data = joblib.load('individual_methods/fourier/fourier_embedding_data.joblib')
embeddings = fourier_data['embeddings']
metadata = fourier_data['drug_metadata']
```

### Load Combined Dataset
```python
# All methods together
combined_data = joblib.load('combined/all_methods_combined.joblib')
all_embeddings = combined_data['embeddings']
drug_metadata = combined_data['drug_metadata']
```

### Load Toxicity-Optimized
```python
# Compact toxicity predictor
tox_system = joblib.load('toxicity_optimized/toxicity_embedding_system.joblib')
tox_embedding = tox_system['embedding']  # 15 dimensions
```

## Citation

Generated using hierarchical embedding analysis of liver organoid time series data.
Methods include traditional (Fourier, SAX), statistical (catch22, TSFresh), and 
custom organoid-specific features.
"""
    
    readme_path = embeddings_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  Saved comprehensive README: {readme_path}")
    
    # Summary
    print(f"\n📁 All embeddings saved to: {embeddings_dir}")
    print(f"📊 Individual methods: {len(drug_embeddings)} methods")
    print(f"🔬 Combined dataset: {combined_matrix.shape}")
    print(f"🎯 Toxicity-optimized: 15 dimensions")


def main():
    """Main function."""
    save_all_drug_embeddings()


if __name__ == "__main__":
    main()