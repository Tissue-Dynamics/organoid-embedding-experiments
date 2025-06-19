# Organoid Embedding Experiments

A comprehensive framework for comparing embedding methods on liver organoid oxygen time series data. This project evaluates traditional methods (DTW, Fourier, SAX), feature-based approaches (TSFresh, catch22), and deep learning methods on ~3.15M oxygen measurements from 240 drug treatments.

## 🚀 Quick Start

```bash
# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Supabase credentials

# Run drug embedding analysis
python scripts/drug_embeddings/fast_drug_analysis.py
```

## 📂 Project Structure

```
organoid-embedding-experiments/
├── data/                      # Data processing modules
│   ├── loaders/              # Database loaders
│   └── preprocessing/        # Cleaning, normalization, interpolation
├── embeddings/               # Embedding methods
│   ├── traditional/          # DTW, Fourier, SAX
│   ├── features/            # TSFresh, catch22, custom features
│   └── deep_learning/       # Autoencoders, transformers, triplet networks
├── evaluation/              # Metrics and visualization
│   ├── metrics/            # Clustering, neighborhood preservation
│   └── visualization/      # Plotting utilities
├── experiments/            # Experiment configurations and runners
│   ├── configs/           # YAML experiment configs
│   └── run_experiment.py  # Main experiment runner
├── scripts/               # Analysis and utility scripts
│   ├── analysis/         # Data analysis scripts
│   ├── database/         # Database connection utilities
│   ├── drug_embeddings/  # Drug embedding pipeline
│   └── data_exploration/ # Exploratory scripts
├── results/              # Output files
│   ├── figures/         # Generated visualizations
│   └── data/           # Processed data and embeddings
└── docs/               # Documentation
```

## 🔬 Key Features

### 1. **Hierarchical Drug Embeddings**
- Well-level time series → Concentration aggregation → Drug signatures
- Captures dose-response relationships and temporal dynamics
- 150-373 dimensional embeddings per drug

### 2. **Embedding Methods**
- **Traditional**: Dynamic Time Warping, Fourier Transform, SAX
- **Feature-based**: TSFresh (700+ features), catch22 (22 features), Custom organoid features
- **Deep Learning**: LSTM/CNN Autoencoders, Transformers, Triplet Networks

### 3. **Drug Response Analysis**
- Automatic clustering of drugs by response patterns
- Toxicity scoring based on multiple indicators
- Dose-response curve fitting and parameterization

## 📊 Results

The analysis identified 4 distinct drug clusters and successfully detected known toxic compounds. See [docs/DRUG_EMBEDDING_SUMMARY.md](docs/DRUG_EMBEDDING_SUMMARY.md) for detailed results.

## 🛠️ Usage Examples

### Run Full Drug Analysis
```python
# Analyze all drugs with sufficient data
python scripts/drug_embeddings/fast_drug_analysis.py
```

### Process Single Drug
```python
# Demo with one drug
python scripts/drug_embeddings/demo_drug_embedding.py
```

### Run Embedding Experiments
```python
# Compare all embedding methods
python experiments/run_experiment.py

# Quick test with subset
python experiments/run_experiment.py --config-name quick_test

# Deep learning only
python experiments/run_experiment.py --config-name deep_learning_only
```

## 📈 Key Results

- **4 drug clusters** identified based on response patterns
- **Top toxic drugs**: Rucaparib phosphate, Gemcitibine, Amiodarone
- **Sanofi compounds** form tight cluster with high similarity (>0.97)
- **Best features**: Dose-response correlation, temporal response change

## 🔧 Configuration

Experiments are configured via YAML files in `experiments/configs/`:
- `config.yaml` - Full comparison of all methods
- `quick_test.yaml` - Fast subset for testing
- `deep_learning_only.yaml` - Neural network methods only

## 📝 Citation

If you use this code in your research, please cite:
```
@software{organoid_embeddings,
  title = {Organoid Embedding Experiments},
  year = {2024},
  url = {https://github.com/yourusername/organoid-embedding-experiments}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.