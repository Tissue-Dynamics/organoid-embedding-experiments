#!/usr/bin/env python3
"""
Hierarchical SAX Feature Extraction

PURPOSE:
    Converts time series data into symbolic representations at multiple resolutions
    to capture qualitative patterns and shape characteristics. SAX provides
    noise-robust pattern recognition complementary to statistical features.

METHODOLOGY:
    - Transforms oxygen consumption to symbolic strings at 3 resolutions:
      * Coarse (4 segments, 3 symbols): Major trajectory types
      * Medium (8 segments, 4 symbols): Intermediate patterns  
      * Fine (16 segments, 6 symbols): Detailed shape features
    - Extracts pattern features: symbol frequencies, entropy, transitions
    - Analyzes pattern complexity using Lempel-Ziv and run-length encoding
    - Identifies motifs: monotonic trends, peaks, valleys
    - Tracks pattern evolution across drug concentrations

INPUTS:
    - Database connection via DATABASE_URL environment variable
    - Queries raw oxygen consumption data from database
    - Requires minimum 50 timepoints per well for reliable SAX

OUTPUTS:
    - results/data/hierarchical_sax_features_wells.parquet
      Well-level SAX features at all resolutions with raw SAX strings
    - results/data/hierarchical_sax_features_drugs.parquet
      Drug-concentration aggregated features with replicate consistency
    - results/data/sax_pattern_evolution.parquet (if sufficient data)
      Pattern change analysis across concentration gradients

REQUIREMENTS:
    - numpy, pandas, pyts, scipy, tqdm, joblib
    - Database connection with oxygen consumption data
    - pyts.approximation.SymbolicAggregateApproximation for SAX transform
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyts.approximation import SymbolicAggregateApproximation
from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
results_dir.mkdir(parents=True, exist_ok=True)
fig_dir = project_root / "results" / "figures" / "sax_features"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HIERARCHICAL SAX FEATURE EXTRACTION")
print("=" * 80)

class HierarchicalSAX:
    """
    Multi-resolution SAX transformation with pattern analysis
    """
    
    def __init__(self, resolutions):
        """
        Args:
            resolutions: List of (n_segments, n_symbols) tuples for each resolution level
        """
        self.resolutions = resolutions
        self.sax_transformers = {}
        
        # Create SAX transformer for each resolution
        for level, (n_segments, n_symbols) in enumerate(resolutions):
            self.sax_transformers[f'level_{level}'] = SymbolicAggregateApproximation(
                n_bins=n_symbols,
                strategy='quantile',  # Use quantile for better symbol distribution
                alphabet=[chr(ord('a') + i) for i in range(n_symbols)]
            )
    
    def transform(self, time_series):
        """
        Transform time series at multiple resolutions
        
        Returns:
            Dict with SAX strings at each resolution level
        """
        # Reshape for pyts (needs 2D input)
        ts = time_series.reshape(1, -1)
        
        sax_representations = {}
        for level, (n_segments, n_symbols) in enumerate(self.resolutions):
            try:
                # Transform to SAX
                sax_array = self.sax_transformers[f'level_{level}'].fit_transform(ts)
                sax_string = ''.join(sax_array[0])
                sax_representations[f'level_{level}'] = {
                    'string': sax_string,
                    'n_segments': n_segments,
                    'n_symbols': n_symbols
                }
            except Exception as e:
                print(f"Error in SAX transformation at level {level}: {e}")
                # Fallback to constant string
                sax_representations[f'level_{level}'] = {
                    'string': 'a' * n_segments,
                    'n_segments': n_segments,
                    'n_symbols': n_symbols
                }
        
        return sax_representations
    
    def extract_pattern_features(self, sax_representations):
        """
        Extract features from multi-resolution SAX representations
        """
        features = {}
        
        for level, sax_data in sax_representations.items():
            sax_string = sax_data['string']
            n_symbols = sax_data['n_symbols']
            
            # 1. Symbol frequency features
            symbol_counts = Counter(sax_string)
            total_symbols = len(sax_string)
            
            for i in range(n_symbols):
                symbol = chr(ord('a') + i)
                freq = symbol_counts.get(symbol, 0) / total_symbols if total_symbols > 0 else 0
                features[f'{level}_symbol_{symbol}_freq'] = freq
            
            # 2. Symbol entropy (pattern diversity)
            if total_symbols > 0:
                probs = np.array([symbol_counts.get(chr(ord('a') + i), 0) for i in range(n_symbols)]) / total_symbols
                probs = probs[probs > 0]
                symbol_entropy = entropy(probs, base=2) if len(probs) > 0 else 0
            else:
                symbol_entropy = 0
            features[f'{level}_entropy'] = symbol_entropy
            
            # 3. Transition patterns
            if len(sax_string) > 1:
                transitions = [sax_string[i:i+2] for i in range(len(sax_string)-1)]
                transition_counts = Counter(transitions)
                
                # Most common transitions
                if transition_counts:
                    most_common_transition, count = transition_counts.most_common(1)[0]
                    features[f'{level}_dominant_transition'] = count / len(transitions)
                    features[f'{level}_n_unique_transitions'] = len(transition_counts)
                else:
                    features[f'{level}_dominant_transition'] = 0
                    features[f'{level}_n_unique_transitions'] = 0
                
                # Ascending/descending patterns
                ascending = sum(1 for t in transitions if ord(t[1]) > ord(t[0]))
                descending = sum(1 for t in transitions if ord(t[1]) < ord(t[0]))
                stable = sum(1 for t in transitions if t[0] == t[1])
                
                features[f'{level}_ascending_ratio'] = ascending / len(transitions)
                features[f'{level}_descending_ratio'] = descending / len(transitions)
                features[f'{level}_stable_ratio'] = stable / len(transitions)
            
            # 4. Pattern complexity (Lempel-Ziv)
            features[f'{level}_complexity'] = self._lempel_ziv_complexity(sax_string)
            
            # 5. Run length statistics
            run_lengths = []
            if len(sax_string) > 0:
                current_symbol = sax_string[0]
                current_run = 1
                
                for i in range(1, len(sax_string)):
                    if sax_string[i] == current_symbol:
                        current_run += 1
                    else:
                        run_lengths.append(current_run)
                        current_symbol = sax_string[i]
                        current_run = 1
                run_lengths.append(current_run)
                
                features[f'{level}_mean_run_length'] = np.mean(run_lengths)
                features[f'{level}_max_run_length'] = np.max(run_lengths)
                features[f'{level}_run_length_variance'] = np.var(run_lengths)
            
            # 6. Pattern motifs (3-symbol patterns)
            if len(sax_string) >= 3:
                motifs = [sax_string[i:i+3] for i in range(len(sax_string)-2)]
                motif_counts = Counter(motifs)
                
                # Pattern type classification
                monotonic_up = sum(1 for m in motifs if m[0] < m[1] < m[2])
                monotonic_down = sum(1 for m in motifs if m[0] > m[1] > m[2])
                valley = sum(1 for m in motifs if m[0] > m[1] < m[2])
                peak = sum(1 for m in motifs if m[0] < m[1] > m[2])
                
                total_motifs = len(motifs)
                features[f'{level}_monotonic_up_ratio'] = monotonic_up / total_motifs if total_motifs > 0 else 0
                features[f'{level}_monotonic_down_ratio'] = monotonic_down / total_motifs if total_motifs > 0 else 0
                features[f'{level}_valley_ratio'] = valley / total_motifs if total_motifs > 0 else 0
                features[f'{level}_peak_ratio'] = peak / total_motifs if total_motifs > 0 else 0
        
        return features
    
    def _lempel_ziv_complexity(self, string):
        """Calculate Lempel-Ziv complexity of a string"""
        i, k, l = 0, 1, 1
        k_max = 1
        n = len(string) - 1
        c = 1
        
        while True:
            if string[i + k - 1] == string[l + k - 1]:
                k = k + 1
                if l + k > n:
                    c = c + 1
                    break
            else:
                if k > k_max:
                    k_max = k
                
                i = i + 1
                
                if i == l:
                    c = c + 1
                    l = l + k_max
                    if l + 1 > n:
                        break
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
        
        return c

def visualize_sax_transformation(time_series, sax_representations, well_id, drug, concentration, hsax, save_path):
    """
    Visualize original time series with SAX approximations at each resolution
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'SAX Transformation - {drug} ({concentration} μM)\nWell: {well_id}', fontsize=16, fontweight='bold')
    
    # Original time series
    ax = axes[0]
    time_points = np.arange(len(time_series))
    ax.plot(time_points, time_series, 'b-', alpha=0.7, linewidth=1, label='Original')
    ax.set_ylabel('O₂ (%)')
    ax.set_title('Original Time Series')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # For each resolution level
    for level, (n_segments, n_symbols) in enumerate(hsax.resolutions):
        ax = axes[level + 1]
        sax_data = sax_representations[f'level_{level}']
        sax_string = sax_data['string']
        
        # Plot original
        ax.plot(time_points, time_series, 'b-', alpha=0.3, linewidth=1, label='Original')
        
        # Create SAX approximation for visualization
        segment_length = len(time_series) // n_segments
        sax_approx = []
        
        for i, symbol in enumerate(sax_string):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(time_series))
            segment = time_series[start_idx:end_idx]
            
            # Map symbol to value (simple mapping for visualization)
            symbol_idx = ord(symbol) - ord('a')
            symbol_value = np.percentile(time_series, (symbol_idx + 0.5) * 100 / n_symbols)
            
            # Draw horizontal line for segment
            segment_times = time_points[start_idx:end_idx]
            ax.plot(segment_times, [symbol_value] * len(segment_times), 'r-', linewidth=3, alpha=0.8)
            
            # Add symbol label
            if i < 20:  # Limit labels for readability
                ax.text(np.mean(segment_times), symbol_value, symbol, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('O₂ (%)')
        ax.set_title(f'Level {level}: {n_segments} segments, {n_symbols} symbols - "{sax_string[:30]}..."')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time Points')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_pattern_distribution(features_df, save_path):
    """
    Visualize distribution of SAX patterns at each resolution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SAX Pattern Distribution Across All Wells', fontsize=16, fontweight='bold')
    
    for level in range(3):
        ax = axes[level]
        
        # Get all SAX strings for this level
        sax_strings = features_df[f'level_{level}_sax_string'].values
        
        # Count 3-gram patterns
        all_patterns = []
        for sax_string in sax_strings:
            patterns = [sax_string[i:i+3] for i in range(len(sax_string)-2)]
            all_patterns.extend(patterns)
        
        # Get top patterns
        pattern_counts = Counter(all_patterns)
        top_patterns = pattern_counts.most_common(20)
        
        if top_patterns:
            patterns, counts = zip(*top_patterns)
            x_pos = np.arange(len(patterns))
            
            bars = ax.bar(x_pos, counts, color=plt.cm.viridis(np.linspace(0, 1, len(patterns))))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(patterns, rotation=45, ha='right')
            ax.set_xlabel('3-Symbol Patterns')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Level {level} - Top 20 Patterns')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_pattern_evolution(drug_features_df, save_path):
    """
    Visualize how SAX patterns evolve across drug concentrations
    """
    # Select top drugs by number of concentrations
    drug_conc_counts = drug_features_df.groupby('drug')['concentration'].nunique()
    top_drugs = drug_conc_counts.nlargest(12).index.tolist()
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('SAX Pattern Evolution Across Concentrations', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, drug in enumerate(top_drugs[:12]):
        ax = axes[idx]
        drug_data = drug_features_df[drug_features_df['drug'] == drug].sort_values('concentration')
        
        # Create pattern evolution matrix for level 1 (medium resolution)
        concentrations = drug_data['concentration'].values
        
        # Extract pattern features
        pattern_features = []
        for _, row in drug_data.iterrows():
            features = []
            for metric in ['entropy', 'complexity', 'monotonic_up_ratio', 'valley_ratio']:
                col = f'level_1_{metric}_mean'
                if col in row:
                    features.append(row[col])
            pattern_features.append(features)
        
        if pattern_features and len(pattern_features[0]) > 0:
            pattern_matrix = np.array(pattern_features).T
            
            # Create heatmap
            im = ax.imshow(pattern_matrix, aspect='auto', cmap='RdBu_r')
            
            # Set labels
            ax.set_xticks(range(len(concentrations)))
            ax.set_xticklabels([f'{c:.2f}' for c in concentrations], rotation=45, ha='right')
            ax.set_yticks(range(len(['Entropy', 'Complexity', 'Monotonic↑', 'Valley'])))
            ax.set_yticklabels(['Entropy', 'Complexity', 'Monotonic↑', 'Valley'])
            ax.set_xlabel('Concentration (μM)')
            ax.set_title(drug[:20])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove empty subplots
    for idx in range(len(top_drugs), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_embedding_space(drug_features_df, save_path):
    """
    Create UMAP and t-SNE visualizations of drugs using SAX features
    """
    # Prepare feature matrix - get all feature columns that end with _mean
    feature_cols = [col for col in drug_features_df.columns 
                   if col.endswith('_mean') and 'sax_string' not in col]
    
    # Debug: print available columns
    print(f"   Available columns: {len(drug_features_df.columns)}")
    print(f"   Feature columns found: {len(feature_cols)}")
    
    if len(feature_cols) == 0:
        print("   Warning: No feature columns found for embedding visualization")
        # Create a dummy plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No features available for embedding', 
                ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Get unique drugs (aggregate across concentrations)
    drug_embeddings = []
    drug_names = []
    
    for drug in drug_features_df['drug'].unique():
        drug_data = drug_features_df[drug_features_df['drug'] == drug]
        # Average features across concentrations
        drug_features = drug_data[feature_cols].mean().values
        drug_embeddings.append(drug_features)
        drug_names.append(drug)
    
    X = np.array(drug_embeddings)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create embeddings
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(drug_names)-1))
    
    umap_embedding = umap_reducer.fit_transform(X_scaled)
    tsne_embedding = tsne_reducer.fit_transform(X_scaled)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Drug Embeddings using SAX Features', fontsize=16, fontweight='bold')
    
    # UMAP plot
    ax = axes[0]
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                        c=range(len(drug_names)), cmap='tab20', s=100, alpha=0.7)
    
    # Add labels for some points
    for i, drug in enumerate(drug_names[::3]):  # Label every 3rd drug
        ax.annotate(drug[:15], (umap_embedding[i*3, 0], umap_embedding[i*3, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP Projection')
    ax.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax = axes[1]
    scatter = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                        c=range(len(drug_names)), cmap='tab20', s=100, alpha=0.7)
    
    # Add labels for some points
    for i, drug in enumerate(drug_names[::3]):  # Label every 3rd drug
        ax.annotate(drug[:15], (tsne_embedding[i*3, 0], tsne_embedding[i*3, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Projection')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_replicate_consistency(features_df, save_path):
    """
    Visualize how consistent SAX patterns are across replicates
    """
    # Group by drug and concentration
    consistency_data = []
    
    for (drug, conc), group in features_df.groupby(['drug', 'concentration']):
        if len(group) >= 2:  # Need at least 2 replicates
            # Compare SAX strings across replicates
            for level in range(3):
                sax_strings = group[f'level_{level}_sax_string'].values
                
                # Calculate pairwise similarity
                similarities = []
                for i in range(len(sax_strings)):
                    for j in range(i+1, len(sax_strings)):
                        s1, s2 = sax_strings[i], sax_strings[j]
                        min_len = min(len(s1), len(s2))
                        if min_len > 0:
                            matches = sum(1 for k in range(min_len) if s1[k] == s2[k])
                            similarity = matches / min_len
                            similarities.append(similarity)
                
                if similarities:
                    consistency_data.append({
                        'drug': drug,
                        'concentration': conc,
                        'level': level,
                        'mean_similarity': np.mean(similarities),
                        'n_replicates': len(group)
                    })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    if len(consistency_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('SAX Pattern Consistency Across Replicates', fontsize=16, fontweight='bold')
        
        for level in range(3):
            ax = axes[level]
            level_data = consistency_df[consistency_df['level'] == level]
            
            if len(level_data) > 0:
                # Create violin plot
                drugs = level_data['drug'].unique()[:20]  # Top 20 drugs
                data_for_plot = [level_data[level_data['drug'] == drug]['mean_similarity'].values 
                               for drug in drugs]
                
                parts = ax.violinplot(data_for_plot, positions=range(len(drugs)), 
                                     widths=0.8, showmeans=True, showmedians=True)
                
                # Customize colors
                for pc in parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(drugs)))
                ax.set_xticklabels(drugs, rotation=45, ha='right')
                ax.set_ylabel('Replicate Similarity')
                ax.set_ylim(0, 1.1)
                ax.set_title(f'Level {level}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add horizontal line at 0.8
                ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% similarity')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# Define hierarchical resolutions
resolutions = [
    (4, 3),    # Coarse: 4 segments, 3 symbols (a,b,c) - major trends
    (8, 4),    # Medium: 8 segments, 4 symbols (a,b,c,d) - intermediate patterns
    (16, 6),   # Fine: 16 segments, 6 symbols (a-f) - detailed patterns
]

print(f"\n📊 SAX Resolutions:")
for i, (n_seg, n_sym) in enumerate(resolutions):
    print(f"   Level {i}: {n_seg} segments, {n_sym} symbols")

# Load oxygen consumption data
print("\n📊 Loading oxygen consumption data...")
import sys
sys.path.append(str(project_root / "src"))
from utils.data_loader import DataLoader

loader = DataLoader()
# Load limited data for faster demonstration
df = loader.load_oxygen_data(limit=3)  # Only 3 plates for demo
print(f"   Loaded {len(df)} measurements")
print(f"   Wells: {df['well_id'].nunique()}")

# Initialize hierarchical SAX
hsax = HierarchicalSAX(resolutions)

# Process each well
print(f"\n🔄 Extracting hierarchical SAX features...")
all_features = []
example_wells = []  # Store some examples for visualization

wells = df['well_id'].unique()
for well_id in tqdm(wells, desc="Processing wells"):
    well_data = df[df['well_id'] == well_id].sort_values('elapsed_hours')
    
    # Get metadata
    drug = well_data['drug'].iloc[0]
    concentration = well_data['concentration'].iloc[0]
    
    # Extract time series
    time_series = well_data['o2'].values
    
    # Skip if too few points
    if len(time_series) < 50:
        continue
    
    # Transform to SAX at multiple resolutions
    sax_representations = hsax.transform(time_series)
    
    # Extract pattern features
    features = hsax.extract_pattern_features(sax_representations)
    
    # Add metadata
    features['well_id'] = well_id
    features['drug'] = drug
    features['concentration'] = concentration
    features['n_timepoints'] = len(time_series)
    
    # Add raw SAX strings for later analysis
    for level, sax_data in sax_representations.items():
        features[f'{level}_sax_string'] = sax_data['string']
    
    all_features.append(features)
    
    # Store first few examples for visualization
    if len(example_wells) < 5 and drug not in ['Unknown', 'DMSO']:
        example_wells.append({
            'well_id': well_id,
            'drug': drug,
            'concentration': concentration,
            'time_series': time_series,
            'sax_representations': sax_representations
        })

# Convert to DataFrame
features_df = pd.DataFrame(all_features)
print(f"\n📊 Extracted features for {len(features_df)} wells")

# Get feature columns (excluding metadata and raw strings)
feature_cols = [col for col in features_df.columns 
               if col not in ['well_id', 'drug', 'concentration', 'n_timepoints'] 
               and not col.endswith('_sax_string')]

print(f"   Total hierarchical SAX features: {len(feature_cols)}")
print(f"   Features per resolution level:")
for i in range(len(resolutions)):
    level_features = [col for col in feature_cols if col.startswith(f'level_{i}_')]
    print(f"      Level {i}: {len(level_features)} features")

# Aggregate to drug level
print("\n🔄 Aggregating to drug level...")
drug_features = []

for drug in features_df['drug'].unique():
    drug_data = features_df[features_df['drug'] == drug]
    
    for conc in sorted(drug_data['concentration'].unique()):
        conc_data = drug_data[drug_data['concentration'] == conc]
        
        if len(conc_data) == 0:
            continue
        
        drug_feat = {
            'drug': drug,
            'concentration': conc,
            'n_wells': len(conc_data)
        }
        
        # Aggregate numerical features
        for col in feature_cols:
            values = conc_data[col].dropna()
            if len(values) > 0:
                drug_feat[f'{col}_mean'] = values.mean()
                drug_feat[f'{col}_std'] = values.std()
                drug_feat[f'{col}_cv'] = values.std() / values.mean() if values.mean() != 0 else 0
        
        # Analyze SAX string consistency across replicates
        for level in range(len(resolutions)):
            sax_strings = conc_data[f'level_{level}_sax_string'].values
            
            # String similarity (fraction of matching positions)
            if len(sax_strings) > 1:
                similarities = []
                for i in range(len(sax_strings)):
                    for j in range(i+1, len(sax_strings)):
                        s1, s2 = sax_strings[i], sax_strings[j]
                        min_len = min(len(s1), len(s2))
                        if min_len > 0:
                            matches = sum(1 for k in range(min_len) if s1[k] == s2[k])
                            similarities.append(matches / min_len)
                
                drug_feat[f'level_{level}_replicate_consistency'] = np.mean(similarities) if similarities else 0
            else:
                drug_feat[f'level_{level}_replicate_consistency'] = 1.0
        
        drug_features.append(drug_feat)

drug_features_df = pd.DataFrame(drug_features)
print(f"   Created features for {len(drug_features_df)} drug-concentration pairs")

# Pattern evolution analysis
print("\n🔄 Analyzing pattern evolution across concentrations...")
pattern_evolution = []

for drug in features_df['drug'].unique():
    drug_data = features_df[features_df['drug'] == drug]
    concentrations = sorted(drug_data['concentration'].unique())
    
    if len(concentrations) < 3:
        continue
    
    # Track how patterns change with concentration
    for level in range(len(resolutions)):
        level_patterns = []
        
        for conc in concentrations:
            conc_data = drug_data[drug_data['concentration'] == conc]
            
            # Get dominant pattern features
            entropy_vals = conc_data[f'level_{level}_entropy'].values
            complexity_vals = conc_data[f'level_{level}_complexity'].values
            
            if len(entropy_vals) > 0:
                level_patterns.append({
                    'concentration': conc,
                    'entropy': np.mean(entropy_vals),
                    'complexity': np.mean(complexity_vals)
                })
        
        if len(level_patterns) >= 3:
            # Calculate pattern progression
            entropies = [p['entropy'] for p in level_patterns]
            complexities = [p['complexity'] for p in level_patterns]
            # Handle zero concentrations
            concs = [p['concentration'] for p in level_patterns]
            log_concs = [np.log10(c) if c > 0 else -3 for c in concs]  # Use -3 for 0 concentration (0.001)
            
            # Fit linear trend with error handling
            try:
                entropy_slope = np.polyfit(log_concs, entropies, 1)[0] if len(log_concs) > 1 else 0
            except:
                entropy_slope = 0
            
            try:
                complexity_slope = np.polyfit(log_concs, complexities, 1)[0] if len(log_concs) > 1 else 0
            except:
                complexity_slope = 0
            
            pattern_evolution.append({
                'drug': drug,
                'resolution_level': level,
                'entropy_dose_response': entropy_slope,
                'complexity_dose_response': complexity_slope,
                'pattern_change_magnitude': np.std(entropies) + np.std(complexities)
            })

pattern_evolution_df = pd.DataFrame(pattern_evolution)

# Save results
print("\n💾 Saving results...")
features_df.to_parquet(results_dir / 'hierarchical_sax_features_wells.parquet', index=False)
drug_features_df.to_parquet(results_dir / 'hierarchical_sax_features_drugs.parquet', index=False)
if len(pattern_evolution_df) > 0:
    pattern_evolution_df.to_parquet(results_dir / 'sax_pattern_evolution.parquet', index=False)

print(f"   Well-level features: {results_dir / 'hierarchical_sax_features_wells.parquet'}")
print(f"   Drug-level features: {results_dir / 'hierarchical_sax_features_drugs.parquet'}")
if len(pattern_evolution_df) > 0:
    print(f"   Pattern evolution: {results_dir / 'sax_pattern_evolution.parquet'}")

# Analysis summary
print("\n📊 FEATURE SUMMARY:")
print(f"   Wells processed: {len(features_df)}")
print(f"   Drugs processed: {features_df['drug'].nunique()}")
print(f"   Total features per well: {len(feature_cols)}")

# Show example features
if len(features_df) > 0:
    example = features_df.iloc[0]
    print(f"\n🔍 Example SAX strings for {example['drug']} at {example['concentration']} μM:")
    for level in range(len(resolutions)):
        sax_string = example[f'level_{level}_sax_string']
        print(f"   Level {level}: {sax_string[:50]}..." if len(sax_string) > 50 else f"   Level {level}: {sax_string}")

# Pattern type distribution
print("\n📈 Pattern type distribution (averaged across all wells):")
for level in range(len(resolutions)):
    print(f"\n   Level {level} ({resolutions[level][0]} segments, {resolutions[level][1]} symbols):")
    
    for pattern_type in ['monotonic_up', 'monotonic_down', 'valley', 'peak']:
        col = f'level_{level}_{pattern_type}_ratio'
        if col in features_df.columns:
            mean_ratio = features_df[col].mean()
            print(f"      {pattern_type}: {mean_ratio:.3f}")

# Generate visualizations
print("\n📊 Generating visualizations...")

# 1. SAX transformation examples
print("   Creating SAX transformation visualizations...")
for i, example in enumerate(example_wells[:3]):
    save_path = fig_dir / f'sax_transformation_example_{i+1}.png'
    visualize_sax_transformation(
        example['time_series'], 
        example['sax_representations'],
        example['well_id'],
        example['drug'],
        example['concentration'],
        hsax,
        save_path
    )
    print(f"      Saved: {save_path.name}")

# 2. Pattern distribution
print("   Creating pattern distribution visualization...")
visualize_pattern_distribution(features_df, fig_dir / 'sax_pattern_distribution.png')
print(f"      Saved: sax_pattern_distribution.png")

# 3. Pattern evolution
print("   Creating pattern evolution visualization...")
visualize_pattern_evolution(drug_features_df, fig_dir / 'sax_pattern_evolution.png')
print(f"      Saved: sax_pattern_evolution.png")

# 4. Embedding space (UMAP and t-SNE)
print("   Creating embedding space visualizations...")
visualize_embedding_space(drug_features_df, fig_dir / 'sax_drug_embeddings.png')
print(f"      Saved: sax_drug_embeddings.png")

# 5. Replicate consistency
print("   Creating replicate consistency visualization...")
visualize_replicate_consistency(features_df, fig_dir / 'sax_replicate_consistency.png')
print(f"      Saved: sax_replicate_consistency.png")

print("\n✅ Hierarchical SAX feature extraction complete!")
print(f"   Generated {len(all_features)} well-level features")
print(f"   Generated {len(drug_features_df)} drug-concentration features")
print(f"   Created 8 visualization figures in: {fig_dir}")
print(f"\n   Next steps:")
print(f"   1. Combine with catch22 features for hybrid representation")
print(f"   2. Correlate SAX patterns with DILI")
print(f"   3. Analyze pattern evolution across doses")
print(f"   4. Use pattern consistency as quality metric")