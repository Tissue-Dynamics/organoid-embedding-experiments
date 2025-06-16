#!/usr/bin/env python3
"""
Compare structural embeddings (from SMILES) with oxygen-based embeddings.
This will help identify if structurally similar drugs have similar toxicity patterns.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
import umap
from dotenv import load_dotenv
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# RDKit for molecular fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, AtomPairs
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Installing...")
    import subprocess
    subprocess.check_call(["uv", "pip", "install", "rdkit"])
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, AtomPairs
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_drug_smiles():
    """Load SMILES data from drugs table."""
    print("Loading drug SMILES data...")
    
    # Connect to database
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    
    parsed = urlparse(database_url)
    attach_query = f"""
    ATTACH 'host={parsed.hostname} port={parsed.port or 5432} dbname={parsed.path.lstrip('/')} 
    user={parsed.username} password={parsed.password}' 
    AS supabase (TYPE POSTGRES, READ_ONLY);
    """
    conn.execute(attach_query)
    
    # Get drug SMILES
    query = """
    SELECT 
        drug,
        smiles,
        molecular_weight,
        logp,
        binary_dili,
        hepatotoxicity_boxed_warning
    FROM supabase.public.drugs
    WHERE smiles IS NOT NULL
    AND smiles != ''
    """
    
    drugs_df = conn.execute(query).df()
    conn.close()
    
    print(f"  Loaded SMILES for {len(drugs_df)} drugs")
    print(f"  DILI positive: {drugs_df['binary_dili'].sum()}")
    print(f"  Hepatotoxicity warnings: {drugs_df['hepatotoxicity_boxed_warning'].sum()}")
    
    return drugs_df


def compute_molecular_fingerprints(drugs_df):
    """Compute various molecular fingerprints from SMILES."""
    print("\nComputing molecular fingerprints...")
    
    fingerprints = {}
    valid_drugs = []
    
    for idx, row in drugs_df.iterrows():
        drug = row['drug']
        smiles = row['smiles']
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  Warning: Invalid SMILES for {drug}: {smiles}")
                continue
                
            # 1. Morgan fingerprints (circular fingerprints, similar to ECFP)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            # 2. MACCS keys (166 structural keys)
            maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
            
            # 3. RDKit fingerprints (Daylight-like)
            rdkit_fp = Chem.RDKFingerprint(mol)
            
            # Convert to numpy arrays
            morgan_arr = np.zeros(2048)
            DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
            
            maccs_arr = np.zeros(167)
            DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
            
            rdkit_arr = np.zeros(2048)
            DataStructs.ConvertToNumpyArray(rdkit_fp, rdkit_arr)
            
            # 4. Basic molecular descriptors
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol)
            ]
            
            fingerprints[drug] = {
                'morgan': morgan_arr,
                'maccs': maccs_arr,
                'rdkit': rdkit_arr,
                'descriptors': np.array(descriptors),
                'combined': np.concatenate([morgan_arr, maccs_arr, descriptors])
            }
            
            valid_drugs.append(drug)
            
        except Exception as e:
            print(f"  Error processing {drug}: {e}")
            continue
    
    print(f"  Successfully computed fingerprints for {len(valid_drugs)} drugs")
    
    return fingerprints, valid_drugs


def load_oxygen_embeddings():
    """Load oxygen-based embeddings from hierarchical results."""
    print("\nLoading oxygen-based embeddings...")
    
    results_file = project_root / "results" / "data" / "hierarchical_embedding_results.joblib"
    results = joblib.load(results_file)
    
    drug_embeddings = results['drug_embeddings']
    drug_metadata = results['drug_metadata']
    
    print(f"  Loaded embeddings for {len(drug_metadata)} drugs")
    print(f"  Available methods: {list(drug_embeddings.keys())}")
    
    return drug_embeddings, drug_metadata


def create_embedding_dataframes(fingerprints, drug_embeddings, drug_metadata):
    """Create aligned DataFrames for structural and oxygen embeddings."""
    print("\nAligning structural and oxygen embeddings...")
    
    # Get common drugs
    structural_drugs = set(fingerprints.keys())
    oxygen_drugs = set(drug_metadata['drug'].values)
    common_drugs = sorted(structural_drugs & oxygen_drugs)
    
    print(f"  Structural embeddings: {len(structural_drugs)} drugs")
    print(f"  Oxygen embeddings: {len(oxygen_drugs)} drugs")
    print(f"  Common drugs: {len(common_drugs)}")
    
    # Create structural embedding DataFrames
    structural_embeddings = {}
    for fp_type in ['morgan', 'maccs', 'rdkit', 'descriptors', 'combined']:
        data = []
        for drug in common_drugs:
            data.append(fingerprints[drug][fp_type])
        structural_embeddings[fp_type] = pd.DataFrame(
            data, index=common_drugs
        )
    
    # Create oxygen embedding DataFrames
    oxygen_embeddings = {}
    for method, embedding_matrix in drug_embeddings.items():
        if embedding_matrix is not None:
            # Get indices of common drugs in the original order
            drug_list = drug_metadata['drug'].values
            indices = [i for i, drug in enumerate(drug_list) if drug in common_drugs]
            
            # Extract embeddings for common drugs
            common_embeddings = embedding_matrix[indices]
            oxygen_embeddings[method] = pd.DataFrame(
                common_embeddings, 
                index=[drug_list[i] for i in indices]
            ).loc[common_drugs]  # Ensure same order
    
    return structural_embeddings, oxygen_embeddings, common_drugs


def compute_embedding_correlations(structural_embeddings, oxygen_embeddings):
    """Compute correlations between structural and oxygen embedding spaces."""
    print("\nComputing embedding space correlations...")
    
    correlations = {}
    
    for struct_name, struct_emb in structural_embeddings.items():
        correlations[struct_name] = {}
        
        # Standardize structural embeddings
        scaler = StandardScaler()
        struct_scaled = scaler.fit_transform(struct_emb)
        
        for oxygen_name, oxygen_emb in oxygen_embeddings.items():
            # Standardize oxygen embeddings
            oxygen_scaled = StandardScaler().fit_transform(oxygen_emb)
            
            # 1. Mantel test (correlation between distance matrices)
            struct_dist = cdist(struct_scaled, struct_scaled, metric='euclidean')
            oxygen_dist = cdist(oxygen_scaled, oxygen_scaled, metric='euclidean')
            
            # Flatten upper triangular matrices
            n = len(struct_dist)
            triu_indices = np.triu_indices(n, k=1)
            struct_dist_flat = struct_dist[triu_indices]
            oxygen_dist_flat = oxygen_dist[triu_indices]
            
            mantel_r, mantel_p = spearmanr(struct_dist_flat, oxygen_dist_flat)
            
            # 2. Procrustes analysis (alignment quality)
            # Reduce dimensions if needed
            min_components = min(struct_scaled.shape[0] - 1, 
                               min(struct_scaled.shape[1], oxygen_scaled.shape[1]))
            
            if struct_scaled.shape[1] > min_components:
                pca = PCA(n_components=min_components)
                struct_reduced = pca.fit_transform(struct_scaled)
            else:
                struct_reduced = struct_scaled
                
            if oxygen_scaled.shape[1] > min_components:
                pca = PCA(n_components=min_components)
                oxygen_reduced = pca.fit_transform(oxygen_scaled)
            else:
                oxygen_reduced = oxygen_scaled
            
            # Simple Procrustes (rotation only)
            from scipy.linalg import orthogonal_procrustes
            R, scale = orthogonal_procrustes(struct_reduced, oxygen_reduced)
            struct_aligned = struct_reduced @ R
            procrustes_error = np.mean((struct_aligned - oxygen_reduced) ** 2)
            
            correlations[struct_name][oxygen_name] = {
                'mantel_r': mantel_r,
                'mantel_p': mantel_p,
                'procrustes_error': procrustes_error,
                'n_drugs': len(struct_emb)
            }
            
            print(f"  {struct_name} vs {oxygen_name}: Mantel r={mantel_r:.3f} (p={mantel_p:.3e})")
    
    return correlations


def create_comparison_visualization(structural_embeddings, oxygen_embeddings, 
                                  drugs_df, common_drugs, output_dir):
    """Create comprehensive visualization comparing structural and oxygen embeddings."""
    print("\nCreating comparison visualizations...")
    
    # Use combined structural embedding and fourier oxygen embedding for main comparison
    struct_emb = structural_embeddings['combined']
    oxygen_emb = oxygen_embeddings['fourier']
    
    # Standardize
    struct_scaled = StandardScaler().fit_transform(struct_emb)
    oxygen_scaled = StandardScaler().fit_transform(oxygen_emb)
    
    # Compute projections
    print("  Computing t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    struct_tsne = tsne.fit_transform(struct_scaled)
    
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30)
    oxygen_tsne = tsne2.fit_transform(oxygen_scaled)
    
    print("  Computing UMAP projections...")
    reducer = umap.UMAP(n_neighbors=15, random_state=42)
    struct_umap = reducer.fit_transform(struct_scaled)
    
    reducer2 = umap.UMAP(n_neighbors=15, random_state=42)
    oxygen_umap = reducer2.fit_transform(oxygen_scaled)
    
    # Get toxicity labels
    drug_toxicity = drugs_df.set_index('drug').loc[common_drugs]
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color by DILI status
    colors = ['#2ecc71' if x == 0 else '#e74c3c' for x in drug_toxicity['binary_dili']]
    
    # 1. Structural t-SNE
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(struct_tsne[:, 0], struct_tsne[:, 1], 
                          c=colors, alpha=0.6, s=100)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Structural Embeddings (t-SNE)\nColored by DILI Risk')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='DILI Negative'),
                      Patch(facecolor='#e74c3c', label='DILI Positive')]
    ax1.legend(handles=legend_elements, loc='best')
    
    # 2. Oxygen t-SNE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(oxygen_tsne[:, 0], oxygen_tsne[:, 1], 
               c=colors, alpha=0.6, s=100)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('Oxygen Embeddings (t-SNE)\nColored by DILI Risk')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='best')
    
    # 3. Structural UMAP
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(struct_umap[:, 0], struct_umap[:, 1], 
               c=colors, alpha=0.6, s=100)
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.set_title('Structural Embeddings (UMAP)\nColored by DILI Risk')
    ax3.grid(True, alpha=0.3)
    
    # 4. Oxygen UMAP
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(oxygen_umap[:, 0], oxygen_umap[:, 1], 
               c=colors, alpha=0.6, s=100)
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.set_title('Oxygen Embeddings (UMAP)\nColored by DILI Risk')
    ax4.grid(True, alpha=0.3)
    
    # 5. Direct comparison scatter
    ax5 = fig.add_subplot(gs[2, :])
    
    # For direct comparison, use first 2 PCs of each
    pca_struct = PCA(n_components=2).fit_transform(struct_scaled)
    pca_oxygen = PCA(n_components=2).fit_transform(oxygen_scaled)
    
    # Plot arrows connecting same drug in both spaces
    for i in range(len(common_drugs)):
        ax5.annotate('', xy=(pca_oxygen[i, 0], pca_oxygen[i, 1]), 
                    xytext=(pca_struct[i, 0], pca_struct[i, 1]),
                    arrowprops=dict(arrowstyle='->', alpha=0.3, color='gray'))
    
    ax5.scatter(pca_struct[:, 0], pca_struct[:, 1], 
               c='blue', alpha=0.6, s=100, label='Structural')
    ax5.scatter(pca_oxygen[:, 0], pca_oxygen[:, 1], 
               c='red', alpha=0.6, s=100, label='Oxygen')
    
    ax5.set_xlabel('PC 1')
    ax5.set_ylabel('PC 2')
    ax5.set_title('Direct Comparison: Structural (blue) vs Oxygen (red) Embeddings\nArrows connect the same drug')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Structural vs Oxygen-Based Drug Embeddings Comparison', 
                fontsize=16, fontweight='bold')
    
    # Save
    output_path = output_dir / 'structural_oxygen_embedding_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return output_path


def create_correlation_heatmap(correlations, output_dir):
    """Create heatmap of correlations between embedding methods."""
    print("\nCreating correlation heatmap...")
    
    # Extract Mantel correlations
    struct_methods = list(correlations.keys())
    oxygen_methods = list(next(iter(correlations.values())).keys())
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(struct_methods), len(oxygen_methods)))
    p_matrix = np.zeros((len(struct_methods), len(oxygen_methods)))
    
    for i, struct in enumerate(struct_methods):
        for j, oxygen in enumerate(oxygen_methods):
            corr_matrix[i, j] = correlations[struct][oxygen]['mantel_r']
            p_matrix[i, j] = correlations[struct][oxygen]['mantel_p']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, 
                xticklabels=oxygen_methods,
                yticklabels=struct_methods,
                annot=True, fmt='.3f',
                cmap='coolwarm', center=0,
                vmin=-0.5, vmax=0.5,
                ax=ax1)
    ax1.set_title('Mantel Correlation (Spearman)\nBetween Distance Matrices')
    ax1.set_xlabel('Oxygen Embedding Method')
    ax1.set_ylabel('Structural Embedding Type')
    
    # Significance heatmap
    sig_matrix = p_matrix < 0.05
    sns.heatmap(sig_matrix, 
                xticklabels=oxygen_methods,
                yticklabels=struct_methods,
                annot=p_matrix, fmt='.3e',
                cmap='RdYlGn_r', vmin=0, vmax=0.1,
                ax=ax2)
    ax2.set_title('P-values\n(Green = p < 0.05)')
    ax2.set_xlabel('Oxygen Embedding Method')
    ax2.set_ylabel('Structural Embedding Type')
    
    plt.suptitle('Correlation Between Structural and Oxygen Embedding Spaces', 
                fontsize=14, fontweight='bold')
    
    # Save
    output_path = output_dir / 'structural_oxygen_correlation_heatmap.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Print summary
    print("\nCorrelation Summary:")
    best_corr = 0
    best_pair = None
    for struct in struct_methods:
        for oxygen in oxygen_methods:
            r = correlations[struct][oxygen]['mantel_r']
            p = correlations[struct][oxygen]['mantel_p']
            if abs(r) > abs(best_corr) and p < 0.05:
                best_corr = r
                best_pair = (struct, oxygen)
            if p < 0.05:
                print(f"  {struct} vs {oxygen}: r={r:.3f} (p={p:.3e}) ***")
    
    if best_pair:
        print(f"\nStrongest correlation: {best_pair[0]} vs {best_pair[1]} (r={best_corr:.3f})")
    
    return output_path


def analyze_toxicity_clustering(structural_embeddings, oxygen_embeddings, 
                               drugs_df, common_drugs, output_dir):
    """Analyze how well each embedding type separates toxic vs non-toxic drugs."""
    print("\nAnalyzing toxicity separation...")
    
    # Get toxicity labels
    drug_toxicity = drugs_df.set_index('drug').loc[common_drugs]
    y_dili = drug_toxicity['binary_dili'].values
    y_hepatotox = drug_toxicity['hepatotoxicity_boxed_warning'].values
    
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    results = []
    
    # Test each embedding
    all_embeddings = {
        **{f'struct_{k}': v for k, v in structural_embeddings.items()},
        **{f'oxygen_{k}': v for k, v in oxygen_embeddings.items()}
    }
    
    for name, embedding in all_embeddings.items():
        # Standardize
        X = StandardScaler().fit_transform(embedding)
        
        # Silhouette score for DILI separation
        if len(np.unique(y_dili)) > 1:
            sil_dili = silhouette_score(X, y_dili)
        else:
            sil_dili = 0
            
        # Silhouette score for hepatotoxicity separation  
        if len(np.unique(y_hepatotox)) > 1:
            sil_hepatotox = silhouette_score(X, y_hepatotox)
        else:
            sil_hepatotox = 0
        
        # KNN classification accuracy
        knn = KNeighborsClassifier(n_neighbors=5)
        dili_scores = cross_val_score(knn, X, y_dili, cv=5, scoring='roc_auc')
        
        if len(np.unique(y_hepatotox)) > 1:
            hepatotox_scores = cross_val_score(knn, X, y_hepatotox, cv=5, scoring='roc_auc')
        else:
            hepatotox_scores = [0]
        
        results.append({
            'Embedding': name,
            'DILI_Silhouette': sil_dili,
            'Hepatotox_Silhouette': sil_hepatotox,
            'DILI_AUC': np.mean(dili_scores),
            'Hepatotox_AUC': np.mean(hepatotox_scores)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('DILI_AUC', ascending=False)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # DILI Silhouette scores
    results_df.plot.barh(x='Embedding', y='DILI_Silhouette', ax=ax1)
    ax1.set_title('DILI Separation (Silhouette Score)')
    ax1.set_xlabel('Silhouette Score')
    
    # Hepatotox Silhouette scores
    results_df.plot.barh(x='Embedding', y='Hepatotox_Silhouette', ax=ax2)
    ax2.set_title('Hepatotoxicity Separation (Silhouette Score)')
    ax2.set_xlabel('Silhouette Score')
    
    # DILI AUC scores
    results_df.plot.barh(x='Embedding', y='DILI_AUC', ax=ax3)
    ax3.set_title('DILI Prediction (5-fold CV AUC)')
    ax3.set_xlabel('AUC Score')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Hepatotox AUC scores
    results_df.plot.barh(x='Embedding', y='Hepatotox_AUC', ax=ax4)
    ax4.set_title('Hepatotoxicity Prediction (5-fold CV AUC)')
    ax4.set_xlabel('AUC Score')
    ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Toxicity Separation Quality: Structural vs Oxygen Embeddings', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'toxicity_separation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results table
    results_path = output_dir / 'toxicity_separation_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"  Saved: {output_path}")
    print(f"  Saved: {results_path}")
    
    # Print summary
    print("\nTop 5 embeddings for DILI prediction:")
    print(results_df[['Embedding', 'DILI_AUC']].head())
    
    return results_df


def main():
    """Main analysis pipeline."""
    print("=== Structural vs Oxygen Embedding Comparison ===\n")
    
    # Create output directory
    output_dir = project_root / "results" / "figures" / "structural_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    drugs_df = load_drug_smiles()
    
    # Compute molecular fingerprints
    fingerprints, valid_drugs = compute_molecular_fingerprints(drugs_df)
    
    # Load oxygen embeddings
    drug_embeddings, drug_metadata = load_oxygen_embeddings()
    
    # Create aligned embeddings
    structural_embeddings, oxygen_embeddings, common_drugs = create_embedding_dataframes(
        fingerprints, drug_embeddings, drug_metadata
    )
    
    # Compute correlations
    correlations = compute_embedding_correlations(structural_embeddings, oxygen_embeddings)
    
    # Create visualizations
    comparison_path = create_comparison_visualization(
        structural_embeddings, oxygen_embeddings, 
        drugs_df, common_drugs, output_dir
    )
    
    heatmap_path = create_correlation_heatmap(correlations, output_dir)
    
    # Analyze toxicity separation
    toxicity_results = analyze_toxicity_clustering(
        structural_embeddings, oxygen_embeddings,
        drugs_df, common_drugs, output_dir
    )
    
    # Save all correlations
    corr_path = output_dir / 'structural_oxygen_correlations.joblib'
    joblib.dump({
        'correlations': correlations,
        'structural_embeddings': structural_embeddings,
        'oxygen_embeddings': oxygen_embeddings,
        'common_drugs': common_drugs
    }, corr_path)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔬 Analyzed {len(common_drugs)} drugs with both structural and oxygen data")


if __name__ == "__main__":
    main()