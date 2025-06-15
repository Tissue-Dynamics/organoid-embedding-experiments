#!/usr/bin/env python3
"""Detailed analysis of Sanofi compounds and their relationships."""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sanofi_compounds():
    """Analyze Sanofi compounds in detail."""
    
    # Load previous results
    results = joblib.load('fast_drug_analysis_results.joblib')
    
    features_df = results['features_df']
    embeddings = results['embeddings']
    drug_names = results['drug_names']
    feature_cols = results['feature_cols']
    
    # Extract Sanofi compounds
    sanofi_mask = np.array([drug.startswith('Sanofi') for drug in drug_names])
    sanofi_drugs = drug_names[sanofi_mask]
    sanofi_embeddings = embeddings[sanofi_mask]
    sanofi_features = features_df[features_df['drug'].str.startswith('Sanofi')]
    
    logger.info(f"\n🔬 Analyzing {len(sanofi_drugs)} Sanofi compounds: {list(sanofi_drugs)}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Sanofi compound similarity matrix
    ax1 = plt.subplot(3, 3, 1)
    similarity = cosine_similarity(sanofi_embeddings)
    sns.heatmap(similarity, xticklabels=sanofi_drugs, yticklabels=sanofi_drugs,
                annot=True, fmt='.2f', cmap='coolwarm', center=0.5, ax=ax1)
    ax1.set_title('Sanofi Compound Similarity Matrix')
    
    # 2. Feature comparison
    ax2 = plt.subplot(3, 3, 2)
    key_features = ['response_range', 'dose_response_correlation', 
                    'overall_response_change', 'avg_variability']
    sanofi_key_features = sanofi_features[key_features]
    
    # Normalize for visualization
    normalized = (sanofi_key_features - sanofi_key_features.mean()) / sanofi_key_features.std()
    
    sns.heatmap(normalized.T, xticklabels=sanofi_drugs, yticklabels=key_features,
                cmap='RdBu_r', center=0, annot=True, fmt='.1f', ax=ax2)
    ax2.set_title('Key Feature Profiles (Normalized)')
    
    # 3. Dose-response patterns
    ax3 = plt.subplot(3, 3, 3)
    dr_corr = sanofi_features['dose_response_correlation'].values
    response_range = sanofi_features['response_range'].values
    
    scatter = ax3.scatter(dr_corr, response_range, s=200, alpha=0.7, 
                         c=range(len(sanofi_drugs)), cmap='tab10')
    
    for i, drug in enumerate(sanofi_drugs):
        ax3.annotate(drug, (dr_corr[i], response_range[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Dose-Response Correlation')
    ax3.set_ylabel('Response Range')
    ax3.set_title('Dose-Response Characteristics')
    ax3.grid(True, alpha=0.3)
    
    # 4. Concentration range comparison
    ax4 = plt.subplot(3, 3, 4)
    conc_range = np.log10(sanofi_features['max_concentration'] / sanofi_features['min_concentration'])
    n_concs = sanofi_features['n_concentrations']
    
    ax4.bar(range(len(sanofi_drugs)), conc_range, alpha=0.7, label='Log Conc Range')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(range(len(sanofi_drugs)), n_concs, 'ro-', label='N Concentrations')
    
    ax4.set_xticks(range(len(sanofi_drugs)))
    ax4.set_xticklabels(sanofi_drugs, rotation=45)
    ax4.set_ylabel('Log10(Max/Min Concentration)')
    ax4_twin.set_ylabel('Number of Concentrations')
    ax4.set_title('Concentration Coverage')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    # 5. Early vs Late response
    ax5 = plt.subplot(3, 3, 5)
    early = sanofi_features['overall_early_response'].values
    late = sanofi_features['overall_late_response'].values
    
    ax5.scatter(early, late, s=200, alpha=0.7)
    
    # Add diagonal line
    lims = [min(early.min(), late.min()), max(early.max(), late.max())]
    ax5.plot(lims, lims, 'k--', alpha=0.5)
    
    for i, drug in enumerate(sanofi_drugs):
        ax5.annotate(drug, (early[i], late[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax5.set_xlabel('Early Response (0-24h)')
    ax5.set_ylabel('Late Response (>72h)')
    ax5.set_title('Temporal Response Patterns')
    ax5.grid(True, alpha=0.3)
    
    # 6. Response change over time
    ax6 = plt.subplot(3, 3, 6)
    response_change = sanofi_features['overall_response_change'].values
    
    bars = ax6.bar(range(len(sanofi_drugs)), response_change, 
                    color=['red' if x < 0 else 'green' for x in response_change])
    ax6.set_xticks(range(len(sanofi_drugs)))
    ax6.set_xticklabels(sanofi_drugs, rotation=45)
    ax6.set_ylabel('Response Change (Late - Early)')
    ax6.set_title('Response Change Over Time')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Variability analysis
    ax7 = plt.subplot(3, 3, 7)
    variability = sanofi_features['avg_variability'].values
    
    ax7.scatter(n_concs, variability, s=200, alpha=0.7)
    for i, drug in enumerate(sanofi_drugs):
        ax7.annotate(drug, (n_concs[i], variability[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax7.set_xlabel('Number of Concentrations Tested')
    ax7.set_ylabel('Average Variability')
    ax7.set_title('Data Quality: Variability vs Coverage')
    ax7.grid(True, alpha=0.3)
    
    # 8. Compound ranking by multiple criteria
    ax8 = plt.subplot(3, 3, 8)
    
    # Create composite score
    scores = pd.DataFrame({
        'Drug': sanofi_drugs,
        'Dose-Response': dr_corr,
        'Low Variability': -variability / variability.max(),
        'Positive Change': response_change / abs(response_change).max(),
        'Data Coverage': n_concs / n_concs.max()
    })
    
    scores['Composite'] = scores[['Dose-Response', 'Low Variability', 
                                  'Positive Change', 'Data Coverage']].mean(axis=1)
    scores = scores.sort_values('Composite', ascending=False)
    
    ax8.barh(range(len(scores)), scores['Composite'])
    ax8.set_yticks(range(len(scores)))
    ax8.set_yticklabels(scores['Drug'])
    ax8.set_xlabel('Composite Quality Score')
    ax8.set_title('Sanofi Compound Ranking')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. Detailed feature correlation
    ax9 = plt.subplot(3, 3, 9)
    feature_subset = sanofi_features[key_features]
    corr_matrix = feature_subset.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax9, square=True)
    ax9.set_title('Feature Correlations (Sanofi compounds)')
    
    plt.tight_layout()
    plt.savefig('sanofi_compound_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("💾 Saved Sanofi analysis to: sanofi_compound_analysis.png")
    
    # Print summary statistics
    logger.info("\n📊 SANOFI COMPOUND SUMMARY:")
    logger.info(f"\nMost similar compounds:")
    sim_upper = similarity[np.triu_indices(len(similarity), k=1)]
    sim_pairs = [(sanofi_drugs[i], sanofi_drugs[j], similarity[i,j]) 
                 for i in range(len(sanofi_drugs)) 
                 for j in range(i+1, len(sanofi_drugs))]
    sim_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for drug1, drug2, sim in sim_pairs[:3]:
        logger.info(f"  {drug1} - {drug2}: {sim:.3f}")
    
    logger.info(f"\nBest dose-response correlations:")
    dr_sorted = sorted(zip(sanofi_drugs, dr_corr), key=lambda x: x[1], reverse=True)
    for drug, corr in dr_sorted[:3]:
        logger.info(f"  {drug}: {corr:.3f}")
    
    logger.info(f"\nLowest variability (most consistent):")
    var_sorted = sorted(zip(sanofi_drugs, variability), key=lambda x: x[1])
    for drug, var in var_sorted[:3]:
        logger.info(f"  {drug}: {var:.2f}")
    
    return sanofi_features, scores

def create_sanofi_network():
    """Create network visualization of Sanofi relationships."""
    import networkx as nx
    
    # Load data
    results = joblib.load('fast_drug_analysis_results.joblib')
    embeddings = results['embeddings']
    drug_names = results['drug_names']
    
    # Get Sanofi compounds
    sanofi_mask = np.array([drug.startswith('Sanofi') for drug in drug_names])
    sanofi_drugs = drug_names[sanofi_mask]
    sanofi_embeddings = embeddings[sanofi_mask]
    
    # Calculate similarities
    similarity = cosine_similarity(sanofi_embeddings)
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for drug in sanofi_drugs:
        G.add_node(drug)
    
    # Add edges (only strong similarities)
    threshold = 0.7
    for i in range(len(sanofi_drugs)):
        for j in range(i+1, len(sanofi_drugs)):
            if similarity[i,j] > threshold:
                G.add_edge(sanofi_drugs[i], sanofi_drugs[j], 
                          weight=similarity[i,j])
    
    # Visualize
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                          alpha=0.7)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Add edge labels
    edge_labels = {(u,v): f"{G[u][v]['weight']:.2f}" for u,v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(f'Sanofi Compound Similarity Network (threshold > {threshold})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('sanofi_network.png', dpi=150, bbox_inches='tight')
    logger.info("💾 Saved network visualization to: sanofi_network.png")

if __name__ == "__main__":
    logger.info("🔬 Starting Sanofi Compound Analysis...")
    
    try:
        sanofi_features, scores = analyze_sanofi_compounds()
        create_sanofi_network()
        
        logger.info("\n✅ Sanofi analysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise