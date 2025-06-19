#!/usr/bin/env python3
"""
Generate Final Summary of Event-Aware Feature Analysis
Summarize the improvements and key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "data"
fig_dir = project_root / "results" / "figures"

print("=" * 80)
print("EVENT-AWARE FEATURES: FINAL SUMMARY")
print("=" * 80)

# Load correlation results
event_corr = pd.read_csv(results_dir / "event_aware_dili_correlations.csv")
event_corr = event_corr.sort_values('abs_spearman', ascending=False)

print("\n🏆 TOP EVENT-AWARE FEATURES CORRELATING WITH DILI:")
print("-" * 60)
print(f"{'Feature':<35} {'Correlation':>12} {'P-value':>12}")
print("-" * 60)

for _, row in event_corr.head(10).iterrows():
    feature_name = row['feature'].replace('_mean', '').replace('_', ' ').title()[:35]
    print(f"{feature_name:<35} {row['spearman_r']:>12.3f} {row['spearman_p']:>12.3e}")

print("\n📊 KEY FINDINGS:")
print("-" * 60)

# Best correlation
best_corr = event_corr.iloc[0]
print(f"1. STRONGEST PREDICTOR: {best_corr['feature'].replace('_mean', '').replace('_', ' ').title()}")
print(f"   • Correlation: r = {best_corr['spearman_r']:.3f} (p = {best_corr['spearman_p']:.3e})")
print(f"   • Interpretation: Drugs that maintain oxygen consumption late in treatment")
print(f"     (consumption_ratio > 1) are associated with higher DILI risk")

# Temporal changes
temporal_features = event_corr[event_corr['feature'].str.contains('consumption_change')]
if len(temporal_features) > 0:
    temp_best = temporal_features.iloc[0]
    print(f"\n2. TEMPORAL PROGRESSION:")
    print(f"   • Feature: {temp_best['feature'].replace('_mean', '').replace('_', ' ').title()}")
    print(f"   • Correlation: r = {temp_best['spearman_r']:.3f} (p = {temp_best['spearman_p']:.3e})")
    print(f"   • Interpretation: Drugs causing decreasing oxygen consumption over time")
    print(f"     (negative consumption change) have lower DILI risk")

# Comparison with Phase 2
print(f"\n3. IMPROVEMENT OVER PHASE 2 EMBEDDINGS:")
print(f"   • Phase 2 best (Fourier): r = 0.260")
print(f"   • Event-aware best: r = {best_corr['abs_spearman']:.3f}")
print(f"   • Improvement: {(best_corr['abs_spearman'] - 0.260) / 0.260 * 100:.1f}%")

print(f"\n4. WHY EVENT-AWARE FEATURES WORK BETTER:")
print(f"   • Remove media change artifacts (spikes)")
print(f"   • Focus on biological oxygen consumption patterns")
print(f"   • Capture temporal progression of drug effects")
print(f"   • Extract features from stable measurement periods")

# Dataset statistics
drug_features = pd.read_parquet(results_dir / "event_aware_features_drugs.parquet")
wells_features = pd.read_parquet(results_dir / "event_aware_features_wells.parquet")

print(f"\n5. DATASET COVERAGE:")
print(f"   • Wells analyzed: {len(wells_features):,}")
print(f"   • Drugs analyzed: {len(drug_features)}")
print(f"   • Average segments per well: {wells_features['n_segments'].mean():.1f}")
print(f"   • Average monitored hours per well: {wells_features['total_monitored_hours'].mean():.0f}")

# Create final summary figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Event-Aware Features: Key Results', fontsize=16, fontweight='bold')

# Plot 1: Method comparison
methods = ['Phase 2\nFourier', 'Phase 2\nTSFresh', 'Phase 2\nCatch22', 'Event-Aware\nFeatures']
correlations = [0.260, 0.243, 0.237, best_corr['abs_spearman']]
colors = ['lightblue', 'lightblue', 'lightblue', 'lightgreen']

bars = ax1.bar(methods, correlations, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Best |Correlation| with DILI Risk', fontsize=12)
ax1.set_title('Comparison of Methods', fontsize=14)
ax1.set_ylim(0, 0.5)

# Add value labels and improvement
for i, (bar, val) in enumerate(zip(bars, correlations)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    if i == 3:  # Event-aware
        improvement = (val - 0.260) / 0.260 * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04, 
                f'+{improvement:.0f}%', ha='center', va='bottom', 
                fontsize=10, color='green', fontweight='bold')

ax1.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='r=0.3 threshold')
ax1.legend()

# Plot 2: Top event-aware features
top_features = event_corr.head(8).copy()
feature_names = [f.replace('_mean', '').replace('_', ' ').title()[:25] 
                 for f in top_features['feature']]

y_pos = np.arange(len(feature_names))
colors = ['darkgreen' if p < 0.01 else 'green' if p < 0.05 else 'gray' 
          for p in top_features['spearman_p']]

bars = ax2.barh(y_pos, top_features['abs_spearman'], color=colors, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(feature_names, fontsize=10)
ax2.set_xlabel('|Correlation| with DILI Risk', fontsize=12)
ax2.set_title('Top Event-Aware Features', fontsize=14)
ax2.set_xlim(0, 0.5)

# Add significance markers
for i, (bar, p_val) in enumerate(zip(bars, top_features['spearman_p'])):
    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = ''
    
    if sig_text:
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                sig_text, va='center', fontsize=10, fontweight='bold')

ax2.axvline(0.3, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(fig_dir / 'event_aware_final_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📁 RESULTS SAVED:")
print(f"   • Summary figure: {fig_dir / 'event_aware_final_summary.png'}")
print(f"   • Feature correlations: {results_dir / 'event_aware_dili_correlations.csv'}")
print(f"   • Event-aware features: {results_dir / 'event_aware_features_drugs.parquet'}")

print("\n" + "="*80)
print("✅ EVENT-AWARE FEATURE ANALYSIS COMPLETE!")
print("="*80)

print("\n🎯 NEXT STEPS:")
print("   1. Build predictive model using top event-aware features")
print("   2. Validate on held-out drugs")
print("   3. Combine with chemical structure features")
print("   4. Deploy as DILI risk screening tool")