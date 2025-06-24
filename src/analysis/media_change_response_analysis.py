#!/usr/bin/env python3
"""
Extract and visualize drug-specific media change response profiles.
Shows how different drugs respond differently to media changes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / "results" / "figures" / "diffusion_analysis"
results_dir.mkdir(parents=True, exist_ok=True)

def connect_to_database():
    """Connect to the PostgreSQL database via DuckDB."""
    DATABASE_URL = "postgresql://postgres.ooqjakwyfawahvnzcllk:eTEEoWWGExovyChe@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    
    conn = duckdb.connect()
    conn.execute("INSTALL postgres;")
    conn.execute("LOAD postgres;")
    conn.execute(f"ATTACH '{DATABASE_URL}' AS supabase (TYPE POSTGRES, READ_ONLY);")
    return conn

def detect_media_changes_simple(df, expected_times=[72, 144, 216]):
    """Use expected media change times with some tolerance."""
    media_changes = []
    
    for plate_id in df['plate_id'].unique():
        plate_data = df[df['plate_id'] == plate_id]
        max_time = plate_data['hours'].max()
        
        for expected_time in expected_times:
            if expected_time < max_time - 24:  # Only if we have data after the change
                media_changes.append({
                    'plate_id': plate_id,
                    'time_hours': expected_time
                })
    
    return pd.DataFrame(media_changes)

def extract_media_response_windows(df, media_changes, window_before=4, window_after=24):
    """Extract oxygen data windows around media changes."""
    response_data = []
    
    for _, change in media_changes.iterrows():
        plate_id = change['plate_id']
        change_time = change['time_hours']
        
        # Get data for this plate around the media change
        plate_data = df[df['plate_id'] == plate_id]
        
        # For each well on this plate
        for well_id in plate_data['well_id'].unique():
            well_data = plate_data[plate_data['well_id'] == well_id]
            
            # Time window
            mask = (well_data['hours'] >= change_time - window_before) & \
                   (well_data['hours'] <= change_time + window_after)
            
            window_data = well_data[mask].copy()
            if len(window_data) < 10:  # Need sufficient data
                continue
            
            # Calculate time relative to media change
            window_data['time_relative'] = window_data['hours'] - change_time
            
            # Get well metadata
            drug = window_data['drug'].iloc[0]
            concentration = window_data['concentration'].iloc[0]
            
            response_data.append({
                'well_id': well_id,
                'drug': drug,
                'concentration': concentration,
                'plate_id': plate_id,
                'media_change_time': change_time,
                'times': window_data['time_relative'].values,
                'oxygen': window_data['oxygen'].values
            })
    
    return response_data

def fit_recovery_model(times, oxygen):
    """Fit exponential recovery model to media change response."""
    # Find the spike point (maximum oxygen around t=0)
    near_change_mask = np.abs(times) < 2
    if not near_change_mask.any():
        # No data near media change time, use wider window
        near_change_mask = np.abs(times) < 4
        if not near_change_mask.any():
            return None
    
    near_change_oxygen = oxygen[near_change_mask]
    near_change_times = times[near_change_mask]
    
    if len(near_change_oxygen) == 0:
        return None
        
    spike_idx = np.argmax(near_change_oxygen)
    spike_time = near_change_times[spike_idx]
    spike_value = near_change_oxygen[spike_idx]
    
    # Baseline (before spike)
    pre_spike = oxygen[times < -1]
    baseline = np.median(pre_spike) if len(pre_spike) > 0 else oxygen[0]
    
    # Only fit recovery after the spike
    post_spike_mask = times > spike_time + 0.5
    if post_spike_mask.sum() < 5:
        return None
    
    t_fit = times[post_spike_mask] - spike_time
    y_fit = oxygen[post_spike_mask]
    
    def exp_recovery(t, recovery_tau, final_level):
        return (spike_value - final_level) * np.exp(-t / recovery_tau) + final_level
    
    try:
        popt, _ = curve_fit(exp_recovery, t_fit, y_fit, 
                           p0=[5.0, baseline],
                           bounds=([0.1, baseline - 20], [50, baseline + 20]))
        
        return {
            'baseline': baseline,
            'spike_height': spike_value - baseline,
            'recovery_tau': popt[0],
            'final_level': popt[1],
            'spike_time': spike_time
        }
    except:
        return None

def analyze_drug_responses(response_windows):
    """Analyze media change responses by drug and concentration."""
    drug_profiles = {}
    
    # Convert to DataFrame for easier grouping
    rows = []
    for window in response_windows:
        rows.append({
            'drug': window['drug'],
            'concentration': window['concentration'],
            'well_id': window['well_id'],
            'window': window
        })
    
    df_windows = pd.DataFrame(rows)
    
    # Group by drug and concentration
    for (drug, conc), group in df_windows.groupby(['drug', 'concentration']):
        if drug not in drug_profiles:
            drug_profiles[drug] = {}
        
        # Collect all response curves for this drug/conc
        all_responses = []
        recovery_params = []
        
        for _, row in group.iterrows():
            window = row['window']
            times = window['times']
            oxygen = window['oxygen']
            
            # Fit recovery model
            params = fit_recovery_model(times, oxygen)
            if params is not None:
                recovery_params.append(params)
            
            all_responses.append((times, oxygen))
        
        if recovery_params:
            # Average parameters
            avg_params = {
                'baseline': np.mean([p['baseline'] for p in recovery_params]),
                'spike_height': np.mean([p['spike_height'] for p in recovery_params]),
                'recovery_tau': np.mean([p['recovery_tau'] for p in recovery_params]),
                'final_level': np.mean([p['final_level'] for p in recovery_params])
            }
            
            drug_profiles[drug][conc] = {
                'responses': all_responses,
                'avg_params': avg_params,
                'n_samples': len(all_responses),
                'all_params': recovery_params
            }
    
    return drug_profiles

def create_drug_response_visualization(drug_profiles):
    """Create comprehensive visualization of drug-specific media responses."""
    
    # Calculate average recovery tau for each drug
    drug_avg_taus = {}
    for drug, conc_data in drug_profiles.items():
        taus = []
        for conc, data in conc_data.items():
            if 'avg_params' in data:
                taus.append(data['avg_params']['recovery_tau'])
        if taus:
            drug_avg_taus[drug] = np.mean(taus)
    
    # Sort drugs by recovery tau
    sorted_drugs = sorted(drug_avg_taus.items(), key=lambda x: x[1])
    
    # Select diverse examples
    n_drugs = len(sorted_drugs)
    if n_drugs >= 9:
        # Select drugs with fast, medium, and slow recovery
        indices = [0, n_drugs//8, n_drugs//4, n_drugs//3, n_drugs//2, 
                  2*n_drugs//3, 3*n_drugs//4, 7*n_drugs//8, n_drugs-1]
        drugs_to_show = [sorted_drugs[i][0] for i in indices]
    else:
        drugs_to_show = [drug for drug, _ in sorted_drugs[:9]]
    
    # Create main figure
    fig = plt.figure(figsize=(20, 16))
    
    # Main grid for individual drug plots
    for idx, drug in enumerate(drugs_to_show[:9]):
        ax = plt.subplot(3, 3, idx + 1)
        
        drug_data = drug_profiles[drug]
        
        # Color by concentration
        concs = sorted(drug_data.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
        
        for conc, color in zip(concs, colors):
            data = drug_data[conc]
            
            # Plot individual responses
            for times, oxygen in data['responses'][:3]:  # Show up to 3 replicates
                ax.plot(times, oxygen, color=color, alpha=0.2, linewidth=1)
            
            # Plot average response
            if len(data['responses']) > 0:
                # Create average curve
                all_times = []
                all_oxygen = []
                for times, oxygen in data['responses']:
                    all_times.extend(times)
                    all_oxygen.extend(oxygen)
                
                # Bin the data
                time_bins = np.arange(-4, 25, 0.5)
                binned_oxygen = []
                
                for i in range(len(time_bins) - 1):
                    mask = (np.array(all_times) >= time_bins[i]) & (np.array(all_times) < time_bins[i+1])
                    if mask.sum() > 0:
                        binned_oxygen.append(np.mean(np.array(all_oxygen)[mask]))
                    else:
                        binned_oxygen.append(np.nan)
                
                time_centers = (time_bins[:-1] + time_bins[1:]) / 2
                
                # Plot average
                valid = ~np.isnan(binned_oxygen)
                ax.plot(time_centers[valid], np.array(binned_oxygen)[valid], 
                       color=color, linewidth=3,
                       label=f'{conc:.2f} μM (τ={data["avg_params"]["recovery_tau"]:.1f}h)')
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Media change')
        ax.set_xlabel('Time from media change (hours)')
        ax.set_ylabel('Oxygen (%)')
        ax.set_title(f'{drug}\nAvg τ = {drug_avg_taus[drug]:.1f}h')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 24)
    
    plt.suptitle('Drug-Specific Media Change Response Profiles', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save main figure
    output_path = results_dir / "drug_specific_media_responses.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary figures
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of recovery times
    all_taus = []
    for drug, conc_data in drug_profiles.items():
        for conc, data in conc_data.items():
            if 'all_params' in data:
                for params in data['all_params']:
                    all_taus.append(params['recovery_tau'])
    
    ax1.hist(all_taus, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.median(all_taus), color='red', linestyle='--', 
                label=f'Median: {np.median(all_taus):.1f}h')
    ax1.set_xlabel('Recovery Time Constant τ (hours)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Media Change Recovery Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery tau vs concentration
    tau_vs_conc = []
    for drug, conc_data in drug_profiles.items():
        for conc, data in conc_data.items():
            if 'avg_params' in data:
                tau_vs_conc.append({
                    'concentration': conc,
                    'tau': data['avg_params']['recovery_tau'],
                    'drug': drug
                })
    
    df_tau = pd.DataFrame(tau_vs_conc)
    
    # Bin concentrations
    conc_bins = [0, 0.1, 1, 10, 100, 1000]
    df_tau['conc_bin'] = pd.cut(df_tau['concentration'], bins=conc_bins)
    
    # Box plot
    df_tau.boxplot(column='tau', by='conc_bin', ax=ax2)
    ax2.set_xlabel('Concentration Range (μM)')
    ax2.set_ylabel('Recovery τ (hours)')
    ax2.set_title('Recovery Time vs Concentration')
    plt.sca(ax2)
    plt.xticks(rotation=45)
    
    # 3. Top drugs by slowest recovery
    sorted_drugs_display = sorted(drug_avg_taus.items(), key=lambda x: x[1], reverse=True)[:15]
    drugs, taus = zip(*sorted_drugs_display)
    
    y_pos = np.arange(len(drugs))
    ax3.barh(y_pos, taus, color=plt.cm.RdYlBu_r(np.array(taus)/max(taus)))
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(drugs)
    ax3.set_xlabel('Average Recovery τ (hours)')
    ax3.set_title('Top 15 Drugs by Slowest Media Change Recovery')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, tau in enumerate(taus):
        ax3.text(tau + 0.2, i, f'{tau:.1f}', va='center', fontsize=9)
    
    # 4. Spike height distribution
    all_spikes = []
    for drug, conc_data in drug_profiles.items():
        for conc, data in conc_data.items():
            if 'all_params' in data:
                for params in data['all_params']:
                    all_spikes.append(params['spike_height'])
    
    ax4.hist(all_spikes, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.median(all_spikes), color='red', linestyle='--',
                label=f'Median: {np.median(all_spikes):.1f}%')
    ax4.set_xlabel('Spike Height (% O₂)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Media Change Spike Heights')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Media Change Response Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = results_dir / "media_recovery_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_path = results_dir / "media_response_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("MEDIA CHANGE RESPONSE STATISTICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total drugs analyzed: {len(drug_profiles)}\n")
        f.write(f"Total recovery parameters fitted: {len(all_taus)}\n")
        f.write(f"Recovery tau range: {min(all_taus):.1f} - {max(all_taus):.1f} hours\n")
        f.write(f"Median recovery tau: {np.median(all_taus):.1f} hours\n")
        f.write(f"Mean recovery tau: {np.mean(all_taus):.1f} ± {np.std(all_taus):.1f} hours\n")
        f.write(f"Spike height range: {min(all_spikes):.1f} - {max(all_spikes):.1f} %O₂\n")
        f.write(f"Median spike height: {np.median(all_spikes):.1f} %O₂\n")
        f.write(f"\nFastest recovering drugs:\n")
        for drug, tau in sorted(drug_avg_taus.items(), key=lambda x: x[1])[:10]:
            f.write(f"  {drug}: {tau:.1f} hours\n")
        f.write(f"\nSlowest recovering drugs:\n")
        for drug, tau in sorted(drug_avg_taus.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  {drug}: {tau:.1f} hours\n")
    
    print(f"Saved visualizations to:")
    print(f"  - {output_path}")
    print(f"  - {summary_path}") 
    print(f"  - {stats_path}")
    
    return drug_profiles

def main():
    """Extract and analyze drug-specific media change responses."""
    print("="*80)
    print("DRUG-SPECIFIC MEDIA CHANGE RESPONSE ANALYSIS")
    print("="*80)
    
    # Connect to database
    print("Connecting to database...")
    conn = connect_to_database()
    
    # Query oxygen data - sample to make it faster
    query = """
    WITH sampled_plates AS (
        SELECT DISTINCT plate_id 
        FROM supabase.public.well_map_data 
        WHERE drug != '' AND drug IS NOT NULL
        ORDER BY plate_id
        LIMIT 10  -- Sample 10 plates for faster analysis
    )
    SELECT 
        w.plate_id || '_' || w.well_number as well_id,
        w.drug,
        w.concentration,
        w.plate_id,
        p.timestamp,
        p.median_o2 as oxygen,
        EXTRACT(EPOCH FROM (p.timestamp - MIN(p.timestamp) OVER (PARTITION BY w.plate_id))) / 3600.0 as hours
    FROM supabase.public.well_map_data w
    JOIN supabase.public.processed_data p
        ON w.plate_id = p.plate_id AND w.well_number = p.well_number
    WHERE w.plate_id IN (SELECT plate_id FROM sampled_plates)
        AND w.drug != '' 
        AND w.drug IS NOT NULL
        AND w.is_excluded = false
        AND p.is_excluded = false
        AND w.concentration > 0
    ORDER BY w.plate_id, w.well_number, p.timestamp
    """
    
    print("Fetching oxygen data (sampled)...")
    df = conn.execute(query).df()
    conn.close()
    
    print(f"Loaded {len(df):,} measurements")
    print(f"Covering {df['drug'].nunique()} drugs")
    print(f"From {df['plate_id'].nunique()} plates")
    
    # Use expected media change times
    print("\nUsing expected media change times...")
    media_changes = detect_media_changes_simple(df)
    print(f"Created {len(media_changes)} potential media change events")
    
    # Extract response windows
    print("\nExtracting response windows...")
    response_windows = extract_media_response_windows(df, media_changes)
    print(f"Extracted {len(response_windows)} response windows")
    
    if len(response_windows) == 0:
        print("No response windows found. Check data availability around expected media change times.")
        return
    
    # Analyze drug-specific responses
    print("\nAnalyzing drug-specific response patterns...")
    drug_profiles = analyze_drug_responses(response_windows)
    print(f"Analyzed {len(drug_profiles)} drugs")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_drug_response_visualization(drug_profiles)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()