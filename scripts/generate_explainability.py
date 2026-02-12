"""
Generate explainability plots showing which vitals contribute to risk scores
Run: python scripts\generate_explainability.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk_breakdown(df, patient_id, save_path=None):
    """
    Show breakdown of risk score components for a patient
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                    patient_data['timestamp'].iloc[0]).dt.total_seconds()
    
    scenario = patient_data['scenario'].iloc[0]
    
    # Calculate risk components (simplified from RiskScorer)
    vital_severity = calculate_vital_severity_simple(patient_data)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Patient {patient_id} - Risk Score Explainability\n{scenario}', 
                 fontsize=14, fontweight='bold')
    
    time = patient_data['time_seconds']
    
    # Plot 1: Total Risk Score
    axes[0].plot(time, patient_data['risk_score'], linewidth=2, color='darkred', label='Total Risk Score')
    axes[0].axhspan(0, 3, alpha=0.1, color='green', label='LOW')
    axes[0].axhspan(3, 6, alpha=0.1, color='yellow', label='MODERATE')
    axes[0].axhspan(6, 8, alpha=0.1, color='orange', label='HIGH')
    axes[0].axhspan(8, 10, alpha=0.1, color='red', label='CRITICAL')
    axes[0].set_ylabel('Risk Score (0-10)')
    axes[0].set_title('Overall Risk Score Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.5, 10.5)
    
    # Plot 2: Vital Contributions
    hr_contrib = calculate_hr_contribution(patient_data)
    spo2_contrib = calculate_spo2_contribution(patient_data)
    bp_contrib = calculate_bp_contribution(patient_data)
    
    axes[1].fill_between(time, 0, hr_contrib, alpha=0.6, label='Heart Rate', color='red')
    axes[1].fill_between(time, hr_contrib, hr_contrib + spo2_contrib, alpha=0.6, label='SpO2', color='blue')
    axes[1].fill_between(time, hr_contrib + spo2_contrib, hr_contrib + spo2_contrib + bp_contrib, 
                        alpha=0.6, label='Blood Pressure', color='purple')
    axes[1].set_ylabel('Risk Contribution')
    axes[1].set_title('Which Vitals Are Driving the Risk Score?')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance Heatmap
    feature_importance = calculate_feature_importance(patient_data)
    
    # Sample every 60 seconds for readability
    sample_indices = range(0, len(patient_data), 60)
    time_labels = [f"{int(t/60)}m" for t in time.iloc[sample_indices]]
    
    im = axes[2].imshow(feature_importance[:, sample_indices], aspect='auto', cmap='RdYlGn_r')
    axes[2].set_yticks(range(len(['HR', 'SpO2', 'SBP', 'DBP', 'Shock Index', 'MAP'])))
    axes[2].set_yticklabels(['HR', 'SpO2', 'SBP', 'DBP', 'Shock Index', 'MAP'])
    axes[2].set_xticks(range(len(sample_indices)))
    axes[2].set_xticklabels(time_labels, rotation=45)
    axes[2].set_title('Feature Abnormality Heatmap (Red = Abnormal, Green = Normal)')
    plt.colorbar(im, ax=axes[2], label='Abnormality Score')
    
    # Plot 4: Alert Triggers
    axes[3].scatter(time[patient_data['should_alert']], 
                   patient_data['risk_score'][patient_data['should_alert']], 
                   c='red', s=100, marker='x', label='Alert Triggered', zorder=5)
    axes[3].plot(time, patient_data['risk_score'], linewidth=1, alpha=0.5, color='gray')
    axes[3].set_ylabel('Risk Score')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('When Were Alerts Triggered and Why?')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Add annotations for alerts
    for idx in patient_data[patient_data['should_alert']].index[:3]:  # First 3 alerts
        row = patient_data.loc[idx]
        t = row['time_seconds']
        r = row['risk_score']
        reason = row.get('alert_reason', 'High risk detected')
        if pd.notna(reason) and reason:
            axes[3].annotate(reason[:30] + '...', 
                           xy=(t, r), xytext=(t+100, r+1),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_vital_severity_simple(df):
    """Simplified vital severity for visualization"""
    severity = pd.Series(0.0, index=df.index)
    severity += np.where(df['heart_rate'] > 120, 3, 0)
    severity += np.where(df['spo2'] < 90, 3, 0)
    severity += np.where(df['sbp'] < 90, 3, 0)
    return severity


def calculate_hr_contribution(df):
    """How much HR contributes to risk"""
    contrib = pd.Series(0.0, index=df.index)
    contrib += np.where(df['heart_rate'] > 140, 3, 0)
    contrib += np.where((df['heart_rate'] > 120) & (df['heart_rate'] <= 140), 2, 0)
    contrib += np.where((df['heart_rate'] > 100) & (df['heart_rate'] <= 120), 1, 0)
    contrib += np.where(df['heart_rate'] < 50, 3, 0)
    return contrib


def calculate_spo2_contribution(df):
    """How much SpO2 contributes to risk"""
    contrib = pd.Series(0.0, index=df.index)
    contrib += np.where(df['spo2'] < 85, 3, 0)
    contrib += np.where((df['spo2'] >= 85) & (df['spo2'] < 90), 2, 0)
    contrib += np.where((df['spo2'] >= 90) & (df['spo2'] < 94), 1, 0)
    return contrib


def calculate_bp_contribution(df):
    """How much BP contributes to risk"""
    contrib = pd.Series(0.0, index=df.index)
    contrib += np.where(df['sbp'] < 80, 3, 0)
    contrib += np.where((df['sbp'] >= 80) & (df['sbp'] < 90), 2, 0)
    contrib += np.where(df['sbp'] > 180, 2, 0)
    return contrib


def calculate_feature_importance(df):
    """Calculate abnormality score for each feature"""
    features = []
    
    # HR abnormality
    hr_score = np.abs((df['heart_rate'] - 80) / 40)  # Normalize around 80 bpm
    features.append(hr_score.values)
    
    # SpO2 abnormality
    spo2_score = np.abs((df['spo2'] - 98) / 10)  # Normalize around 98%
    features.append(spo2_score.values)
    
    # SBP abnormality
    sbp_score = np.abs((df['sbp'] - 120) / 30)
    features.append(sbp_score.values)
    
    # DBP abnormality
    dbp_score = np.abs((df['dbp'] - 80) / 20)
    features.append(dbp_score.values)
    
    # Shock index
    if 'shock_index' in df.columns:
        shock_score = np.abs((df['shock_index'] - 0.7) / 0.3)
        features.append(shock_score.values)
    else:
        features.append(np.zeros(len(df)))
    
    # MAP
    if 'map' in df.columns:
        map_score = np.abs((df['map'] - 90) / 20)
        features.append(map_score.values)
    else:
        features.append(np.zeros(len(df)))
    
    return np.clip(np.array(features), 0, 2)


def main():
    print("=" * 60)
    print("GENERATING EXPLAINABILITY PLOTS")
    print("=" * 60)
    
    # Load predictions with risk scores
    print("\nLoading predictions...")
    df = pd.read_csv('data/processed/predictions_improved.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add should_alert column if missing
    if 'should_alert' not in df.columns:
        df['should_alert'] = (df['risk_level'].isin(['HIGH', 'CRITICAL'])) & (df['confidence'] > 0.6)
    
    print(f"✓ Loaded {len(df)} predictions")
    
    # Create output directory
    os.makedirs('plots/explainability', exist_ok=True)
    
    # Generate plots for key patients
    print("\nGenerating explainability plots...")
    
    for patient_id in [1, 3, 5]:  # Normal, deterioration, acute
        scenario = df[df['patient_id'] == patient_id]['scenario'].iloc[0]
        print(f"\nPatient {patient_id} ({scenario})...")
        
        plot_risk_breakdown(
            df,
            patient_id,
            save_path=f'plots/explainability/patient_{patient_id}_explainability.png'
        )
    
    print("\n" + "=" * 60)
    print("✓ EXPLAINABILITY PLOTS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - plots/explainability/patient_1_explainability.png")
    print("  - plots/explainability/patient_3_explainability.png")
    print("  - plots/explainability/patient_5_explainability.png")
    print("\nThese plots show:")
    print("  1. Overall risk score progression")
    print("  2. Which vitals contribute most to risk")
    print("  3. Feature abnormality heatmap")
    print("  4. When and why alerts were triggered")

if __name__ == "__main__":
    main()