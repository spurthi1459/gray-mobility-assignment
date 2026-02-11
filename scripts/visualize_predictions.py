"""
Visualize anomaly detection and risk scoring results

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_patient_predictions(df, patient_id, save_path=None):
    """Plot anomaly detection and risk scores for a patient"""
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                    patient_data['timestamp'].iloc[0]).dt.total_seconds()
    
    scenario = patient_data['scenario'].iloc[0]
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 14))
    fig.suptitle(f'Patient {patient_id} - {scenario} - Anomaly Detection & Risk Scoring', 
                 fontsize=14, fontweight='bold')
    
    time = patient_data['time_seconds']
    
    # Plot 1: Heart Rate with anomalies highlighted
    axes[0].plot(time, patient_data['heart_rate'], linewidth=0.8, alpha=0.7, label='Heart Rate')
    
    # Highlight anomalies
    anomalies = patient_data['is_anomaly']
    if anomalies.any():
        axes[0].scatter(time[anomalies], patient_data['heart_rate'][anomalies],
                       c='red', s=10, alpha=0.6, label='Detected Anomalies', zorder=5)
    
    axes[0].axhline(y=120, color='orange', linestyle='--', alpha=0.3, label='Tachycardia')
    axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.3, label='Bradycardia')
    axes[0].set_ylabel('Heart Rate (bpm)')
    axes[0].set_title('Heart Rate with Anomaly Detection')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: SpO2 with anomalies
    axes[1].plot(time, patient_data['spo2'], linewidth=0.8, alpha=0.7, color='blue')
    if anomalies.any():
        axes[1].scatter(time[anomalies], patient_data['spo2'][anomalies],
                       c='red', s=10, alpha=0.6, zorder=5)
    axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.3, label='Critical < 90%')
    axes[1].axhline(y=94, color='orange', linestyle='--', alpha=0.3, label='Low < 94%')
    axes[1].set_ylabel('SpO₂ (%)')
    axes[1].set_title('SpO₂ with Anomaly Detection')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Blood Pressure with anomalies
    axes[2].plot(time, patient_data['sbp'], linewidth=0.8, alpha=0.7, 
                label='Systolic', color='darkred')
    axes[2].plot(time, patient_data['dbp'], linewidth=0.8, alpha=0.7, 
                label='Diastolic', color='red')
    if anomalies.any():
        axes[2].scatter(time[anomalies], patient_data['sbp'][anomalies],
                       c='black', s=10, alpha=0.5, zorder=5)
    axes[2].axhline(y=90, color='orange', linestyle='--', alpha=0.3, label='Hypotension')
    axes[2].set_ylabel('Blood Pressure (mmHg)')
    axes[2].set_title('Blood Pressure with Anomaly Detection')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Anomaly Score (continuous)
    axes[3].plot(time, patient_data['anomaly_score'], linewidth=0.8, color='purple')
    axes[3].fill_between(time, patient_data['anomaly_score'], alpha=0.3, color='purple')
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Anomaly Threshold')
    axes[3].set_ylabel('Anomaly Score')
    axes[3].set_title('Anomaly Score (lower = more anomalous)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Risk Score with levels
    axes[4].plot(time, patient_data['risk_score'], linewidth=1.2, color='darkgreen', label='Risk Score')
    axes[4].fill_between(time, patient_data['risk_score'], alpha=0.3, color='green')
    
    # Risk level zones
    axes[4].axhspan(0, 3, alpha=0.1, color='green', label='LOW')
    axes[4].axhspan(3, 6, alpha=0.1, color='yellow', label='MODERATE')
    axes[4].axhspan(6, 8, alpha=0.1, color='orange', label='HIGH')
    axes[4].axhspan(8, 10, alpha=0.1, color='red', label='CRITICAL')
    
    axes[4].set_ylabel('Risk Score (0-10)')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].set_title('Patient Risk Score')
    axes[4].legend(loc='upper right')
    axes[4].set_ylim(-0.5, 10.5)
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_risk_comparison(df, save_path=None):
    """Compare risk scores across all patients"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Risk Score Comparison - All Patients', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, patient_id in enumerate(sorted(df['patient_id'].unique())):
        patient_data = df[df['patient_id'] == patient_id].copy()
        patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                        patient_data['timestamp'].iloc[0]).dt.total_seconds()
        scenario = patient_data['scenario'].iloc[0]
        
        time = patient_data['time_seconds']
        
        # Plot risk score
        axes[idx].plot(time, patient_data['risk_score'], linewidth=1)
        axes[idx].fill_between(time, patient_data['risk_score'], alpha=0.3)
        
        # Risk zones
        axes[idx].axhspan(0, 3, alpha=0.1, color='green')
        axes[idx].axhspan(3, 6, alpha=0.1, color='yellow')
        axes[idx].axhspan(6, 8, alpha=0.1, color='orange')
        axes[idx].axhspan(8, 10, alpha=0.1, color='red')
        
        # Stats
        avg_risk = patient_data['risk_score'].mean()
        max_risk = patient_data['risk_score'].max()
        
        axes[idx].set_title(f'Patient {patient_id}\n{scenario}\nAvg: {avg_risk:.2f}, Max: {max_risk:.2f}')
        axes[idx].set_ylabel('Risk Score')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylim(-0.5, 10.5)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    print("=" * 60)
    print("VISUALIZING ANOMALY DETECTION RESULTS")
    print("=" * 60)
    
    # Load predictions
    print("\nLoading predictions...")
    df = pd.read_csv('data/processed/predictions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f" Loaded {len(df)} predictions")
    
    # Create output directory
    os.makedirs('plots/anomaly_detection', exist_ok=True)
    
    # Plot individual patients
    print("\n" + "=" * 60)
    print("GENERATING DETAILED PATIENT PLOTS")
    print("=" * 60)
    
    for patient_id in [1, 3, 5]:  # Normal, deterioration, acute
        print(f"\nPlotting Patient {patient_id}...")
        plot_patient_predictions(
            df,
            patient_id,
            save_path=f'plots/anomaly_detection/patient_{patient_id}_predictions.png'
        )
    
    # Comparison plot
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOT")
    print("=" * 60)
    
    print("\nCreating risk score comparison...")
    plot_risk_comparison(
        df,
        save_path='plots/anomaly_detection/risk_comparison_all_patients.png'
    )
    
    print("\n" + "=" * 60)
    print(" VISUALIZATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - plots/anomaly_detection/patient_1_predictions.png")
    print("  - plots/anomaly_detection/patient_3_predictions.png")
    print("  - plots/anomaly_detection/patient_5_predictions.png")
    print("  - plots/anomaly_detection/risk_comparison_all_patients.png")

if __name__ == "__main__":
    main()