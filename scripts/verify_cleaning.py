"""
Verify artifact detection quality
Run: python scripts\verify_cleaning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import matplotlib.pyplot as plt

def verify_cleaning():
    print("=" * 60)
    print("VERIFYING ARTIFACT CLEANING QUALITY")
    print("=" * 60)
    
    # Load cleaned data
    df = pd.read_csv('data/processed/cleaned_vitals.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("\n1. Checking for remaining NaN values...")
    for vital in ['heart_rate_cleaned', 'spo2_cleaned', 'sbp_cleaned', 'dbp_cleaned']:
        nan_count = df[vital].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        print(f"   {vital}: {nan_count} NaN ({nan_pct:.2f}%)")
    
    print("\n2. Checking physiological validity...")
    issues = {
        'heart_rate_cleaned': (df['heart_rate_cleaned'] < 30) | (df['heart_rate_cleaned'] > 220),
        'spo2_cleaned': (df['spo2_cleaned'] < 60) | (df['spo2_cleaned'] > 100),
        'sbp_cleaned': (df['sbp_cleaned'] < 50) | (df['sbp_cleaned'] > 260),
        'dbp_cleaned': (df['dbp_cleaned'] < 30) | (df['dbp_cleaned'] > 160)
    }
    
    for vital, mask in issues.items():
        issue_count = mask.sum()
        print(f"   {vital}: {issue_count} potentially invalid values")
    
    print("\n3. Artifact removal effectiveness...")
    total_artifacts = df['artifact_any'].sum()
    total_samples = len(df)
    removal_rate = (total_artifacts / total_samples) * 100
    
    print(f"   Total artifacts detected: {total_artifacts}")
    print(f"   Total samples: {total_samples}")
    print(f"   Artifact rate: {removal_rate:.2f}%")
    
    print("\n4. Per-patient cleaning summary...")
    for patient_id in sorted(df['patient_id'].unique()):
        patient_df = df[df['patient_id'] == patient_id]
        scenario = patient_df['scenario'].iloc[0]
        artifacts = patient_df['artifact_any'].sum()
        pct = (artifacts / len(patient_df)) * 100
        
        # Check if medical signal preserved
        hr_before = patient_df['heart_rate'].mean()
        hr_after = patient_df['heart_rate_cleaned'].mean()
        hr_diff = abs(hr_before - hr_after)
        
        print(f"\n   Patient {patient_id} ({scenario}):")
        print(f"     Artifacts: {artifacts} ({pct:.1f}%)")
        print(f"     HR before: {hr_before:.1f} bpm")
        print(f"     HR after: {hr_after:.1f} bpm")
        print(f"     Difference: {hr_diff:.1f} bpm (should be small)")
        
        if hr_diff > 5:
            print(f"     ⚠ Warning: Large change in mean HR - check if medical signal preserved")
        else:
            print(f"     ✓ Medical signal preserved")
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION COMPLETE")
    print("=" * 60)
    
    # Create comparison plot
    print("\nGenerating comparison visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Patient 1 (normal)
    p1 = df[df['patient_id'] == 1].copy()
    p1['time'] = (p1['timestamp'] - p1['timestamp'].iloc[0]).dt.total_seconds()
    
    axes[0].plot(p1['time'], p1['heart_rate'], alpha=0.5, label='Before', linewidth=0.5)
    axes[0].plot(p1['time'], p1['heart_rate_cleaned'], label='After', linewidth=0.8)
    axes[0].set_title('Patient 1 (Normal) - Heart Rate Cleaning')
    axes[0].set_ylabel('Heart Rate (bpm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Patient 5 (acute)
    p5 = df[df['patient_id'] == 5].copy()
    p5['time'] = (p5['timestamp'] - p5['timestamp'].iloc[0]).dt.total_seconds()
    
    axes[1].plot(p5['time'], p5['heart_rate'], alpha=0.5, label='Before', linewidth=0.5)
    axes[1].plot(p5['time'], p5['heart_rate_cleaned'], label='After', linewidth=0.8)
    axes[1].set_title('Patient 5 (Acute Cardiac Event) - Heart Rate Cleaning')
    axes[1].set_ylabel('Heart Rate (bpm)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/artifact_detection/cleaning_verification.png', dpi=150)
    print("✓ Saved: plots/artifact_detection/cleaning_verification.png")
    plt.show()
    
    return True

if __name__ == "__main__":
    verify_cleaning()