"""
Run artifact detection on generated data
Run: python scripts\detect_artifacts.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.preprocessing.artifact_detector import (
    ArtifactDetector,
    plot_artifact_detection,
    plot_all_vitals_comparison
)

def main():
    print("=" * 60)
    print("ARTIFACT DETECTION")
    print("=" * 60)
    
    # Load generated data
    print("\nLoading data...")
    df = pd.read_csv('data/raw/ambulance_vitals_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Loaded {len(df)} records")
    
    # Initialize detector
    detector = ArtifactDetector(sampling_rate=1)
    
    # Detect artifacts
    print("\nDetecting artifacts...")
    df_with_artifacts = detector.detect_all_artifacts(df)
    
    # Clean signals
    print("Cleaning signals...")
    df_cleaned = detector.clean_signal(df_with_artifacts, method='interpolate')
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('plots/artifact_detection', exist_ok=True)
    
    # Save cleaned data
    df_cleaned.to_csv('data/processed/cleaned_vitals.csv', index=False)
    print("✓ Saved cleaned data to data/processed/cleaned_vitals.csv")
    
    # Generate summary for each patient
    print("\n" + "=" * 60)
    print("ARTIFACT SUMMARY BY PATIENT")
    print("=" * 60)
    
    for patient_id in df_cleaned['patient_id'].unique():
        summary = detector.get_artifact_summary(df_cleaned, patient_id)
        scenario = df_cleaned[df_cleaned['patient_id'] == patient_id]['scenario'].iloc[0]
        
        print(f"\nPatient {patient_id} ({scenario}):")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Range artifacts: {summary['range_artifacts']}")
        print(f"  Spike artifacts: {summary['spike_artifacts']}")
        print(f"  Dropout artifacts: {summary['dropout_artifacts']}")
        print(f"  Motion artifacts: {summary['motion_artifacts']}")
        print(f"  Total artifacts: {summary['total_artifacts']} ({summary['artifact_percentage']:.2f}%)")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING ARTIFACT DETECTION PLOTS")
    print("=" * 60)
    
    # Detailed plots for select patients
    for patient_id in [1, 3, 5]:  # Normal, deterioration, acute
        print(f"\nPlotting Patient {patient_id}...")
        
        # Heart rate artifact detection
        plot_artifact_detection(
            df_cleaned,
            patient_id,
            vital='heart_rate',
            save_path=f'plots/artifact_detection/patient_{patient_id}_hr_artifacts.png'
        )
        
        # All vitals comparison
        plot_all_vitals_comparison(
            df_cleaned,
            patient_id,
            save_path=f'plots/artifact_detection/patient_{patient_id}_all_vitals.png'
        )
    
    print("\n" + "=" * 60)
    print("✓ ARTIFACT DETECTION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/processed/cleaned_vitals.csv")
    print("  - plots/artifact_detection/*.png")
    print("\nNext steps:")
    print("  1. Review artifact detection plots")
    print("  2. Verify cleaning quality")
    print("  3. Move to Part 2: Anomaly Detection")

if __name__ == "__main__":
    main()