"""
Generate synthetic ambulance vitals data

"""

import sys
import os

# Add src to path (Windows compatible)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.vitals_generator import (
    generate_all_patients, 
    plot_patient_vitals
)
import json

def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC AMBULANCE VITALS DATA")
    print("=" * 60)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('plots/patient_vitals', exist_ok=True)
    
    # Generate data
    print("\nGenerating patient data...")
    all_data, individual_patients = generate_all_patients(duration_minutes=30)
    
    # Save combined data
    all_data.to_csv('data/raw/ambulance_vitals_data.csv', index=False)
    print(f"\n Saved combined data: {len(all_data)} records")
    
    # Save individual patient files
    print("\nSaving individual patient files...")
    for i, df in enumerate(individual_patients, 1):
        df.to_csv(f'data/raw/patient_{i}_vitals.csv', index=False)
        scenario = df['scenario'].iloc[0]
        print(f"  Patient {i}: {scenario}")
    
    # Save metadata
    metadata = {
        "generation_date": str(all_data['timestamp'].iloc[0]),
        "duration_minutes": 30,
        "sampling_rate_hz": 1,
        "num_patients": len(individual_patients),
        "total_records": len(all_data),
        "scenarios": all_data.groupby('scenario').size().to_dict()
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\n Saved metadata.json")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Patients: {all_data['patient_id'].nunique()}")
    print(f"Total Records: {len(all_data)}")
    print(f"Duration per patient: 30 minutes")
    print(f"Sampling rate: 1 Hz")
    print("\nScenarios:")
    for scenario, count in all_data.groupby('scenario').size().items():
        print(f"  - {scenario}: {count} records")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    for patient_id in range(1, 7):
        print(f"Plotting Patient {patient_id}...")
        plot_patient_vitals(
            all_data, 
            patient_id,
            save_path=f'plots/patient_vitals/patient_{patient_id}.png'
        )
    print("\n" + "=" * 60)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/raw/ambulance_vitals_data.csv")
    print("  - data/raw/patient_1_vitals.csv ... patient_6_vitals.csv")
    print("  - data/metadata.json")
    print("  - plots/patient_vitals/patient_1.png ... patient_6.png")

if __name__ == "__main__":
    main()