import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class AmbulanceVitalsGenerator:
    """
    Generate realistic synthetic patient vital signs for ambulance transport.
    Includes normal patterns, distress scenarios, and sensor artifacts.
    """
    
    def __init__(self, duration_minutes=30, sampling_rate=1):
        """
        Args:
            duration_minutes: Length of transport simulation
            sampling_rate: Samples per second (1 Hz = 1 sample/second)
        """
        self.duration_minutes = duration_minutes
        self.sampling_rate = sampling_rate
        self.n_samples = duration_minutes * 60 * sampling_rate
        self.time = np.arange(0, self.n_samples) / sampling_rate
        
    def _add_baseline_noise(self, signal, noise_level=0.5):
        """Add realistic physiological variation"""
        return signal + np.random.normal(0, noise_level, len(signal))
    
    def _add_respiratory_variation(self, signal, resp_rate=16, amplitude=2):
        """Add respiratory variation to heart rate and SpO2"""
        resp_component = amplitude * np.sin(2 * np.pi * (resp_rate / 60) * self.time)
        return signal + resp_component
    
    def _add_motion_artifacts(self, signal, artifact_prob=0.05, artifact_magnitude=10):
        """Simulate motion artifacts from ambulance movement and patient shifts"""
        artifacts = np.zeros(len(signal))
        artifact_mask = np.random.random(len(signal)) < artifact_prob
        artifacts[artifact_mask] = np.random.normal(0, artifact_magnitude, np.sum(artifact_mask))
        return signal + artifacts
    
    def _add_sensor_dropouts(self, signal, dropout_prob=0.02):
        """Simulate sensor connection issues"""
        dropout_mask = np.random.random(len(signal)) < dropout_prob
        signal_copy = signal.copy()
        signal_copy[dropout_mask] = np.nan
        return signal_copy
    
    def _add_bump_spikes(self, signal, bump_times, spike_magnitude=15):
        """Simulate sudden spikes from road bumps"""
        signal_copy = signal.copy()
        for bump_time in bump_times:
            bump_idx = int(bump_time * self.sampling_rate)
            if bump_idx < len(signal_copy):
                # Short spike with exponential decay
                spike_duration = int(2 * self.sampling_rate)  # 2 second spike
                end_idx = min(bump_idx + spike_duration, len(signal_copy))
                decay = np.exp(-np.arange(end_idx - bump_idx) / self.sampling_rate)
                signal_copy[bump_idx:end_idx] += spike_magnitude * decay
        return signal_copy
    
    def generate_normal_transport(self, patient_id=1, age=45, add_artifacts=True):
        """Generate vitals for normal, stable transport"""
        
        # Baseline vitals (age-adjusted)
        base_hr = 75 - (age - 45) * 0.1  # Slight age adjustment
        base_spo2 = 98
        base_sbp = 120 + (age - 45) * 0.3
        base_dbp = 80 + (age - 45) * 0.2
        
        # Generate base signals
        hr = np.ones(self.n_samples) * base_hr
        spo2 = np.ones(self.n_samples) * base_spo2
        sbp = np.ones(self.n_samples) * base_sbp
        dbp = np.ones(self.n_samples) * base_dbp
        
        # Add physiological variations
        hr = self._add_respiratory_variation(hr, resp_rate=16, amplitude=3)
        hr = self._add_baseline_noise(hr, noise_level=2)
        
        spo2 = self._add_baseline_noise(spo2, noise_level=0.5)
        
        sbp = self._add_baseline_noise(sbp, noise_level=3)
        dbp = self._add_baseline_noise(dbp, noise_level=2)
        
        # Generate motion/vibration signal (vehicle movement)
        motion = np.random.normal(0.5, 0.2, self.n_samples)  # Low baseline
        
        if add_artifacts:
            # Simulate road bumps at random times
            n_bumps = np.random.randint(5, 15)
            bump_times = np.random.uniform(60, self.duration_minutes * 60 - 60, n_bumps)
            
            hr = self._add_bump_spikes(hr, bump_times, spike_magnitude=10)
            spo2 = self._add_bump_spikes(spo2, bump_times, spike_magnitude=-3)
            
            # Add motion artifacts
            hr = self._add_motion_artifacts(hr, artifact_prob=0.03, artifact_magnitude=5)
            spo2 = self._add_motion_artifacts(spo2, artifact_prob=0.03, artifact_magnitude=2)
            
            # Sensor dropouts
            spo2 = self._add_sensor_dropouts(spo2, dropout_prob=0.01)
            
            # Increase motion during bumps
            for bump_time in bump_times:
                bump_idx = int(bump_time * self.sampling_rate)
                if bump_idx < len(motion):
                    motion[bump_idx:min(bump_idx + 3 * self.sampling_rate, len(motion))] += 2.0
        
        # Clip to realistic ranges
        hr = np.clip(hr, 50, 120)
        spo2 = np.clip(spo2, 85, 100)
        sbp = np.clip(sbp, 90, 160)
        dbp = np.clip(dbp, 60, 100)
        motion = np.clip(motion, 0, 5)
        
        return self._create_dataframe(patient_id, hr, spo2, sbp, dbp, motion, 
                                       scenario="normal_transport")
    
    def generate_gradual_deterioration(self, patient_id=2, age=65, deterioration_type="shock"):
        """
        Generate vitals showing gradual deterioration (e.g., hemorrhagic shock, sepsis)
        """
        base_hr = 80
        base_spo2 = 97
        base_sbp = 130
        base_dbp = 85
        
        # Initialize arrays
        hr = np.ones(self.n_samples) * base_hr
        spo2 = np.ones(self.n_samples) * base_spo2
        sbp = np.ones(self.n_samples) * base_sbp
        dbp = np.ones(self.n_samples) * base_dbp
        
        if deterioration_type == "shock":
            # Gradual decrease in BP, increase in HR, decrease in SpO2
            deterioration_curve = np.linspace(0, 1, self.n_samples) ** 1.5
            
            hr += deterioration_curve * 35  # HR increases to ~115
            spo2 -= deterioration_curve * 8  # SpO2 drops to ~89
            sbp -= deterioration_curve * 40  # SBP drops to ~90
            dbp -= deterioration_curve * 25  # DBP drops to ~60
            
        elif deterioration_type == "respiratory":
            # Gradual hypoxia
            deterioration_curve = np.linspace(0, 1, self.n_samples) ** 2
            
            hr += deterioration_curve * 25
            spo2 -= deterioration_curve * 12  # Significant SpO2 drop
            sbp += deterioration_curve * 15  # Slight BP increase (compensation)
        
        # Add physiological variations
        hr = self._add_respiratory_variation(hr, resp_rate=22, amplitude=4)  # Faster breathing
        hr = self._add_baseline_noise(hr, noise_level=3)
        spo2 = self._add_baseline_noise(spo2, noise_level=1)
        sbp = self._add_baseline_noise(sbp, noise_level=4)
        dbp = self._add_baseline_noise(dbp, noise_level=3)
        
        # Motion signal - patient restlessness increases
        motion = 0.5 + 0.5 * deterioration_curve + np.random.normal(0, 0.3, self.n_samples)
        
        # Add artifacts
        n_bumps = np.random.randint(7, 12)
        bump_times = np.random.uniform(60, self.duration_minutes * 60 - 60, n_bumps)
        hr = self._add_bump_spikes(hr, bump_times, spike_magnitude=12)
        spo2 = self._add_sensor_dropouts(spo2, dropout_prob=0.02)
        
        # Clip values
        hr = np.clip(hr, 60, 140)
        spo2 = np.clip(spo2, 75, 100)
        sbp = np.clip(sbp, 70, 180)
        dbp = np.clip(dbp, 45, 110)
        motion = np.clip(motion, 0, 5)
        
        return self._create_dataframe(patient_id, hr, spo2, sbp, dbp, motion,
                                       scenario=f"deterioration_{deterioration_type}")
    
    def generate_acute_event(self, patient_id=3, age=58, event_type="cardiac"):
        """
        Generate vitals with sudden acute event (cardiac arrest, PE, arrhythmia)
        """
        base_hr = 78
        base_spo2 = 96
        base_sbp = 125
        base_dbp = 82
        
        # Start normal
        hr = np.ones(self.n_samples) * base_hr
        spo2 = np.ones(self.n_samples) * base_spo2
        sbp = np.ones(self.n_samples) * base_sbp
        dbp = np.ones(self.n_samples) * base_dbp
        
        # Event occurs randomly between 5-20 minutes
        event_time = np.random.uniform(5 * 60, 20 * 60)
        event_idx = int(event_time * self.sampling_rate)
        
        if event_type == "cardiac":
            # Sudden tachycardia or bradycardia, BP drop, SpO2 drop
            hr[event_idx:] = 140 + np.random.normal(0, 10, len(hr[event_idx:]))  # Sudden tachycardia
            spo2[event_idx:] = 88 - np.linspace(0, 5, len(spo2[event_idx:]))
            sbp[event_idx:] = 95 - np.linspace(0, 15, len(sbp[event_idx:]))
            dbp[event_idx:] = 60 - np.linspace(0, 10, len(dbp[event_idx:]))
            
        elif event_type == "arrhythmia":
            # Irregular heart rate with high variability
            hr[event_idx:] = 110 + 20 * np.sin(2 * np.pi * 0.1 * self.time[event_idx:])
            hr[event_idx:] += np.random.normal(0, 15, len(hr[event_idx:]))
        
        # Add normal variations
        hr = self._add_respiratory_variation(hr, resp_rate=18, amplitude=3)
        hr = self._add_baseline_noise(hr, noise_level=2)
        spo2 = self._add_baseline_noise(spo2, noise_level=0.8)
        sbp = self._add_baseline_noise(sbp, noise_level=3)
        dbp = self._add_baseline_noise(dbp, noise_level=2)
        
        # Motion - sudden increase at event
        motion = np.random.normal(0.5, 0.2, self.n_samples)
        motion[event_idx:] += 1.5  # Patient distress
        
        # Add artifacts
        n_bumps = np.random.randint(6, 10)
        bump_times = np.random.uniform(60, self.duration_minutes * 60 - 60, n_bumps)
        hr = self._add_bump_spikes(hr, bump_times, spike_magnitude=15)
        spo2 = self._add_sensor_dropouts(spo2, dropout_prob=0.03)
        
        # Clip values
        hr = np.clip(hr, 40, 180)
        spo2 = np.clip(spo2, 70, 100)
        sbp = np.clip(sbp, 60, 200)
        dbp = np.clip(dbp, 40, 120)
        motion = np.clip(motion, 0, 5)
        
        return self._create_dataframe(patient_id, hr, spo2, sbp, dbp, motion,
                                       scenario=f"acute_{event_type}", 
                                       event_time=event_time)
    
    def _create_dataframe(self, patient_id, hr, spo2, sbp, dbp, motion, 
                          scenario, event_time=None):
        """Create pandas DataFrame with proper timestamps"""
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i/self.sampling_rate) 
                     for i in range(self.n_samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'patient_id': patient_id,
            'heart_rate': hr,
            'spo2': spo2,
            'sbp': sbp,
            'dbp': dbp,
            'motion_vibration': motion,
            'scenario': scenario
        })
        
        if event_time:
            df['event_time_sec'] = event_time
        
        return df


# Generate datasets for multiple patients
def generate_all_patients(duration_minutes=30):
    """Generate complete dataset with multiple patient scenarios"""
    
    generator = AmbulanceVitalsGenerator(duration_minutes=duration_minutes)
    
    patients_data = []
    
    # Patient 1: Normal transport
    print("Generating Patient 1: Normal transport...")
    df1 = generator.generate_normal_transport(patient_id=1, age=45)
    patients_data.append(df1)
    
    # Patient 2: Normal transport (different age)
    print("Generating Patient 2: Normal transport (elderly)...")
    df2 = generator.generate_normal_transport(patient_id=2, age=72)
    patients_data.append(df2)
    
    # Patient 3: Gradual deterioration (shock)
    print("Generating Patient 3: Hemorrhagic shock...")
    df3 = generator.generate_gradual_deterioration(patient_id=3, age=55, 
                                                   deterioration_type="shock")
    patients_data.append(df3)
    
    # Patient 4: Gradual deterioration (respiratory)
    print("Generating Patient 4: Respiratory distress...")
    df4 = generator.generate_gradual_deterioration(patient_id=4, age=68, 
                                                   deterioration_type="respiratory")
    patients_data.append(df4)
    
    # Patient 5: Acute cardiac event
    print("Generating Patient 5: Acute cardiac event...")
    df5 = generator.generate_acute_event(patient_id=5, age=62, event_type="cardiac")
    patients_data.append(df5)
    
    # Patient 6: Arrhythmia
    print("Generating Patient 6: Arrhythmia...")
    df6 = generator.generate_acute_event(patient_id=6, age=58, event_type="arrhythmia")
    patients_data.append(df6)
    
    # Combine all patients
    all_data = pd.concat(patients_data, ignore_index=True)
    
    return all_data, patients_data


# Visualization function
def plot_patient_vitals(df, patient_id, save_path=None):
    """Plot all vitals for a single patient"""
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                    patient_data['timestamp'].iloc[0]).dt.total_seconds()
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.suptitle(f'Patient {patient_id} - {patient_data["scenario"].iloc[0]}', 
                 fontsize=14, fontweight='bold')
    
    # Heart Rate
    axes[0].plot(patient_data['time_seconds'], patient_data['heart_rate'], 
                 linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Heart Rate (bpm)')
    axes[0].axhline(y=60, color='g', linestyle='--', alpha=0.3, label='Normal range')
    axes[0].axhline(y=100, color='g', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # SpO2
    axes[1].plot(patient_data['time_seconds'], patient_data['spo2'], 
                 linewidth=0.5, alpha=0.7, color='blue')
    axes[1].set_ylabel('SpO₂ (%)')
    axes[1].axhline(y=95, color='g', linestyle='--', alpha=0.3, label='Normal > 95%')
    axes[1].axhline(y=90, color='orange', linestyle='--', alpha=0.3, label='Warning < 90%')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Blood Pressure
    axes[2].plot(patient_data['time_seconds'], patient_data['sbp'], 
                 linewidth=0.5, alpha=0.7, label='Systolic', color='red')
    axes[2].plot(patient_data['time_seconds'], patient_data['dbp'], 
                 linewidth=0.5, alpha=0.7, label='Diastolic', color='darkred')
    axes[2].set_ylabel('Blood Pressure (mmHg)')
    axes[2].axhline(y=120, color='g', linestyle='--', alpha=0.3)
    axes[2].axhline(y=80, color='g', linestyle='--', alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Motion/Vibration
    axes[3].plot(patient_data['time_seconds'], patient_data['motion_vibration'], 
                 linewidth=0.5, alpha=0.7, color='purple')
    axes[3].set_ylabel('Motion/Vibration')
    axes[3].grid(True, alpha=0.3)
    
    # MAP (Mean Arterial Pressure)
    patient_data['map'] = (patient_data['sbp'] + 2 * patient_data['dbp']) / 3
    axes[4].plot(patient_data['time_seconds'], patient_data['map'], 
                 linewidth=0.5, alpha=0.7, color='green')
    axes[4].set_ylabel('MAP (mmHg)')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].axhline(y=70, color='orange', linestyle='--', alpha=0.3, label='Critical < 70')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Generate data
    print("=" * 60)
    print("AMBULANCE VITALS DATA GENERATOR")
    print("=" * 60)
    
    all_data, individual_patients = generate_all_patients(duration_minutes=30)
    
    # Save to CSV
    all_data.to_csv('ambulance_vitals_data.csv', index=False)
    print(f"\n Saved combined data: {len(all_data)} records")
    
    # Save individual patient files
    for i, df in enumerate(individual_patients, 1):
        df.to_csv(f'patient_{i}_vitals.csv', index=False)
        print(f" Saved patient {i}: {df['scenario'].iloc[0]}")
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(all_data.groupby('scenario')[['heart_rate', 'spo2', 'sbp', 'dbp']].describe())
    
    # Plot example patients
    print("\nGenerating visualizations...")
    for patient_id in [1, 3, 5]:  # Normal, deterioration, acute
        plot_patient_vitals(all_data, patient_id, 
                           save_path=f'patient_{patient_id}_vitals_plot.png')
    
    print("\n✓ Data generation complete!")
    print(f"Total patients: {all_data['patient_id'].nunique()}")
    print(f"Total duration per patient: 30 minutes")
    print(f"Sampling rate: 1 Hz (1 sample/second)")