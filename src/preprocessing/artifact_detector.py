"""
Artifact Detection for Ambulance Vitals
Detects and handles motion artifacts, sensor dropouts, and spurious spikes
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt

class ArtifactDetector:
    """
    Detect and handle artifacts in physiological signals
    
    Artifact types handled:
    1. Motion artifacts (sudden spikes from ambulance bumps)
    2. Sensor dropouts (missing/NaN values)
    3. Physiologically impossible values
    4. High-frequency noise
    """
    
    def __init__(self, sampling_rate=1):
        """
        Args:
            sampling_rate: Samples per second (default 1 Hz)
        """
        self.sampling_rate = sampling_rate
        
        # Define physiologically valid ranges
        self.valid_ranges = {
            'heart_rate': (40, 200),      # bpm
            'spo2': (70, 100),             # percentage
            'sbp': (60, 250),              # mmHg
            'dbp': (40, 150),              # mmHg
        }
        
        # Define maximum rate of change per second (for spike detection)
        self.max_rate_of_change = {
            'heart_rate': 30,   # bpm/sec
            'spo2': 5,          # %/sec
            'sbp': 40,          # mmHg/sec
            'dbp': 30,          # mmHg/sec
        }
    
    def detect_all_artifacts(self, df, patient_id=None):
        """
        Detect all types of artifacts in the data
        
        Returns:
            DataFrame with artifact flags and cleaned signals
        """
        if patient_id is not None:
            df = df[df['patient_id'] == patient_id].copy()
        else:
            df = df.copy()
        
        # Initialize artifact flags
        df['artifact_range'] = False
        df['artifact_spike'] = False
        df['artifact_dropout'] = False
        df['artifact_motion'] = False
        df['artifact_any'] = False
        
        vitals = ['heart_rate', 'spo2', 'sbp', 'dbp']
        
        for vital in vitals:
            # 1. Range-based artifacts (physiologically impossible)
            range_mask = self._detect_range_artifacts(df[vital], vital)
            df['artifact_range'] |= range_mask
            
            # 2. Spike artifacts (sudden large changes)
            spike_mask = self._detect_spike_artifacts(df[vital], vital)
            df['artifact_spike'] |= spike_mask
            
            # 3. Dropout artifacts (missing data)
            dropout_mask = df[vital].isna()
            df['artifact_dropout'] |= dropout_mask
        
        # 4. Motion-correlated artifacts
        motion_mask = self._detect_motion_artifacts(df)
        df['artifact_motion'] = motion_mask
        
        # Combine all artifact flags
        df['artifact_any'] = (df['artifact_range'] | 
                             df['artifact_spike'] | 
                             df['artifact_dropout'] |
                             df['artifact_motion'])
        
        return df
    
    def _detect_range_artifacts(self, signal_data, vital_name):
        """Detect values outside physiologically valid ranges"""
        min_val, max_val = self.valid_ranges[vital_name]
        return (signal_data < min_val) | (signal_data > max_val)
    
    def _detect_spike_artifacts(self, signal_data, vital_name):
        """
        Detect sudden spikes using rate of change analysis
        Spikes are rapid changes that reverse quickly (bump artifacts)
        """
        # Calculate rate of change (derivative)
        rate_of_change = np.abs(signal_data.diff() * self.sampling_rate)
        
        # Threshold based on max physiological change rate
        max_change = self.max_rate_of_change[vital_name]
        
        # Initial spike detection
        spike_mask = rate_of_change > max_change
        
        # Refine: true spikes reverse direction quickly
        # Check if signal returns to baseline within 3 seconds
        refined_mask = spike_mask.copy()
        
        for i in np.where(spike_mask)[0]:
            if i < len(signal_data) - 3:
                # Check if signal reverses within 3 samples
                window = signal_data.iloc[i:i+4]
                if len(window) >= 4:
                    # If spike goes up then down (or vice versa), it's likely artifact
                    changes = window.diff().dropna()
                    if len(changes) >= 2:
                        sign_changes = (changes > 0).astype(int).diff().abs().sum()
                        if sign_changes >= 1:  # Direction reversal
                            refined_mask.iloc[i:i+3] = True
        
        return refined_mask.fillna(False)
    
    def _detect_motion_artifacts(self, df):
        """
        Detect artifacts correlated with high motion/vibration
        Motion > 2.0 often indicates bumps that affect sensors
        """
        if 'motion_vibration' not in df.columns:
            return pd.Series(False, index=df.index)
        
        # High motion threshold
        motion_threshold = 2.0
        high_motion = df['motion_vibration'] > motion_threshold
        
        # Extend motion artifact flag to ±2 seconds around high motion
        motion_mask = high_motion.copy()
        
        for i in np.where(high_motion)[0]:
            start = max(0, i - 2)
            end = min(len(df), i + 3)
            motion_mask.iloc[start:end] = True
        
        return motion_mask
    
    def clean_signal(self, df, patient_id=None, method='interpolate'):
        """
        Clean signals by handling detected artifacts
        
        Args:
            df: DataFrame with artifact flags
            patient_id: Optional patient ID to filter
            method: 'interpolate', 'median_filter', or 'remove'
        
        Returns:
            DataFrame with cleaned signals
        """
        if patient_id is not None:
            df = df[df['patient_id'] == patient_id].copy()
        else:
            df = df.copy()
        
        vitals = ['heart_rate', 'spo2', 'sbp', 'dbp']
        
        for vital in vitals:
            # Create cleaned column
            cleaned_col = f'{vital}_cleaned'
            df[cleaned_col] = df[vital].copy()
            
            if method == 'interpolate':
                # Replace artifacts with NaN, then interpolate
                df.loc[df['artifact_any'], cleaned_col] = np.nan
                df[cleaned_col] = df[cleaned_col].interpolate(method='linear', limit=5)
                
                # If still NaN at edges, forward/backward fill
                df[cleaned_col] = df[cleaned_col].fillna(method='ffill').fillna(method='bfill')
            
            elif method == 'median_filter':
                # Apply median filter to smooth artifacts
                window_size = 5
                df[cleaned_col] = signal.medfilt(df[vital].fillna(method='ffill'), 
                                                kernel_size=window_size)
            
            elif method == 'remove':
                # Simply mark artifacts as NaN (for manual review)
                df.loc[df['artifact_any'], cleaned_col] = np.nan
        
        return df
    
    def get_artifact_summary(self, df, patient_id=None):
        """
        Get summary statistics of detected artifacts
        """
        if patient_id is not None:
            df = df[df['patient_id'] == patient_id].copy()
        
        total_samples = len(df)
        
        summary = {
            'total_samples': total_samples,
            'range_artifacts': df['artifact_range'].sum(),
            'spike_artifacts': df['artifact_spike'].sum(),
            'dropout_artifacts': df['artifact_dropout'].sum(),
            'motion_artifacts': df['artifact_motion'].sum(),
            'total_artifacts': df['artifact_any'].sum(),
            'artifact_percentage': (df['artifact_any'].sum() / total_samples) * 100
        }
        
        return summary


def plot_artifact_detection(df, patient_id, vital='heart_rate', save_path=None):
    """
    Plot before/after comparison showing artifact detection and cleaning
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                    patient_data['timestamp'].iloc[0]).dt.total_seconds()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'Patient {patient_id} - {vital.upper()} Artifact Detection', 
                 fontsize=14, fontweight='bold')
    
    time = patient_data['time_seconds']
    
    # Plot 1: Original signal with artifacts highlighted
    axes[0].plot(time, patient_data[vital], linewidth=0.5, alpha=0.7, label='Original')
    
    # Highlight different artifact types
    artifact_mask = patient_data['artifact_any']
    if artifact_mask.any():
        axes[0].scatter(time[artifact_mask], 
                       patient_data[vital][artifact_mask],
                       c='red', s=20, alpha=0.5, label='Artifacts', zorder=5)
    
    axes[0].set_ylabel(f'{vital.replace("_", " ").title()}')
    axes[0].set_title('Original Signal with Detected Artifacts')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Artifact flags
    axes[1].fill_between(time, 0, patient_data['artifact_range'].astype(int), 
                         alpha=0.3, label='Range', color='orange')
    axes[1].fill_between(time, 0, patient_data['artifact_spike'].astype(int), 
                         alpha=0.3, label='Spike', color='red')
    axes[1].fill_between(time, 0, patient_data['artifact_motion'].astype(int), 
                         alpha=0.3, label='Motion', color='purple')
    axes[1].set_ylabel('Artifact Type')
    axes[1].set_title('Artifact Detection Flags')
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Before vs After cleaning
    cleaned_col = f'{vital}_cleaned'
    axes[2].plot(time, patient_data[vital], linewidth=0.5, alpha=0.5, 
                label='Before (with artifacts)', color='gray')
    axes[2].plot(time, patient_data[cleaned_col], linewidth=0.8, 
                label='After (cleaned)', color='blue')
    axes[2].set_ylabel(f'{vital.replace("_", " ").title()}')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Before vs After Artifact Removal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_all_vitals_comparison(df, patient_id, save_path=None):
    """
    Compare all vitals before and after artifact removal
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    patient_data['time_seconds'] = (patient_data['timestamp'] - 
                                    patient_data['timestamp'].iloc[0]).dt.total_seconds()
    
    vitals = ['heart_rate', 'spo2', 'sbp', 'dbp']
    vital_labels = ['Heart Rate (bpm)', 'SpO₂ (%)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)']
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f'Patient {patient_id} - Artifact Detection Results', 
                 fontsize=14, fontweight='bold')
    
    time = patient_data['time_seconds']
    
    for idx, (vital, label) in enumerate(zip(vitals, vital_labels)):
        # Left column: Before
        axes[idx, 0].plot(time, patient_data[vital], linewidth=0.5, alpha=0.7)
        artifact_mask = patient_data['artifact_any']
        if artifact_mask.any():
            axes[idx, 0].scatter(time[artifact_mask], 
                               patient_data[vital][artifact_mask],
                               c='red', s=10, alpha=0.5, zorder=5)
        axes[idx, 0].set_ylabel(label)
        axes[idx, 0].set_title(f'{label} - Before (with artifacts)')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Right column: After
        cleaned_col = f'{vital}_cleaned'
        axes[idx, 1].plot(time, patient_data[cleaned_col], 
                         linewidth=0.5, alpha=0.7, color='blue')
        axes[idx, 1].set_ylabel(label)
        axes[idx, 1].set_title(f'{label} - After (cleaned)')
        axes[idx, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()