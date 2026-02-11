"""
Anomaly Detection Model for Ambulance Vitals
Detects early warning signals beyond simple threshold breaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import joblib
import warnings
warnings.filterwarnings('ignore')

class VitalsAnomalyDetector:
    """
    Multi-method anomaly detector for patient vitals
    Combines statistical methods and machine learning
    """
    
    def __init__(self, window_size=60, contamination=0.1):
        """
        Args:
            window_size: Sliding window size in seconds for feature extraction
            contamination: Expected proportion of anomalies (for Isolation Forest)
        """
        self.window_size = window_size
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = None
        self.trained = False
        
    def extract_features(self, df, patient_id=None):
        """
        Extract time-series features from vitals for anomaly detection
        
        Features include:
        - Rolling statistics (mean, std, min, max)
        - Trends (slopes)
        - Rate of change
        - Vital combinations (e.g., Shock Index)
        """
        if patient_id is not None:
            df = df[df['patient_id'] == patient_id].copy()
        else:
            df = df.copy()
        
        # Use cleaned vitals if available
        vitals = ['heart_rate', 'spo2', 'sbp', 'dbp']
        
        # Check if cleaned versions exist
        for vital in vitals:
            cleaned_col = f'{vital}_cleaned'
            if cleaned_col in df.columns:
                df[vital] = df[cleaned_col]
        
        features_list = []
        
        # Rolling window features
        for vital in vitals:
            # Rolling mean
            df[f'{vital}_mean_{self.window_size}s'] = (
                df[vital].rolling(window=self.window_size, min_periods=1).mean()
            )
            
            # Rolling std (variability)
            df[f'{vital}_std_{self.window_size}s'] = (
                df[vital].rolling(window=self.window_size, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{vital}_min_{self.window_size}s'] = (
                df[vital].rolling(window=self.window_size, min_periods=1).min()
            )
            df[f'{vital}_max_{self.window_size}s'] = (
                df[vital].rolling(window=self.window_size, min_periods=1).max()
            )
            
            # Rate of change (trend)
            df[f'{vital}_roc'] = df[vital].diff()
            
            # Z-score (deviation from normal)
            df[f'{vital}_zscore'] = zscore(df[vital].fillna(method='ffill'), nan_policy='omit')
        
        # Derived features (medical insights)
        
        # 1. Shock Index (HR / SBP) - indicator of shock
        # Normal: 0.5-0.7, Shock: > 1.0
        df['shock_index'] = df['heart_rate'] / df['sbp'].replace(0, np.nan)
        
        # 2. Mean Arterial Pressure (MAP)
        # Critical if < 70 mmHg
        df['map'] = (df['sbp'] + 2 * df['dbp']) / 3
        
        # 3. Pulse Pressure (SBP - DBP)
        # Narrow pulse pressure can indicate shock
        df['pulse_pressure'] = df['sbp'] - df['dbp']
        
        # 4. HR-SpO2 inverse correlation (high HR, low SpO2 = distress)
        df['hr_spo2_ratio'] = df['heart_rate'] / df['spo2'].replace(0, np.nan)
        
        # Feature columns for ML model
        feature_cols = [
            # Rolling statistics
            'heart_rate_mean_60s', 'heart_rate_std_60s',
            'spo2_mean_60s', 'spo2_std_60s',
            'sbp_mean_60s', 'sbp_std_60s',
            'dbp_mean_60s', 'dbp_std_60s',
            
            # Rate of change
            'heart_rate_roc', 'spo2_roc', 'sbp_roc', 'dbp_roc',
            
            # Z-scores
            'heart_rate_zscore', 'spo2_zscore', 'sbp_zscore', 'dbp_zscore',
            
            # Derived medical features
            'shock_index', 'map', 'pulse_pressure', 'hr_spo2_ratio'
        ]
        
        return df, feature_cols
    
    def train(self, df_normal):
        """
        Train the anomaly detector on NORMAL transport data only
        
        Args:
            df_normal: DataFrame containing only normal/stable transport data
        """
        print("Training anomaly detector on normal data...")
        
        # Extract features
        df_features, feature_cols = self.extract_features(df_normal)
        
        # Get feature matrix
        X = df_features[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.iso_forest.fit(X_scaled)
        self.feature_cols = feature_cols
        self.trained = True
        
        print(f" Training complete on {len(X)} samples")
        print(f" Features used: {len(feature_cols)}")
        
        return self
    
    def predict(self, df):
        """
        Predict anomalies on new data
        
        Returns:
            DataFrame with anomaly scores and predictions
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        df_features, _ = self.extract_features(df)
        
        # Get feature matrix
        X = df_features[self.feature_cols].copy()
        
        # CRITICAL FIX: Fill NaN values before scaling
        # This happens with single data points that lack historical context
        # Use multiple strategies to ensure no NaN remains
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Double-check: replace any remaining NaN with 0
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.iso_forest.predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = self.iso_forest.score_samples(X_scaled)
        
        # Add to dataframe
        df_features['anomaly_prediction'] = predictions
        df_features['anomaly_score'] = anomaly_scores
        df_features['is_anomaly'] = (predictions == -1)
        
        return df_features
    
    def detect_statistical_anomalies(self, df):
        """
        Simple statistical anomaly detection (threshold-based)
        This runs WITHOUT training and serves as a baseline
        """
        df_features, _ = self.extract_features(df)
        
        # Define clinical thresholds for emergency
        emergency_conditions = (
            (df_features['heart_rate'] > 120) |  # Tachycardia
            (df_features['heart_rate'] < 50) |   # Bradycardia
            (df_features['spo2'] < 90) |         # Hypoxia
            (df_features['sbp'] < 90) |          # Hypotension
            (df_features['sbp'] > 180) |         # Hypertension crisis
            (df_features['shock_index'] > 1.0) | # Shock
            (df_features['map'] < 70)            # Critical MAP
        )
        
        df_features['statistical_anomaly'] = emergency_conditions
        
        return df_features
    
    def save_model(self, filepath='models/anomaly_detector_v1.pkl'):
        """Save trained model"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'iso_forest': self.iso_forest,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'window_size': self.window_size,
            'contamination': self.contamination
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/anomaly_detector_v1.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.iso_forest = model_data['iso_forest']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.window_size = model_data['window_size']
        self.contamination = model_data['contamination']
        self.trained = True
        
        print(f" Model loaded from {filepath}")
        return self