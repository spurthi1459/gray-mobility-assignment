"""
Evaluation Metrics for Anomaly Detection System
Calculate precision, recall, false alert rate, and latency
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyEvaluator:
    """Evaluate anomaly detection and alert quality"""
    
    def __init__(self):
        self.results = {}
    
    def create_ground_truth(self, df):
        """
        Create ground truth labels based on scenario type
        
        Normal transport = 0 (no anomaly)
        Deterioration/Acute = 1 (anomaly)
        """
        df = df.copy()
        
        # Ground truth: abnormal scenarios should be labeled as anomalies
        df['ground_truth'] = (
            df['scenario'].str.contains('deterioration|acute')
        ).astype(int)
        
        # For deterioration scenarios, only later stages are true anomalies
        # First 5 minutes are often still stable
        for patient_id in df['patient_id'].unique():
            mask = (df['patient_id'] == patient_id) & (df['scenario'].str.contains('deterioration'))
            if mask.any():
                patient_data = df[mask]
                total_samples = len(patient_data)
                # Mark first 300 seconds (5 min) as normal even in deterioration
                first_300 = int(total_samples * 300 / 1800)  # 5 min out of 30 min
                indices = patient_data.index[:first_300]
                df.loc[indices, 'ground_truth'] = 0
        
        # For acute events, only post-event is true anomaly
        for patient_id in df['patient_id'].unique():
            mask = (df['patient_id'] == patient_id) & (df['scenario'].str.contains('acute'))
            if mask.any() and 'event_time_sec' in df.columns:
                patient_data = df[mask]
                event_time = patient_data['event_time_sec'].iloc[0]
                
                # Create time column
                patient_data_copy = patient_data.copy()
                patient_data_copy['time_sec'] = (
                    (patient_data_copy['timestamp'] - patient_data_copy['timestamp'].iloc[0])
                    .dt.total_seconds()
                )
                
                # Only post-event is anomaly
                pre_event_mask = patient_data_copy['time_sec'] < event_time
                df.loc[patient_data_copy[pre_event_mask].index, 'ground_truth'] = 0
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate precision, recall, F1, false positive rate"""
        
        # Create ground truth if not exists
        if 'ground_truth' not in df.columns:
            df = self.create_ground_truth(df)
        
        y_true = df['ground_truth'].values
        y_pred = df['is_anomaly'].values
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # False positive rate (false alarms)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False negative rate (missed detections - DANGEROUS!)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(df)
        }
        
        return metrics
    
    def calculate_alert_latency(self, df):
        """
        Calculate how quickly alerts trigger after true event onset
        
        Alert latency = time from event start to first alert
        Lower is better (early detection)
        """
        latencies = {}
        
        df = df.copy()
        if 'ground_truth' not in df.columns:
            df = self.create_ground_truth(df)
        
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id].copy()
            scenario = patient_data['scenario'].iloc[0]
            
            # Only calculate for abnormal scenarios
            if 'deterioration' not in scenario and 'acute' not in scenario:
                continue
            
            # Find first true anomaly
            true_anomaly_indices = patient_data[patient_data['ground_truth'] == 1].index
            if len(true_anomaly_indices) == 0:
                continue
            
            first_true_anomaly_idx = true_anomaly_indices[0]
            
            # Find first alert
            alert_indices = patient_data[patient_data['is_anomaly'] == True].index
            if len(alert_indices) == 0:
                latencies[patient_id] = {
                    'scenario': scenario,
                    'latency_seconds': None,
                    'status': 'MISSED - No alert triggered'
                }
                continue
            
            first_alert_idx = alert_indices[0]
            
            # Calculate time difference
            time_true = patient_data.loc[first_true_anomaly_idx, 'timestamp']
            time_alert = patient_data.loc[first_alert_idx, 'timestamp']
            
            latency = (time_alert - time_true).total_seconds()
            
            if latency < 0:
                status = f'EARLY - Alert {abs(latency):.0f}s before event'
            elif latency == 0:
                status = 'IMMEDIATE - Alert at event start'
            else:
                status = f'DELAYED - Alert {latency:.0f}s after event'
            
            latencies[patient_id] = {
                'scenario': scenario,
                'latency_seconds': latency,
                'status': status
            }
        
        return latencies
    
    def evaluate_by_scenario(self, df):
        """Calculate metrics separately for each scenario type"""
        
        if 'ground_truth' not in df.columns:
            df = self.create_ground_truth(df)
        
        scenario_metrics = {}
        
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            
            y_true = scenario_data['ground_truth'].values
            y_pred = scenario_data['is_anomaly'].values
            
            if len(np.unique(y_true)) < 2:
                # If all same class, skip some metrics
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = 0 if y_true[0] == 1 else np.nan
            else:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            scenario_metrics[scenario] = {
                'precision': precision,
                'recall': recall,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        return scenario_metrics
    
    def plot_confusion_matrix(self, df, save_path=None):
        """Plot confusion matrix"""
        
        if 'ground_truth' not in df.columns:
            df = self.create_ground_truth(df)
        
        y_true = df['ground_truth'].values
        y_pred = df['is_anomaly'].values
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Anomaly Detection')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def analyze_failure_cases(self, df, n_cases=3):
        """
        Identify and analyze failure cases
        
        Returns:
            - False Positives (normal labeled as anomaly)
            - False Negatives (anomaly missed)
        """
        if 'ground_truth' not in df.columns:
            df = self.create_ground_truth(df)
        
        # False Positives
        fp_data = df[(df['ground_truth'] == 0) & (df['is_anomaly'] == True)]
        
        # False Negatives
        fn_data = df[(df['ground_truth'] == 1) & (df['is_anomaly'] == False)]
        
        failures = {
            'false_positives': {
                'count': len(fp_data),
                'examples': fp_data.head(n_cases)[['patient_id', 'timestamp', 'scenario', 
                                                   'heart_rate', 'spo2', 'sbp', 'risk_score']]
            },
            'false_negatives': {
                'count': len(fn_data),
                'examples': fn_data.head(n_cases)[['patient_id', 'timestamp', 'scenario',
                                                   'heart_rate', 'spo2', 'sbp', 'risk_score']]
            }
        }
        
        return failures