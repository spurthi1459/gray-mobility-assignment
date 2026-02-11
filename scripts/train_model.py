"""
Train anomaly detection model with improvements

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.models.anomaly_detector import VitalsAnomalyDetector
from src.models.risk_scorer import RiskScorer
from src.evaluation.metrics import AnomalyEvaluator

def main():
    print("=" * 60)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("=" * 60)
    
    # Load cleaned data
    print("\nLoading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_vitals.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Loaded {len(df)} records")
    
    # Separate normal vs abnormal data
    df_normal = df[df['scenario'] == 'normal_transport'].copy()
    df_abnormal = df[df['scenario'] != 'normal_transport'].copy()
    
    print(f"\nNormal transport data: {len(df_normal)} records")
    print(f"Abnormal data: {len(df_abnormal)} records")
    
    # ========== TRAIN ORIGINAL MODEL ==========
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)
    
    detector_original = VitalsAnomalyDetector(window_size=60, contamination=0.1)
    detector_original.train(df_normal)
    
    os.makedirs('models', exist_ok=True)
    detector_original.save_model('models/anomaly_detector_v1.pkl')
    
    df_pred_original = detector_original.predict(df)
    scorer = RiskScorer()
    df_original = scorer.calculate_risk_score(df_pred_original)
    df_original.to_csv('data/processed/predictions.csv', index=False)
    
    # Evaluate original
    evaluator = AnomalyEvaluator()
    df_eval_original = evaluator.create_ground_truth(df_original)
    metrics_original = evaluator.calculate_metrics(df_eval_original)
    
    print("\n BASELINE MODEL RESULTS:")
    print(f"  Precision: {metrics_original['precision']:.3f}")
    print(f"  Recall: {metrics_original['recall']:.3f}")
    print(f"  F1 Score: {metrics_original['f1_score']:.3f}")
    print(f"  False Positive Rate: {metrics_original['false_positive_rate']:.3f}")
    print(f"  False Negatives: {metrics_original['false_negatives']} (CRITICAL!)")
    print(f"  False Positives: {metrics_original['false_positives']} (False alarms)")
    
    # ========== TRAIN IMPROVED MODEL ==========
    print("\n" + "=" * 60)
    print("TRAINING IMPROVED MODEL (Lower FP Rate)")
    print("=" * 60)
    
    # Improved model with lower contamination
    detector_improved = VitalsAnomalyDetector(window_size=60, contamination=0.05)
    detector_improved.train(df_normal)
    detector_improved.save_model('models/anomaly_detector_v2_improved.pkl')
    
    df_pred_improved = detector_improved.predict(df)
    df_improved = scorer.calculate_risk_score(df_pred_improved)
    
    # Apply temporal smoothing to reduce transient false positives
    print("\nApplying temporal smoothing (30-second window)...")
    for patient_id in df_improved['patient_id'].unique():
        mask = df_improved['patient_id'] == patient_id
        patient_data = df_improved[mask].copy()
        
        # Smooth anomaly predictions: require 50% of last 30 seconds to be anomalies
        smoothed = patient_data['is_anomaly'].astype(int).rolling(
            window=30, 
            min_periods=15
        ).mean()
        
        # Fill NaN values with original predictions
        smoothed = smoothed.fillna(patient_data['is_anomaly'].astype(int))
        
        df_improved.loc[mask, 'is_anomaly'] = (smoothed > 0.5)
    
    # Ensure no NaN values and correct dtype
    df_improved['is_anomaly'] = df_improved['is_anomaly'].fillna(False).astype(bool)
    
    df_improved.to_csv('data/processed/predictions_improved.csv', index=False)
    
    # Evaluate improved
    df_eval_improved = evaluator.create_ground_truth(df_improved)
    metrics_improved = evaluator.calculate_metrics(df_eval_improved)
    
    print("\n📊 IMPROVED MODEL RESULTS:")
    print(f"  Precision: {metrics_improved['precision']:.3f}")
    print(f"  Recall: {metrics_improved['recall']:.3f}")
    print(f"  F1 Score: {metrics_improved['f1_score']:.3f}")
    print(f"  False Positive Rate: {metrics_improved['false_positive_rate']:.3f}")
    print(f"  False Negatives: {metrics_improved['false_negatives']} (CRITICAL!)")
    print(f"  False Positives: {metrics_improved['false_positives']} (False alarms)")
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    fp_reduction = metrics_original['false_positives'] - metrics_improved['false_positives']
    fp_reduction_pct = (fp_reduction / metrics_original['false_positives']) * 100 if metrics_original['false_positives'] > 0 else 0
    
    precision_gain = ((metrics_improved['precision'] - metrics_original['precision']) 
                     / metrics_original['precision']) * 100 if metrics_original['precision'] > 0 else 0
    
    recall_change = metrics_improved['recall'] - metrics_original['recall']
    
    print(f"\n{'Metric':<25} {'Baseline':<12} {'Improved':<12} {'Change':<12}")
    print("-" * 60)
    print(f"{'Precision':<25} {metrics_original['precision']:<12.3f} "
          f"{metrics_improved['precision']:<12.3f} "
          f"{precision_gain:+.1f}%")
    print(f"{'Recall':<25} {metrics_original['recall']:<12.3f} "
          f"{metrics_improved['recall']:<12.3f} "
          f"{recall_change:+.3f}")
    print(f"{'F1 Score':<25} {metrics_original['f1_score']:<12.3f} "
          f"{metrics_improved['f1_score']:<12.3f}")
    print(f"{'FP Rate':<25} {metrics_original['false_positive_rate']:<12.3f} "
          f"{metrics_improved['false_positive_rate']:<12.3f}")
    print(f"{'False Positives':<25} {metrics_original['false_positives']:<12} "
          f"{metrics_improved['false_positives']:<12} "
          f"{-fp_reduction:+}")
    print(f"{'False Negatives':<25} {metrics_original['false_negatives']:<12} "
          f"{metrics_improved['false_negatives']:<12}")
    
    print(f"\n IMPROVEMENTS:")
    print(f"   False alarms reduced by {fp_reduction} ({fp_reduction_pct:.1f}%)")
    print(f"   Precision increased by {precision_gain:.1f}%")
    
    if recall_change < -0.02:
        print(f"\n⚠️  WARNING: Recall decreased by {-recall_change:.3f}")
        print("  Recommendation: Use baseline model for maximum safety")
    else:
        print(f"  ➤ Recall maintained (change: {recall_change:+.3f})")
        print("\n✅ RECOMMENDATION: Use improved model")
    
    # ========== PER-PATIENT RESULTS ==========
    print("\n" + "=" * 60)
    print("IMPROVED MODEL - RESULTS BY PATIENT")
    print("=" * 60)
    
    for patient_id in sorted(df['patient_id'].unique()):
        patient_data = df_improved[df_improved['patient_id'] == patient_id]
        scenario = patient_data['scenario'].iloc[0]
        
        anomaly_rate = (patient_data['is_anomaly'].sum() / len(patient_data)) * 100
        avg_risk = patient_data['risk_score'].mean()
        max_risk = patient_data['risk_score'].max()
        critical_count = (patient_data['risk_level'] == 'CRITICAL').sum()
        
        print(f"\nPatient {patient_id} ({scenario}):")
        print(f"  Anomaly rate: {anomaly_rate:.1f}%")
        print(f"  Average risk score: {avg_risk:.2f}/10")
        print(f"  Max risk score: {max_risk:.2f}/10")
        print(f"  Critical alerts: {critical_count}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/anomaly_detector_v1.pkl (baseline)")
    print("  - models/anomaly_detector_v2_improved.pkl (improved)")
    print("  - data/processed/predictions.csv (baseline)")
    print("  - data/processed/predictions_improved.csv (improved)")


if __name__ == "__main__":
    main()