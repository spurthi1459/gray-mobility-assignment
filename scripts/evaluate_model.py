"""
Evaluate anomaly detection model performance

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.evaluation.metrics import AnomalyEvaluator

def main():
    print("=" * 60)
    print("EVALUATING ANOMALY DETECTION MODEL")
    print("=" * 60)
    
    # Load predictions
    print("\nLoading predictions...")
    df = pd.read_csv('data/processed/predictions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Loaded {len(df)} predictions")
    
    # Initialize evaluator
    evaluator = AnomalyEvaluator()
    
    # Create ground truth labels
    print("\nCreating ground truth labels...")
    df = evaluator.create_ground_truth(df)
    print("✓ Ground truth created")
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    
    metrics = evaluator.calculate_metrics(df)
    
    print(f"\nPrecision: {metrics['precision']:.3f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.3f} (CRITICAL!)")
    print(f"Specificity: {metrics['specificity']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP): {metrics['true_positives']}")
    print(f"  True Negatives (TN): {metrics['true_negatives']}")
    print(f"  False Positives (FP): {metrics['false_positives']} (False alarms)")
    print(f"  False Negatives (FN): {metrics['false_negatives']} (Missed events - DANGEROUS!)")
    
    # Alert latency
    print("\n" + "=" * 60)
    print("ALERT LATENCY ANALYSIS")
    print("=" * 60)
    
    latencies = evaluator.calculate_alert_latency(df)
    
    for patient_id, data in latencies.items():
        print(f"\nPatient {patient_id} ({data['scenario']}):")
        print(f"  {data['status']}")
    
    # By-scenario metrics
    print("\n" + "=" * 60)
    print("METRICS BY SCENARIO")
    print("=" * 60)
    
    scenario_metrics = evaluator.evaluate_by_scenario(df)
    
    for scenario, metrics in scenario_metrics.items():
        print(f"\n{scenario}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    # Failure analysis
    print("\n" + "=" * 60)
    print("FAILURE CASE ANALYSIS")
    print("=" * 60)
    
    failures = evaluator.analyze_failure_cases(df, n_cases=3)
    
    print(f"\nFalse Positives (False Alarms): {failures['false_positives']['count']}")
    print("\nExample False Positive Cases:")
    print(failures['false_positives']['examples'].to_string())
    
    print(f"\n\nFalse Negatives (Missed Detections): {failures['false_negatives']['count']}")
    print("\nExample False Negative Cases:")
    print(failures['false_negatives']['examples'].to_string())
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 60)
    
    os.makedirs('plots/evaluation', exist_ok=True)
    
    print("\nPlotting confusion matrix...")
    evaluator.plot_confusion_matrix(df, save_path='plots/evaluation/confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 60)
if __name__ == "__main__":
    main()