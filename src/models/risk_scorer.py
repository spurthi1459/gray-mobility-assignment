"""
Risk Scoring System for Patient Triage
Combines multiple vitals, trends, and confidence into actionable risk score
"""

import numpy as np
import pandas as pd

class RiskScorer:
    """
    Calculate patient risk score for triage and alert prioritization
    
    Risk levels:
    - LOW (0-3): Stable, routine transport
    - MODERATE (4-6): Requires monitoring
    - HIGH (7-8): Urgent attention needed
    - CRITICAL (9-10): Immediate intervention
    """
    
    def __init__(self):
        # Weight factors for different components
        self.weights = {
            'vital_severity': 0.4,      # How abnormal are current vitals
            'trend_direction': 0.3,     # Is patient deteriorating
            'anomaly_confidence': 0.2,  # ML model confidence
            'vital_combination': 0.1    # Multiple abnormal vitals
        }
    
    def calculate_risk_score(self, df):
        """
        Calculate comprehensive risk score for each time point
        
        Returns:
            DataFrame with risk_score (0-10), risk_level, and confidence
        """
        df = df.copy()
        
        # Component 1: Vital Severity Score (0-10)
        vital_score = self._calculate_vital_severity(df)
        
        # Component 2: Trend Direction Score (0-10)
        trend_score = self._calculate_trend_score(df)
        
        # Component 3: Anomaly Confidence (0-10)
        if 'anomaly_score' in df.columns:
            # Convert anomaly scores to 0-10 scale
            # Isolation Forest scores are typically in range [-0.5, 0.5]
            anomaly_conf = 10 * (1 - (df['anomaly_score'] + 0.5))
            anomaly_conf = anomaly_conf.clip(0, 10)
        else:
            anomaly_conf = 0
        
        # Component 4: Multiple Vital Abnormality (0-10)
        combination_score = self._calculate_combination_score(df)
        
        # Weighted combination
        risk_score = (
            self.weights['vital_severity'] * vital_score +
            self.weights['trend_direction'] * trend_score +
            self.weights['anomaly_confidence'] * anomaly_conf +
            self.weights['vital_combination'] * combination_score
        )
        
        # Clip to 0-10 range
        risk_score = risk_score.clip(0, 10)
        
        # Assign risk levels
        risk_level = pd.cut(
            risk_score,
            bins=[-0.1, 3, 6, 8, 10],
            labels=['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        )
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(df)
        
        df['risk_score'] = risk_score
        df['risk_level'] = risk_level
        df['confidence'] = confidence
        
        return df
    
    def _calculate_vital_severity(self, df):
        """
        Score how abnormal current vitals are (0-10)
        Based on clinical severity ranges
        """
        severity = pd.Series(0.0, index=df.index)
        
        # Heart Rate scoring
        hr = df['heart_rate']
        severity += np.where(hr > 140, 3, 0)  # Severe tachycardia
        severity += np.where((hr > 120) & (hr <= 140), 2, 0)
        severity += np.where((hr > 100) & (hr <= 120), 1, 0)
        severity += np.where(hr < 40, 3, 0)   # Severe bradycardia
        severity += np.where((hr >= 40) & (hr < 50), 2, 0)
        
        # SpO2 scoring
        spo2 = df['spo2']
        severity += np.where(spo2 < 85, 3, 0)  # Critical hypoxia
        severity += np.where((spo2 >= 85) & (spo2 < 90), 2, 0)
        severity += np.where((spo2 >= 90) & (spo2 < 94), 1, 0)
        
        # Blood Pressure scoring
        sbp = df['sbp']
        severity += np.where(sbp < 80, 3, 0)   # Severe hypotension
        severity += np.where((sbp >= 80) & (sbp < 90), 2, 0)
        severity += np.where(sbp > 200, 3, 0)  # Hypertensive crisis
        severity += np.where((sbp > 180) & (sbp <= 200), 2, 0)
        
        # MAP scoring
        if 'map' in df.columns:
            map_val = df['map']
            severity += np.where(map_val < 65, 3, 0)  # Critical
            severity += np.where((map_val >= 65) & (map_val < 70), 2, 0)
        
        return severity.clip(0, 10)
    
    def _calculate_trend_score(self, df):
        """
        Score based on deteriorating trends (0-10)
        Worse if vitals are getting worse over time
        """
        trend = pd.Series(0.0, index=df.index)
        
        window = 300  # 5 minute window
        
        # HR trend (increasing = bad if already high, decreasing = bad if already low)
        if 'heart_rate' in df.columns:
            hr_trend = df['heart_rate'].rolling(window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            # Increasing HR when already elevated
            trend += np.where((df['heart_rate'] > 100) & (hr_trend > 0.1), 2, 0)
            # Decreasing HR when already low
            trend += np.where((df['heart_rate'] < 60) & (hr_trend < -0.1), 2, 0)
        
        # SpO2 trend (decreasing = bad)
        if 'spo2' in df.columns:
            spo2_trend = df['spo2'].rolling(window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            trend += np.where(spo2_trend < -0.05, 3, 0)  # Declining SpO2 is critical
        
        # BP trend (decreasing = bad)
        if 'sbp' in df.columns:
            sbp_trend = df['sbp'].rolling(window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            trend += np.where(sbp_trend < -0.2, 3, 0)  # Dropping BP
        
        return trend.clip(0, 10)
    
    def _calculate_combination_score(self, df):
        """
        Score based on multiple simultaneous abnormalities (0-10)
        Multiple abnormal vitals = higher risk
        """
        abnormal_count = pd.Series(0, index=df.index)
        
        # Count abnormal vitals
        abnormal_count += (df['heart_rate'] > 120) | (df['heart_rate'] < 50)
        abnormal_count += (df['spo2'] < 94)
        abnormal_count += (df['sbp'] < 90) | (df['sbp'] > 180)
        
        if 'shock_index' in df.columns:
            abnormal_count += (df['shock_index'] > 1.0)
        
        if 'map' in df.columns:
            abnormal_count += (df['map'] < 70)
        
        # Convert count to score
        score = abnormal_count * 2.5  # Each abnormal vital adds 2.5 points
        
        return score.clip(0, 10)
    
    def _calculate_confidence(self, df):
        """
        Confidence in the risk score (0-1)
        Lower confidence if:
        - Recent artifacts detected
        - Missing data
        - High motion/vibration
        """
        confidence = pd.Series(1.0, index=df.index)
        
        # Reduce confidence if artifacts present
        if 'artifact_any' in df.columns:
            recent_artifacts = df['artifact_any'].rolling(60, min_periods=1).sum()
            confidence -= (recent_artifacts / 60) * 0.3  # Max 30% reduction
        
        # Reduce confidence if high motion
        if 'motion_vibration' in df.columns:
            high_motion = df['motion_vibration'] > 2.0
            confidence -= high_motion * 0.2  # 20% reduction for high motion
        
        return confidence.clip(0, 1)
    
    def should_alert(self, risk_score, risk_level, confidence, suppress_threshold=0.5):
        """
        Decide whether to trigger an alert
        
        Args:
            risk_score: Current risk score (0-10)
            risk_level: Risk level (LOW/MODERATE/HIGH/CRITICAL)
            confidence: Confidence in score (0-1)
            suppress_threshold: Minimum confidence to alert
        
        Returns:
            bool: True if alert should be triggered
        """
        # Don't alert if confidence too low
        if confidence < suppress_threshold:
            return False
        
        # Always alert on CRITICAL
        if risk_level == 'CRITICAL':
            return True
        
        # Alert on HIGH if confidence is reasonable
        if risk_level == 'HIGH' and confidence >= 0.6:
            return True
        
        # Alert on MODERATE only if confidence is high
        if risk_level == 'MODERATE' and confidence >= 0.8 and risk_score >= 5:
            return True
        
        return False