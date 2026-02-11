"""
FastAPI service for anomaly detection in ambulance vitals
Accepts real-time vital signs and returns anomaly predictions + risk scores
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.anomaly_detector import VitalsAnomalyDetector
from src.models.risk_scorer import RiskScorer

# Initialize FastAPI app
app = FastAPI(
    title="Gray Mobility Smart Ambulance API",
    description="Real-time anomaly detection and risk scoring for ambulance patient vitals",
    version="1.0.0"
)

# Load trained model on startup
detector = VitalsAnomalyDetector()
scorer = RiskScorer()

try:
    detector.load_model('models/anomaly_detector_v2_improved.pkl')
    print(" Loaded improved anomaly detection model")
except Exception as e:
    print(f" Could not load improved model, using baseline: {e}")
    detector.load_model('models/anomaly_detector_v1.pkl')

# Request model
class VitalsInput(BaseModel):
    """Input: Real-time patient vitals"""
    patient_id: int = Field(..., description="Patient identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp (ISO format), defaults to now")
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in bpm")
    spo2: float = Field(..., ge=0, le=100, description="SpO2 percentage")
    sbp: float = Field(..., ge=0, le=300, description="Systolic blood pressure (mmHg)")
    dbp: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mmHg)")
    motion_vibration: Optional[float] = Field(0.5, ge=0, le=10, description="Motion/vibration level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": 101,
                "timestamp": "2026-02-10T14:30:00",
                "heart_rate": 85.0,
                "spo2": 97.0,
                "sbp": 120.0,
                "dbp": 80.0,
                "motion_vibration": 0.5
            }
        }

class VitalsBatchInput(BaseModel):
    """Input: Batch of vitals (for time-series context)"""
    vitals: List[VitalsInput] = Field(..., description="List of vital readings (chronological order)")

# Response model
class PredictionOutput(BaseModel):
    """Output: Anomaly prediction and risk assessment"""
    patient_id: int
    timestamp: str
    is_anomaly: bool = Field(..., description="Whether vitals are anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (lower = more anomalous)")
    risk_score: float = Field(..., ge=0, le=10, description="Risk score (0-10 scale)")
    risk_level: str = Field(..., description="Risk level: LOW, MODERATE, HIGH, CRITICAL")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction (0-1)")
    should_alert: bool = Field(..., description="Whether to trigger an alert")
    alert_reason: Optional[str] = Field(None, description="Reason for alert")
    vitals_summary: dict = Field(..., description="Summary of input vitals")

# Helper functions
def create_dataframe_from_input(vitals_list: List[VitalsInput]) -> pd.DataFrame:
    """Convert input vitals to DataFrame for model processing"""
    data = []
    for v in vitals_list:
        timestamp = v.timestamp if v.timestamp else datetime.now().isoformat()
        data.append({
            'patient_id': v.patient_id,
            'timestamp': timestamp,
            'heart_rate': v.heart_rate,
            'spo2': v.spo2,
            'sbp': v.sbp,
            'dbp': v.dbp,
            'motion_vibration': v.motion_vibration,
            'scenario': 'real_time'
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add cleaned columns (same as input for real-time)
    df['heart_rate_cleaned'] = df['heart_rate']
    df['spo2_cleaned'] = df['spo2']
    df['sbp_cleaned'] = df['sbp']
    df['dbp_cleaned'] = df['dbp']
    
    return df

def determine_alert_reason(row: pd.Series) -> Optional[str]:
    """Determine why an alert should be triggered"""
    reasons = []
    
    if row['risk_level'] == 'CRITICAL':
        reasons.append("CRITICAL risk level")
    
    if row['heart_rate'] > 140:
        reasons.append("Severe tachycardia (HR > 140)")
    elif row['heart_rate'] < 40:
        reasons.append("Severe bradycardia (HR < 40)")
    
    if row['spo2'] < 90:
        reasons.append("Critical hypoxia (SpO2 < 90%)")
    
    if row['sbp'] < 90:
        reasons.append("Hypotension (SBP < 90)")
    
    if 'shock_index' in row and row['shock_index'] > 1.0:
        reasons.append("Shock index > 1.0")
    
    if 'map' in row and row['map'] < 70:
        reasons.append("Critical MAP < 70")
    
    return "; ".join(reasons) if reasons else None

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Gray Mobility Smart Ambulance API",
        "version": "1.0.0",
        "model": "Improved Anomaly Detector v2"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": detector.trained,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(vitals: VitalsInput):
    """
    Predict anomaly and risk for a single vital reading
    
    Note: Single readings have limited context. For better accuracy,
    use /predict_batch with historical vitals.
    """
    try:
        # Convert to DataFrame
        df = create_dataframe_from_input([vitals])
        
        # Run prediction
        df_pred = detector.predict(df)
        df_risk = scorer.calculate_risk_score(df_pred)
        
        # Fill NaN values that occur due to lack of historical context
        df_risk = df_risk.fillna({
            'anomaly_score': 0,
            'shock_index': 0,
            'map': 0,
            'pulse_pressure': 0,
            'hr_spo2_ratio': 0,
            'is_anomaly': False
        })
        
        # Get result
        row = df_risk.iloc[0]
        
        # Determine if alert should trigger
        should_alert = scorer.should_alert(
            row['risk_score'],
            row['risk_level'],
            row['confidence']
        )
        
        alert_reason = determine_alert_reason(row) if should_alert else None
        
        return PredictionOutput(
            patient_id=int(row['patient_id']),
            timestamp=row['timestamp'].isoformat(),
            is_anomaly=bool(row.get('is_anomaly', False)),
            anomaly_score=float(row.get('anomaly_score', 0)),
            risk_score=float(row['risk_score']),
            risk_level=str(row['risk_level']),
            confidence=float(row['confidence']),
            should_alert=should_alert,
            alert_reason=alert_reason,
            vitals_summary={
                'heart_rate': float(row['heart_rate']),
                'spo2': float(row['spo2']),
                'sbp': float(row['sbp']),
                'dbp': float(row['dbp']),
                'map': float(row.get('map', 0)),
                'shock_index': float(row.get('shock_index', 0))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=List[PredictionOutput])
async def predict_batch(batch: VitalsBatchInput):
    """
    Predict anomalies for a batch of vitals (time-series)
    
    Recommended: Send last 5 minutes of vitals for best context
    """
    try:
        if len(batch.vitals) == 0:
            raise HTTPException(status_code=400, detail="Empty vitals list")
        
        # Convert to DataFrame
        df = create_dataframe_from_input(batch.vitals)
        
        # Run prediction
        df_pred = detector.predict(df)
        df_risk = scorer.calculate_risk_score(df_pred)
        
        # Fill NaN values
        df_risk = df_risk.fillna({
            'anomaly_score': 0,
            'shock_index': 0,
            'map': 0,
            'pulse_pressure': 0,
            'hr_spo2_ratio': 0,
            'is_anomaly': False
        })
        
        # Convert to response format
        results = []
        for idx, row in df_risk.iterrows():
            should_alert = scorer.should_alert(
                row['risk_score'],
                row['risk_level'],
                row['confidence']
            )
            
            alert_reason = determine_alert_reason(row) if should_alert else None
            
            results.append(PredictionOutput(
                patient_id=int(row['patient_id']),
                timestamp=row['timestamp'].isoformat(),
                is_anomaly=bool(row.get('is_anomaly', False)),
                anomaly_score=float(row.get('anomaly_score', 0)),
                risk_score=float(row['risk_score']),
                risk_level=str(row['risk_level']),
                confidence=float(row['confidence']),
                should_alert=should_alert,
                alert_reason=alert_reason,
                vitals_summary={
                    'heart_rate': float(row['heart_rate']),
                    'spo2': float(row['spo2']),
                    'sbp': float(row['sbp']),
                    'dbp': float(row['dbp']),
                    'map': float(row.get('map', 0)),
                    'shock_index': float(row.get('shock_index', 0))
                }
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Isolation Forest (Improved)",
        "features_count": len(detector.feature_cols) if detector.trained else 0,
        "window_size": detector.window_size,
        "contamination": detector.contamination,
        "trained": detector.trained,
        "performance": {
            "precision": 0.811,
            "recall": 0.989,
            "f1_score": 0.891,
            "false_positive_rate": 0.194
        }
    }