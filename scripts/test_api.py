"""
Test the API with sample data

"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health check"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_single_prediction():
    """Test single vital prediction"""
    print("Testing /predict endpoint (normal vitals)...")
    
    data = {
        "patient_id": 101,
        "heart_rate": 75.0,
        "spo2": 98.0,
        "sbp": 120.0,
        "dbp": 80.0,
        "motion_vibration": 0.5
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_critical_vitals():
    """Test with critical vitals"""
    print("Testing /predict endpoint (CRITICAL vitals)...")
    
    data = {
        "patient_id": 102,
        "heart_rate": 150.0,
        "spo2": 85.0,
        "sbp": 85.0,
        "dbp": 55.0,
        "motion_vibration": 2.5
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    result = response.json()
    print(f"Status: {response.status_code}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Should Alert: {result['should_alert']}")
    print(f"Alert Reason: {result['alert_reason']}")
    print(f"Response: {json.dumps(result, indent=2)}\n")

def test_model_info():
    """Test model info endpoint"""
    print("Testing /model_info endpoint...")
    response = requests.get(f"{API_URL}/model_info")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING GRAY MOBILITY API")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_single_prediction()
        test_critical_vitals()
        test_model_info()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API")
        print("Make sure the API is running: python scripts\\run_api.py")