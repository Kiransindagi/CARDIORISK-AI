import requests
import time
import sys

def run_integration_test():
    base_url = "http://localhost:8000"
    print("Starting integration test...")
    
    # 1. Test Health
    try:
        health = requests.get(f"{base_url}/health")
        health.raise_for_status()
        h_data = health.json()
        assert h_data["status"] == "healthy"
        print("PASS: Health endpoint OK")
    except Exception as e:
        print(f"FAIL: Health endpoint failed: {e}")
        sys.exit(1)
        
    # 2. Test Model Info
    try:
        info = requests.get(f"{base_url}/model-info")
        info.raise_for_status()
        i_data = info.json()
        assert "model_name" in i_data
        print("PASS: Model info endpoint OK")
    except Exception as e:
        print(f"FAIL: Model info endpoint failed: {e}")
        sys.exit(1)
        
    # 3. Test Prediction
    valid_payload = {
        "age": 50, 
        "sex": "Male", 
        "chest_pain_type": "Typical Angina", 
        "resting_bp": 120, 
        "cholesterol": 200,
        "fasting_blood_sugar": "No", 
        "resting_ecg": "Normal", 
        "max_heart_rate": 150, 
        "exercise_induced_angina": "No",
        "st_depression": 0.0, 
        "st_slope": "Flat", 
        "num_major_vessels": 0, 
        "thalassemia": "Normal"
    }
    
    try:
        start = time.time()
        pred = requests.post(f"{base_url}/predict", json=valid_payload)
        pred.raise_for_status()
        p_data = pred.json()
        end = time.time()
        
        assert "prediction" in p_data
        assert 0.0 <= p_data["risk_probability"] <= 1.0
        assert "explanation" in p_data
        assert "risk_increasing_factors" in p_data["explanation"]
        
        print(f"PASS: Prediction endpoint OK ({(end - start)*1000:.1f}ms)")
    except Exception as e:
        print(f"FAIL: Prediction endpoint failed: {e}")
        if 'pred' in locals():
            print(f"Response: {pred.text}")
        sys.exit(1)
        
    print("Integration test passed successfully!")

if __name__ == "__main__":
    run_integration_test()
