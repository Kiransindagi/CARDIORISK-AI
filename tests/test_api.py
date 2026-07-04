import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.api.main import app

# Create client
client = TestClient(app)

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["explainer_loaded"] is True
        assert "model_version" in data

def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "threshold" in data

def test_predict_valid():
    valid_payload = {
        "age": 63,
        "sex": "Male",
        "chest_pain_type": "Typical Angina",
        "resting_bp": 145,
        "cholesterol": 233,
        "fasting_blood_sugar": "Yes",
        "resting_ecg": "Left Ventricular Hypertrophy",
        "max_heart_rate": 150,
        "exercise_induced_angina": "No",
        "st_depression": 2.3,
        "st_slope": "Downsloping",
        "num_major_vessels": 0,
        "thalassemia": "Fixed Defect"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert "risk_probability" in data
        assert 0.0 <= data["risk_probability"] <= 1.0
        assert "explanation" in data
        assert "risk_increasing_factors" in data["explanation"]
        assert "disclaimer" in data

def test_predict_invalid():
    invalid_payload = {
        "age": 10, # below 18
        "sex": "Unknown", # invalid category
        "chest_pain_type": "Typical Angina",
        "resting_bp": 145,
        "cholesterol": 233,
        "fasting_blood_sugar": "Yes",
        "resting_ecg": "Left Ventricular Hypertrophy",
        "max_heart_rate": 150,
        "exercise_induced_angina": "No",
        "st_depression": 2.3,
        "st_slope": "Downsloping",
        "num_major_vessels": 0,
        "thalassemia": "Fixed Defect"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert len(data["error"]["details"]) == 2 # age and sex are invalid
