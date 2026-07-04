import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, clean_data
from src.data.split import split_development_test
from src.features.feature_schema import FEATURES_ORDER
from src.models.explain import get_explainer, get_local_explanation
from scripts.train_model import specificity_score

@pytest.fixture(scope="module")
def df():
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    return clean_data(load_raw_data(raw_data_path))

def test_specificity_implementation():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    # tn=1, fp=1, fn=1, tp=1
    # specificity = tn / (tn + fp) = 1 / 2 = 0.5
    assert specificity_score(y_true, y_pred) == 0.5

def test_final_model_and_threshold(df):
    model_path = PROJECT_ROOT / "artifacts" / "model" / "final_pipeline.joblib"
    thresh_path = PROJECT_ROOT / "artifacts" / "model" / "decision_threshold.json"
    
    if not model_path.exists():
        pytest.skip("Model not trained yet")
        
    model = joblib.load(model_path)
    with open(thresh_path, "r") as f:
        thresh_data = json.load(f)
        
    threshold = thresh_data['selected_threshold']
    assert 0.0 <= threshold <= 1.0
    
    X_dev, _, _, _ = split_development_test(df)
    
    # Check 13-feature input works
    X_sample = X_dev.iloc[[0]]
    assert list(X_sample.columns) == FEATURES_ORDER
    
    proba = model.predict_proba(X_sample)[:, 1][0]
    assert 0.0 <= proba <= 1.0
    
    # Check threshold logic
    pred = 1 if proba >= threshold else 0
    assert pred in [0, 1]

def test_shap_explainer(df):
    model_path = PROJECT_ROOT / "artifacts" / "model" / "final_pipeline.joblib"
    if not model_path.exists():
        pytest.skip("Model not trained yet")
        
    model = joblib.load(model_path)
    X_dev, _, _, _ = split_development_test(df)
    
    # Small test
    X_train_summary = X_dev.iloc[:5]
    
    def predict_fn(X_array):
        if isinstance(X_array, np.ndarray):
            X_df = pd.DataFrame(X_array, columns=FEATURES_ORDER)
        else:
            X_df = X_array
        return model.predict_proba(X_df)[:, 1]
        
    import shap
    explainer = shap.KernelExplainer(predict_fn, X_train_summary)
    
    X_instance = X_dev.iloc[[0]]
    local_exp = get_local_explanation(explainer, X_instance, FEATURES_ORDER)
    
    assert "risk_increasing_factors" in local_exp
    assert "risk_decreasing_factors" in local_exp
    assert "base_value" in local_exp
    assert isinstance(local_exp["base_value"], float)
