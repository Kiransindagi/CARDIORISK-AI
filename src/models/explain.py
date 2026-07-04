import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.feature_schema import FEATURES_ORDER, CATEGORY_MAPPING

def get_explainer(model, X_train):
    """
    Returns a SHAP explainer appropriate for the model type.
    We are using CalibratedClassifierCV wrapping LogisticRegression.
    For this complex pipeline, KernelExplainer or a custom approach is needed, 
    but SHAP's LinearExplainer can be used on the base LogisticRegression model.
    To keep it simple and universally compatible, we can use KernelExplainer 
    or ExactExplainer on a subset of data.
    """
    # Use KernelExplainer for black-box/complex pipelines
    # Background summary to speed up KernelExplainer
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer

def get_local_explanation(explainer, X_instance, feature_names):
    """
    Returns structured local explanation for a single instance.
    """
    # For KernelExplainer, shap_values returns a list for multi-class/probability outputs
    # For binary classification predict_proba, shap_values[1] is the probability of class 1
    shap_values = explainer.shap_values(X_instance)
    
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        contributions = shap_values[0]
        base_value = explainer.expected_value
        
    increasing_factors = []
    decreasing_factors = []
    
    for i, feature in enumerate(feature_names):
        val = X_instance.iloc[0, i]
        contrib = contributions[i]
        
        item = {
            "feature": feature,
            "label": feature.replace("_", " ").title(),
            "value": float(val) if pd.api.types.is_numeric_dtype(type(val)) else val,
            "contribution": float(contrib),
            "direction": "increasing" if contrib > 0 else "decreasing"
        }
        
        if contrib > 0:
            increasing_factors.append(item)
        elif contrib < 0:
            decreasing_factors.append(item)
            
    # Sort by absolute magnitude of contribution
    increasing_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
    decreasing_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return {
        "risk_increasing_factors": increasing_factors,
        "risk_decreasing_factors": decreasing_factors,
        "base_value": float(base_value)
    }
