import json
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("dependencies")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class AppState:
    model = None
    metadata = None
    threshold = None
    explainer = None
    is_ready = False

app_state = AppState()

def load_artifacts():
    try:
        logger.info("Loading model artifacts...")
        artifacts_dir = PROJECT_ROOT / "artifacts" / "model"
        
        model_path = artifacts_dir / "final_pipeline.joblib"
        metadata_path = artifacts_dir / "model_metadata.json"
        threshold_path = artifacts_dir / "decision_threshold.json"
        
        if not all(p.exists() for p in [model_path, metadata_path, threshold_path]):
            logger.error("One or more critical artifacts are missing.")
            return False
            
        app_state.model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            app_state.metadata = json.load(f)
        with open(threshold_path, "r") as f:
            app_state.threshold = json.load(f)["selected_threshold"]
            
        logger.info("Initializing SHAP explainer...")
        # To avoid high latency and large memory usage in production API, 
        # we load a cached background dataset.
        from src.data.load_data import load_raw_data, clean_data
        from src.data.split import split_development_test
        df = clean_data(load_raw_data(PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"))
        X_dev, _, _, _ = split_development_test(df)
        X_train_summary = shap.sample(X_dev, 25) # Small representative background
        
        features_order = app_state.metadata["feature_order"]
        
        def predict_fn(X_array):
            if isinstance(X_array, np.ndarray):
                X_df = pd.DataFrame(X_array, columns=features_order)
            else:
                X_df = X_array
            return app_state.model.predict_proba(X_df)[:, 1]
            
        app_state.explainer = shap.KernelExplainer(predict_fn, X_train_summary)
        
        app_state.is_ready = True
        logger.info("Artifacts and Explainer successfully loaded.")
        return True
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        return False

def get_model():
    return app_state.model

def get_metadata():
    return app_state.metadata

def get_threshold():
    return app_state.threshold

def get_explainer():
    return app_state.explainer
