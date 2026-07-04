from fastapi import APIRouter
from src.api.dependencies import app_state

router = APIRouter()

@router.get("/model-info")
def model_info():
    if not app_state.is_ready:
        return {"error": "Model not ready"}
        
    return {
        "model_name": app_state.metadata["model_name"],
        "version": app_state.metadata["model_version"],
        "feature_count": app_state.metadata["feature_count"],
        "threshold": app_state.metadata["selected_threshold"],
        "calibration_status": app_state.metadata["calibration_status"],
        "intended_use": "Educational and portfolio demonstration of structured-data ML.",
        "dataset_name": app_state.metadata["dataset_identifier"],
        "disclaimer": "This application is an educational machine learning decision-support demonstration. It must never claim to diagnose disease, recommend treatment, or replace a qualified healthcare professional."
    }
