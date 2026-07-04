from fastapi import APIRouter
from src.api.dependencies import app_state

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy" if app_state.is_ready else "unhealthy",
        "model_loaded": app_state.model is not None,
        "explainer_loaded": app_state.explainer is not None,
        "model_version": app_state.metadata.get("model_version", "unknown") if app_state.metadata else "unknown"
    }
