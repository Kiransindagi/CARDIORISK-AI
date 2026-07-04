from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.dependencies import load_artifacts
from src.api.routes import health, model_info, predict
from src.utils.logger import RequestIDMiddleware, get_logger
import uvicorn

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    success = load_artifacts()
    if not success:
        logger.error("Failed to load artifacts. API will fail health checks.")
    yield
    logger.info("Application shutting down...")

app = FastAPI(
    title="CardioRisk AI API",
    description="Educational machine learning decision-support demonstration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on request: {exc.errors()}")
    errors = []
    for err in exc.errors():
        field = ".".join([str(loc) for loc in err["loc"] if loc != "body"])
        errors.append({
            "field": field,
            "message": err["msg"]
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "One or more fields are invalid.",
                "details": errors
            }
        },
    )

app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
