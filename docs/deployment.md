# CardioRisk AI Deployment Guide

This document outlines the required deployment steps for the local and production environments of CardioRisk AI.

## Architecture 1: LOCAL (Docker Compose)

The local architecture runs both the frontend and backend containers in a single network using Docker Compose.

**Structure:**
- Docker Compose
  - Frontend Container (Next.js)
  - Backend Container (FastAPI + Model Artifact + SHAP Explainer)

**Setup Instructions:**
1. Ensure Docker Desktop is installed and running on Windows.
2. Clone the repository.
3. In the root directory, create a frontend environment file `frontend/.env` based on `frontend/.env.example` if not already present.
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```
4. Run the full stack:
   ```bash
   docker compose up --build
   ```
5. **Access the application:**
   - Frontend: `http://localhost:3000`
   - Backend API Docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`
6. Stop the application:
   ```bash
   docker compose down
   ```

---

## Architecture 2: PRODUCTION

Production requires separating the frontend (Next.js) to a hosting platform like Vercel, and the backend (FastAPI) to a container-capable platform (e.g., AWS AppRunner, Google Cloud Run, Railway, or Render).

**Structure:**
- Vercel Frontend (Next.js)
  - Interacts via HTTPS with the backend.
- Container-capable Platform Backend (FastAPI)
  - Runs the model artifact and handles SHAP computations.

### Frontend Deployment (Vercel)
1. Push the code to a GitHub repository.
2. Import the project in Vercel. Vercel will automatically detect the Next.js framework.
3. **Environment Variables**:
   - Set `NEXT_PUBLIC_API_BASE_URL` to the HTTPS URL of your deployed backend API (e.g., `https://api.cardiorisk.example.com`).
4. Since `NEXT_PUBLIC_*` variables are embedded at build time, ensure the variable is set *before* Vercel builds the project.
5. Deploy.

### Backend Deployment (Container Hosting)
1. Provide the root repository directory (or Docker image) to the container-hosting provider.
2. **Required Configurations**:
   - **Port**: The application exposes port `8000` by default via the Dockerfile CMD (`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`).
   - **Startup Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` (Defined in the Dockerfile).
   - **Health Check Path**: `/health`
   - **CORS Origin**: Update the allowed origins in `src/api/main.py` if strict CORS is required. By default, it allows `*` (which is suitable for demonstration, but should be updated to the Vercel URL in production).
   - **Artifacts**: The repository must be deployed with the trained artifacts in the `/artifacts/` folder since the backend needs to load `final_pipeline.joblib`, `model_metadata.json`, and `decision_threshold.json` on startup.
3. **Memory & Cold-start Considerations**:
   - SHAP `KernelExplainer` uses a small representative background dataset, but still requires some memory. Recommend at least `1GB` of RAM for the backend container.
   - Initial loading of joblib and SHAP artifacts happens entirely during startup. Fast scaling may experience cold starts of 1-3 seconds.
