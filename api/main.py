from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from .database.database import init_db
from .models.model_loader import model_loader
from .routers import prediction, monitoring
from .models.schemas import HealthResponse
from datetime import datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting API...")

    # Initialiser la base de données
    init_db()

    # Charger le modèle
    print("Loading ML model...")
    model_loaded = model_loader.load_model_from_mlflow()

    if not model_loaded:
        print("Warning: Model could not be loaded from MLflow")
        print("Trying to load from file if available...")
        # Vous pouvez ajouter un fallback ici si vous avez un fichier model.pkl

    if model_loader.is_loaded():
        print(f"Model loaded successfully: {model_loader.model_name}")
    else:
        print("No model loaded - predictions will fail")

    yield

    # Shutdown
    print("Shutting down API...")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="API de prédiction de risque de crédit avec monitoring complet",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
app.include_router(monitoring.router)


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    model_info = model_loader.get_model_info()

    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        api_version="1.0.0",
        model_loaded=model_loader.is_loaded(),
        model_version=model_info.get("model_version"),
        timestamp=datetime.utcnow()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Désactiver en production
    )
