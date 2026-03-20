from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Schéma pour les requêtes de prédiction
    Accepte un dictionnaire de features
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire des features du modèle",
        example={
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5,
            "DAYS_BIRTH": -9461,
            "DAYS_EMPLOYED": -637
        }
    )

    @validator("features")
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        if not all(isinstance(k, str) for k in v.keys()):
            raise ValueError("All feature names must be strings")
        return v


class PredictionResponse(BaseModel):
    """
    Schéma pour les réponses de prédiction
    """
    request_id: str = Field(..., description="Identifiant unique de la requête")
    timestamp: datetime = Field(..., description="Timestamp de la prédiction")

    # Outputs du modèle
    prediction_score: float = Field(
        ...,
        description="Score de prédiction (probabilité de défaut)",
        ge=0.0,
        le=1.0
    )
    prediction_class: int = Field(
        ...,
        description="Classe prédite (0: pas de défaut, 1: défaut)",
        ge=0,
        le=1
    )
    threshold: float = Field(
        default=0.5,
        description="Seuil utilisé pour la classification"
    )

    # Métadonnées
    model_version: str = Field(..., description="Version du modèle utilisé")
    api_version: str = Field(..., description="Version de l'API")

    # Métriques
    inference_time_ms: float = Field(..., description="Temps d'inférence en ms")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-03-02T10:30:00Z",
                "prediction_score": 0.23,
                "prediction_class": 0,
                "threshold": 0.5,
                "model_version": "LogisticRegression_v1.0",
                "api_version": "1.0.0",
                "inference_time_ms": 12.5
            }
        }


class HealthResponse(BaseModel):
    """
    Schéma pour le endpoint de health check
    """
    status: str = Field(..., description="Status de l'API")
    api_version: str = Field(..., description="Version de l'API")
    model_loaded: bool = Field(..., description="Modèle chargé ou non")
    model_version: Optional[str] = Field(None, description="Version du modèle chargé")
    timestamp: datetime = Field(..., description="Timestamp du health check")


class StatsResponse(BaseModel):
    """
    Schéma pour les statistiques du monitoring
    """
    total_predictions: int
    avg_latency_ms: float
    avg_inference_time_ms: float
    error_rate: float
    predictions_by_class: Dict[int, int]
    avg_prediction_score: float
    period_start: Optional[datetime]
    period_end: Optional[datetime]


class ErrorResponse(BaseModel):
    """
    Schéma pour les erreurs
    """
    request_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    status_code: int
