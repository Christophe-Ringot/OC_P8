from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import time
import uuid
from datetime import datetime

from ..models.schemas import PredictionRequest, PredictionResponse, ErrorResponse
from ..models.model_loader import model_loader
from ..database.database import get_db
from ..database.db_models import PredictionLog

router = APIRouter(prefix="/api/v1", tags=["predictions"])

API_VERSION = "1.0.0"
DEFAULT_THRESHOLD = 0.5


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    # Générer un request_id unique
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    start_time = time.time()

    # Variables pour le logging
    status_code = 200
    error_message = None
    error_type = None
    prediction_score = None
    prediction_class = None
    inference_time_ms = 0
    n_features = len(request.features)
    missing_values_count = sum(1 for v in request.features.values() if v is None or (isinstance(v, float) and v != v))

    try:
        # Vérifier si le modèle est chargé
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Mesurer le temps d'inférence
        inference_start = time.time()
        result = model_loader.predict(request.features)
        inference_time_ms = (time.time() - inference_start) * 1000

        # Extraire les résultats
        prediction_score = result["prediction_score"]
        prediction_class = 1 if prediction_score >= DEFAULT_THRESHOLD else 0

        # Calculer la latence totale
        latency_ms = (time.time() - start_time) * 1000

        # Logger dans la base de données
        log_entry = PredictionLog(
            request_id=request_id,
            timestamp=timestamp,
            model_version=model_loader.model_version or "unknown",
            api_version=API_VERSION,
            environment="dev",  # TODO: récupérer depuis env variable
            input_features=request.features,
            n_features=n_features,
            missing_values_count=missing_values_count,
            schema_version="v1.0",
            prediction_score=prediction_score,
            prediction_class=prediction_class,
            threshold=DEFAULT_THRESHOLD,
            status_code=status_code,
            latency_ms=latency_ms,
            inference_time_ms=inference_time_ms,
            error_message=error_message,
            error_type=error_type
        )
        db.add(log_entry)
        db.commit()

        # Retourner la réponse
        return PredictionResponse(
            request_id=request_id,
            timestamp=timestamp,
            prediction_score=prediction_score,
            prediction_class=prediction_class,
            threshold=DEFAULT_THRESHOLD,
            model_version=model_loader.model_version or "unknown",
            api_version=API_VERSION,
            inference_time_ms=inference_time_ms
        )

    except HTTPException as he:
        status_code = he.status_code
        error_message = he.detail
        error_type = "HTTPException"
        raise

    except Exception as e:
        status_code = 500
        error_message = str(e)
        error_type = type(e).__name__

        # Logger l'erreur
        latency_ms = (time.time() - start_time) * 1000
        log_entry = PredictionLog(
            request_id=request_id,
            timestamp=timestamp,
            model_version=model_loader.model_version or "unknown",
            api_version=API_VERSION,
            environment="dev",
            input_features=request.features,
            n_features=n_features,
            missing_values_count=missing_values_count,
            schema_version="v1.0",
            prediction_score=0.0,  # Valeur par défaut en cas d'erreur
            prediction_class=None,
            threshold=DEFAULT_THRESHOLD,
            status_code=status_code,
            latency_ms=latency_ms,
            inference_time_ms=inference_time_ms,
            error_message=error_message,
            error_type=error_type
        )
        db.add(log_entry)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        from sqlalchemy import func

        # Nombre total de prédictions
        total_predictions = db.query(func.count(PredictionLog.id)).scalar()

        if total_predictions == 0:
            return {
                "total_predictions": 0,
                "message": "No predictions logged yet"
            }

        # Statistiques techniques
        avg_latency = db.query(func.avg(PredictionLog.latency_ms)).scalar()
        avg_inference_time = db.query(func.avg(PredictionLog.inference_time_ms)).scalar()

        # Taux d'erreur
        error_count = db.query(func.count(PredictionLog.id)).filter(
            PredictionLog.status_code != 200
        ).scalar()
        error_rate = (error_count / total_predictions) * 100 if total_predictions > 0 else 0

        # Distribution des classes
        class_0_count = db.query(func.count(PredictionLog.id)).filter(
            PredictionLog.prediction_class == 0
        ).scalar()
        class_1_count = db.query(func.count(PredictionLog.id)).filter(
            PredictionLog.prediction_class == 1
        ).scalar()

        # Score moyen
        avg_score = db.query(func.avg(PredictionLog.prediction_score)).scalar()

        # Période
        period_start = db.query(func.min(PredictionLog.timestamp)).scalar()
        period_end = db.query(func.max(PredictionLog.timestamp)).scalar()

        return {
            "total_predictions": total_predictions,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0,
            "avg_inference_time_ms": round(avg_inference_time, 2) if avg_inference_time else 0,
            "error_rate": round(error_rate, 2),
            "predictions_by_class": {
                0: class_0_count or 0,
                1: class_1_count or 0
            },
            "avg_prediction_score": round(avg_score, 4) if avg_score else 0,
            "period_start": period_start,
            "period_end": period_end
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stats: {str(e)}"
        )


@router.get("/model/features")
async def get_model_features():
    try:
        features = model_loader.get_expected_features()

        if features is None:
            return {
                "message": "Feature names not available",
                "note": "The model may not have been trained with feature names"
            }

        return {
            "total_features": len(features),
            "features": features,
            "note": "You can provide a partial set of features. Missing features will be filled with default value (0)"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching features: {str(e)}"
        )
