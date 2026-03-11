from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    # Traçabilité
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    api_version = Column(String(20), nullable=False)
    environment = Column(String(20), default="dev", nullable=False)

    # Inputs (stockés en JSON pour flexibilité)
    input_features = Column(JSON, nullable=False)
    n_features = Column(Integer, nullable=False)
    missing_values_count = Column(Integer, default=0)
    schema_version = Column(String(20), default="v1.0")

    # Outputs du modèle
    prediction_score = Column(Float, nullable=False)
    prediction_class = Column(Integer, nullable=True)
    threshold = Column(Float, default=0.5)

    # Métriques techniques
    status_code = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    inference_time_ms = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<PredictionLog(request_id={self.request_id}, timestamp={self.timestamp})>"
