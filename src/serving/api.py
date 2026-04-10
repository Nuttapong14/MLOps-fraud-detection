"""
Phase 3 — FastAPI Inference Service
Production-ready REST API with Prometheus metrics, health checks, and structured logging.

Run locally:
  uvicorn src.serving.api:app --port 8080 --reload

Docker:
  docker build -t fraud-detector . && docker run -p 8080:8080 fraud-detector
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

from src.monitoring.prediction_logger import log_prediction
from src.serving.model_loader import get_feature_columns, get_params, load_production_model

# ── Prometheus Metrics ────────────────────────────────────────────────────────

PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total prediction requests",
    ["result", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_duration_seconds",
    "End-to-end prediction latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FRAUD_SCORE = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Request / Response Schemas ────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """Input schema — mirrors Feast feature columns."""
    transaction_id: int = Field(..., description="Unique transaction ID")
    amount: float = Field(..., ge=0, description="Transaction amount (USD)")
    hour_of_day: int = Field(..., ge=0, le=23)
    amount_log: float = Field(..., description="log1p(amount) — pre-computed")
    amount_zscore: float = Field(..., description="Z-score of amount")
    rolling_count_1h: int = Field(default=0, ge=0)
    rolling_amount_1h: float = Field(default=0.0, ge=0.0)
    v_features: list[float] = Field(..., min_length=28, max_length=28,
                                     description="V1–V28 PCA features")

    @field_validator("v_features")
    @classmethod
    def check_v_features_length(cls, v: list[float]) -> list[float]:
        if len(v) != 28:
            raise ValueError("v_features must have exactly 28 elements (V1–V28)")
        return v


class PredictionResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_fraud: bool
    fraud_threshold: float
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# ── App Lifecycle ─────────────────────────────────────────────────────────────

MODEL = None
MODEL_VERSION = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    global MODEL, MODEL_VERSION
    MODEL = load_production_model()
    # Extract version from MLflow tags if available
    try:
        import mlflow
        params = get_params()
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{params['serving']['model_name']}'")
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        MODEL_VERSION = f"v{latest.version}"
    except Exception:
        MODEL_VERSION = "v1"
    print(f"[api] Model {MODEL_VERSION} loaded and ready")
    yield
    print("[api] Shutting down")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection powered by XGBoost + MLflow",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
    )


@app.get("/ready")
def ready() -> dict[str, str]:
    """Kubernetes readiness probe — fails if model not loaded."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    start = time.perf_counter()
    params = get_params()
    threshold = params["serving"]["fraud_threshold"]

    # Build feature vector in the exact order the model was trained on:
    # Amount, hour_of_day, amount_log, amount_zscore, v1-v28, rolling_amount_1h, rolling_count_1h
    feature_values = (
        [request.amount,
         float(request.hour_of_day),
         request.amount_log,
         request.amount_zscore]
        + request.v_features                    # v1–v28 (28 values)
        + [request.rolling_amount_1h,
           float(request.rolling_count_1h)]
    )
    feature_cols = get_feature_columns()
    X = pd.DataFrame([feature_values], columns=feature_cols)

    proba = float(MODEL.predict_proba(X)[0][1])
    is_fraud = proba > threshold
    latency_ms = (time.perf_counter() - start) * 1000

    # Record Prometheus metrics
    result_label = "fraud" if is_fraud else "legitimate"
    PREDICTION_COUNTER.labels(result=result_label, model_version=MODEL_VERSION).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    FRAUD_SCORE.observe(proba)

    # Log to SQLite for drift monitoring
    log_prediction(
        transaction_id=request.transaction_id,
        amount=request.amount,
        hour_of_day=request.hour_of_day,
        amount_log=request.amount_log,
        amount_zscore=request.amount_zscore,
        rolling_count_1h=request.rolling_count_1h,
        rolling_amount_1h=request.rolling_amount_1h,
        fraud_probability=proba,
        is_fraud=is_fraud,
        model_version=MODEL_VERSION,
    )

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(proba, 6),
        is_fraud=is_fraud,
        fraud_threshold=threshold,
        latency_ms=round(latency_ms, 2),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: list[TransactionRequest]) -> list[PredictionResponse]:
    """Batch inference — more efficient than N single requests."""
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000")
    return [predict(r) for r in requests]


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "fraud-detection-api",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
    }
