"""
Phase 3 — BentoML Service Definition
Packages the MLflow model + dependencies into a portable BentoML Service.

Build + containerize:
  bentoml build
  bentoml containerize fraud_detector:latest \
    --platform linux/amd64 \
    -t asia-southeast1-docker.pkg.dev/$PROJECT/mlops/fraud-detector:$VERSION

Run locally:
  bentoml serve src.serving.service:svc --port 3000
"""

import os

import bentoml
import mlflow.sklearn
import numpy as np
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel

from src.serving.model_loader import get_feature_columns, get_params, load_production_model

# ── BentoML Runner ────────────────────────────────────────────────────────────
# Wraps the MLflow model as a BentoML runner (async-capable, batchable)

params = get_params()
MODEL_NAME = params["serving"]["model_name"]
FRAUD_THRESHOLD = params["serving"]["fraud_threshold"]

# Save MLflow model into BentoML model store (one-time operation)
# In CI/CD this would be done during the build step
try:
    fraud_model = bentoml.mlflow.get(f"{MODEL_NAME}:latest")
except bentoml.exceptions.NotFound:
    print(f"[service] Importing {MODEL_NAME} from MLflow into BentoML store...")
    fraud_model = bentoml.mlflow.import_model(
        MODEL_NAME,
        model_uri=f"models:/{MODEL_NAME}@Production",
        signatures={"predict_proba": {"batchable": True}},
    )

fraud_runner = fraud_model.to_runner()

# ── Service Definition ────────────────────────────────────────────────────────

svc = bentoml.Service("fraud_detector", runners=[fraud_runner])


class TransactionInput(BaseModel):
    transaction_id: int
    v_features: list[float]        # V1–V28
    amount_log: float
    amount_zscore: float
    hour_of_day: int
    rolling_count_1h: int = 0
    rolling_amount_1h: float = 0.0


class FraudPrediction(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_fraud: bool


@svc.api(input=JSON(pydantic_model=TransactionInput),
         output=JSON(pydantic_model=FraudPrediction))
async def predict(transaction: TransactionInput) -> FraudPrediction:
    feature_cols = get_feature_columns()
    feature_values = (
        transaction.v_features +
        [transaction.amount_log,
         transaction.amount_zscore,
         float(transaction.hour_of_day),
         float(transaction.rolling_count_1h),
         transaction.rolling_amount_1h]
    )
    X = pd.DataFrame([feature_values], columns=feature_cols)

    # Async runner call — non-blocking, BentoML handles batching
    proba = await fraud_runner.predict_proba.async_run(X)
    fraud_prob = float(proba[0][1])

    return FraudPrediction(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(fraud_prob, 6),
        is_fraud=fraud_prob > FRAUD_THRESHOLD,
    )
