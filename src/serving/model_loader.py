"""
Shared model loading logic for both BentoML and FastAPI serving.
Loads the Production model from MLflow registry with a local cache.
"""

import os
from functools import lru_cache
from pathlib import Path

import mlflow.sklearn
import yaml

ROOT = Path(__file__).parent.parent.parent
PARAMS_FILE = ROOT / "params.yaml"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


@lru_cache(maxsize=1)
def get_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_production_model():
    """Load Production model from MLflow. Cached — only loads once per process."""
    params = get_params()
    model_name = params["serving"]["model_name"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Try alias first (set by register.py), fall back to latest version
    try:
        model_uri = f"models:/{model_name}@Production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[model_loader] Loaded {model_name}@Production from {MLFLOW_TRACKING_URI}")
    except Exception:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{model_name}/{latest.version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[model_loader] Loaded {model_name} v{latest.version} (fallback)")

    return model


def get_feature_columns() -> list[str]:
    """Return the feature columns in the exact order the model was trained on.

    Order matches train.py: [c for c in train.columns if c not in drop_cols]
    drop_cols = ["transaction_id", "event_timestamp", "Time", "Class"]
    Resulting order: Amount, hour_of_day, amount_log, amount_zscore, v1-v28,
                     rolling_amount_1h, rolling_count_1h
    """
    v_cols = [f"v{i}" for i in range(1, 29)]
    return (
        ["Amount", "hour_of_day", "amount_log", "amount_zscore"]
        + v_cols
        + ["rolling_amount_1h", "rolling_count_1h"]
    )
