"""
Phase 2 — Test Set Evaluation
Loads the latest Production model from MLflow and evaluates on held-out test set.

Run via DVC:  dvc repro evaluate
Run directly: python src/training/evaluate.py [--min-roc-auc 0.90]
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent.parent
PARAMS_FILE = ROOT / "params.yaml"
METRICS_OUT = ROOT / "data" / "metrics" / "eval_report.json"


def load_latest_model(model_name: str):
    """Load latest model version from MLflow registry (any stage)."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'. Run train.py first.")

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model_uri = f"models:/{model_name}/{latest.version}"
    print(f"[evaluate] Loading model: {model_name} v{latest.version} ({model_uri})")
    return mlflow.sklearn.load_model(model_uri), latest


def main(min_roc_auc: float | None = None) -> None:
    params = yaml.safe_load(open(PARAMS_FILE))
    target = params["prepare"]["target_col"]
    model_name = params["serving"]["model_name"]
    threshold = min_roc_auc or params["thresholds"]["min_roc_auc"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    drop_cols = ["transaction_id", "event_timestamp", "Time", target]
    feature_cols = [c for c in test.columns if c not in drop_cols]
    X_test, y_test = test[feature_cols], test[target]

    model, version = load_latest_model(model_name)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > params["serving"]["fraud_threshold"]).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    report = {
        "model_name": model_name,
        "model_version": version.version,
        "test_rows": len(X_test),
        "fraud_count": int(y_test.sum()),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "passed_threshold": roc_auc >= threshold,
    }

    print(f"\n[evaluate] Test Set Results")
    print(f"  ROC-AUC  : {roc_auc:.4f}  (threshold: >={threshold})")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  F1       : {report['test_f1']:.4f}")
    print(f"  Precision: {report['test_precision']:.4f}")
    print(f"  Recall   : {report['test_recall']:.4f}")
    print(f"\n{classification_report(y_test, pred, target_names=['legit', 'fraud'])}")

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(report, f, indent=2)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics({
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
            "test_f1": report["test_f1"],
        })

    if not report["passed_threshold"]:
        raise SystemExit(
            f"[evaluate] FAILED: test_roc_auc={roc_auc:.4f} < threshold {threshold}"
        )

    print(f"[evaluate] ✓ Model passed all thresholds. Report: {METRICS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-roc-auc", type=float, default=None)
    args = parser.parse_args()
    main(min_roc_auc=args.min_roc_auc)
