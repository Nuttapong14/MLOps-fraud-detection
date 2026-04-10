"""
Phase 2 — Model Training
XGBoost + Optuna hyperparameter search, tracked in MLflow.

Run via DVC:  dvc repro train
Run directly: python src/training/train.py [--n-trials N] [--experiment-name NAME]
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
PARAMS_FILE = ROOT / "params.yaml"
METRICS_OUT = ROOT / "data" / "metrics" / "train_report.json"


def load_data(params: dict) -> tuple:
    target = params["prepare"]["target_col"]
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    val = pd.read_parquet(ROOT / "data" / "processed" / "val.parquet")

    # Drop non-feature columns
    drop_cols = ["transaction_id", "event_timestamp", "Time", target]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train, y_train = train[feature_cols], train[target]
    X_val, y_val = val[feature_cols], val[target]

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[train] Features: {len(feature_cols)}  |  "
          f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  "
          f"scale_pos_weight: {pos_weight:.1f}")

    return X_train, y_train, X_val, y_val, feature_cols, pos_weight


def make_objective(X_train, y_train, X_val, y_val, pos_weight, parent_run_id):
    """Optuna objective — each trial is a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": pos_weight,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run(nested=True, tags={"trial": str(trial.number)}):
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            proba = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, proba)
            roc_auc = roc_auc_score(y_val, proba)

            mlflow.log_params(params)
            mlflow.log_metrics({"val_pr_auc": pr_auc, "val_roc_auc": roc_auc})

        return pr_auc

    return objective


def train_final_model(best_params, X_train, y_train, X_val, y_val, pos_weight):
    """Retrain with best params on train+val combined."""
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    params = {**best_params, "scale_pos_weight": pos_weight,
              "random_state": 42, "n_jobs": -1}
    model = XGBClassifier(**params)
    model.fit(X_all, y_all, verbose=False)
    return model


def compute_metrics(model, X, y, prefix: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y, proba)),
        f"{prefix}_pr_auc": float(average_precision_score(y, proba)),
        f"{prefix}_f1": float(f1_score(y, pred, zero_division=0)),
        f"{prefix}_precision": float(precision_score(y, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, pred, zero_division=0)),
    }


def main(n_trials: int | None = None, experiment_name: str = "fraud-detection") -> None:
    params = yaml.safe_load(open(PARAMS_FILE))
    train_params = params["train"]
    thresholds = params["thresholds"]
    n_trials = n_trials or train_params["n_trials"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_val, y_val, feature_cols, pos_weight = load_data(params)

    with mlflow.start_run(run_name=f"xgboost-optuna-{n_trials}trials") as run:
        print(f"[train] MLflow run: {run.info.run_id}")
        print(f"[train] Starting Optuna search ({n_trials} trials) ...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            make_objective(X_train, y_train, X_val, y_val, pos_weight, run.info.run_id),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_params
        print(f"[train] Best val PR-AUC: {study.best_value:.4f}")
        print(f"[train] Best params: {best}")

        # Final model on train+val
        model = train_final_model(best, X_train, y_train, X_val, y_val, pos_weight)

        val_metrics = compute_metrics(model, X_val, y_val, "val")
        mlflow.log_params({**best, "n_trials": n_trials, "n_features": len(feature_cols)})
        mlflow.log_metrics({**val_metrics, "best_optuna_pr_auc": study.best_value})
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=params["serving"]["model_name"],
            input_example=X_val.iloc[:1],
        )

        # Gate: fail if below threshold
        if val_metrics["val_roc_auc"] < thresholds["min_roc_auc"]:
            raise SystemExit(
                f"[train] FAILED: val_roc_auc={val_metrics['val_roc_auc']:.4f} "
                f"< threshold {thresholds['min_roc_auc']}"
            )

        report = {
            "run_id": run.info.run_id,
            **val_metrics,
            "best_params": best,
            "n_trials": n_trials,
        }
        METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_OUT, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[train] ✓ val_roc_auc={val_metrics['val_roc_auc']:.4f}  "
              f"val_pr_auc={val_metrics['val_pr_auc']:.4f}")
        print(f"[train] Model registered as '{params['serving']['model_name']}' in MLflow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default="fraud-detection")
    args = parser.parse_args()
    main(n_trials=args.n_trials, experiment_name=args.experiment_name)
