"""
Phase 2 — Model Registry Promotion
Promotes the latest model version to Production stage in MLflow.
Only promotes if test metrics pass the production threshold.

Run directly: python src/training/register.py
"""

import json
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).parent.parent.parent
PARAMS_FILE = ROOT / "params.yaml"
EVAL_REPORT = ROOT / "data" / "metrics" / "eval_report.json"


def main() -> None:
    params = yaml.safe_load(open(PARAMS_FILE))
    model_name = params["serving"]["model_name"]
    min_roc_auc = params["thresholds"]["min_roc_auc"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()

    # Read eval report to confirm model passed threshold
    if not EVAL_REPORT.exists():
        raise RuntimeError("Run evaluate.py first before promoting.")

    with open(EVAL_REPORT) as f:
        eval_report = json.load(f)

    test_roc_auc = eval_report["test_roc_auc"]
    model_version = eval_report["model_version"]

    if test_roc_auc < min_roc_auc:
        raise SystemExit(
            f"[register] Refusing to promote: test_roc_auc={test_roc_auc:.4f} "
            f"< threshold {min_roc_auc}. Retrain and improve the model first."
        )

    # Archive any existing Production versions
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for v in all_versions:
        if v.tags.get("stage") == "Production":
            client.set_model_version_tag(model_name, v.version, "stage", "Archived")
            print(f"[register] Archived previous Production: v{v.version}")

    # Promote latest version to Production
    client.set_model_version_tag(model_name, model_version, "stage", "Production")
    client.set_registered_model_alias(model_name, "Production", model_version)

    print(f"[register] ✓ Promoted {model_name} v{model_version} to Production")
    print(f"[register]   test_roc_auc={test_roc_auc:.4f}  "
          f"test_pr_auc={eval_report['test_pr_auc']:.4f}")
    print(f"[register]   Load with: mlflow.sklearn.load_model('models:/{model_name}@Production')")


if __name__ == "__main__":
    main()
