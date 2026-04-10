"""
Phase 4 — Data & Model Drift Monitor using Evidently AI
Compares recent predictions against training distribution baseline.

Run manually:  python src/monitoring/drift_monitor.py
Run via Argo:  triggered by Alertmanager webhook (see pipelines/events/)

Drift is measured using Population Stability Index (PSI):
  PSI < 0.1  → No significant drift
  PSI 0.1–0.2 → Moderate drift, monitor closely
  PSI > 0.2  → Significant drift → trigger retrain
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PARAMS_FILE = ROOT / "params.yaml"
REFERENCE_DATA = ROOT / "data" / "processed" / "train.parquet"
REPORTS_DIR = ROOT / "data" / "drift_reports"

PUSHGATEWAY_URL = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091")
ARGO_WEBHOOK_URL = os.getenv("ARGO_WEBHOOK_URL", "http://argo-events:12000/trigger-retrain")

# Features to monitor for drift (subset — most interpretable ones)
MONITORED_FEATURES = [
    "amount", "hour_of_day", "amount_log",
    "amount_zscore", "rolling_count_1h", "rolling_amount_1h",
]


def load_reference() -> pd.DataFrame:
    """Load training data as drift baseline."""
    df = pd.read_parquet(REFERENCE_DATA)
    # Rename to match prediction log column names
    return df.rename(columns={"Amount": "amount"})[MONITORED_FEATURES + ["Class"]]


def load_current(hours: int = 1) -> pd.DataFrame:
    """Load recent predictions from the prediction log DB."""
    sys.path.insert(0, str(ROOT))
    from src.monitoring.prediction_logger import load_recent_predictions
    df = load_recent_predictions(hours=hours)
    if df.empty:
        return df
    df = df.rename(columns={"is_fraud": "Class"})
    return df[MONITORED_FEATURES + ["Class", "fraud_probability"]]


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift between two Series.

    PSI = sum((P_current - P_reference) * ln(P_current / P_reference))

    Thresholds:
      < 0.1  : stable
      0.1–0.2: moderate shift
      > 0.2  : significant shift → retrain
    """
    # Create bins from reference distribution
    breakpoints = pd.cut(reference, bins=bins, retbins=True)[1]
    breakpoints[0] = -float("inf")
    breakpoints[-1] = float("inf")

    ref_counts = pd.cut(reference, bins=breakpoints).value_counts(sort=False) + 1e-4
    cur_counts = pd.cut(current, bins=breakpoints).value_counts(sort=False) + 1e-4

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = ((cur_pct - ref_pct) * (cur_pct / ref_pct).apply(lambda x: x if x > 0 else 1e-10).apply(
        __import__("math").log
    )).sum()
    return float(abs(psi))


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run full Evidently drift report for detailed analysis."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        # Evidently 0.7: metrics is a list of result objects with direct attributes
        metric_result = report.metrics[0] if report.metrics else None
        drift_share = float(getattr(metric_result, "drift_share", 0.0))
        dataset_drift = drift_share > 0.5

        return {"dataset_drift": dataset_drift, "drift_share": drift_share}
    except Exception as e:
        print(f"[drift] Evidently report failed: {e}")
        return {"dataset_drift": False, "drift_share": 0.0}


def push_to_prometheus(metrics: dict) -> None:
    """Push drift metrics to Prometheus Pushgateway."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        for feature, psi in metrics.get("feature_psi", {}).items():
            g = Gauge(f"drift_psi_{feature}", f"PSI drift score for {feature}",
                      registry=registry)
            g.set(psi)

        max_psi = Gauge("drift_psi_max", "Maximum PSI across all features", registry=registry)
        max_psi.set(metrics.get("max_psi", 0.0))

        push_to_gateway(PUSHGATEWAY_URL, job="drift-monitor", registry=registry)
        print(f"[drift] Pushed metrics to Prometheus Pushgateway: {PUSHGATEWAY_URL}")
    except Exception as e:
        print(f"[drift] Pushgateway push failed (OK if not running): {e}")


def trigger_retrain(reason: str) -> None:
    """POST to Argo Events webhook to trigger retraining workflow."""
    try:
        import urllib.request
        payload = json.dumps({"reason": reason, "timestamp": datetime.now(tz=timezone.utc).isoformat()})
        req = urllib.request.Request(
            ARGO_WEBHOOK_URL,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print(f"[drift] ✓ Retrain triggered via Argo Events: {reason}")
    except Exception as e:
        print(f"[drift] Argo webhook unavailable (OK locally): {e}")
        print(f"[drift] Would trigger retrain: {reason}")


def main(window_hours: int = 1, min_samples: int = 50) -> dict:
    import yaml
    params = yaml.safe_load(open(PARAMS_FILE))
    psi_threshold = params["thresholds"]["drift_psi_threshold"]

    print(f"[drift] Loading reference data from {REFERENCE_DATA} ...")
    reference = load_reference()

    print(f"[drift] Loading current window ({window_hours}h) from prediction log ...")
    current = load_current(hours=window_hours)

    if len(current) < min_samples:
        print(f"[drift] Only {len(current)} predictions in window "
              f"(need {min_samples}). Skipping drift check.")
        return {"status": "insufficient_data", "count": len(current)}

    print(f"[drift] Computing PSI for {len(MONITORED_FEATURES)} features ...")
    feature_psi = {}
    for feature in MONITORED_FEATURES:
        if feature in current.columns and feature in reference.columns:
            psi = compute_psi(reference[feature].dropna(), current[feature].dropna())
            feature_psi[feature] = round(psi, 4)
            status = "🔴 DRIFT" if psi > psi_threshold else ("🟡 WARN" if psi > 0.1 else "🟢 OK")
            print(f"  {feature:25s}: PSI={psi:.4f}  {status}")

    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    drift_detected = max_psi > psi_threshold

    # Run full Evidently report for HTML artifact
    evidently_result = run_evidently_report(
        reference[MONITORED_FEATURES].rename(columns={"amount": "Amount"}),
        current[MONITORED_FEATURES].rename(columns={"amount": "Amount"}),
    )

    result = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "window_hours": window_hours,
        "current_samples": len(current),
        "feature_psi": feature_psi,
        "max_psi": max_psi,
        "drift_detected": drift_detected,
        "psi_threshold": psi_threshold,
        "evidently": evidently_result,
    }

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "latest_drift.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    push_to_prometheus(result)

    if drift_detected:
        reason = f"max_psi={max_psi:.4f} > threshold={psi_threshold} on features: " \
                 f"{[k for k, v in feature_psi.items() if v > psi_threshold]}"
        print(f"\n[drift] 🚨 DRIFT DETECTED: {reason}")
        trigger_retrain(reason)
    else:
        print(f"\n[drift] ✓ No significant drift detected (max_psi={max_psi:.4f})")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()
    result = main(window_hours=args.window_hours, min_samples=args.min_samples)
    print(f"\n[drift] Report: {REPORTS_DIR / 'latest_drift.json'}")
