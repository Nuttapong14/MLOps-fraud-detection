"""
Phase 1 — Data Validation with Pandera
Validates train.parquet schema, value ranges, and data quality.
Fails the DVC pipeline if data doesn't meet expectations.

Run via DVC:  dvc repro validate
Run directly: python src/features/validate.py
"""

import json
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

ROOT = Path(__file__).parent.parent.parent
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
METRICS_OUT = ROOT / "data" / "metrics" / "validation_report.json"

# ── Schema Definition ─────────────────────────────────────────────────────────

TRANSACTION_SCHEMA = DataFrameSchema(
    columns={
        "transaction_id": Column(int, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
        "Amount": Column(float, Check.ge(0.0), nullable=False),
        "hour_of_day": Column(int, Check.in_range(0, 23), nullable=False),
        "amount_log": Column(float, Check.ge(0.0), nullable=False),
        "amount_zscore": Column(float, nullable=False),
        "rolling_count_1h": Column(int, Check.ge(0), nullable=False),
        "rolling_amount_1h": Column(float, Check.ge(0.0), nullable=False),
        # V1–V28 PCA features: no hard range (PCA output varies)
        **{
            f"v{i}": Column(float, nullable=False)
            for i in range(1, 29)
        },
    },
    checks=[
        # Dataset-level checks
        Check(lambda df: df["Class"].mean() < 0.05,
              error="Fraud rate unexpectedly high (>5%) — possible label leakage"),
        Check(lambda df: len(df) > 1000,
              error="Training set too small (<1000 rows)"),
    ],
    coerce=True,
)


def main() -> None:
    print(f"[validate] Reading {TRAIN_PARQUET} ...")
    df = pd.read_parquet(TRAIN_PARQUET)

    errors = []
    passed = True

    try:
        TRANSACTION_SCHEMA.validate(df, lazy=True)
        print("[validate] ✓ Schema validation passed")
    except pa.errors.SchemaErrors as e:
        passed = False
        errors = e.failure_cases.to_dict(orient="records")
        print(f"[validate] ✗ Schema errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err}")

    # Additional data quality checks
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    if null_cols:
        passed = False
        errors.append({"check": "null_values", "columns": null_cols})
        print(f"[validate] ✗ Null values found: {null_cols}")

    fraud_rate = float(df["Class"].mean())
    duplicate_count = int(df.duplicated(subset=["transaction_id"]).sum())

    report = {
        "passed": passed,
        "rows": len(df),
        "columns": len(df.columns),
        "fraud_rate": fraud_rate,
        "duplicate_transaction_ids": duplicate_count,
        "null_columns": null_cols,
        "schema_errors": errors,
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[validate] Fraud rate: {fraud_rate:.4%}  |  Rows: {len(df):,}")
    print(f"[validate] Report saved to {METRICS_OUT}")

    if not passed:
        raise SystemExit(f"[validate] FAILED — fix data issues before training. See {METRICS_OUT}")


if __name__ == "__main__":
    main()
