"""
Phase 1 — Data Preparation
Loads raw Kaggle CSV, engineers features with Polars, splits into train/val/test.

Run via DVC:  dvc repro prepare
Run directly: python src/features/prepare.py
"""

import json
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw" / "creditcard.csv"
DATA_OUT = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def engineer_features(df: pl.LazyFrame, params: dict) -> pl.LazyFrame:
    """
    Add engineered features on top of raw V1–V28 PCA columns.

    Features added:
    - transaction_id   : unique row id (required by Feast entity)
    - event_timestamp  : required by Feast offline store
    - hour_of_day      : transaction hour (0–23), proxy for time-of-day fraud pattern
    - amount_log       : log1p(Amount) — normalises the heavily right-skewed distribution
    - amount_zscore    : z-score of Amount — scale-invariant amount signal
    - rolling_count_1h : global count of transactions in the preceding 1-hour window
    - rolling_amount_1h: global sum of Amount in the preceding 1-hour window

    Rolling strategy: GLOBAL window (not per-card) because the Kaggle dataset
    anonymizes card IDs into PCA features V1–V28, making per-card grouping impossible.
    Global rolling still captures temporal fraud bursts (e.g., coordinated attacks).
    """
    clip_upper = params["features"]["amount_clip_upper"]

    # Pass 1: scalar features — no sort dependency
    df = (
        df.with_row_index("transaction_id")
        .with_columns([
            (pl.lit("2024-01-01").str.to_datetime() +
             pl.duration(seconds=pl.col("Time").cast(pl.Int64))
             ).alias("event_timestamp"),
            (pl.col("Time").cast(pl.Int64) % 86400 // 3600
             ).cast(pl.Int64).alias("hour_of_day"),
            pl.col("Amount").clip(upper_bound=clip_upper).log1p().alias("amount_log"),
            (
                (pl.col("Amount") - pl.col("Amount").mean()) /
                (pl.col("Amount").std() + 1e-8)
            ).alias("amount_zscore"),
            *[pl.col(f"V{i}").alias(f"v{i}") for i in range(1, 29)],
        ])
        .drop([f"V{i}" for i in range(1, 29)])
    )

    # Pass 2: rolling velocity — requires sort by event_timestamp
    # Polars 1.x API: rolling_sum_by(by_col, window_size)
    # closed="left" excludes the current row (preceding 1h window only)
    # fill_null(0) handles the first rows where no 1h history exists yet
    df = df.sort("event_timestamp").with_columns([
        pl.col("Amount")
          .rolling_sum_by("event_timestamp", window_size="1h", closed="left")
          .fill_null(0.0)
          .alias("rolling_amount_1h"),
        pl.col("Amount").is_not_null().cast(pl.Int64)
          .rolling_sum_by("event_timestamp", window_size="1h", closed="left")
          .fill_null(0)
          .alias("rolling_count_1h"),
    ])

    return df


def split(df: pl.DataFrame, params: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Stratified-ish split: shuffle then slice by index."""
    p = params["prepare"]
    total = len(df)
    test_n = int(total * p["test_size"])
    val_n = int(total * p["val_size"])

    # Shuffle with fixed seed for reproducibility
    df = df.sample(fraction=1.0, shuffle=True, seed=p["random_state"])

    test = df.slice(0, test_n)
    val = df.slice(test_n, val_n)
    train = df.slice(test_n + val_n)

    return train, val, test


def main() -> None:
    params = load_params()
    target = params["prepare"]["target_col"]

    print(f"[prepare] Loading {DATA_RAW} ...")
    df = (
        pl.scan_csv(
            DATA_RAW,
            schema_overrides={"Time": pl.Float64, "Amount": pl.Float64},
            infer_schema_length=10000,
        )
        .pipe(engineer_features, params)
        .collect()
    )

    print(f"[prepare] Total rows: {len(df):,}  |  Fraud rate: "
          f"{df[target].mean():.4%}")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = split(df, params)

    train.write_parquet(DATA_OUT / "train.parquet")
    val.write_parquet(DATA_OUT / "val.parquet")
    test.write_parquet(DATA_OUT / "test.parquet")

    # Write split stats for DVC metrics
    stats = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "fraud_rate_train": float(train[target].mean()),
        "fraud_rate_test": float(test[target].mean()),
        "feature_count": len(df.columns),
    }
    (ROOT / "data" / "metrics").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "metrics" / "prepare_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[prepare] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"[prepare] Saved parquet files to {DATA_OUT}/")


if __name__ == "__main__":
    main()
