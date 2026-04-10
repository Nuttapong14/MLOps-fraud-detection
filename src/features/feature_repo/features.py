from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, UnixTimestamp

# ── Entity ────────────────────────────────────────────────────────────────────

transaction = Entity(
    name="transaction_id",
    description="Unique ID for each credit card transaction",
    join_keys=["transaction_id"],
)

# ── Data Source ───────────────────────────────────────────────────────────────

transaction_source = FileSource(
    name="transaction_source",
    path="data/processed/train.parquet",   # DVC-managed parquet
    timestamp_field="event_timestamp",
)

# ── Feature View ──────────────────────────────────────────────────────────────

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=[
        # Engineered features (from prepare.py)
        Field(name="amount_log", dtype=Float64),
        Field(name="amount_zscore", dtype=Float64),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="rolling_count_1h", dtype=Int64),
        Field(name="rolling_amount_1h", dtype=Float64),
        # PCA features from Kaggle dataset (V1–V28)
        *[Field(name=f"v{i}", dtype=Float64) for i in range(1, 29)],
    ],
    source=transaction_source,
    online=True,
    tags={"team": "mlops", "use_case": "fraud_detection"},
)
