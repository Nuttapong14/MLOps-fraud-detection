"""
Prediction Logger — stores inference requests to SQLite for drift monitoring.
In production this would be Postgres or BigQuery, but SQLite works for local dev.

The drift monitor reads from this DB to get "current window" data.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "predictions.db"


def init_db() -> None:
    """Create predictions table if it doesn't exist."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER,
                timestamp TEXT NOT NULL,
                amount REAL,
                hour_of_day INTEGER,
                amount_log REAL,
                amount_zscore REAL,
                rolling_count_1h INTEGER,
                rolling_amount_1h REAL,
                fraud_probability REAL,
                is_fraud INTEGER,
                model_version TEXT
            )
        """)


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    transaction_id: int,
    amount: float,
    hour_of_day: int,
    amount_log: float,
    amount_zscore: float,
    rolling_count_1h: int,
    rolling_amount_1h: float,
    fraud_probability: float,
    is_fraud: bool,
    model_version: str,
) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO predictions
              (transaction_id, timestamp, amount, hour_of_day, amount_log,
               amount_zscore, rolling_count_1h, rolling_amount_1h,
               fraud_probability, is_fraud, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id,
            datetime.now(tz=timezone.utc).isoformat(),
            amount, hour_of_day, amount_log, amount_zscore,
            rolling_count_1h, rolling_amount_1h,
            fraud_probability, int(is_fraud), model_version,
        ))


def load_recent_predictions(hours: int = 24) -> "pd.DataFrame":
    import pandas as pd
    with get_conn() as conn:
        return pd.read_sql(f"""
            SELECT * FROM predictions
            WHERE timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        """, conn)


# Auto-init on import
init_db()
