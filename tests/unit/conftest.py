"""Shared fixtures for unit tests."""

import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def sample_raw_df() -> pl.DataFrame:
    """Minimal Kaggle-schema DataFrame for testing prepare.py."""
    import numpy as np

    n = 1100  # must be > 1000 to pass Pandera dataset-level size check
    rng = np.random.default_rng(42)
    data = {
        "Time": rng.uniform(0, 172800, n).tolist(),   # 2 days of seconds
        "Amount": rng.exponential(scale=80, size=n).tolist(),
        "Class": ([1] * 40) + ([0] * (n - 40)),      # ~3.6% fraud (< 5% schema threshold)
        **{f"V{i}": rng.normal(0, 1, n).tolist() for i in range(1, 29)},
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_processed_df(sample_raw_df, default_params) -> pd.DataFrame:
    """Processed DataFrame (Pandas) for validate.py tests."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.features.prepare import engineer_features
    return (
        sample_raw_df.lazy()
        .pipe(engineer_features, default_params)
        .collect()
        .to_pandas()
    )


@pytest.fixture
def default_params() -> dict:
    return {
        "prepare": {"test_size": 0.2, "val_size": 0.1, "random_state": 42, "target_col": "Class"},
        "features": {"amount_clip_upper": 10000.0, "rolling_window_seconds": 3600},
    }
