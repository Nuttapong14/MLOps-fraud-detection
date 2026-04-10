"""Unit tests for src/features/prepare.py"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.prepare import engineer_features, split


class TestEngineerFeatures:
    def test_adds_transaction_id(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert "transaction_id" in result.columns

    def test_adds_event_timestamp(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert "event_timestamp" in result.columns
        assert result["event_timestamp"].dtype == pl.Datetime

    def test_hour_of_day_range(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_amount_log_non_negative(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert result["amount_log"].min() >= 0.0

    def test_drops_uppercase_v_columns(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        uppercase_v_cols = [c for c in result.columns if c.startswith("V")]
        assert len(uppercase_v_cols) == 0, f"Found uppercase V cols: {uppercase_v_cols}"

    def test_lowercase_v_columns_present(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        for i in range(1, 29):
            assert f"v{i}" in result.columns, f"Missing column v{i}"

    def test_no_nulls(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        null_counts = result.null_count().row(0)
        assert all(c == 0 for c in null_counts), f"Nulls found: {result.null_count()}"

    def test_row_count_preserved(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert len(result) == len(sample_raw_df)

    def test_amount_clip_applied(self, default_params):
        """Amounts above clip_upper should be clipped before log transform."""
        df = pl.DataFrame({"Time": [0.0], "Amount": [99999.0], "Class": [0],
                           **{f"V{i}": [0.0] for i in range(1, 29)}})
        clip = default_params["features"]["amount_clip_upper"]
        result = df.lazy().pipe(engineer_features, default_params).collect()
        import math
        expected_log = math.log1p(clip)
        assert abs(result["amount_log"][0] - expected_log) < 1e-6


class TestRollingFeatures:
    def test_rolling_count_non_negative(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert result["rolling_count_1h"].min() >= 0

    def test_rolling_amount_non_negative(self, sample_raw_df, default_params):
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        assert result["rolling_amount_1h"].min() >= 0.0

    def test_rolling_count_excludes_current_row(self, default_params):
        """First transaction in time should have rolling_count_1h = 0 (no preceding rows)."""
        # Single row: no preceding window → count should be 0
        df = pl.DataFrame({"Time": [0.0], "Amount": [50.0], "Class": [0],
                           **{f"V{i}": [0.0] for i in range(1, 29)}})
        result = df.lazy().pipe(engineer_features, default_params).collect()
        assert result["rolling_count_1h"][0] == 0

    def test_rolling_sorted_by_timestamp(self, sample_raw_df, default_params):
        """Output must be sorted by event_timestamp (rolling requires this)."""
        result = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        timestamps = result["event_timestamp"].to_list()
        assert timestamps == sorted(timestamps)


class TestSplit:
    def test_split_sizes(self, sample_raw_df, default_params):
        processed = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        train, val, test = split(processed, default_params)
        assert len(train) + len(val) + len(test) == len(processed)

    def test_no_overlap(self, sample_raw_df, default_params):
        processed = sample_raw_df.lazy().pipe(engineer_features, default_params).collect()
        train, val, test = split(processed, default_params)
        train_ids = set(train["transaction_id"].to_list())
        val_ids = set(val["transaction_id"].to_list())
        test_ids = set(test["transaction_id"].to_list())
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
