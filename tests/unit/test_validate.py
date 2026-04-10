"""Unit tests for src/features/validate.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.validate import TRANSACTION_SCHEMA


class TestTransactionSchema:
    def test_valid_dataframe_passes(self, sample_processed_df):
        TRANSACTION_SCHEMA.validate(sample_processed_df)

    def test_negative_amount_fails(self, sample_processed_df):
        import pandera as pa
        bad_df = sample_processed_df.copy()
        bad_df.loc[0, "Amount"] = -1.0
        with pytest.raises(pa.errors.SchemaErrors):
            TRANSACTION_SCHEMA.validate(bad_df, lazy=True)

    def test_invalid_class_value_fails(self, sample_processed_df):
        import pandera as pa
        bad_df = sample_processed_df.copy()
        bad_df.loc[0, "Class"] = 2          # only 0 or 1 allowed
        with pytest.raises(pa.errors.SchemaErrors):
            TRANSACTION_SCHEMA.validate(bad_df, lazy=True)

    def test_hour_out_of_range_fails(self, sample_processed_df):
        import pandera as pa
        bad_df = sample_processed_df.copy()
        bad_df.loc[0, "hour_of_day"] = 25   # 0–23 only
        with pytest.raises(pa.errors.SchemaErrors):
            TRANSACTION_SCHEMA.validate(bad_df, lazy=True)

    def test_all_v_columns_present(self, sample_processed_df):
        """Schema should reject DataFrame missing any V feature."""
        import pandera as pa
        bad_df = sample_processed_df.drop(columns=["v1"])
        with pytest.raises(pa.errors.SchemaErrors):
            TRANSACTION_SCHEMA.validate(bad_df, lazy=True)
