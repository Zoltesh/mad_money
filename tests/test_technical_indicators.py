"""Tests for technical indicators ordering and timestamp requirements."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.technical_indicators.core import add_indicator


def _sample_ohlcv_df(n: int = 12) -> pl.DataFrame:
    """Build deterministic timestamped OHLCV test data."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


def test_add_indicator_requires_timestamp_column() -> None:
    """Reject inputs without timestamp to enforce chronological semantics."""
    df = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="requires a 'timestamp' column"):
        add_indicator(df, "rsi", timeframe="5m", base_timeframe="5m", timeperiod=3)


def test_add_indicator_sorts_unsorted_input_before_computing() -> None:
    """Unsorted input should produce the same result as sorted input."""
    sorted_df = _sample_ohlcv_df()
    unsorted_df = sorted_df.reverse()

    out_from_sorted = add_indicator(
        sorted_df, "rsi", timeframe="15m", base_timeframe="5m", timeperiod=3
    )
    out_from_unsorted = add_indicator(
        unsorted_df, "rsi", timeframe="15m", base_timeframe="5m", timeperiod=3
    )

    # Function guarantees chronological calculations and returns chronological rows.
    assert_frame_equal(out_from_unsorted, out_from_sorted)


def test_add_indicator_synthetic_volume_indicator_mfi_raises_value_error() -> None:
    """Synthetic MFI request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported"):
        add_indicator(df, "mfi", timeframe="15m", base_timeframe="5m", timeperiod=14)


def test_add_indicator_synthetic_volume_indicator_ad_raises_value_error() -> None:
    """Synthetic AD request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported"):
        add_indicator(df, "ad", timeframe="15m", base_timeframe="5m")


def test_add_indicator_synthetic_volume_indicator_adosc_raises_value_error() -> None:
    """Synthetic ADOSC request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported"):
        add_indicator(
            df,
            "adosc",
            timeframe="15m",
            base_timeframe="5m",
            fastperiod=3,
            slowperiod=10,
        )


def test_add_indicator_base_timeframe_volume_indicator_succeeds() -> None:
    """Base-timeframe MFI still computes successfully."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(
        df, "mfi", timeframe="5m", base_timeframe="5m", timeperiod=14
    )

    assert "mfi_14_5m" in result.columns


def test_add_indicator_synthetic_timeframe_adx() -> None:
    """Synthetic ADX (OHLC-only) still works after volume guard."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(
        df, "adx", timeframe="15m", base_timeframe="5m", timeperiod=14
    )

    assert "adx_14_15m" in result.columns
