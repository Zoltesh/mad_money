"""Tests for standalone local OHLCV range querying."""

from datetime import UTC, datetime

import pytest

from src.data import CoinbaseDataClient
from src.data.ohlcv_query import load_ohlcv_range


def test_load_ohlcv_range_loads_all_when_unbounded(tmp_path, sample_df_factory):
    """Query without bounds should return all stored rows."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 2, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 3, 1, 0, 0, tzinfo=UTC),
    )
    client.save(df, "AAVE/USDC", "5m")

    loaded = load_ohlcv_range(
        data_dir=str(tmp_path), symbol="AAVE/USDC", timeframe="5m"
    )

    assert len(loaded) == 3
    assert loaded["timestamp"].to_list() == sorted(df["timestamp"].to_list())


def test_load_ohlcv_range_supports_start_end_and_bounded(tmp_path, sample_df_factory):
    """Query should support start-only, end-only, and bounded windows."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
    )
    client.save(df, "AAVE/USDC", "5m")

    start_only = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="5m",
        start="2024-01-01T00:10:00",
    )
    end_only = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="5m",
        end="2024-01-01T00:10:00",
    )
    bounded = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="5m",
        start="2024-01-01T00:05:00",
        end="2024-01-01T00:10:00",
    )

    assert start_only["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
    ]
    assert end_only["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
    ]
    assert bounded["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
    ]


def test_load_ohlcv_range_handles_month_boundary_with_time_filter(
    tmp_path, sample_df_factory
):
    """Range filtering should work across month partitions with date+time."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2024, 1, 31, 23, 55, tzinfo=UTC),
        datetime(2024, 2, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 2, 1, 0, 5, tzinfo=UTC),
    )
    client.save(df, "AAVE/USDC", "5m")

    loaded = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="5m",
        start="2024-01-31T23:58:00",
        end="2024-02-01T00:01:00",
    )

    assert loaded["timestamp"].to_list() == [datetime(2024, 2, 1, 0, 0, tzinfo=UTC)]


def test_load_ohlcv_range_floors_end_to_timeframe_boundary(tmp_path, sample_df_factory):
    """End should be aligned to candle open at-or-before the requested instant."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2026, 2, 28, 22, 0, tzinfo=UTC),
        datetime(2026, 2, 28, 23, 0, tzinfo=UTC),
        datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
    )
    client.save(df, "AAVE/USDC", "1h")

    loaded = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="1h",
        end="2026-02-28T23:59:59",
    )

    assert loaded["timestamp"].to_list() == [
        datetime(2026, 2, 28, 22, 0, tzinfo=UTC),
        datetime(2026, 2, 28, 23, 0, tzinfo=UTC),
    ]


def test_load_ohlcv_range_normalizes_symbol_forms(tmp_path, sample_df_factory):
    """Dashed and slash symbol inputs should resolve to the same storage path."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
    client.save(df, "AAVE/USDC", "5m")

    slash = load_ohlcv_range(
        data_dir=str(tmp_path), symbol="AAVE/USDC", timeframe="5m"
    )
    dashed = load_ohlcv_range(
        data_dir=str(tmp_path), symbol="aave-usdc", timeframe="5m"
    )

    assert slash.equals(dashed)


def test_load_ohlcv_range_validates_inputs(tmp_path):
    """Invalid timeframe/date/range values should fail fast."""
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        load_ohlcv_range(
            data_dir=str(tmp_path),
            symbol="AAVE/USDC",
            timeframe="7m",
        )

    with pytest.raises(ValueError, match="Unable to parse date"):
        load_ohlcv_range(
            data_dir=str(tmp_path),
            symbol="AAVE/USDC",
            timeframe="5m",
            start="2024/01/01",
        )

    with pytest.raises(ValueError, match="start must be less than or equal to end"):
        load_ohlcv_range(
            data_dir=str(tmp_path),
            symbol="AAVE/USDC",
            timeframe="1h",
            start="2024-01-01T01:30:00",
            end="2024-01-01T01:10:00",
        )


def test_load_ohlcv_range_returns_empty_ohlcv_frame_for_missing_data(tmp_path):
    """Missing partitions should return an empty OHLCV-shaped frame."""
    loaded = load_ohlcv_range(
        data_dir=str(tmp_path),
        symbol="AAVE/USDC",
        timeframe="5m",
    )

    assert loaded.is_empty()
    assert loaded.columns == ["timestamp", "open", "high", "low", "close", "volume"]
