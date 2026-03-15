"""Lean storage tests for OHLCV parquet persistence and update flow."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from src.data import CoinbaseDataClient


def test_save_creates_partition_files_without_month_leakage(tmp_path, sample_df_factory):
    """Save should create month partitions without cross-month contamination."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2025, 1, 31, 23, 0, tzinfo=UTC),
        datetime(2025, 2, 1, 0, 0, tzinfo=UTC),
    )
    client.save(df, "BTC/USD", "1h")

    jan_path = tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "01.parquet"
    feb_path = tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "02.parquet"
    assert jan_path.exists()
    assert feb_path.exists()

    jan = pl.read_parquet(jan_path)
    feb = pl.read_parquet(feb_path)
    assert all(ts.month == 1 for ts in jan["timestamp"].to_list())
    assert all(ts.month == 2 for ts in feb["timestamp"].to_list())


def test_save_appends_and_dedupes_keep_last(tmp_path, sample_df_factory):
    """Appending duplicate timestamps should keep the latest record."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    first = sample_df_factory(ts)
    second = pl.DataFrame(
        {
            "timestamp": [ts],
            "open": [99.0],
            "high": [99.0],
            "low": [99.0],
            "close": [99.0],
            "volume": [99.0],
        }
    )
    client.save(first, "BTC/USD", "1h")
    client.save(second, "BTC/USD", "1h")

    loaded = client.load("BTC/USD", "1h", year=2025, month=1)
    assert len(loaded) == 1
    assert loaded["close"][0] == 99.0


def test_load_supports_year_month_filters(tmp_path, sample_df_factory):
    """Load should support all-data, year-only, and year+month scopes."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    df = sample_df_factory(
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 2, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    )
    client.save(df, "BTC/USD", "1h")

    all_data = client.load("BTC/USD", "1h")
    y2025 = client.load("BTC/USD", "1h", year=2025)
    jan2025 = client.load("BTC/USD", "1h", year=2025, month=1)

    assert len(all_data) == 3
    assert len(y2025) == 2
    assert len(jan2025) == 1
    assert jan2025["timestamp"][0].year == 2025
    assert jan2025["timestamp"][0].month == 1


def test_save_filters_cross_month_rows_from_existing_file(tmp_path, sample_df_factory):
    """Save should remove wrong-month rows from an existing month file."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    feb_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "02.parquet"
    )
    feb_path.parent.mkdir(parents=True, exist_ok=True)

    seeded = sample_df_factory(
        datetime(2025, 1, 31, 23, 0, tzinfo=UTC),
        datetime(2025, 2, 1, 0, 0, tzinfo=UTC),
    )
    seeded.write_parquet(feb_path)

    client.save(
        sample_df_factory(datetime(2025, 2, 1, 1, 0, tzinfo=UTC)),
        "BTC/USD",
        "1h",
    )

    reloaded = pl.read_parquet(feb_path)
    assert len(reloaded) == 2
    assert all(ts.month == 2 for ts in reloaded["timestamp"].to_list())


@pytest.mark.asyncio
async def test_update_async_load_append_overwrite(tmp_path, sample_df_factory):
    """update_async should merge latest data and persist combined result."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    existing = sample_df_factory(
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 1, 0, tzinfo=UTC),
    )
    client.save(existing, "BTC/USD", "1h")

    latest = sample_df_factory(
        datetime(2025, 1, 1, 1, 0, tzinfo=UTC),  # duplicate
        datetime(2025, 1, 1, 2, 0, tzinfo=UTC),  # new
    )
    with patch.object(client, "fetch_latest", AsyncMock(return_value=latest)):
        updated = await client.update_async("BTC/USD", "1h")

    assert len(updated) == 3
    persisted = client.load("BTC/USD", "1h", year=2025, month=1)
    assert len(persisted) == 3


def test_load_validates_month_requires_year(tmp_path):
    """Load should reject month filter without year."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="year must be specified"):
        client.load("BTC/USD", "1h", month=1)
