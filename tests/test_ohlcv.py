"""Tests for OHLCV module."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from src.data import OHLCV_SCHEMA, CoinbaseDataClient
from src.data.ohlcv import SUPPORTED_TIMEFRAMES


def test_client_import():
    """Test that CoinbaseDataClient can be imported."""
    client = CoinbaseDataClient()
    assert client is not None
    assert client.data_dir == "./data"
    assert client.max_concurrency == 10
    assert client.rate_limit_backoff == 1.0


def test_schema_import():
    """Test that OHLCV_SCHEMA can be imported."""
    assert OHLCV_SCHEMA is not None
    assert isinstance(OHLCV_SCHEMA, dict)


def test_schema_has_six_columns():
    """Test that OHLCV_SCHEMA has all 6 required columns."""
    expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    assert set(OHLCV_SCHEMA.keys()) == expected_columns


def test_schema_column_types():
    """Test that OHLCV_SCHEMA has correct column types."""
    assert OHLCV_SCHEMA["timestamp"] == pl.Datetime
    assert OHLCV_SCHEMA["open"] == pl.Float64
    assert OHLCV_SCHEMA["high"] == pl.Float64
    assert OHLCV_SCHEMA["low"] == pl.Float64
    assert OHLCV_SCHEMA["close"] == pl.Float64
    assert OHLCV_SCHEMA["volume"] == pl.Float64


def test_client_custom_parameters():
    """Test CoinbaseDataClient with custom parameters."""
    client = CoinbaseDataClient(
        data_dir="/custom/path",
        max_concurrency=5,
        rate_limit_backoff=2.0,
    )
    assert client.data_dir == "/custom/path"
    assert client.max_concurrency == 5
    assert client.rate_limit_backoff == 2.0


def test_supported_timeframes():
    """Test that supported timeframes are defined."""
    expected = ["1m", "5m", "15m", "30m", "1h", "2h", "6h", "1d"]
    assert SUPPORTED_TIMEFRAMES == expected


def test_fetch_latest_returns_only_new(tmp_path):
    """Test that fetch_latest returns only new candles not already stored."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # First, save some existing data
    existing_df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 5, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 10, tzinfo=UTC),
            ],
            "open": [42000.0, 42050.0, 42100.0],
            "high": [42100.0, 42150.0, 42200.0],
            "low": [41900.0, 41950.0, 42000.0],
            "close": [42050.0, 42100.0, 42150.0],
            "volume": [1000.0, 800.0, 600.0],
        }
    )
    client.save(existing_df, "BTC/USD", "5m")

    # Get existing timestamps to exclude
    exclude_timestamps = existing_df["timestamp"].to_list()

    # Mock fetch_latest to return some overlapping and some new data
    # Use UTC-aware datetimes for creating timestamps to ensure consistent timezone handling
    ts_10_10 = int(datetime(2025, 1, 15, 10, 10, tzinfo=UTC).timestamp() * 1000)
    ts_10_15 = int(datetime(2025, 1, 15, 10, 15, tzinfo=UTC).timestamp() * 1000)
    ts_10_20 = int(datetime(2025, 1, 15, 10, 20, tzinfo=UTC).timestamp() * 1000)

    mock_candles = [
        [ts_10_10, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],  # Already stored
        [ts_10_15, 42150.0, 42250.0, 42100.0, 42200.0, 500.0],  # New
        [ts_10_20, 42200.0, 42300.0, 42150.0, 42250.0, 400.0],  # New
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        # Call fetch_latest with exclude_timestamps
        result = asyncio.run(
            client.fetch_latest("BTC/USD", "5m", exclude_timestamps=exclude_timestamps)
        )

    # Should return only the 2 new candles (filtering out already stored)
    assert len(result) == 2
    # Verify the timestamps are the new ones
    timestamps = result["timestamp"].to_list()
    expected_ts_1 = datetime(2025, 1, 15, 10, 15, tzinfo=UTC)
    expected_ts_2 = datetime(2025, 1, 15, 10, 20, tzinfo=UTC)
    assert expected_ts_1 in timestamps
    assert expected_ts_2 in timestamps


def test_fetch_latest_returns_dataframe():
    """Test that fetch_latest returns a Polars DataFrame with correct schema."""
    client = CoinbaseDataClient()

    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(client.fetch_latest("BTC/USD", "1m"))

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_update_combines_and_deduplicates(tmp_path):
    """Test that update correctly combines and deduplicates data."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Save existing data
    existing_df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 5, tzinfo=UTC),
            ],
            "open": [42000.0, 42050.0],
            "high": [42100.0, 42150.0],
            "low": [41900.0, 41950.0],
            "close": [42050.0, 42100.0],
            "volume": [1000.0, 800.0],
        }
    )
    client.save(existing_df, "BTC/USD", "5m")

    # Mock fetch_latest to return overlapping and new data
    # Use UTC-aware datetimes for consistent timezone handling
    ts_10_05 = int(datetime(2025, 1, 15, 10, 5, tzinfo=UTC).timestamp() * 1000)
    ts_10_10 = int(datetime(2025, 1, 15, 10, 10, tzinfo=UTC).timestamp() * 1000)
    ts_10_15 = int(datetime(2025, 1, 15, 10, 15, tzinfo=UTC).timestamp() * 1000)

    mock_candles = [
        [ts_10_05, 42050.0, 42150.0, 41950.0, 42100.0, 800.0],  # Duplicate
        [ts_10_10, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],  # New
        [ts_10_15, 42150.0, 42250.0, 42100.0, 42200.0, 500.0],  # New
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        result = client.update("BTC/USD", "5m")

    # Should have 4 total rows (2 existing + 2 new, deduplicated)
    assert len(result) == 4
    # Should be sorted by timestamp
    assert result["timestamp"].is_sorted()

    # Verify saved file has correct data
    loaded = client.load("BTC/USD", "5m")
    assert len(loaded) == 4


def test_client_has_fetch_latest_method():
    """Test that CoinbaseDataClient has fetch_latest method."""
    client = CoinbaseDataClient()
    assert hasattr(client, "fetch_latest")
    assert callable(client.fetch_latest)


def test_client_has_update_method():
    """Test that CoinbaseDataClient has update method."""
    client = CoinbaseDataClient()
    assert hasattr(client, "update")
    assert callable(client.update)


def test_client_validate_timeframe():
    """Test timeframe validation."""
    client = CoinbaseDataClient()

    # Valid timeframes should not raise
    for tf in ["1m", "5m", "15m", "30m", "1h", "2h", "6h", "1d"]:
        client._validate_timeframe(tf)

    # Invalid timeframe should raise
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        client._validate_timeframe("999m")


def test_parse_date():
    """Test date parsing."""
    client = CoinbaseDataClient()

    # Test ISO format
    dt = client._parse_date("2024-01-01T00:00:00Z")
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1

    # Test date only
    dt = client._parse_date("2024-01-01")
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1

    # Test invalid date
    with pytest.raises(ValueError, match="Unable to parse date"):
        client._parse_date("invalid-date")


def test_fetch_returns_dataframe():
    """Test that fetch method returns a Polars DataFrame."""
    client = CoinbaseDataClient()

    # Mock the exchange's fetch_ohlcv method
    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(client.fetch("BTC/USD", "1m", "2024-01-01"))

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_respects_date_range():
    """Test that fetch respects start and end date range."""
    client = CoinbaseDataClient()

    # Mock candles with timestamps
    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],  # 2024-01-01 00:00
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],  # 2024-01-01 00:01
        [1704067320000, 42100.0, 42200.0, 42050.0, 42150.0, 600.0],  # 2024-01-01 00:02
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        # Fetch with end date that should filter out the last candle
        result = asyncio.run(
            client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01T00:01:00")
        )

    # Should only have 2 candles (filtered by end_date)
    assert len(result) == 2
    # Verify timestamps are within range
    assert result["timestamp"][0] <= datetime(
        2024, 1, 1, 0, 1, 0, tzinfo=result["timestamp"][0].tzinfo
    )


def test_save_creates_parquet(tmp_path):
    """Test that save method creates parquet file."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create sample OHLCV data
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [1000.0],
        }
    )

    # Save the data
    client.save(df, "BTC/USD", "1h")

    # Verify parquet file was created
    expected_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "01.parquet"
    )
    assert expected_path.exists()

    # Verify data was saved correctly
    loaded_df = pl.read_parquet(expected_path)
    assert len(loaded_df) == 1
    assert loaded_df["close"][0] == 42050.0


def test_save_correct_directory_structure(tmp_path):
    """Test that save creates correct directory structure."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create sample OHLCV data spanning multiple months
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2025, 2, 20, 15, 0, tzinfo=UTC),
            ],
            "open": [42000.0, 43000.0],
            "high": [42100.0, 43100.0],
            "low": [41900.0, 42900.0],
            "close": [42050.0, 43050.0],
            "volume": [1000.0, 800.0],
        }
    )

    # Save the data
    client.save(df, "BTC/USD", "1h")

    # Verify directory structure
    jan_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "01.parquet"
    )
    feb_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "02.parquet"
    )

    assert jan_path.exists()
    assert feb_path.exists()

    # Verify correct data in each file
    jan_df = pl.read_parquet(jan_path)
    feb_df = pl.read_parquet(feb_path)

    assert len(jan_df) == 1
    assert len(feb_df) == 1
    assert jan_df["timestamp"][0].month == 1
    assert feb_df["timestamp"][0].month == 2


def test_load_parquet(tmp_path):
    """Test that load method reads saved parquet files."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create and save sample OHLCV data
    save_df = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [1000.0],
        }
    )
    client.save(save_df, "BTC/USD", "1h")

    # Load the data
    loaded_df = client.load("BTC/USD", "1h")

    # Verify data was loaded correctly
    assert isinstance(loaded_df, pl.DataFrame)
    assert len(loaded_df) == 1
    assert loaded_df["close"][0] == 42050.0
    assert loaded_df.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_load_with_year_month_filter(tmp_path):
    """Test loading with year and month filters."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create and save sample OHLCV data spanning multiple months
    save_df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2025, 2, 20, 15, 0, tzinfo=UTC),
                datetime(2025, 3, 10, 20, 0, tzinfo=UTC),
            ],
            "open": [42000.0, 43000.0, 44000.0],
            "high": [42100.0, 43100.0, 44100.0],
            "low": [41900.0, 42900.0, 43900.0],
            "close": [42050.0, 43050.0, 44050.0],
            "volume": [1000.0, 800.0, 600.0],
        }
    )
    client.save(save_df, "BTC/USD", "1h")

    # Test loading all data
    all_data = client.load("BTC/USD", "1h")
    assert len(all_data) == 3

    # Test loading by year only (should get all months for that year)
    year_data = client.load("BTC/USD", "1h", year=2025)
    assert len(year_data) == 3

    # Test loading by year and month (January)
    jan_data = client.load("BTC/USD", "1h", year=2025, month=1)
    assert len(jan_data) == 1
    assert jan_data["timestamp"][0].month == 1

    # Test loading by year and month (February)
    feb_data = client.load("BTC/USD", "1h", year=2025, month=2)
    assert len(feb_data) == 1
    assert feb_data["timestamp"][0].month == 2

    # Test loading non-existent month returns empty
    empty_data = client.load("BTC/USD", "1h", year=2025, month=12)
    assert len(empty_data) == 0
    assert isinstance(empty_data, pl.DataFrame)
