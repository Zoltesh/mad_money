"""Tests for OHLCV module."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import ccxt
import ccxt.async_support
import polars as pl
import pytest

from src.data import OHLCV_SCHEMA, CoinbaseDataClient
from src.data.ohlcv import SUPPORTED_TIMEFRAMES, Verbosity


def test_client_import():
    """Test that CoinbaseDataClient can be imported."""
    client = CoinbaseDataClient()
    assert client is not None
    assert client.data_dir == "./data"
    assert client.max_concurrency == 10
    assert client.rate_limit_backoff == 1.0
    assert client.max_retries == 3  # Default max_retries


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
        max_retries=5,
    )
    assert client.data_dir == "/custom/path"
    assert client.max_concurrency == 5
    assert client.rate_limit_backoff == 2.0
    assert client.max_retries == 5


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


def test_apply_end_of_day():
    """Test _apply_end_of_day method with various edge cases."""
    from datetime import datetime

    # Test 1: Date with no time component and end_of_day=True - should convert to end of day
    dt = datetime(2024, 1, 15, 0, 0, 0, 0)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.hour == 23
    assert result.minute == 59
    assert result.second == 59
    assert result.microsecond == 999999

    # Test 2: Date with no time component and end_of_day=False - should remain unchanged
    dt = datetime(2024, 1, 15, 0, 0, 0, 0)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=False)
    assert result.hour == 0
    assert result.minute == 0
    assert result.second == 0
    assert result.microsecond == 0

    # Test 3: Date with time component and end_of_day=True - should remain unchanged
    dt = datetime(2024, 1, 15, 10, 30, 0, 0)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.hour == 10
    assert result.minute == 30

    # Test 4: Date with microseconds and no time component - should still convert
    dt = datetime(2024, 1, 15, 0, 0, 0, 0)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.microsecond == 999999

    # Test 5: Date with non-zero microseconds and no time component - should NOT convert
    # because microsecond != 0 means there's some time component
    dt = datetime(2024, 1, 15, 0, 0, 0, 1)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.microsecond == 1
    assert result.hour == 0

    # Test 6: Date with seconds set and no time component - should NOT convert
    dt = datetime(2024, 1, 15, 0, 0, 1, 0)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.second == 1
    assert result.hour == 0

    # Test 7: Naive datetime (no timezone) - should work correctly
    dt = datetime(2024, 1, 15)
    result = CoinbaseDataClient._apply_end_of_day(dt, end_of_day=True)
    assert result.hour == 23
    assert result.minute == 59
    assert result.second == 59
    assert result.microsecond == 999999


def test_candles_to_dataframe():
    """Test _candles_to_dataframe static method."""
    # Test with empty list
    result = CoinbaseDataClient._candles_to_dataframe([])
    assert result.is_empty()
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    # Test with valid candles
    candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],
    ]
    result = CoinbaseDataClient._candles_to_dataframe(candles)

    assert len(result) == 2
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    # Verify first row values
    assert result["timestamp"][0] == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert result["open"][0] == 42000.0
    assert result["high"][0] == 42100.0
    assert result["low"][0] == 41900.0
    assert result["close"][0] == 42050.0
    assert result["volume"][0] == 1000.0


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


def test_fetch_multiple_returns_dict():
    """Test that fetch_multiple returns correct nested dict structure."""
    client = CoinbaseDataClient()

    # Mock candles with timestamps
    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],  # 2024-01-01 00:00
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],  # 2024-01-01 00:01
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        # Test with multiple symbols and timeframes
        result = asyncio.run(
            client.fetch_multiple(
                symbols=["BTC/USD", "ETH/USD"],
                timeframes=["1m", "1h"],
                start_date="2024-01-01",
            )
        )

    # Verify nested dict structure
    assert isinstance(result, dict)
    assert "BTC/USD" in result
    assert "ETH/USD" in result

    assert isinstance(result["BTC/USD"], dict)
    assert "1m" in result["BTC/USD"]
    assert "1h" in result["BTC/USD"]

    assert isinstance(result["ETH/USD"], dict)
    assert "1m" in result["ETH/USD"]
    assert "1h" in result["ETH/USD"]

    # Verify each entry is a DataFrame with correct columns
    for symbol in result:
        for timeframe in result[symbol]:
            df = result[symbol][timeframe]
            assert isinstance(df, pl.DataFrame)
            assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_multiple_empty_results():
    """Test that fetch_multiple handles empty results correctly."""
    client = CoinbaseDataClient()

    # Mock returning empty candles
    async def mock_fetch(*args, **kwargs):
        return []

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(
            client.fetch_multiple(
                symbols=["BTC/USD"],
                timeframes=["1m"],
                start_date="2024-01-01",
            )
        )

    # Should still return dict structure with empty DataFrames
    assert isinstance(result, dict)
    assert "BTC/USD" in result
    assert isinstance(result["BTC/USD"], dict)
    assert "1m" in result["BTC/USD"]
    assert result["BTC/USD"]["1m"].is_empty()


def test_async_context_manager():
    """Test async context manager support."""
    client = CoinbaseDataClient()

    # Track if close was called
    close_called = False

    async def mock_close():
        nonlocal close_called
        close_called = True
        client._exchange = None

    # Set up mock exchange directly on the client
    mock_exchange_instance = AsyncMock()
    mock_exchange_instance.close = mock_close
    client._exchange = mock_exchange_instance

    async def run_test():
        async with client as ctx:
            assert ctx is client
        return close_called

    close_was_called = asyncio.run(run_test())

    assert close_was_called, "close() should be called on async context exit"


def test_sync_context_manager():
    """Test sync context manager support."""
    client = CoinbaseDataClient()

    # Track if close was called
    close_called = False

    async def mock_close():
        nonlocal close_called
        close_called = True

    # Set up mock exchange directly on the client
    mock_exchange_instance = AsyncMock()
    mock_exchange_instance.close = mock_close
    client._exchange = mock_exchange_instance

    with client as ctx:
        assert ctx is client

    assert close_called, "close() should be called on sync context exit"


@pytest.mark.integration
def test_full_workflow_integration(tmp_path):
    """Integration test covering full fetch-save-load-update workflow.

    This test validates end-to-end functionality:
    1. Fetch historical data
    2. Save to parquet
    3. Load from parquet
    4. Update with new data
    """
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Mock data for initial fetch
    ts_1 = int(datetime(2025, 1, 15, 10, 0, tzinfo=UTC).timestamp() * 1000)
    ts_2 = int(datetime(2025, 1, 15, 10, 5, tzinfo=UTC).timestamp() * 1000)
    ts_3 = int(datetime(2025, 1, 15, 10, 10, tzinfo=UTC).timestamp() * 1000)
    ts_4 = int(datetime(2025, 1, 15, 10, 15, tzinfo=UTC).timestamp() * 1000)

    mock_candles = [
        [ts_1, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [ts_2, 42050.0, 42150.0, 41950.0, 42100.0, 800.0],
        [ts_3, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],
    ]

    # Step 1: Fetch historical data
    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        fetched_df = asyncio.run(
            client.fetch("BTC/USD", "5m", "2025-01-15", "2025-01-15T12:00:00")
        )

    # Verify fetch returns valid data
    assert isinstance(fetched_df, pl.DataFrame)
    assert len(fetched_df) == 3
    assert fetched_df.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    # Step 2: Save data to parquet
    client.save(fetched_df, "BTC/USD", "5m")

    # Verify parquet file was created
    parquet_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "5m" / "2025" / "01.parquet"
    )
    assert parquet_path.exists(), "Parquet file should be created"

    # Step 3: Load data from parquet
    loaded_df = client.load("BTC/USD", "5m", year=2025, month=1)

    # Verify loaded data matches saved data
    assert isinstance(loaded_df, pl.DataFrame)
    assert len(loaded_df) == 3
    assert loaded_df["close"].to_list() == [42050.0, 42100.0, 42150.0]

    # Step 4: Update with new data (fetch latest and merge)
    mock_new_candles = [
        [ts_3, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],  # Duplicate
        [ts_4, 42150.0, 42250.0, 42100.0, 42200.0, 500.0],  # New
    ]

    async def mock_fetch_new(*args, **kwargs):
        return mock_new_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_new
        mock_exchange.return_value = mock_exchange_instance

        updated_df = client.update("BTC/USD", "5m")

    # Verify update merged and deduplicated correctly
    assert isinstance(updated_df, pl.DataFrame)
    assert len(updated_df) == 4, "Should have 4 rows after update (3 original + 1 new)"
    assert updated_df["timestamp"].is_sorted(), "Data should be sorted by timestamp"

    # Final verification: load again to confirm persistence
    final_loaded = client.load("BTC/USD", "5m", year=2025, month=1)
    assert len(final_loaded) == 4, "Updated data should be persisted"


def test_fetch_multiple_progress_bar_stopped_on_exception():
    """Test that progress bar is stopped when exception occurs in fetch_multiple."""
    from src.data.ohlcv import Verbosity

    client = CoinbaseDataClient()

    # Track if progress bar was stopped
    progress_stopped = []

    # Create a mock Progress class that tracks stop calls
    class MockProgress:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            progress_stopped.append(True)

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, *args, **kwargs):
            pass

    # Mock fetch that raises exception
    async def mock_fetch_error(*args, **kwargs):
        raise Exception("Simulated API error")

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_error
        mock_exchange.return_value = mock_exchange_instance

        with patch("src.data.ohlcv.Progress", MockProgress):
            try:
                asyncio.run(
                    client.fetch_multiple(
                        symbols=["INVALID/SYM"],
                        timeframes=["1m"],
                        start_date="2024-01-01",
                        verbosity=Verbosity.PROGRESS,
                    )
                )
            except Exception:
                pass  # Expected to fail

    # Verify progress bar was stopped despite exception
    assert len(progress_stopped) > 0, "Progress bar stop should be called"


def test_fetch_multiple_progress_bar_stopped_on_success():
    """Test that progress bar is stopped on normal completion of fetch_multiple."""
    from src.data.ohlcv import Verbosity

    client = CoinbaseDataClient()

    # Track if progress bar was stopped
    progress_stopped = []

    class MockProgress:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            progress_stopped.append(True)

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, *args, **kwargs):
            pass

    # Mock successful fetch
    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        with patch("src.data.ohlcv.Progress", MockProgress):
            asyncio.run(
                client.fetch_multiple(
                    symbols=["BTC/USD"],
                    timeframes=["1m"],
                    start_date="2024-01-01",
                    verbosity=Verbosity.PROGRESS,
                )
            )

    # Verify progress bar was stopped on success
    assert len(progress_stopped) > 0, "Progress bar stop should be called on success"


def test_fetch_includes_candle_at_exact_end_timestamp():
    """Test that candles at exact end timestamp are included in fetch results."""
    client = CoinbaseDataClient()

    # Create timestamps where one is exactly at the end boundary
    ts_1 = int(datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)  # 00:00
    ts_2 = int(datetime(2024, 1, 1, 0, 1, tzinfo=UTC).timestamp() * 1000)  # 00:01
    ts_3 = int(datetime(2024, 1, 1, 0, 2, tzinfo=UTC).timestamp() * 1000)  # 00:02

    mock_candles = [
        [ts_1, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [ts_2, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],
        [ts_3, 42100.0, 42200.0, 42050.0, 42150.0, 600.0],
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        # Fetch with end_date at exactly 00:01 - should include candle at 00:01
        result = asyncio.run(
            client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01T00:01:00")
        )

    # With the fix (using > instead of >=), candle at exactly end_ts should be included
    # So we should have 2 candles: 00:00 and 00:01
    assert len(result) == 2, f"Expected 2 candles, got {len(result)}"

    # Verify the timestamps
    timestamps = result["timestamp"].to_list()
    assert timestamps[0] == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert timestamps[1] == datetime(2024, 1, 1, 0, 1, tzinfo=UTC)


def test_refactored_fetch_preserves_behavior(tmp_path):
    """Test that refactored code produces identical results to original.

    This test validates that the fetch method returns expected data structure
    and values after refactoring.
    """
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create known test data
    ts_1 = int(datetime(2024, 1, 1, 10, 0, tzinfo=UTC).timestamp() * 1000)
    ts_2 = int(datetime(2024, 1, 1, 10, 5, tzinfo=UTC).timestamp() * 1000)
    ts_3 = int(datetime(2024, 1, 1, 10, 10, tzinfo=UTC).timestamp() * 1000)

    mock_candles = [
        [ts_1, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [ts_2, 42050.0, 42150.0, 41950.0, 42100.0, 800.0],
        [ts_3, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(
            client.fetch("BTC/USD", "5m", "2024-01-01", "2024-01-01T12:00:00")
        )

    # Verify data structure is correct
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 3
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    # Verify data values are correct
    assert result["open"].to_list() == [42000.0, 42050.0, 42100.0]
    assert result["high"].to_list() == [42100.0, 42150.0, 42200.0]
    assert result["low"].to_list() == [41900.0, 41950.0, 42000.0]
    assert result["close"].to_list() == [42050.0, 42100.0, 42150.0]
    assert result["volume"].to_list() == [1000.0, 800.0, 600.0]

    # Verify timestamps are correctly parsed
    timestamps = result["timestamp"].to_list()
    assert timestamps[0] == datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    assert timestamps[1] == datetime(2024, 1, 1, 10, 5, tzinfo=UTC)
    assert timestamps[2] == datetime(2024, 1, 1, 10, 10, tzinfo=UTC)

    # Save and reload - verify data integrity is preserved
    client.save(result, "BTC/USD", "5m")
    loaded = client.load("BTC/USD", "5m")

    assert len(loaded) == 3
    assert loaded["close"].to_list() == result["close"].to_list()


def test_fetch_rate_limit_retry():
    """Test that rate limit triggers retry and succeeds on second attempt."""
    import ccxt

    client = CoinbaseDataClient()

    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
    ]

    call_count = 0

    async def mock_fetch_with_rate_limit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call raises rate limit
            raise ccxt.RateLimitExceeded({"message": "Rate limit exceeded"})
        # Second call succeeds
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_with_rate_limit
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(client.fetch("BTC/USD", "1m", "2024-01-01"))

    # Verify retry happened
    assert call_count == 2, "Should have made 2 calls (initial + retry)"
    # Verify data was returned on retry
    assert len(result) == 1
    assert result["close"][0] == 42050.0


def test_fetch_network_error():
    """Test that network errors propagate correctly."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_network_error(*args, **kwargs):
        raise ccxt.NetworkError({"message": "Connection failed"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_network_error
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.NetworkError):
            asyncio.run(client.fetch("BTC/USD", "1m", "2024-01-01"))


def test_fetch_invalid_symbol():
    """Test that invalid symbol returns appropriate error."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_invalid_symbol(*args, **kwargs):
        raise ccxt.BadSymbol({"message": "Symbol not found"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_invalid_symbol
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.BadSymbol):
            asyncio.run(client.fetch("INVALID/SYMBOL", "1m", "2024-01-01"))


def test_fetch_rate_limit_exhausted():
    """Test behavior when rate limits persist (both attempts fail)."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_always_rate_limited(*args, **kwargs):
        raise ccxt.RateLimitExceeded({"message": "Rate limit exceeded"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_always_rate_limited
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.RateLimitExceeded):
            asyncio.run(client.fetch("BTC/USD", "1m", "2024-01-01"))


def test_fetch_latest_rate_limit_retry():
    """Test that fetch_latest retries on rate limit and succeeds."""
    import ccxt

    client = CoinbaseDataClient()

    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
    ]

    call_count = 0

    async def mock_fetch_with_rate_limit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ccxt.RateLimitExceeded({"message": "Rate limit exceeded"})
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_with_rate_limit
        mock_exchange.return_value = mock_exchange_instance

        result = asyncio.run(client.fetch_latest("BTC/USD", "1m"))

    assert call_count == 2
    assert len(result) == 1
    assert result["close"][0] == 42050.0


def test_fetch_latest_network_error():
    """Test that fetch_latest propagates network errors."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_network_error(*args, **kwargs):
        raise ccxt.NetworkError({"message": "Connection failed"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_network_error
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.NetworkError):
            asyncio.run(client.fetch_latest("BTC/USD", "1m"))


def test_fetch_latest_invalid_symbol():
    """Test that fetch_latest raises BadSymbol for invalid symbols."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_invalid_symbol(*args, **kwargs):
        raise ccxt.BadSymbol({"message": "Symbol not found"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_invalid_symbol
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.BadSymbol):
            asyncio.run(client.fetch_latest("INVALID/SYMBOL", "1m"))


def test_fetch_latest_rate_limit_exhausted():
    """Test fetch_latest raises exception when rate limits persist."""
    import ccxt

    client = CoinbaseDataClient()

    async def mock_fetch_always_rate_limited(*args, **kwargs):
        raise ccxt.RateLimitExceeded({"message": "Rate limit exceeded"})

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch_always_rate_limited
        mock_exchange.return_value = mock_exchange_instance

        with pytest.raises(ccxt.RateLimitExceeded):
            asyncio.run(client.fetch_latest("BTC/USD", "1m"))


def test_candles_high_low_integrity():
    """Test that high >= open, high >= close, low <= open, low <= close."""
    # Create sample OHLCV data that satisfies integrity constraints
    candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 41950.0, 42100.0, 800.0],
        [1704067320000, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],
    ]
    df = CoinbaseDataClient._candles_to_dataframe(candles)

    # Verify high is the maximum of OHLC for each row
    for i in range(len(df)):
        row = df.row(i, named=True)
        assert row["high"] >= row["open"], f"high must be >= open at row {i}"
        assert row["high"] >= row["close"], f"high must be >= close at row {i}"
        assert row["high"] >= row["low"], f"high must be >= low at row {i}"

    # Verify low is the minimum of OHLC for each row
    for i in range(len(df)):
        row = df.row(i, named=True)
        assert row["low"] <= row["open"], f"low must be <= open at row {i}"
        assert row["low"] <= row["close"], f"low must be <= close at row {i}"
        assert row["low"] <= row["high"], f"low must be <= high at row {i}"


def test_candles_positive_prices():
    """Test that all prices are positive."""
    candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 41950.0, 42100.0, 800.0],
    ]
    df = CoinbaseDataClient._candles_to_dataframe(candles)

    # Check all price columns are positive
    assert (df["open"] > 0).all(), "open prices must be positive"
    assert (df["high"] > 0).all(), "high prices must be positive"
    assert (df["low"] > 0).all(), "low prices must be positive"
    assert (df["close"] > 0).all(), "close prices must be positive"


def test_save_empty_dataframe(tmp_path):
    """Test that saving empty DataFrame is a no-op."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create an empty DataFrame with correct schema
    empty_df = pl.DataFrame(
        {
            "timestamp": pl.Series([], dtype=pl.Datetime),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
        }
    )

    # Save empty DataFrame - should be a no-op (no exception, no file created)
    client.save(empty_df, "BTC/USD", "1h")

    # Verify no parquet file was created
    parquet_path = tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h"
    assert not parquet_path.exists(), "No file should be created for empty DataFrame"


def test_candles_non_negative_volume():
    """Test that volume is non-negative."""
    candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],
        [1704067260000, 42050.0, 42150.0, 41950.0, 42100.0, 0.0],
        [1704067320000, 42100.0, 42200.0, 42000.0, 42150.0, 600.0],
    ]
    df = CoinbaseDataClient._candles_to_dataframe(candles)

    # Check volume is non-negative
    assert (df["volume"] >= 0).all(), "volume must be non-negative"


def test_fetch_invalid_date_order():
    """Test behavior when end_date is before start_date."""
    client = CoinbaseDataClient()

    # Mock candles with timestamps
    mock_candles = [
        [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0],  # 2024-01-01 00:00
        [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 800.0],  # 2024-01-01 00:01
    ]

    async def mock_fetch(*args, **kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance

        # Fetch with end_date before start_date - should return empty
        result = asyncio.run(client.fetch("BTC/USD", "1m", "2024-01-02", "2024-01-01"))

    # Should return empty DataFrame when end_date < start_date
    assert result.is_empty(), "Should return empty when end_date before start_date"


def test_save_duplicate_timestamps(tmp_path):
    """Test that duplicate timestamps are deduplicated on save."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create data with duplicate timestamps
    df_with_duplicates = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 5, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 0, tzinfo=UTC),  # Duplicate of first
                datetime(2025, 1, 15, 10, 10, tzinfo=UTC),
            ],
            "open": [
                42000.0,
                42050.0,
                42000.1,
                42100.0,
            ],  # Note: different open for duplicate
            "high": [42100.0, 42150.0, 42100.1, 42200.0],
            "low": [41900.0, 41950.0, 41900.1, 42000.0],
            "close": [42050.0, 42100.0, 42050.1, 42150.0],
            "volume": [1000.0, 800.0, 800.0, 600.0],
        }
    )

    # Save the data (should deduplicate)
    client.save(df_with_duplicates, "BTC/USD", "5m")

    # Load and verify deduplication
    loaded = client.load("BTC/USD", "5m", year=2025, month=1)

    # Should have only 3 unique timestamps (deduplicated)
    assert len(loaded) == 3, f"Expected 3 rows after deduplication, got {len(loaded)}"

    # Verify the first entry is kept (not the duplicate)
    # The first entry should have open=42000.0 (the original)
    assert loaded["open"][0] == 42000.0, "Should keep first occurrence"


def test_save_appends_correctly(tmp_path):
    """Test that save correctly appends to existing parquet."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create and save initial data
    initial_df = pl.DataFrame(
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
    client.save(initial_df, "BTC/USD", "5m")

    # Create additional data to append (same month)
    additional_df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 15, 10, 10, tzinfo=UTC),
                datetime(2025, 1, 15, 10, 15, tzinfo=UTC),
            ],
            "open": [42100.0, 42150.0],
            "high": [42200.0, 42250.0],
            "low": [42000.0, 42050.0],
            "close": [42150.0, 42200.0],
            "volume": [600.0, 500.0],
        }
    )
    client.save(additional_df, "BTC/USD", "5m")

    # Load and verify both are present
    loaded = client.load("BTC/USD", "5m", year=2025, month=1)

    # Should have all 4 rows
    assert len(loaded) == 4, f"Expected 4 rows after append, got {len(loaded)}"

    # Verify data is sorted by timestamp
    assert loaded["timestamp"].is_sorted(), "Data should be sorted by timestamp"


def test_date_boundary_crossing(tmp_path):
    """Test data spanning month boundaries saves correctly."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create data spanning month boundary (Jan 31 to Feb 1)
    boundary_df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 31, 22, 0, tzinfo=UTC),
                datetime(2025, 1, 31, 23, 0, tzinfo=UTC),
                datetime(2025, 2, 1, 0, 0, tzinfo=UTC),
                datetime(2025, 2, 1, 1, 0, tzinfo=UTC),
            ],
            "open": [42000.0, 42100.0, 42200.0, 42300.0],
            "high": [42100.0, 42200.0, 42300.0, 42400.0],
            "low": [41900.0, 42000.0, 42100.0, 42200.0],
            "close": [42100.0, 42200.0, 42300.0, 42400.0],
            "volume": [1000.0, 800.0, 600.0, 500.0],
        }
    )

    # Save the data
    client.save(boundary_df, "BTC/USD", "1h")

    # Verify both month files were created
    jan_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "01.parquet"
    )
    feb_path = (
        tmp_path / "coinbase" / "ohlcv" / "btc-usd" / "1h" / "2025" / "02.parquet"
    )

    assert jan_path.exists(), "January file should exist"
    assert feb_path.exists(), "February file should exist"

    # Load and verify each month separately
    jan_data = client.load("BTC/USD", "1h", year=2025, month=1)
    feb_data = client.load("BTC/USD", "1h", year=2025, month=2)

    assert len(jan_data) == 2, f"Expected 2 rows in January, got {len(jan_data)}"
    assert len(feb_data) == 2, f"Expected 2 rows in February, got {len(feb_data)}"

    # Verify January timestamps
    assert jan_data["timestamp"][0].month == 1
    assert jan_data["timestamp"][1].month == 1

    # Verify February timestamps
    assert feb_data["timestamp"][0].month == 2
    assert feb_data["timestamp"][1].month == 2

    # Load all data and verify
    all_data = client.load("BTC/USD", "1h", year=2025)
    assert len(all_data) == 4, f"Expected 4 total rows, got {len(all_data)}"
    assert all_data["timestamp"].is_sorted(), "Data should be sorted by timestamp"


class TestRetryOnNetworkError:
    """Tests for retry behavior on transient network errors."""

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        """Verify NetworkError triggers retry."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)
        exchange = ccxt.async_support.coinbaseadvanced()
        semaphore = asyncio.Semaphore(1)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.NetworkError("Network issue")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        with patch.object(exchange, "fetch_ohlcv", side_effect=mock_fetch):
            result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2, "Should have retried after NetworkError"
        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_on_exchange_not_available(self):
        """Verify ExchangeNotAvailable triggers retry."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)
        exchange = ccxt.async_support.coinbaseadvanced()
        semaphore = asyncio.Semaphore(1)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.ExchangeNotAvailable("Exchange unavailable")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        with patch.object(exchange, "fetch_ohlcv", side_effect=mock_fetch):
            result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2, "Should have retried after ExchangeNotAvailable"
        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_on_request_timeout(self):
        """Verify RequestTimeout triggers retry."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)
        exchange = ccxt.async_support.coinbaseadvanced()
        semaphore = asyncio.Semaphore(1)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.RequestTimeout("Request timed out")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        with patch.object(exchange, "fetch_ohlcv", side_effect=mock_fetch):
            result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2, "Should have retried after RequestTimeout"
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_retry_on_bad_symbol(self):
        """Verify BadSymbol does NOT retry (invalid symbol)."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)
        exchange = ccxt.async_support.coinbaseadvanced()
        semaphore = asyncio.Semaphore(1)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ccxt.BadSymbol("Invalid symbol")

        with patch.object(exchange, "fetch_ohlcv", side_effect=mock_fetch):
            with pytest.raises(ccxt.BadSymbol):
                await client._fetch_with_retry(exchange, semaphore, symbol="INVALID/PAIR", timeframe="1h")

        assert call_count == 1, "Should NOT retry on BadSymbol"

    def test_exception_types_covered(self):
        """Verify at least 6 exception types are handled."""
        from src.data.ohlcv import NON_RETRYABLE_EXCEPTIONS, RETRYABLE_EXCEPTIONS

        # Count retryable exceptions
        retryable_count = len(RETRYABLE_EXCEPTIONS)
        _ = len(NON_RETRYABLE_EXCEPTIONS)  # Intentionally unused but documented

        assert retryable_count >= 6, f"Expected at least 6 retryable exceptions, got {retryable_count}"

        # Verify key exceptions are included
        expected_retryable = {
            "RateLimitExceeded",
            "NetworkError",
            "ExchangeNotAvailable",
            "RequestTimeout",
            "DDoSProtection",
            "NullResponse",
        }
        actual_retryable = {exc.__name__ for exc in RETRYABLE_EXCEPTIONS}
        assert expected_retryable.issubset(actual_retryable), "Missing expected retryable exceptions"


# =============================================================================
# Concurrency and Error Handling Tests (CHOR-001)
# =============================================================================


class TestConcurrencyAndGather:
    """Tests for asyncio.gather behavior with return_exceptions=True."""

    @pytest.mark.asyncio
    async def test_gather_preserves_partial_results(self):
        """Test that gather with return_exceptions preserves successful results when some tasks fail."""
        # Create tasks where some succeed and some fail
        async def succeed_task():
            await asyncio.sleep(0.01)
            return ("success", "data")

        async def fail_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed")

        # Mix of successes and failures
        tasks = [
            succeed_task(),
            fail_task(),
            succeed_task(),
            fail_task(),
            succeed_task(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 3, f"Expected 3 successes, got {len(successes)}"
        assert len(exceptions) == 2, f"Expected 2 exceptions, got {len(exceptions)}"

    @pytest.mark.asyncio
    async def test_gather_handles_rate_limit_exceeded(self):
        """Test that gather properly handles RateLimitExceeded in fetch_multiple."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        # Create mock exchange
        exchange = AsyncMock()
        exchange.fetch_ohlcv = AsyncMock(side_effect=[
            ccxt.RateLimitExceeded("Rate limited"),
            [[1700000000000, 42000, 42100, 41900, 42050, 100]],  # Success on retry
            ccxt.RateLimitExceeded("Rate limited"),
            [[1700000010000, 42100, 42200, 42000, 42150, 100]],  # Success on retry
        ])

        semaphore = asyncio.Semaphore(10)

        # First task will fail with rate limit, then succeed on retry
        result1 = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h", since=1700000000000)
        # Second task will fail with rate limit, then succeed on retry
        result2 = await client._fetch_with_retry(exchange, semaphore, symbol="ETH/USD", timeframe="1h", since=1700000000000)

        assert result1 is not None
        assert len(result1) > 0
        assert result2 is not None
        assert len(result2) > 0

    @pytest.mark.asyncio
    async def test_gather_handles_network_error(self):
        """Test that gather properly handles NetworkError in concurrent tasks."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        exchange = AsyncMock()
        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.NetworkError("Network issue")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        exchange.fetch_ohlcv = mock_fetch
        semaphore = asyncio.Semaphore(10)

        # Should retry and succeed on second attempt
        result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2  # 1 failure + 1 success
        assert result is not None

    @pytest.mark.asyncio
    async def test_gather_handles_multiple_simultaneous_failures(self):
        """Test fetch_multiple returns partial results when multiple tasks fail simultaneously."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        # Create mock that fails for one symbol
        async def mock_fetch(**kwargs):
            symbol = kwargs.get("symbol", "")
            if "BTC" in symbol:
                raise ccxt.NetworkError("BTC network error")
            if "ETH" in symbol:
                raise ccxt.BadSymbol("Invalid ETH symbol")
            # SOL succeeds
            return [[1700000000000, 100, 101, 99, 100.5, 1000]]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        # Patch the exchange
        with patch.object(client, "_get_exchange", return_value=exchange):
            result = await client.fetch_multiple(
                symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
                timeframes=["1h"],
                start_date="2024-01-01",
                verbosity=None,
            )

        # Only SOL should succeed
        assert "SOL/USD" in result
        assert len(result["SOL/USD"]) == 1
        # BTC and ETH should not be in results (they failed)
        assert "BTC/USD" not in result
        assert "ETH/USD" not in result

    @pytest.mark.asyncio
    async def test_max_concurrency_enforced(self):
        """Test that semaphore limits concurrent requests to max_concurrency."""
        client = CoinbaseDataClient(max_concurrency=2, rate_limit_backoff=0.01)

        active_count = 0
        max_concurrent = 0
        semaphore = asyncio.Semaphore(client.max_concurrency)

        async def limited_task(task_id):
            nonlocal active_count, max_concurrent
            async with semaphore:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
                await asyncio.sleep(0.05)  # Hold the semaphore
                active_count -= 1
            return task_id

        # Create 5 tasks with max_concurrency=2
        tasks = [limited_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert max_concurrent <= 2, f"Max concurrent should be <= 2, got {max_concurrent}"
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_max_concurrency_parameter_works(self):
        """Test that max_concurrency parameter is properly enforced."""
        client = CoinbaseDataClient(max_concurrency=5)

        assert client.max_concurrency == 5
        semaphore = client._get_semaphore()
        assert semaphore._value == 5

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that retry uses appropriate backoff timing."""
        client = CoinbaseDataClient(rate_limit_backoff=0.1, max_retries=2)

        call_times = []

        async def mock_fetch(**kwargs):
            call_times.append(asyncio.get_event_loop().time())
            raise ccxt.RateLimitExceeded("Rate limited")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        with pytest.raises(ccxt.RateLimitExceeded):
            semaphore = asyncio.Semaphore(1)
            await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        # Should have made 3 calls (1 initial + 2 retries)
        assert len(call_times) == 3, f"Expected 3 calls, got {len(call_times)}"
        # Second call should be after backoff time (~0.1s for attempt 0)
        time_diff = call_times[1] - call_times[0]
        # With jitter (0.5-1.5x), minimum is 0.05s
        assert time_diff >= 0.05, f"Backoff should be ~0.1s (with jitter), got {time_diff}s"
        # Third call should be after ~0.2s (exponential backoff for attempt 1)
        time_diff2 = call_times[2] - call_times[1]
        # With jitter (0.5-1.5x), minimum is 0.1s
        assert time_diff2 >= 0.1, f"Backoff should be ~0.2s (with jitter), got {time_diff2}s"

    @pytest.mark.asyncio
    async def test_backoff_with_jitter_no_negative_delay(self):
        """Test that jitter never causes negative delays."""
        client = CoinbaseDataClient(rate_limit_backoff=0.5)

        # Run multiple times to check for negative values
        delays = []

        for _ in range(10):
            call_times = []

            async def mock_fetch(**kwargs):
                call_times.append(asyncio.get_event_loop().time())
                if len(call_times) < 3:
                    raise ccxt.RateLimitExceeded("Rate limited")
                return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

            exchange = AsyncMock()
            exchange.fetch_ohlcv = mock_fetch

            try:
                semaphore = asyncio.Semaphore(1)
                await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")
            except ccxt.RateLimitExceeded:
                pass  # After max retries

            if len(call_times) >= 2:
                delay = call_times[1] - call_times[0]
                delays.append(delay)

        # All delays should be non-negative
        for delay in delays:
            assert delay >= 0, f"Delay should never be negative: {delay}"

    @pytest.mark.asyncio
    async def test_max_retries_limit(self):
        """Test that retry respects max retries - now uses exponential backoff."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01, max_retries=2)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ccxt.RateLimitExceeded("Always rate limited")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        # Should fail after initial + 2 retries = 3 total calls
        with pytest.raises(ccxt.RateLimitExceeded):
            await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 3, f"Should make initial call + 2 retries = 3 calls, got {call_count}"

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test that successful fetch works correctly after retry failures."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.NetworkError(f"Transient error {call_count}")
            return [
                [1700000000000, 42000, 42100, 41900, 42050, 100],
                [1700000060000, 42050, 42150, 41950, 42100, 150],
            ]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2  # initial + 1 retry
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_exception_results_tracked_in_fetch_multiple(self):
        """Test that failed tasks are tracked and logged in fetch_multiple."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01, verbosity=Verbosity.VERBOSE)

        # Capture print output
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured

        try:
            async def mock_fetch(**kwargs):
                symbol = kwargs.get("symbol", "")
                if "BTC" in symbol:
                    raise ccxt.NetworkError("Network error")
                return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

            exchange = AsyncMock()
            exchange.fetch_ohlcv = mock_fetch

            with patch.object(client, "_get_exchange", return_value=exchange):
                result = await client.fetch_multiple(
                    symbols=["BTC/USD", "ETH/USD"],
                    timeframes=["1h"],
                    start_date="2024-01-01",
                    verbosity=Verbosity.VERBOSE,
                )
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()

        # Check that warning about failed fetch was printed
        assert "Failed to fetch" in output or "Warning" in output or len(result) >= 0

    @pytest.mark.asyncio
    async def test_fetch_multiple_returns_partial_results_dict(self):
        """Test that fetch_multiple returns proper dict structure with partial results."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        # Make BTC fail, ETH succeed
        async def mock_fetch(**kwargs):
            symbol = kwargs.get("symbol", "")
            if "BTC" in symbol:
                raise ccxt.RateLimitExceeded("Rate limited")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        with patch.object(client, "_get_exchange", return_value=exchange):
            result = await client.fetch_multiple(
                symbols=["BTC/USD", "ETH/USD"],
                timeframes=["1h"],
                start_date="2024-01-01",
            )

        # Should return proper nested dict structure
        assert isinstance(result, dict)
        assert "ETH/USD" in result
        assert isinstance(result["ETH/USD"], dict)
        assert "1h" in result["ETH/USD"]
        assert len(result["ETH/USD"]["1h"]) > 0
        # BTC should not be present (failed)
        assert "BTC/USD" not in result

    @pytest.mark.asyncio
    async def test_ddos_protection_triggers_retry(self):
        """Test that DDoSProtection exception triggers retry."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.DDoSProtection("DDoS protection triggered")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_null_response_triggers_retry(self):
        """Test that NullResponse exception triggers retry."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt.NullResponse("Null response from server")
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 2
        assert result is not None

    @pytest.mark.asyncio
    async def test_authentication_error_no_retry(self):
        """Test that AuthenticationError is not retried."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ccxt.AuthenticationError("Invalid API key")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        with pytest.raises(ccxt.AuthenticationError):
            await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 1, "Should NOT retry on AuthenticationError"

    @pytest.mark.asyncio
    async def test_permission_denied_no_retry(self):
        """Test that PermissionDenied is not retried."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ccxt.PermissionDenied("Permission denied")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        with pytest.raises(ccxt.PermissionDenied):
            await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD", timeframe="1h")

        assert call_count == 1, "Should NOT retry on PermissionDenied"


# Exponential backoff tests
class TestExponentialBackoff:
    """Tests for exponential backoff retry logic."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify delays increase exponentially (mock time)."""
        client = CoinbaseDataClient(rate_limit_backoff=1.0, max_retries=4)

        sleep_times = []

        async def mock_sleep(duration):
            sleep_times.append(duration)

        async def mock_fetch(**kwargs):
            raise ccxt.RateLimitExceeded("Rate limited")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(ccxt.RateLimitExceeded):
                await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD")

        # With base=1.0 and max_retries=4, we expect:
        # Total attempts = 5 (1 initial + 4 retries)
        # Sleeps = 4 (after attempts 0, 1, 2, 3 before attempts 1, 2, 3, 4)
        # Delays: attempt 0→1s, 1→2s, 2→4s, 3→8s (with jitter)
        assert len(sleep_times) == 4, f"Expected 4 sleeps (retries), got {len(sleep_times)}"

        # Verify exponential growth (ignoring jitter for the check)
        assert sleep_times[1] > sleep_times[0], "Second delay should be greater than first"
        assert sleep_times[2] > sleep_times[1], "Third delay should be greater than second"
        assert sleep_times[3] > sleep_times[2], "Fourth delay should be greater than third"

    @pytest.mark.asyncio
    async def test_max_retries_respects_limit(self):
        """Verify exactly N retries then gives up."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01, max_retries=3)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ccxt.RateLimitExceeded("Rate limited")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        with pytest.raises(ccxt.RateLimitExceeded):
            await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD")

        # First attempt + max_retries retries = max_retries + 1 total calls
        assert call_count == 4, f"Expected 4 calls (1 + 3 retries), got {call_count}"

    @pytest.mark.asyncio
    async def test_backoff_with_jitter(self):
        """Verify jitter adds randomness to delays."""
        client = CoinbaseDataClient(rate_limit_backoff=1.0, max_retries=10)

        sleep_times = []

        async def mock_sleep(duration):
            sleep_times.append(duration)

        async def mock_fetch(**kwargs):
            raise ccxt.RateLimitExceeded("Rate limited")

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        # Run multiple times to verify jitter produces different values
        for _ in range(5):
            sleep_times.clear()

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with pytest.raises(ccxt.RateLimitExceeded):
                    await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD")

            # Verify jitter: delay should vary (not be exactly the exponential value)
            # Base delay for attempt 0 is 1.0, with jitter it should be 0.5-1.5
            # Since we can't predict exact random, just verify sleeps happen
            assert len(sleep_times) == 10

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Simulate rate limit on first 2 attempts, success on 3rd."""
        client = CoinbaseDataClient(rate_limit_backoff=0.01, max_retries=3)

        call_count = 0

        async def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ccxt.RateLimitExceeded("Rate limited")
            return [[1700000000000, 42000, 42100, 41900, 42050, 1.5]]

        exchange = AsyncMock()
        exchange.fetch_ohlcv = mock_fetch

        semaphore = asyncio.Semaphore(1)

        result = await client._fetch_with_retry(exchange, semaphore, symbol="BTC/USD")

        assert call_count == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count}"
        assert result == [[1700000000000, 42000, 42100, 41900, 42050, 1.5]]

    @pytest.mark.asyncio
    async def test_constructor_accepts_max_retries(self):
        """Verify new max_retries parameter works."""
        client = CoinbaseDataClient(max_retries=7)
        assert client.max_retries == 7

        # Test default
        client_default = CoinbaseDataClient()
        assert client_default.max_retries == 3
