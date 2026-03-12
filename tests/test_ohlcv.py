"""Tests for OHLCV module."""

import asyncio
from datetime import datetime
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


def test_client_has_fetch_method():
    """Test that CoinbaseDataClient has fetch method."""
    client = CoinbaseDataClient()
    assert hasattr(client, "fetch")
    assert callable(client.fetch)


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
