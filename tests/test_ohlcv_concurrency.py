"""Concurrency tests for OHLCV fetch_multiple with exception handling."""

from datetime import UTC, datetime
from unittest.mock import patch

import polars as pl
import pytest

from src.data import CoinbaseDataClient


def create_sample_df(symbol: str) -> pl.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    return pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [1000.0],
        }
    )


@pytest.mark.asyncio
async def test_gather_preserves_partial_results(tmp_path):
    """Test that when 8/10 tasks succeed, 8 results are returned."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    call_count = 0

    async def mock_fetch_with_partial_failure(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First 8 calls succeed, last 2 fail
        if call_count <= 8:
            return create_sample_df("test")
        else:
            raise Exception("Simulated failure")

    with patch.object(client, "fetch", side_effect=mock_fetch_with_partial_failure):
        symbols = [f"TEST{i}/USD" for i in range(10)]
        timeframes = ["1h"]

        result = await client.fetch_multiple(
            symbols=symbols,
            timeframes=timeframes,
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

    # Should have 8 successful results
    assert len(result) == 8
    for symbol, timeframes_dict in result.items():
        assert "1h" in timeframes_dict
        assert len(timeframes_dict["1h"]) > 0


@pytest.mark.asyncio
async def test_gather_handles_rate_limit_exception(tmp_path):
    """Test that RateLimitExceeded on one task allows others to complete."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    call_count = 0

    async def mock_fetch_with_rate_limit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # 3rd call raises rate limit error
        if call_count == 3:
            raise Exception("RateLimitExceeded: Too many requests")
        return create_sample_df("test")

    with patch.object(client, "fetch", side_effect=mock_fetch_with_rate_limit):
        # Use 4 unique symbols so we can track each result separately
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
        timeframes = ["1h"]

        result = await client.fetch_multiple(
            symbols=symbols,
            timeframes=timeframes,
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

    # Should have 3 successful results (one failed)
    assert len(result) == 3
    # Verify no exceptions propagated
    for symbol, timeframes_dict in result.items():
        assert isinstance(timeframes_dict, dict)


@pytest.mark.asyncio
async def test_gather_handles_network_error(tmp_path):
    """Test that NetworkError on one task allows others to complete."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    call_count = 0

    async def mock_fetch_with_network_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # 2nd call raises network error
        if call_count == 2:
            raise Exception("NetworkError: Connection failed")
        return create_sample_df("test")

    with patch.object(client, "fetch", side_effect=mock_fetch_with_network_error):
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        timeframes = ["1h"]

        result = await client.fetch_multiple(
            symbols=symbols,
            timeframes=timeframes,
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

    # Should have 2 successful results
    assert len(result) == 2


@pytest.mark.asyncio
async def test_gather_handles_multiple_exceptions(tmp_path):
    """Test that multiple failed tasks still preserve partial results."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    call_count = 0

    async def mock_fetch_multiple_failures(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on calls 2, 4, 6
        if call_count in (2, 4, 6):
            raise Exception("Simulated failure")
        return create_sample_df("test")

    with patch.object(client, "fetch", side_effect=mock_fetch_multiple_failures):
        symbols = [f"TEST{i}/USD" for i in range(8)]
        timeframes = ["1h"]

        result = await client.fetch_multiple(
            symbols=symbols,
            timeframes=timeframes,
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

    # Should have 5 successful results (3 failed)
    assert len(result) == 5


@pytest.mark.asyncio
async def test_fetch_multiple_with_failures(tmp_path):
    """Integration test - one symbol fails, others succeed."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))

    async def mock_fetch(symbol, *args, **kwargs):
        # BTC/USD always fails
        if symbol == "BTC/USD":
            raise Exception("BTC API failure")
        return create_sample_df(symbol)

    with patch.object(client, "fetch", side_effect=mock_fetch):
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        timeframes = ["1h", "1d"]

        result = await client.fetch_multiple(
            symbols=symbols,
            timeframes=timeframes,
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

    # BTC/USD should be missing, ETH and SOL should be present
    assert "BTC/USD" not in result
    assert "ETH/USD" in result
    assert "SOL/USD" in result

    # Check both timeframes for successful symbols
    assert "1h" in result["ETH/USD"]
    assert "1d" in result["ETH/USD"]
    assert "1h" in result["SOL/USD"]
    assert "1d" in result["SOL/USD"]
