"""Lean behavior tests for OHLCV fetching and concurrency."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import ccxt
import polars as pl
import pytest

from src.data import CoinbaseDataClient
from src.data.ohlcv import Verbosity


@pytest.mark.asyncio
async def test_fetch_continues_across_short_batches():
    """Short batches are valid and should not terminate fetch."""
    client = CoinbaseDataClient()
    start_ts = int(datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    end_ts = int(datetime(2024, 1, 1, 0, 3, tzinfo=UTC).timestamp() * 1000)
    call_since_values = []

    async def mock_fetch(*args, **kwargs):
        since = kwargs["since"]
        call_since_values.append(since)
        if since > end_ts:
            return []
        return [[since, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0]]

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        result = await client.fetch(
            "BTC/USD", "1m", "2024-01-01", end_date="2024-01-01 00:03:00"
        )

    assert len(result) == 4
    assert call_since_values[0] == start_ts
    assert call_since_values[-1] == end_ts


@pytest.mark.asyncio
async def test_fetch_raises_non_retryable_error():
    """Non-retryable errors should propagate immediately."""
    client = CoinbaseDataClient()

    async def mock_fetch(**kwargs):
        raise ccxt.BadSymbol("Invalid symbol")

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        with pytest.raises(ccxt.BadSymbol):
            await client.fetch("INVALID/PAIR", "1m", "2024-01-01", "2024-01-01 02:00")


@pytest.mark.asyncio
async def test_fetch_retries_on_rate_limit_then_succeeds():
    """Retryable errors should retry and then succeed."""
    client = CoinbaseDataClient(rate_limit_backoff=0.01)
    call_count = 0
    candles = [[1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0]]

    async def mock_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ccxt.RateLimitExceeded("Rate limited")
        return candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        result = await client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01 00:00:00")

    assert call_count >= 2
    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_guard_exits_on_non_advancing_response():
    """Fetch should exit when exchange response does not advance."""
    client = CoinbaseDataClient()
    call_count = 0

    async def mock_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        since = kwargs["since"]
        return [[since - 60000, 1.0, 1.0, 1.0, 1.0, 1.0]]

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        result = await asyncio.wait_for(
            client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01 00:10:00"),
            timeout=1.0,
        )

    assert call_count == 1
    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_includes_exact_end_timestamp():
    """Candle exactly on end_date boundary should be included."""
    client = CoinbaseDataClient()
    ts_1 = int(datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    ts_2 = int(datetime(2024, 1, 1, 0, 1, tzinfo=UTC).timestamp() * 1000)
    ts_3 = int(datetime(2024, 1, 1, 0, 2, tzinfo=UTC).timestamp() * 1000)
    mock_candles = [
        [ts_1, 1.0, 1.0, 1.0, 1.0, 1.0],
        [ts_2, 2.0, 2.0, 2.0, 2.0, 2.0],
        [ts_3, 3.0, 3.0, 3.0, 3.0, 3.0],
    ]

    async def mock_fetch(**kwargs):
        return mock_candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        result = await client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01 00:01:00")

    timestamps = result["timestamp"].to_list()
    assert datetime(2024, 1, 1, 0, 1, tzinfo=UTC) in timestamps
    assert datetime(2024, 1, 1, 0, 2, tzinfo=UTC) not in timestamps


@pytest.mark.asyncio
async def test_fetch_multiple_returns_partial_results_on_failures():
    """fetch_multiple should preserve successful combos when others fail."""
    client = CoinbaseDataClient()

    async def mock_fetch(symbol, *args, **kwargs):
        if symbol == "BTC/USD":
            raise Exception("Simulated failure")
        return pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )

    with patch.object(client, "fetch", side_effect=mock_fetch):
        result = await client.fetch_multiple(
            symbols=["BTC/USD", "ETH/USD"],
            timeframes=["1h", "1d"],
            start_date="2025-01-01",
            end_date="2025-01-07",
        )

    assert "BTC/USD" not in result
    assert "ETH/USD" in result
    assert set(result["ETH/USD"].keys()) == {"1h", "1d"}


@pytest.mark.asyncio
async def test_fetch_multiple_marks_failure_in_progress():
    """Failed combos should be marked in shared progress updates."""
    client = CoinbaseDataClient()
    progress_updates = []

    class MockProgress:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, task_id, **kwargs):
            progress_updates.append(kwargs)

    async def mock_fetch(symbol, *args, **kwargs):
        if symbol == "BTC/USD":
            raise Exception("Failure")
        return pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )

    with patch.object(client, "fetch", side_effect=mock_fetch):
        with patch("src.data.ohlcv.Progress", MockProgress):
            await client.fetch_multiple(
                symbols=["BTC/USD", "ETH/USD"],
                timeframes=["1h"],
                start_date="2025-01-01",
                verbosity=Verbosity.PROGRESS,
            )

    assert any(update.get("completed") == 0 for update in progress_updates)


@pytest.mark.asyncio
async def test_fetch_and_save_streams_batches(tmp_path):
    """fetch_and_save should stream-write batches without collecting full result."""
    client = CoinbaseDataClient(data_dir=str(tmp_path))
    saved_batches = []

    async def mock_save_async(df, symbol, timeframe):
        saved_batches.append((symbol, timeframe, len(df)))

    with patch.object(client, "save_async", side_effect=mock_save_async):
        with patch.object(client, "fetch") as mock_fetch:
            async def run_fetch(**kwargs):
                on_batch = kwargs["on_batch"]
                await on_batch(
                    pl.DataFrame(
                        {
                            "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)],
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        }
                    )
                )
                return pl.DataFrame()

            mock_fetch.side_effect = run_fetch
            await client.fetch_and_save("BTC/USD", "1h", "2025-01-01", "2025-01-02")

    assert saved_batches == [("BTC/USD", "1h", 1)]


@pytest.mark.asyncio
async def test_fetch_multiple_and_save_runs_all_combinations():
    """fetch_multiple_and_save should dispatch every symbol/timeframe pair."""
    client = CoinbaseDataClient()
    calls = []

    async def mock_fetch_and_save(symbol, timeframe, start_date, end_date, verbosity):
        calls.append((symbol, timeframe, start_date, end_date))

    with patch.object(client, "fetch_and_save", side_effect=mock_fetch_and_save):
        await client.fetch_multiple_and_save(
            symbols=["BTC/USD", "ETH/USD"],
            timeframes=["1h", "1d"],
            start_date="2025-01-01",
            end_date="2025-01-02",
        )

    assert len(calls) == 4
    assert ("BTC/USD", "1h", "2025-01-01", "2025-01-02") in calls
