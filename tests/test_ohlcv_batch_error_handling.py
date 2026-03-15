"""Tests for batch error handling in OHLCV fetch."""

import asyncio
from unittest.mock import AsyncMock, patch

import ccxt
import pytest

from src.data import CoinbaseDataClient


@pytest.fixture
def client():
    """Create a test client."""
    return CoinbaseDataClient(max_retries=0)  # No retries for faster tests


class TestBatchErrorHandling:
    """Tests for per-batch error handling in fetch()."""

    @pytest.mark.asyncio
    async def test_fetch_continues_on_rate_limit(self, client):
        """Test that fetch continues to next batch when a batch fails with rate limit."""
        # Return 300 candles per batch to simulate more data available
        # Batch 1: succeeds, Batch 2: fails, Batch 3: succeeds
        base_ts = 1704067200000  # 2024-01-01 00:00:00

        def make_candles(start_idx: int) -> list:
            return [[base_ts + i * 60000, 42000.0 + i, 42100.0 + i, 41900.0 + i, 42050.0 + i, 100.0]
                    for i in range(start_idx, start_idx + 300)]

        batch2_exception = ccxt.RateLimitExceeded("Rate limit exceeded")

        call_count = 0

        async def mock_fetch(since: int, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise batch2_exception
            # Return candles starting from the since parameter
            start_idx = (since - base_ts) // 60000
            return make_candles(start_idx)

        with patch.object(client, "_get_exchange") as mock_get_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_get_exchange.return_value = mock_exchange_instance

            with patch.object(client, "_get_semaphore") as mock_get_semaphore:
                mock_sem = AsyncMock()
                mock_sem.__aenter__ = AsyncMock(return_value=None)
                mock_sem.__aexit__ = AsyncMock(return_value=None)
                mock_get_semaphore.return_value = mock_sem

                result = await client.fetch(
                    symbol="BTC/USD",
                    timeframe="1m",
                    start_date="2024-01-01",
                    end_date="2024-01-01 02:00",
                )

        # Should have returned partial data (batches 1 and 3)
        # Total candles from start to end_date: 2 hours * 60 + 1 = 121 candles
        assert len(result) == 121
        # First candle should be from start_date
        # Note: Polars stores timestamps in microseconds, so we multiply by 1000
        assert result["timestamp"].dt.timestamp()[0] == 1704067200000 * 1000

    @pytest.mark.asyncio
    async def test_fetch_raises_on_bad_symbol(self, client):
        """Test that fetch raises immediately on non-retryable errors like BadSymbol."""
        async def mock_fetch(**kwargs):
            raise ccxt.BadSymbol("Invalid symbol")

        with patch.object(client, "_get_exchange") as mock_get_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_get_exchange.return_value = mock_exchange_instance

            with patch.object(client, "_get_semaphore") as mock_get_semaphore:
                mock_sem = AsyncMock()
                mock_sem.__aenter__ = AsyncMock(return_value=None)
                mock_sem.__aexit__ = AsyncMock(return_value=None)
                mock_get_semaphore.return_value = mock_sem

                with pytest.raises(ccxt.BadSymbol):
                    await client.fetch(
                        symbol="INVALID/PAIR",
                        timeframe="1m",
                        start_date="2024-01-01",
                        end_date="2024-01-01 02:00",
                    )

    @pytest.mark.asyncio
    async def test_fetch_all_batches_fail(self, client):
        """Test that fetch returns empty DataFrame when all batches fail."""
        async def mock_fetch(**kwargs):
            raise ccxt.RateLimitExceeded("Rate limit")

        with patch.object(client, "_get_exchange") as mock_get_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_get_exchange.return_value = mock_exchange_instance

            with patch.object(client, "_get_semaphore") as mock_get_semaphore:
                mock_sem = AsyncMock()
                mock_sem.__aenter__ = AsyncMock(return_value=None)
                mock_sem.__aexit__ = AsyncMock(return_value=None)
                mock_get_semaphore.return_value = mock_sem

                result = await client.fetch(
                    symbol="BTC/USD",
                    timeframe="1m",
                    start_date="2024-01-01",
                    end_date="2024-01-01 02:00",
                )

        # Should return empty DataFrame with correct schema
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    @pytest.mark.asyncio
    async def test_fetch_partial_success_with_multiple_failures(self, client):
        """Test that partial data is returned when some batches fail."""
        base_ts = 1704067200000

        def make_candles(start_idx: int) -> list:
            return [[base_ts + i * 60000, 42000.0 + i, 42100.0 + i, 41900.0 + i, 42050.0 + i, 100.0]
                    for i in range(start_idx, start_idx + 300)]

        call_count = 0

        async def mock_fetch(since: int, **kwargs):
            nonlocal call_count
            call_count += 1
            # Batches 1, 3 succeed; batch 2 fails
            if call_count == 2:
                raise ccxt.RateLimitExceeded("Rate limit")
            start_idx = (since - base_ts) // 60000
            return make_candles(start_idx)

        with patch.object(client, "_get_exchange") as mock_get_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_get_exchange.return_value = mock_exchange_instance

            with patch.object(client, "_get_semaphore") as mock_get_semaphore:
                mock_sem = AsyncMock()
                mock_sem.__aenter__ = AsyncMock(return_value=None)
                mock_sem.__aexit__ = AsyncMock(return_value=None)
                mock_get_semaphore.return_value = mock_sem

                result = await client.fetch(
                    symbol="BTC/USD",
                    timeframe="1m",
                    start_date="2024-01-01",
                    end_date="2024-01-01 02:00",
                )

        # Should have returned data from successful batches
        assert len(result) >= 1


class TestFetchMultipleProgressBar:
    """Tests for progress bar behavior in fetch_multiple."""

    @pytest.mark.asyncio
    async def test_fetch_multiple_progress_bar_updated_on_failure(self):
        """Test that progress bar is updated when a task fails in fetch_multiple."""
        from src.data.ohlcv import Verbosity

        client = CoinbaseDataClient(max_retries=0)

        # Track progress bar updates
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

        # Mock fetch that raises non-retryable exception for BTC
        # (so it propagates up to fetch_multiple instead of being caught per-batch)
        async def mock_fetch(**kwargs):
            symbol = kwargs.get("symbol", "")
            if "BTC" in symbol:
                raise ccxt.BadSymbol("Invalid symbol")  # Non-retryable
            return [[1700000000000, 42000, 42100, 41900, 42050, 100]]

        with patch.object(client, "_get_exchange") as mock_get_exchange:
            mock_exchange_instance = AsyncMock()
            mock_exchange_instance.fetch_ohlcv = mock_fetch
            mock_get_exchange.return_value = mock_exchange_instance

            with patch.object(client, "_get_semaphore") as mock_get_semaphore:
                mock_sem = AsyncMock()
                mock_sem.__aenter__ = AsyncMock(return_value=None)
                mock_sem.__aexit__ = AsyncMock(return_value=None)
                mock_get_semaphore.return_value = mock_sem

                with patch("src.data.ohlcv.Progress", MockProgress):
                    result = await client.fetch_multiple(
                        symbols=["BTC/USD", "ETH/USD"],
                        timeframes=["1h"],
                        start_date="2024-01-01",
                        verbosity=Verbosity.PROGRESS,
                    )

        # Verify progress bar was updated for failed task
        # Should have update calls with completed=0 for failed task
        failed_updates = [u for u in progress_updates if u.get("completed") == 0]
        assert len(failed_updates) > 0, "Progress bar should be updated for failed task"

        # Verify partial results returned
        assert len(result) == 1  # Only ETH should succeed
        assert "ETH/USD" in result
