"""Lean behavior tests for OHLCV fetching and concurrency."""

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_fetch_retries_on_retryable_exchange_error_then_succeeds():
    """Transient ExchangeError payloads should be retried."""
    client = CoinbaseDataClient(rate_limit_backoff=0.01)
    call_count = 0
    candles = [[1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 1000.0]]

    async def mock_fetch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ccxt.ExchangeError(
                'coinbaseadvanced {"error":"UNAVAILABLE","message":"Something went wrong"}'
            )
        return candles

    with patch.object(client, "_get_exchange") as mock_exchange:
        mock_exchange_instance = AsyncMock()
        mock_exchange_instance.fetch_ohlcv = mock_fetch
        mock_exchange.return_value = mock_exchange_instance
        result = await client.fetch("BTC/USD", "1m", "2024-01-01", "2024-01-01 00:00:00")

    assert call_count >= 2
    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_retry_releases_semaphore_during_backoff():
    """A retrying request should not hold the semaphore while sleeping."""
    client = CoinbaseDataClient(
        max_retries=1,
        rate_limit_backoff=0.05,
        min_request_interval=0.0,
    )
    semaphore = asyncio.Semaphore(1)
    attempts = {1: 0, 2: 0}
    events = []

    async def mock_fetch(**kwargs):
        since = kwargs["since"]
        attempts[since] += 1
        events.append((since, attempts[since], time.perf_counter()))
        if since == 1 and attempts[since] == 1:
            raise ccxt.RateLimitExceeded("Rate limited")
        return [[since, 1.0, 1.0, 1.0, 1.0, 1.0]]

    exchange = AsyncMock()
    exchange.fetch_ohlcv = mock_fetch

    task_a = asyncio.create_task(
        client._fetch_with_retry(  # noqa: SLF001 - behavior validation
            exchange, semaphore, symbol="BTC/USD", timeframe="1m", since=1, limit=1
        )
    )
    await asyncio.sleep(0.005)
    task_b = asyncio.create_task(
        client._fetch_with_retry(  # noqa: SLF001 - behavior validation
            exchange, semaphore, symbol="ETH/USD", timeframe="1m", since=2, limit=1
        )
    )

    await asyncio.gather(task_a, task_b)

    first_retry_idx = next(
        i for i, (since, attempt, _) in enumerate(events) if since == 1 and attempt == 2
    )
    second_request_idx = next(
        i for i, (since, attempt, _) in enumerate(events) if since == 2 and attempt == 1
    )
    assert second_request_idx < first_retry_idx


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
async def test_fetch_verbose_streaming_reports_total_candles(capsys):
    """Verbose fetch should report fetched count even when not collecting results."""
    client = CoinbaseDataClient(
        batch_concurrency=1,
        enable_intra_combo_concurrency=False,
        min_request_interval=0.0,
        verbosity=Verbosity.VERBOSE,
    )
    start_ts = int(datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)

    async def mock_fetch_batch(_exchange, _semaphore, _symbol, _timeframe, since):
        if since == start_ts:
            return [[start_ts, 1.0, 1.0, 1.0, 1.0, 1.0]]
        return []

    with patch.object(client, "_get_exchange", return_value=object()):
        with patch.object(client, "_fetch_batch", side_effect=mock_fetch_batch):
            result = await client.fetch(
                symbol="BTC/USD",
                timeframe="1m",
                start_date="2024-01-01 00:00:00",
                end_date="2024-01-01 00:00:00",
                verbosity=Verbosity.VERBOSE,
                collect_results=False,
            )

    captured = capsys.readouterr().out
    assert result.is_empty()
    assert "Completed fetch for BTC/USD 1m: 1 candles fetched" in captured


@pytest.mark.asyncio
async def test_fetch_multiple_and_save_runs_all_combinations():
    """fetch_multiple_and_save should dispatch every symbol/timeframe pair."""
    client = CoinbaseDataClient()
    calls = []

    async def mock_fetch_and_save(
        symbol,
        timeframe,
        start_date,
        end_date,
        verbosity,
        **_kwargs,
    ):
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


@pytest.mark.asyncio
async def test_fetch_multiple_and_save_continues_when_one_combination_fails():
    """fetch_multiple_and_save should not abort on a single combination failure."""
    client = CoinbaseDataClient()
    calls = []

    async def mock_fetch_and_save(
        symbol,
        timeframe,
        start_date,
        end_date,
        verbosity,
        **_kwargs,
    ):
        calls.append((symbol, timeframe))
        if symbol == "BTC/USD" and timeframe == "1h":
            raise ccxt.ExchangeError("coinbaseadvanced {\"error\":\"UNAVAILABLE\"}")

    with patch.object(client, "fetch_and_save", side_effect=mock_fetch_and_save):
        await client.fetch_multiple_and_save(
            symbols=["BTC/USD", "ETH/USD"],
            timeframes=["1h", "1d"],
            start_date="2025-01-01",
            end_date="2025-01-02",
        )

    assert len(calls) == 4
    assert ("ETH/USD", "1h") in calls


@pytest.mark.asyncio
async def test_fetch_single_combo_uses_concurrent_bounded_batches():
    """Bounded single-combo fetch should run multiple batch windows concurrently."""
    client = CoinbaseDataClient(
        batch_concurrency=3,
        batch_queue_size=6,
        min_request_interval=0.0,
    )
    active_fetches = 0
    max_active_fetches = 0
    called_since: list[int] = []

    async def mock_fetch_batch(_exchange, _semaphore, _symbol, _timeframe, since):
        nonlocal active_fetches, max_active_fetches
        called_since.append(since)
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        await asyncio.sleep(0.02)
        active_fetches -= 1
        return [
            [since + (i * 60_000), 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(300)
        ]

    expected_start = int(datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    batch_span_ms = 300 * 60 * 1000  # 300 candles x 1m timeframe
    expected_windows = [
        expected_start,
        expected_start + batch_span_ms,
        expected_start + (2 * batch_span_ms),
    ]

    with patch.object(client, "_get_exchange", return_value=object()):
        with patch.object(client, "_fetch_batch", side_effect=mock_fetch_batch):
            result = await client.fetch(
                symbol="BTC/USD",
                timeframe="1m",
                start_date="2024-01-01 00:00:00",
                end_date="2024-01-01 12:00:00",
            )

    assert max_active_fetches > 1
    assert sorted(called_since) == expected_windows
    assert len(result) == 721


@pytest.mark.asyncio
async def test_fetch_unbounded_stays_sequential_despite_batch_concurrency():
    """Unbounded fetch should keep legacy sequential progression behavior."""
    client = CoinbaseDataClient(
        batch_concurrency=4,
        batch_queue_size=8,
        min_request_interval=0.0,
    )
    active_fetches = 0
    max_active_fetches = 0
    call_count = 0
    called_since: list[int] = []

    async def mock_fetch_batch(_exchange, _semaphore, _symbol, _timeframe, since):
        nonlocal active_fetches, max_active_fetches, call_count
        called_since.append(since)
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        await asyncio.sleep(0.01)
        active_fetches -= 1
        call_count += 1
        if call_count >= 3:
            return []
        return [[since, 1.0, 1.0, 1.0, 1.0, 1.0]]

    with patch.object(client, "_get_exchange", return_value=object()):
        with patch.object(client, "_fetch_batch", side_effect=mock_fetch_batch):
            result = await client.fetch(
                symbol="BTC/USD",
                timeframe="1m",
                start_date="2024-01-01",
                end_date=None,
            )

    assert max_active_fetches == 1
    assert len(called_since) == 3
    assert called_since[1] > called_since[0]
    assert called_since[2] > called_since[1]
    assert len(result) == 2


@pytest.mark.asyncio
async def test_fetch_concurrent_bounded_handles_sparse_short_batches():
    """Concurrent bounded mode should still advance through sparse short batches."""
    client = CoinbaseDataClient(
        batch_concurrency=3,
        min_request_interval=0.0,
    )
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
            "BTC/USD",
            "1m",
            "2024-01-01",
            end_date="2024-01-01 00:03:00",
        )

    assert len(result) == 4
    assert min(call_since_values) == start_ts
    assert end_ts in call_since_values


def test_shared_progress_updates_each_batch_immediately():
    """Shared progress should render every processed batch without large jumps."""
    client = CoinbaseDataClient()
    shared_progress = MagicMock()

    pending = client._update_progress(
        pending_advance=0,
        progress_task_id=123,
        shared_progress=shared_progress,
        progress_tracker=None,
        use_shared_progress=True,
        candles_so_far=300,
        expected_candles=1200,
    )

    assert pending == 0
    shared_progress.update.assert_called_once_with(123, advance=1)


def test_activity_progress_tracks_active_completed_and_failed():
    """Activity counters should reflect active, completed, and failed updates."""
    client = CoinbaseDataClient()
    shared_progress = MagicMock()
    activity_state = {
        "task_id": 99,
        "active": 0,
        "completed": 0,
        "failed": 0,
        "total": 10,
    }

    client._update_activity_progress(
        shared_progress,
        activity_state,
        active_delta=1,
    )
    client._update_activity_progress(
        shared_progress,
        activity_state,
        advance=2,
        active_delta=-1,
        failed_increment=1,
    )

    assert activity_state["active"] == 0
    assert activity_state["completed"] == 2
    assert activity_state["failed"] == 1
    assert shared_progress.update.call_count == 2
    assert shared_progress.update.call_args_list[-1].kwargs["advance"] == 2
