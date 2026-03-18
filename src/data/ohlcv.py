"""OHLCV data retrieval from Coinbase."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import ccxt
import ccxt.async_support
import polars as pl
from rich.progress import Progress

from src.data.ohlcv_fetch import execute_fetch, execute_fetch_multiple
from src.data.ohlcv_storage import load_partitions, save_partitions, symbol_to_path
from src.data.progress import (
    TIMEFRAME_SECONDS,
    ProgressTracker,
    RichProgressManager,
    _is_test_environment,
    build_shared_progress,
    calculate_expected_batches,
    create_activity_state,
    format_activity_description,
    get_progress_color,
)

logger = logging.getLogger(__name__)


class Verbosity(Enum):
    """Verbosity levels for output control."""

    DISABLED = "disabled"
    PROGRESS = "progress"
    VERBOSE = "verbose"


# Schema for OHLCV data with required columns
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Supported timeframes
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "6h", "1d"]

# Coinbase API limit
COINBASE_CANDLE_LIMIT = 300

# Progress update interval for shared Rich progress updates.
# Keep this at 1 to provide immediate visual feedback during concurrent fetches.
PROGRESS_UPDATE_INTERVAL = 1

# Safety guard for unbounded historical fetches with persistent failures when end_date is omitted
MAX_CONSECUTIVE_RETRYABLE_FAILURES_NO_END_DATE = 5

# Retryable exceptions that should trigger a retry attempt
RETRYABLE_EXCEPTIONS = (
    ccxt.RateLimitExceeded,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
    ccxt.NullResponse,
)

# Permanent exceptions that should not be retried - log and re-raise
NON_RETRYABLE_EXCEPTIONS = (
    ccxt.BadSymbol,
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.InvalidNonce,
)


class CoinbaseDataClient:
    """Client for fetching OHLCV data from Coinbase."""

    @staticmethod
    def _calculate_expected_candles(start_ts: int, end_ts: int, timeframe: str) -> int:
        """Calculate the expected number of candles for a time range.

        Args:
            start_ts: Start timestamp in milliseconds.
            end_ts: End timestamp in milliseconds.
            timeframe: Timeframe string (e.g., "1m", "5m", "1h").

        Returns:
            Expected number of candles as an integer.
        """
        return (end_ts - start_ts) // (1000 * TIMEFRAME_SECONDS[timeframe])

    def __init__(
        self,
        data_dir: str = "./data",
        max_concurrency: int = 10,
        min_request_interval: float = 0.06,
        rate_limit_backoff: float = 1.0,
        max_retries: int = 3,
        batch_concurrency: int = 1,
        batch_queue_size: int | None = None,
        enable_intra_combo_concurrency: bool = True,
        api_key: str | None = None,
        private_key: str | None = None,
        verbosity: Verbosity | None = None,
    ) -> None:
        """Initialize the Coinbase data client.

        Args:
            data_dir: Directory for storing fetched data.
            max_concurrency: Maximum number of concurrent requests.
            min_request_interval: Minimum seconds between request starts across all
                workers. Adds gentle global pacing to reduce 429 rate limits.
            rate_limit_backoff: Initial backoff time in seconds for rate limiting.
            max_retries: Maximum number of retry attempts with exponential backoff.
            batch_concurrency: Concurrent batch fetches per symbol/timeframe combo
                for bounded historical ranges.
            batch_queue_size: Max in-flight batch write tasks per combo. Defaults to
                max(2, batch_concurrency * 2) when not provided.
            enable_intra_combo_concurrency: Enable concurrent bounded batch fetching
                within a single symbol/timeframe combo.
            api_key: Optional Coinbase API key for authenticated requests.
            private_key: Optional Coinbase private key (PEM format) for authenticated requests.
            verbosity: Verbosity level for output control. If None, auto-detects test environment.
        """
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.min_request_interval = min_request_interval
        self.rate_limit_backoff = rate_limit_backoff
        self.max_retries = max_retries
        self.batch_concurrency = max(1, batch_concurrency)
        default_batch_queue_size = max(2, self.batch_concurrency * 2)
        self.batch_queue_size = (
            default_batch_queue_size
            if batch_queue_size is None
            else max(1, batch_queue_size)
        )
        self.enable_intra_combo_concurrency = enable_intra_combo_concurrency
        self._api_key = api_key
        self._private_key = private_key

        # Auto-detect verbosity: disable in test environment if not explicitly set
        if verbosity is not None:
            # Explicitly provided - always use it
            self.verbosity = verbosity
        elif _is_test_environment():
            # Not provided and in test environment - disable by default
            self.verbosity = Verbosity.DISABLED
        else:
            # Not provided and not in test environment - default to progress
            self.verbosity = Verbosity.PROGRESS

        self._exchange: ccxt.async_support.coinbaseadvanced | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._request_gate: asyncio.Lock | None = None
        self._save_locks: dict[str, asyncio.Lock] = {}
        self._save_locks_guard = asyncio.Lock()
        self._next_request_at: float = 0.0

    @classmethod
    def from_settings(cls, settings, **kwargs) -> "CoinbaseDataClient":
        """Create a CoinbaseDataClient from a settings object.

        Args:
            settings: Settings object with coinbase_api_key and coinbase_private_key.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            CoinbaseDataClient instance with credentials from settings.
        """
        return cls(
            api_key=settings.coinbase_api_key,
            private_key=settings.coinbase_private_key,
            **kwargs,
        )

    def _get_exchange(self) -> ccxt.async_support.coinbaseadvanced:
        """Get or create the ccxt exchange instance."""
        if self._exchange is None:
            if self._api_key and self._private_key:
                self._exchange = ccxt.async_support.coinbaseadvanced(
                    {
                        "apiKey": self._api_key,
                        "secret": self._private_key,
                    }
                )
            else:
                self._exchange = ccxt.async_support.coinbaseadvanced()
        return self._exchange

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def _get_request_gate(self) -> asyncio.Lock:
        """Get or create the request pacing lock."""
        if self._request_gate is None:
            self._request_gate = asyncio.Lock()
        return self._request_gate

    async def _wait_for_request_slot(self) -> None:
        """Enforce minimum spacing between outbound request starts."""
        if self.min_request_interval <= 0:
            return

        loop = asyncio.get_running_loop()
        gate = self._get_request_gate()
        async with gate:
            now = loop.time()
            delay = self._next_request_at - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = loop.time()
            self._next_request_at = (
                max(now, self._next_request_at) + self.min_request_interval
            )

    def _resolve_verbosity(self, verbosity: Verbosity | None) -> Verbosity:
        """Resolve effective verbosity level.

        Args:
            verbosity: Optional verbosity override. If None, uses instance default.

        Returns:
            Effective Verbosity level to use.
        """
        return verbosity if verbosity is not None else self.verbosity

    @staticmethod
    def _progress_tracker_factory(**kwargs: Any) -> ProgressTracker:
        """Factory for progress tracker instances used by fetch engine."""
        return RichProgressManager(**kwargs)

    @staticmethod
    async def _gather_with_exceptions(tasks: list[Any]) -> list[Any]:
        """Gather tasks while preserving exceptions in results."""
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _apply_end_of_day(dt: datetime, end_of_day: bool) -> datetime:
        """Apply end-of-day conversion if datetime has no time component.

        Args:
            dt: The datetime to potentially modify.
            end_of_day: If True and datetime has no time component, set to 23:59:59.999.

        Returns:
            Modified datetime if conditions met, otherwise unchanged.
        """
        if (
            end_of_day
            and dt.hour == 0
            and dt.minute == 0
            and dt.second == 0
            and dt.microsecond == 0
        ):
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt

    @staticmethod
    def _parse_date(date_str: str, end_of_day: bool = False) -> datetime:
        """Parse date string to datetime.

        Args:
            date_str: Date string in various formats.
            end_of_day: If True and date has no time component, set time to 23:59:59.999.

        Returns:
            datetime object in UTC.
        """
        # Try parsing as ISO format first
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
            return CoinbaseDataClient._apply_end_of_day(dt, end_of_day)
        except ValueError:
            pass

        # Try parsing common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                dt = CoinbaseDataClient._apply_end_of_day(dt, end_of_day)
                return dt.replace(tzinfo=UTC)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_str}")

    @staticmethod
    def _candles_to_dataframe(candles: list[list[float]]) -> pl.DataFrame:
        """Convert ccxt candle list to Polars DataFrame.

        Args:
            candles: List of ccxt candles, where each candle is a list of
                [timestamp_ms, open, high, low, close, volume].

        Returns:
            Polars DataFrame with OHLCV columns matching OHLCV_SCHEMA.
        """
        if not candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        return pl.DataFrame(
            {
                "timestamp": [
                    datetime.fromtimestamp(c[0] / 1000, tz=UTC) for c in candles
                ],
                "open": [c[1] for c in candles],
                "high": [c[2] for c in candles],
                "low": [c[3] for c in candles],
                "close": [c[4] for c in candles],
                "volume": [c[5] for c in candles],
            },
            schema=OHLCV_SCHEMA,
        )

    def _validate_timeframe(self, timeframe: str) -> None:
        """Validate that the timeframe is supported.

        Args:
            timeframe: The timeframe to validate.

        Raises:
            ValueError: If timeframe is not supported.
        """
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. Supported: {SUPPORTED_TIMEFRAMES}"
            )

    def _update_progress(
        self,
        pending_advance: int,
        progress_task_id: Any,
        shared_progress: Progress | None,
        progress_tracker: ProgressTracker | None,
        use_shared_progress: bool,
        candles_so_far: int,
        expected_candles: int,
        activity_state: dict[str, Any] | None = None,
    ) -> int:
        """Update progress after a batch is processed.

        Args:
            pending_advance: Current pending advance count.
            progress_task_id: Task ID from shared Rich Progress.
            shared_progress: Shared Rich Progress instance.
            progress_tracker: Individual ProgressTracker instance.
            use_shared_progress: Whether to use shared progress.
            candles_so_far: Number of candles fetched so far.
            expected_candles: Expected total candles.
            activity_state: Shared aggregate activity counters for batch fetches.

        Returns:
            Updated pending_advance count (0 if flushed, else same as input).
        """
        if use_shared_progress and shared_progress is not None:
            # Batch updates to reduce contention
            pending_advance += 1
            if pending_advance >= PROGRESS_UPDATE_INTERVAL:
                shared_progress.update(progress_task_id, advance=pending_advance)
                self._update_activity_progress(
                    shared_progress,
                    activity_state,
                    advance=pending_advance,
                )
                return 0
        elif progress_tracker is not None:
            progress_tracker.update(
                n=1,
                candles_so_far=candles_so_far,
                total_candles=expected_candles,
            )
        return pending_advance

    def _flush_progress(
        self,
        pending_advance: int,
        progress_task_id: Any,
        shared_progress: Progress | None,
        progress_tracker: ProgressTracker | None,
        use_shared_progress: bool,
        extra: int = 0,
        activity_state: dict[str, Any] | None = None,
    ) -> None:
        """Flush pending progress updates.

        Args:
            pending_advance: Number of batches to advance the progress.
            progress_task_id: Task ID from shared Rich Progress.
            shared_progress: Shared Rich Progress instance.
            progress_tracker: Individual ProgressTracker instance.
            use_shared_progress: Whether to use shared progress.
            extra: Additional count to add to the advance (default 0).
            activity_state: Shared aggregate activity counters for batch fetches.
        """
        total_advance = pending_advance + extra
        if use_shared_progress and shared_progress is not None and total_advance > 0:
            shared_progress.update(progress_task_id, advance=total_advance)
            self._update_activity_progress(
                shared_progress,
                activity_state,
                advance=total_advance,
            )
        elif progress_tracker is not None and total_advance > 0:
            progress_tracker.update(n=total_advance)

    @staticmethod
    def _update_activity_progress(
        shared_progress: Progress | None,
        activity_state: dict[str, Any] | None,
        *,
        advance: int = 0,
        failed_increment: int = 0,
        active_delta: int = 0,
    ) -> None:
        """Update aggregate activity status displayed in shared progress."""
        if shared_progress is None or activity_state is None:
            return

        previous_active = activity_state["active"]
        previous_completed = activity_state["completed"]
        previous_failed = activity_state["failed"]

        activity_state["active"] = max(0, previous_active + active_delta)
        activity_state["completed"] = min(
            activity_state["total"],
            previous_completed + advance,
        )
        activity_state["failed"] = max(0, previous_failed + failed_increment)

        # Avoid no-op renders that can create duplicate aggregate status lines in
        # terminals that do not fully support Rich's in-place redraw behavior.
        if (
            advance == 0
            and activity_state["active"] == previous_active
            and activity_state["completed"] == previous_completed
            and activity_state["failed"] == previous_failed
        ):
            return

        shared_progress.update(
            activity_state["task_id"],
            advance=advance,
            description=format_activity_description(
                active_requests=activity_state["active"],
                completed_batches=activity_state["completed"],
                total_batches=activity_state["total"],
                failed_batches=activity_state["failed"],
            ),
        )

    async def _fetch_batch(
        self,
        exchange: ccxt.async_support.coinbaseadvanced,
        semaphore: asyncio.Semaphore,
        symbol: str,
        timeframe: str,
        since: int,
    ) -> list[list[float]]:
        """Fetch a single batch of OHLCV data with rate limit handling.

        Args:
            exchange: The ccxt exchange instance.
            semaphore: Concurrency semaphore.
            symbol: Trading pair symbol.
            timeframe: Timeframe for candles.
            since: Start timestamp in milliseconds.

        Returns:
            List of candles, or empty list if no data.
        """
        return await self._fetch_with_retry(
            exchange,
            semaphore,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=COINBASE_CANDLE_LIMIT,
        )

    async def _fetch_with_retry(
        self,
        exchange: ccxt.async_support.coinbaseadvanced,
        semaphore: asyncio.Semaphore,
        **kwargs: Any,
    ) -> Any:
        """Fetch OHLCV data with exponential backoff retry handling.

        Args:
            exchange: The ccxt exchange instance.
            semaphore: Concurrency semaphore.
            **kwargs: Arguments passed to exchange.fetch_ohlcv.

        Returns:
            List of candles from fetch_ohlcv.

        Raises:
            The last encountered exception after max_retries exhausted.
        """
        last_exception: Exception | None = None

        # Loop through initial attempt + max_retries retries
        for attempt in range(self.max_retries + 1):
            try:
                async with semaphore:
                    await self._wait_for_request_slot()
                    return await exchange.fetch_ohlcv(**kwargs)
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                # Skip sleeping on last attempt (no more retries after this)
                if attempt < self.max_retries:
                    # Calculate exponential backoff: base * 2^attempt
                    delay = self.rate_limit_backoff * (2**attempt)
                    # Add jitter while preserving monotonic backoff growth.
                    jitter = delay * (1.0 + (0.5 * random.random()))
                    log_retry = (
                        logger.warning
                        if self.verbosity == Verbosity.VERBOSE
                        else logger.info
                    )
                    log_retry(
                        "Rate limit hit (attempt %d/%d): %s - %s. Retrying in %.2fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        type(e).__name__,
                        e,
                        jitter,
                    )
                    await asyncio.sleep(jitter)
            except ccxt.ExchangeError as e:
                # Some Coinbase transient backend failures are raised as ExchangeError
                # (for example payloads containing "UNAVAILABLE"), so retry those.
                if not self._is_retryable_exchange_error(e):
                    raise
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.rate_limit_backoff * (2**attempt)
                    jitter = delay * (1.0 + (0.5 * random.random()))
                    log_retry = (
                        logger.warning
                        if self.verbosity == Verbosity.VERBOSE
                        else logger.info
                    )
                    log_retry(
                        "Retryable exchange error (attempt %d/%d): %s. Retrying in %.2fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        jitter,
                    )
                    await asyncio.sleep(jitter)
            except NON_RETRYABLE_EXCEPTIONS as e:
                # Permanent error - log and re-raise
                logger.warning("Non-retryable exception: %s - %s", type(e).__name__, e)
                raise

        # All retries exhausted
        logger.error(
            "Max retries (%d) exhausted for fetch_ohlcv. Last error: %s - %s",
            self.max_retries,
            type(last_exception).__name__,
            last_exception,
        )
        # last_exception must be set since we only get here after catching RETRYABLE_EXCEPTIONS
        raise last_exception  # type: ignore[arg-type]

    @staticmethod
    def _is_retryable_exchange_error(error: ccxt.ExchangeError) -> bool:
        """Return True when an ExchangeError looks transient/retryable."""
        message = str(error).upper()
        retryable_markers = (
            "UNAVAILABLE",
            "SOMETHING WENT WRONG",
            "INTERNAL SERVER ERROR",
            "SERVICE UNAVAILABLE",
            "TIMEOUT",
            "TOO MANY REQUESTS",
            "429",
            "5XX",
        )
        return any(marker in message for marker in retryable_markers)

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
        progress_task_id: Any = None,
        shared_progress: Progress | None = None,
        activity_state: dict[str, Any] | None = None,
        on_batch: Callable[[pl.DataFrame], Awaitable[None]] | None = None,
        collect_results: bool = True,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from Coinbase.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h").
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval. Defaults to latest.
                Note: short/sparse batches do not imply completion by themselves.
            verbosity: Override verbosity level for this call.
            progress_task_id: Task ID from shared Rich Progress (for fetch_multiple).
            shared_progress: Shared Rich Progress instance (for fetch_multiple).
            activity_state: Shared aggregate activity counters for batch fetches.
            on_batch: Optional async callback executed for each fetched batch DataFrame.
            collect_results: Whether to collect all rows in-memory and return full DataFrame.

        Returns:
            Polars DataFrame with OHLCV columns.

        Raises:
            ValueError: If timeframe is not supported.
        """
        return await execute_fetch(
            client=self,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            verbosity=verbosity,
            progress_task_id=progress_task_id,
            shared_progress=shared_progress,
            activity_state=activity_state,
            on_batch=on_batch,
            collect_results=collect_results,
            verbosity_disabled=Verbosity.DISABLED,
            verbosity_verbose=Verbosity.VERBOSE,
            ohlcv_schema=OHLCV_SCHEMA,
            timeframe_seconds=TIMEFRAME_SECONDS,
            calculate_expected_batches=calculate_expected_batches,
            coinbase_candle_limit=COINBASE_CANDLE_LIMIT,
            max_consecutive_retryable_failures_no_end_date=MAX_CONSECUTIVE_RETRYABLE_FAILURES_NO_END_DATE,
            retryable_exceptions=RETRYABLE_EXCEPTIONS,
            non_retryable_exceptions=NON_RETRYABLE_EXCEPTIONS,
            logger=logger,
        )

    @staticmethod
    def _symbol_to_path(symbol: str) -> str:
        """Convert symbol to path-friendly format.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD", "BTC-USDC").

        Returns:
            Path-friendly string (e.g., "btc-usd").
        """
        return symbol_to_path(symbol)

    def save(
        self,
        df: pl.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Save OHLCV data to partitioned parquet files.

        Args:
            df: Polars DataFrame with OHLCV data.
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "1h").

        Raises:
            ValueError: If DataFrame is empty or missing required columns.
        """
        save_partitions(
            data_dir=self.data_dir,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
        )

    async def save_async(
        self,
        df: pl.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Asynchronously write a DataFrame to partitioned parquet files.

        Saves for the same symbol/timeframe are serialized to avoid overlapping
        read-modify-write cycles to the same parquet partitions.
        """
        lock = await self._get_save_lock(symbol, timeframe)
        async with lock:
            await asyncio.to_thread(self.save, df, symbol, timeframe)

    async def _get_save_lock(self, symbol: str, timeframe: str) -> asyncio.Lock:
        """Return a stable async lock for a symbol/timeframe save stream."""
        key = f"{self._symbol_to_path(symbol)}|{timeframe}"
        async with self._save_locks_guard:
            lock = self._save_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._save_locks[key] = lock
            return lock

    async def fetch_latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        exclude_timestamps: list[datetime] | None = None,
    ) -> pl.DataFrame:
        """Fetch the latest OHLCV candles from Coinbase.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h").
            limit: Number of candles to fetch. Defaults to 300.
            exclude_timestamps: Optional list of timestamps to exclude from results.

        Returns:
            Polars DataFrame with OHLCV columns.

        Raises:
            ValueError: If timeframe is not supported.
        """
        self._validate_timeframe(timeframe)

        exchange = self._get_exchange()
        semaphore = self._get_semaphore()

        # Fetch latest candles without date range
        candles = await self._fetch_with_retry(
            exchange,
            semaphore,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Convert to Polars DataFrame
        df = self._candles_to_dataframe(candles)

        # Filter out excluded timestamps if provided
        if exclude_timestamps:
            df = df.filter(~pl.col("timestamp").is_in(exclude_timestamps))

        return df

    async def update_async(self, symbol: str, timeframe: str) -> pl.DataFrame:
        """Update stored OHLCV data with latest candles.

        Loads existing data, fetches the latest candles, combines them,
        removes duplicates, and saves back to storage.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "1h").

        Returns:
            Polars DataFrame with the updated OHLCV data.
        """
        # Load existing data
        existing_df = self.load(symbol, timeframe)

        # Get existing timestamps to exclude
        exclude_timestamps = (
            existing_df["timestamp"].to_list() if not existing_df.is_empty() else None
        )

        # Fetch latest candles
        latest_df = await self.fetch_latest(
            symbol, timeframe, exclude_timestamps=exclude_timestamps
        )

        if existing_df.is_empty():
            combined_df = latest_df
        elif latest_df.is_empty():
            combined_df = existing_df
        else:
            # Combine and deduplicate
            combined_df = pl.concat([existing_df, latest_df])
            combined_df = combined_df.unique(subset=["timestamp"], keep="last")
            combined_df = combined_df.sort("timestamp")

        # Save the combined data
        if not combined_df.is_empty():
            self.save(combined_df, symbol, timeframe)

        return combined_df

    def update(self, symbol: str, timeframe: str) -> pl.DataFrame:
        """Sync wrapper for update_async()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.update_async(symbol, timeframe))
        raise RuntimeError(
            "update() cannot be called from an active event loop. "
            "Use await update_async(...) instead."
        )

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    # Async context manager support
    async def __aenter__(self) -> "CoinbaseDataClient":
        """Async context manager entry - initializes exchange connection."""
        self._get_exchange()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures connection is closed."""
        await self.close()

    # Sync context manager support
    def __enter__(self) -> "CoinbaseDataClient":
        """Sync context manager entry - initializes exchange connection."""
        self._get_exchange()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit - ensures connection is closed."""
        asyncio.run(self.close())

    def load(
        self,
        symbol: str,
        timeframe: str,
        year: int | None = None,
        month: int | None = None,
    ) -> pl.DataFrame:
        """Load OHLCV data from partitioned parquet files.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "1h").
            year: Optional year to filter by.
            month: Optional month to filter by (requires year).

        Returns:
            Polars DataFrame with OHLCV data.

        Raises:
            ValueError: If month is specified without year.
        """
        return load_partitions(
            data_dir=self.data_dir,
            ohlcv_schema=OHLCV_SCHEMA,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            month=month,
        )

    async def fetch_multiple(
        self,
        symbols: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
    ) -> dict[str, dict[str, pl.DataFrame]]:
        """Fetch OHLCV data for multiple symbols and timeframes concurrently.

        Args:
            symbols: List of trading pair symbols (e.g., ["BTC/USD", "ETH/USD"]).
            timeframes: List of timeframes (e.g., ["1m", "1h"]).
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval. Defaults to latest.
            verbosity: Override verbosity level for this call.

        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        return await execute_fetch_multiple(
            client=self,
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            verbosity=verbosity,
            verbosity_disabled=Verbosity.DISABLED,
            verbosity_verbose=Verbosity.VERBOSE,
            calculate_expected_batches=calculate_expected_batches,
            logger=logger,
            progress_class=Progress,
            get_progress_color=get_progress_color,
        )

    async def fetch_and_save(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
        progress_task_id: Any = None,
        shared_progress: Progress | None = None,
        activity_state: dict[str, Any] | None = None,
    ) -> None:
        """Fetch OHLCV data and stream-write each batch to parquet asynchronously."""

        async def _write_batch(batch_df: pl.DataFrame) -> None:
            if not batch_df.is_empty():
                await self.save_async(batch_df, symbol, timeframe)

        await self.fetch(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            verbosity=verbosity,
            progress_task_id=progress_task_id,
            shared_progress=shared_progress,
            activity_state=activity_state,
            on_batch=_write_batch,
            collect_results=False,
        )

    async def fetch_multiple_and_save(
        self,
        symbols: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
    ) -> None:
        """Fetch and persist multiple symbol/timeframe combinations concurrently."""
        effective_verbosity = self._resolve_verbosity(verbosity)
        combinations = [(s, t) for s in symbols for t in timeframes]

        if effective_verbosity == Verbosity.VERBOSE:
            end_display = end_date if end_date else "latest"
            print(
                f"Starting batch fetch for {len(combinations)} symbol/timeframe combinations "
                f"from {start_date} to {end_display}..."
            )

        shared_progress: Progress | None = None
        activity_state: dict[str, Any] | None = None
        task_ids: dict[tuple[str, str], Any] = {}
        if effective_verbosity != Verbosity.DISABLED:
            shared_progress = build_shared_progress(Progress)
            shared_progress.start()

            start_ts = int(self._parse_date(start_date).timestamp() * 1000)
            if end_date:
                end_ts = int(
                    self._parse_date(end_date, end_of_day=True).timestamp() * 1000
                )
            else:
                end_ts = int(datetime.now(UTC).timestamp() * 1000)

            total_expected_batches = 0
            expected_batches_by_combo: dict[tuple[str, str], int] = {}
            for symbol, timeframe in combinations:
                expected_batches = calculate_expected_batches(
                    start_ts, end_ts, timeframe
                )
                total_expected_batches += expected_batches
                expected_batches_by_combo[(symbol, timeframe)] = expected_batches

            activity_state = create_activity_state(
                shared_progress, total_expected_batches
            )

            for symbol, timeframe in combinations:
                color = get_progress_color(symbol, timeframe)
                description = f"[{color}]{symbol} {timeframe}[/{color}]"
                expected_batches = expected_batches_by_combo[(symbol, timeframe)]
                task_id = shared_progress.add_task(description, total=expected_batches)
                task_ids[(symbol, timeframe)] = task_id

        tasks = []
        task_metadata = []
        for symbol, timeframe in combinations:
            task_id = task_ids.get((symbol, timeframe))
            tasks.append(
                self.fetch_and_save(
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                    effective_verbosity,
                    progress_task_id=task_id,
                    shared_progress=shared_progress,
                    activity_state=activity_state,
                )
            )
            task_metadata.append((symbol, timeframe))

        results = await self._gather_with_exceptions(tasks)
        failed_tasks = [
            (task_metadata[i], result)
            for i, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if failed_tasks:
            for (symbol, timeframe), exc in failed_tasks:
                task_id = task_ids.get((symbol, timeframe))
                if shared_progress is not None and task_id is not None:
                    shared_progress.update(task_id, completed=0)
                    shared_progress.update(
                        task_id,
                        description=f"[red]{symbol} {timeframe} (failed)[/red]",
                    )
                    self._update_activity_progress(
                        shared_progress,
                        activity_state,
                        failed_increment=1,
                    )
                logger.warning(
                    "Failed to fetch_and_save %s %s: %s: %s",
                    symbol,
                    timeframe,
                    type(exc).__name__,
                    exc,
                )

        if effective_verbosity == Verbosity.VERBOSE:
            print("Completed batch fetch and save")

        if shared_progress is not None:
            shared_progress.stop()
