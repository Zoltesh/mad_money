"""OHLCV data retrieval from Coinbase."""

import asyncio
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import ccxt
import ccxt.async_support
import polars as pl
from rich.progress import Progress

from src.data.progress import (
    TIMEFRAME_SECONDS,
    ProgressTracker,
    RichProgressManager,
    _is_test_environment,
    calculate_expected_batches,
)


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
        rate_limit_backoff: float = 1.0,
        api_key: str | None = None,
        private_key: str | None = None,
        verbosity: Verbosity | None = None,
    ) -> None:
        """Initialize the Coinbase data client.

        Args:
            data_dir: Directory for storing fetched data.
            max_concurrency: Maximum number of concurrent requests.
            rate_limit_backoff: Backoff time in seconds for rate limiting.
            api_key: Optional Coinbase API key for authenticated requests.
            private_key: Optional Coinbase private key (PEM format) for authenticated requests.
            verbosity: Verbosity level for output control. If None, auto-detects test environment.
        """
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.rate_limit_backoff = rate_limit_backoff
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
            # If no time component and end_of_day=True, set to end of day
            if (
                end_of_day
                and dt.hour == 0
                and dt.minute == 0
                and dt.second == 0
                and dt.microsecond == 0
            ):
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            return dt
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
                # If no time component and end_of_day=True, set to end of day
                if (
                    end_of_day
                    and dt.hour == 0
                    and dt.minute == 0
                    and dt.second == 0
                    and dt.microsecond == 0
                ):
                    dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                return dt.replace(tzinfo=UTC)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_str}")

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

    def _flush_progress_update(
        self,
        pending_advance: int,
        progress_task_id: Any,
        shared_progress: Progress | None,
        progress_tracker: ProgressTracker | None,
        use_shared_progress: bool,
        extra: int = 0,
    ) -> int:
        """Flush pending progress updates.

        Args:
            pending_advance: Number of batches to advance the progress.
            progress_task_id: Task ID from shared Rich Progress.
            shared_progress: Shared Rich Progress instance.
            progress_tracker: Individual ProgressTracker instance.
            use_shared_progress: Whether to use shared progress.
            extra: Additional count to add to the advance (default 0).

        Returns:
            0 to reset pending_advance.
        """
        total_advance = pending_advance + extra
        if use_shared_progress and shared_progress is not None and total_advance > 0:
            shared_progress.update(progress_task_id, advance=total_advance)
        elif progress_tracker is not None and total_advance > 0:
            progress_tracker.update(n=total_advance)
        return 0

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
        progress_task_id: Any = None,
        shared_progress: Progress | None = None,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from Coinbase.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h").
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval. Defaults to latest.
            verbosity: Override verbosity level for this call.
            progress_task_id: Task ID from shared Rich Progress (for fetch_multiple).
            shared_progress: Shared Rich Progress instance (for fetch_multiple).

        Returns:
            Polars DataFrame with OHLCV columns.

        Raises:
            ValueError: If timeframe is not supported.
        """
        # Use instance verbosity if not overridden
        effective_verbosity = verbosity if verbosity is not None else self.verbosity
        self._validate_timeframe(timeframe)

        exchange = self._get_exchange()
        semaphore = self._get_semaphore()

        start_ts = int(self._parse_date(start_date).timestamp() * 1000)
        end_ts: int | None = None
        current_ts = datetime.now(UTC).timestamp() * 1000
        if end_date:
            end_ts = int(self._parse_date(end_date, end_of_day=True).timestamp() * 1000)
        else:
            end_ts = int(current_ts)

        # Calculate expected batches for progress tracking
        expected_batches = calculate_expected_batches(start_ts, end_ts, timeframe)
        expected_candles = self._calculate_expected_candles(start_ts, end_ts, timeframe)

        # Determine if we use shared progress or create our own
        use_shared_progress = (
            shared_progress is not None and progress_task_id is not None
        )

        # Initialize progress tracker if verbosity is enabled and not using shared
        progress_tracker: ProgressTracker | None = None
        if effective_verbosity != Verbosity.DISABLED and not use_shared_progress:
            progress_tracker = RichProgressManager(
                total=expected_batches,
                symbol=symbol,
                timeframe=timeframe,
                verbosity=effective_verbosity,
            )
            progress_tracker.start()

            # Verbose mode: print start message
            if effective_verbosity == Verbosity.VERBOSE:
                end_display = end_date if end_date else "latest"
                print(
                    f"Starting fetch for {symbol} {timeframe} from {start_date} to {end_display}..."
                )

        all_candles: list[list[float]] = []
        since = start_ts
        batch_num = 0
        pending_advance = 0  # For batching progress updates (TASK-008)
        update_interval = 10  # Update progress every N batches

        try:
            while True:
                async with semaphore:
                    try:
                        candles = await exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=since,
                            limit=COINBASE_CANDLE_LIMIT,
                        )
                    except ccxt.RateLimitExceeded:
                        await asyncio.sleep(self.rate_limit_backoff)
                        continue
                    except Exception:
                        # Re-raise other exceptions but handle gracefully
                        raise

                if not candles:
                    # Update progress even for empty result to show completion
                    # Include any pending advances from previous batches
                    pending_advance = self._flush_progress_update(
                        pending_advance,
                        progress_task_id,
                        shared_progress,
                        progress_tracker,
                        use_shared_progress,
                        extra=1,
                    )
                    break

                batch_num += 1
                all_candles.extend(candles)

                # Update progress - either shared or individual
                if use_shared_progress and shared_progress is not None:
                    # Batch updates to reduce contention (TASK-008)
                    pending_advance += 1
                    if pending_advance >= update_interval:
                        shared_progress.update(
                            progress_task_id, advance=pending_advance
                        )
                        pending_advance = 0
                elif progress_tracker is not None:
                    progress_tracker.update(
                        n=1,
                        candles_so_far=len(all_candles),
                        total_candles=expected_candles,
                    )

                # Check if we've reached the end date or latest
                last_candle_ts = candles[-1][0]
                if end_ts and last_candle_ts > end_ts:
                    # Flush any pending progress before breaking
                    pending_advance = self._flush_progress_update(
                        pending_advance,
                        progress_task_id,
                        shared_progress,
                        progress_tracker,
                        use_shared_progress,
                    )
                    break

                # Check if we've reached the latest (no more data)
                if len(candles) < COINBASE_CANDLE_LIMIT:
                    # Flush any pending progress before breaking
                    pending_advance = self._flush_progress_update(
                        pending_advance,
                        progress_task_id,
                        shared_progress,
                        progress_tracker,
                        use_shared_progress,
                    )
                    break

                # Move to next batch using the timestamp of the last candle + 1ms
                since = last_candle_ts + 1
        finally:
            # Always close progress tracker (only if we created our own)
            if progress_tracker is not None:
                progress_tracker.close()

                # Verbose mode: print completion message
                if effective_verbosity == Verbosity.VERBOSE:
                    print(
                        f"Completed fetch for {symbol} {timeframe}: {len(all_candles)} candles fetched"
                    )

        if not all_candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Filter to end_date if specified
        if end_date:
            end_ts_filter = int(
                self._parse_date(end_date, end_of_day=True).timestamp() * 1000
            )
            all_candles = [c for c in all_candles if c[0] <= end_ts_filter]

        # Convert to Polars DataFrame
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime.fromtimestamp(c[0] / 1000, tz=UTC) for c in all_candles
                ],
                "open": [c[1] for c in all_candles],
                "high": [c[2] for c in all_candles],
                "low": [c[3] for c in all_candles],
                "close": [c[4] for c in all_candles],
                "volume": [c[5] for c in all_candles],
            },
            schema=OHLCV_SCHEMA,
        )

        return df

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
        # Handle empty DataFrame
        if df.is_empty():
            return

        # Validate required columns
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        df_cols = set(df.columns)
        if not required_cols.issubset(df_cols):
            missing = required_cols - df_cols
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Normalize symbol to path-friendly format (e.g., BTC/USD -> btc-usd)
        pair_path = symbol.replace("/", "-").lower()

        # Extract year and month for partitioning
        df_with_partition = df.with_columns(
            [
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
            ]
        )

        # Group by year-month and save each partition
        for (year, month), partition_df in df_with_partition.group_by(
            ["year", "month"]
        ):
            # Build directory path
            dir_path = (
                Path(self.data_dir)
                / "coinbase"
                / "ohlcv"
                / pair_path
                / timeframe
                / str(year)
            )
            dir_path.mkdir(parents=True, exist_ok=True)

            # Build file path
            file_path = dir_path / f"{month:02d}.parquet"

            # Load existing data if file exists and append with deduplication
            if file_path.exists():
                existing_df = pl.read_parquet(file_path)
                # Combine and remove duplicates based on timestamp
                combined_df = pl.concat(
                    [existing_df, partition_df.drop(["year", "month"])]
                )
                combined_df = combined_df.unique(subset=["timestamp"], keep="first")
                combined_df.write_parquet(file_path)
            else:
                partition_df.drop(["year", "month"]).write_parquet(file_path)

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
        async with semaphore:
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
            except ccxt.RateLimitExceeded:
                await asyncio.sleep(self.rate_limit_backoff)
                candles = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
            except Exception:
                raise

        if not candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Convert to Polars DataFrame
        df = pl.DataFrame(
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

        # Filter out excluded timestamps if provided
        if exclude_timestamps:
            df = df.filter(~pl.col("timestamp").is_in(exclude_timestamps))

        return df

    def update(self, symbol: str, timeframe: str) -> pl.DataFrame:
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

        # Fetch latest candles (run the async method)
        latest_df = asyncio.run(
            self.fetch_latest(symbol, timeframe, exclude_timestamps=exclude_timestamps)
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
        # Validate month without year
        if month is not None and year is None:
            raise ValueError("year must be specified when month is provided")

        # Validate month range
        if month is not None and (month < 1 or month > 12):
            raise ValueError(f"month must be between 1 and 12, got {month}")

        # Normalize symbol to path-friendly format
        pair_path = symbol.replace("/", "-").lower()

        # Build base path
        base_path = Path(self.data_dir) / "coinbase" / "ohlcv" / pair_path / timeframe

        # If no base directory exists, return empty DataFrame
        if not base_path.exists():
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Collect all parquet files to read
        parquet_files: list[Path] = []

        if year is not None:
            # Look in specific year directory
            year_path = base_path / str(year)
            if year_path.exists():
                parquet_files = list(year_path.glob("*.parquet"))
        else:
            # No year specified - scan all year directories
            for year_dir in base_path.iterdir():
                if year_dir.is_dir():
                    parquet_files.extend(year_dir.glob("*.parquet"))

        if not parquet_files:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Filter by month if specified
        if month is not None:
            month_str = f"{month:02d}.parquet"
            parquet_files = [f for f in parquet_files if f.name == month_str]

            if not parquet_files:
                return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Read and combine all parquet files
        dfs = []
        for file_path in sorted(parquet_files):
            df = pl.read_parquet(file_path)
            dfs.append(df)

        if not dfs:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        combined_df = pl.concat(dfs)

        # Sort by timestamp
        combined_df = combined_df.sort("timestamp")

        # Remove duplicates based on timestamp
        combined_df = combined_df.unique(subset=["timestamp"], keep="first")

        return combined_df

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
        # Use instance verbosity if not overridden
        effective_verbosity = verbosity if verbosity is not None else self.verbosity

        # Build list of all symbol/timeframe combinations
        combinations = [(s, t) for s in symbols for t in timeframes]

        # Verbose mode: print batch start message
        if effective_verbosity == Verbosity.VERBOSE:
            end_display = end_date if end_date else "latest"
            print(
                f"Starting batch fetch for {len(combinations)} symbol/timeframe combinations "
                f"from {start_date} to {end_display}..."
            )

        # Create shared Rich Progress if verbosity is enabled
        shared_progress: Progress | None = None
        task_ids: dict[tuple[str, str], Any] = {}
        if effective_verbosity != Verbosity.DISABLED:
            from rich.progress import (
                BarColumn,
                SpinnerColumn,
                TextColumn,
                TimeRemainingColumn,
            )

            from src.data.progress import get_progress_color

            # Create one progress for all tasks
            shared_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} batches)"),
                TimeRemainingColumn(),
            )
            shared_progress.start()

            # Add all tasks upfront with expected batches
            # Calculate expected batches for each timeframe
            start_ts = int(self._parse_date(start_date).timestamp() * 1000)
            end_ts: int
            if end_date:
                end_ts = int(
                    self._parse_date(end_date, end_of_day=True).timestamp() * 1000
                )
            else:
                end_ts = int(datetime.now(UTC).timestamp() * 1000)

            for symbol, timeframe in combinations:
                color = get_progress_color(symbol, timeframe)
                description = f"[{color}]{symbol} {timeframe}[/{color}]"
                # Calculate expected batches for this specific timeframe
                expected_batches = calculate_expected_batches(
                    start_ts, end_ts, timeframe
                )
                task_id = shared_progress.add_task(description, total=expected_batches)
                task_ids[(symbol, timeframe)] = task_id

        # Build list of all fetch tasks with their task IDs
        async def fetch_one(
            symbol: str, timeframe: str, task_id: Any
        ) -> tuple[str, str, pl.DataFrame]:
            df = await self.fetch(
                symbol,
                timeframe,
                start_date,
                end_date,
                effective_verbosity,
                progress_task_id=task_id,
                shared_progress=shared_progress,
            )
            return (symbol, timeframe, df)

        # Create all tasks
        tasks = []
        for symbol, timeframe in combinations:
            task_id = task_ids.get((symbol, timeframe))
            tasks.append(fetch_one(symbol, timeframe, task_id))

        # Run all concurrently
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # Stop shared progress regardless of success or failure
            if shared_progress is not None:
                shared_progress.stop()

        # Build nested dict structure
        result_dict: dict[str, dict[str, pl.DataFrame]] = {}
        total_candles = 0
        for symbol, timeframe, df in results:
            if symbol not in result_dict:
                result_dict[symbol] = {}
            result_dict[symbol][timeframe] = df
            total_candles += len(df)

        # Verbose mode: print batch completion message
        if effective_verbosity == Verbosity.VERBOSE:
            print(f"Completed batch fetch: {total_candles} total candles fetched")

        return result_dict
