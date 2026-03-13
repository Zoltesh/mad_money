"""OHLCV data retrieval from Coinbase."""

import asyncio
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import ccxt
import ccxt.async_support
import polars as pl


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

    def __init__(
        self,
        data_dir: str = "./data",
        max_concurrency: int = 10,
        rate_limit_backoff: float = 1.0,
        api_key: str | None = None,
        private_key: str | None = None,
        verbosity: Verbosity = Verbosity.DISABLED,
    ) -> None:
        """Initialize the Coinbase data client.

        Args:
            data_dir: Directory for storing fetched data.
            max_concurrency: Maximum number of concurrent requests.
            rate_limit_backoff: Backoff time in seconds for rate limiting.
            api_key: Optional Coinbase API key for authenticated requests.
            private_key: Optional Coinbase private key (PEM format) for authenticated requests.
            verbosity: Verbosity level for output control.
        """
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.rate_limit_backoff = rate_limit_backoff
        self._api_key = api_key
        self._private_key = private_key
        self.verbosity = verbosity
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

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        verbosity: Verbosity | None = None,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from Coinbase.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h").
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval. Defaults to latest.
            verbosity: Override verbosity level for this call.

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
        end_ts = None
        if end_date:
            end_ts = int(self._parse_date(end_date, end_of_day=True).timestamp() * 1000)

        all_candles: list[list[float]] = []
        since = start_ts

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
                break

            all_candles.extend(candles)

            # Check if we've reached the end date or latest
            last_candle_ts = candles[-1][0]
            if end_ts and last_candle_ts >= end_ts:
                break

            # Check if we've reached the latest (no more data)
            if len(candles) < COINBASE_CANDLE_LIMIT:
                break

            # Move to next batch using the timestamp of the last candle + 1ms
            since = last_candle_ts + 1

        if not all_candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Filter to end_date if specified
        if end_ts:
            all_candles = [c for c in all_candles if c[0] <= end_ts]

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

        # Build list of all fetch tasks
        async def fetch_one(
            symbol: str, timeframe: str
        ) -> tuple[str, str, pl.DataFrame]:
            df = await self.fetch(symbol, timeframe, start_date, end_date, effective_verbosity)
            return (symbol, timeframe, df)

        # Create all tasks
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append(fetch_one(symbol, timeframe))

        # Run all concurrently
        results = await asyncio.gather(*tasks)

        # Build nested dict structure
        result_dict: dict[str, dict[str, pl.DataFrame]] = {}
        for symbol, timeframe, df in results:
            if symbol not in result_dict:
                result_dict[symbol] = {}
            result_dict[symbol][timeframe] = df

        return result_dict
