"""OHLCV data retrieval from Coinbase."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import ccxt.async_support
import polars as pl

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
    ) -> None:
        """Initialize the Coinbase data client.

        Args:
            data_dir: Directory for storing fetched data.
            max_concurrency: Maximum number of concurrent requests.
            rate_limit_backoff: Backoff time in seconds for rate limiting.
        """
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.rate_limit_backoff = rate_limit_backoff
        self._exchange: ccxt.async_support.coinbaseadvanced | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def _get_exchange(self) -> ccxt.async_support.coinbaseadvanced:
        """Get or create the ccxt exchange instance."""
        if self._exchange is None:
            self._exchange = ccxt.async_support.coinbaseadvanced()
        return self._exchange

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse date string to datetime.

        Args:
            date_str: Date string in various formats.

        Returns:
            datetime object in UTC.
        """
        # Try parsing as ISO format first
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
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
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from Coinbase.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD").
            timeframe: Timeframe for candles (e.g., "1m", "5m", "1h").
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval. Defaults to latest.

        Returns:
            Polars DataFrame with OHLCV columns.

        Raises:
            ValueError: If timeframe is not supported.
        """
        self._validate_timeframe(timeframe)

        exchange = self._get_exchange()
        semaphore = self._get_semaphore()

        start_ts = int(self._parse_date(start_date).timestamp() * 1000)
        end_ts = None
        if end_date:
            end_ts = int(self._parse_date(end_date).timestamp() * 1000)

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

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
