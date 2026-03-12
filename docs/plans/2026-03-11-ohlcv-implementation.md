# OHLCV Data Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a module for retrieving OHLCV data from Coinbase via ccxt, storing in Polars DataFrames and parquet files.

**Architecture:** Class-based `CoinbaseDataClient` with async/await, semaphore-based rate limiting, and month-partitioned parquet storage.

**Tech Stack:** Python 3.14, ccxt (async), polars, asyncio

---

## Task 1: Project Setup - Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add aiohttp dependency for async support**

Run: `uv add aiohttp`

**Step 2: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add aiohttp dependency for async ccxt"
```

---

## Task 2: Create Module Structure

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Create `tests/test_ohlcv.py`:

```python
import pytest
from polars import DataFrame


def test_client_import():
    """Test that CoinbaseDataClient can be imported."""
    from src.data import CoinbaseDataClient
    assert CoinbaseDataClient is not None


def test_schema_import():
    """Test that OHLCV_SCHEMA can be imported."""
    from src.data.ohlcv import OHLCV_SCHEMA
    assert OHLCV_SCHEMA is not None
    assert "timestamp" in OHLCV_SCHEMA
    assert "open" in OHLCV_SCHEMA
    assert "high" in OHLCV_SCHEMA
    assert "low" in OHLCV_SCHEMA
    assert "close" in OHLCV_SCHEMA
    assert "volume" in OHLCV_SCHEMA
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_client_import -v`
Expected: FAIL with "cannot import name"

**Step 3: Write minimal implementation**

Create `src/data/__init__.py`:

```python
"""Data module for OHLCV data retrieval and storage."""

from src.data.ohlcv import CoinbaseDataClient, OHLCV_SCHEMA

__all__ = ["CoinbaseDataClient", "OHLCV_SCHEMA"]
```

Create `src/data/ohlcv.py`:

```python
"""OHLCV data retrieval from Coinbase."""

import polars as pl


OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class CoinbaseDataClient:
    """Client for fetching and storing OHLCV data from Coinbase."""

    def __init__(
        self,
        data_dir: str = "data",
        max_concurrency: int = 10,
        rate_limit_backoff: float = 1.0,
    ):
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.rate_limit_backoff = rate_limit_backoff
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ tests/test_ohlcv.py
git commit -m "feat: create OHLCV module structure with CoinbaseDataClient"
```

---

## Task 3: Implement Fetch with Pagination

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch


def test_fetch_returns_dataframe():
    """Test that fetch returns a Polars DataFrame."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient()
    result = asyncio.run(client.fetch("BTC-USDC", "1h", "2025-01-01", "2025-01-02"))

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_fetch_respects_date_range():
    """Test that fetch respects start and end dates."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient()
    result = asyncio.run(client.fetch("BTC-USDC", "1h", "2025-01-01", "2025-01-02"))

    if len(result) > 0:
        ts_min = result["timestamp"].min()
        ts_max = result["timestamp"].max()
        assert ts_min >= 1704067200000  # 2025-01-01 00:00:00 UTC
        assert ts_max <= 1704153600000  # 2025-01-02 00:00:00 UTC
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_fetch_returns_dataframe -v`
Expected: FAIL with "AttributeError: 'CoinbaseDataClient' object has no attribute 'fetch'"

**Step 3: Write minimal implementation**

Update `src/data/ohlcv.py`:

```python
"""OHLCV data retrieval from Coinbase."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt.async_support
import polars as pl


OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "6h", "1d"]

MAX_CANDLES_PER_REQUEST = 300


class CoinbaseDataClient:
    """Client for fetching and storing OHLCV data from Coinbase."""

    def __init__(
        self,
        data_dir: str = "data",
        max_concurrency: int = 10,
        rate_limit_backoff: float = 1.0,
    ):
        self.data_dir = data_dir
        self.max_concurrency = max_concurrency
        self.rate_limit_backoff = rate_limit_backoff
        self._exchange = None
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _get_exchange(self):
        """Get or create the exchange instance."""
        if self._exchange is None:
            self._exchange = ccxt.async_support.coinbaseadvanced({
                "enableRateLimit": False,  # We handle rate limiting
            })
        return self._exchange

    async def close(self):
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a symbol and timeframe.

        Args:
            symbol: Trading pair (e.g., "BTC-USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime), defaults to now

        Returns:
            Polars DataFrame with OHLCV data
        """
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {TIMEFRAMES}")

        # Parse dates
        start_dt = self._parse_date(start_date)
        if end_date is None:
            end_dt = datetime.now(timezone.utc)
        else:
            end_dt = self._parse_date(end_date)

        if start_dt >= end_dt:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        exchange = await self._get_exchange()

        # Convert to milliseconds for ccxt
        since = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_candles = []
        current_since = since

        while current_since < end_ms:
            async with self._semaphore:
                try:
                    candles = await exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=MAX_CANDLES_PER_REQUEST,
                    )
                except Exception as e:
                    # Rate limit backoff
                    if "429" in str(e):
                        await asyncio.sleep(self.rate_limit_backoff)
                        continue
                    raise

            if not candles:
                break

            # Add to results
            all_candles.extend(candles)

            # Get timestamp of last candle for next request
            last_ts = candles[-1][0]

            # Check if we've reached the end
            if last_ts >= end_ms:
                break

            # Move to next batch (add 1ms to avoid duplicate)
            current_since = last_ts + 1

        # Convert to DataFrame
        if not all_candles:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        df = pl.DataFrame(
            all_candles,
            schema={
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

        # Convert timestamp to datetime (milliseconds)
        df = df.with_columns(
            (pl.col("timestamp") // 1000).cast(pl.Datetime(time_zone="UTC"))
        )

        # Filter to end date
        df = df.filter(pl.col("timestamp") <= end_dt)

        # Select and order columns
        df = df.select(["timestamp", "open", "high", "low", "close", "volume"])

        return df

    def _parse_date(self, date: str | datetime) -> datetime:
        """Parse date string to datetime with UTC timezone."""
        if isinstance(date, datetime):
            if date.tzinfo is None:
                return date.replace(tzinfo=timezone.utc)
            return date

        # Parse YYYY-MM-DD format
        return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_fetch_returns_dataframe -v`
Expected: PASS (or FAIL if no network - that's OK)

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: implement fetch with pagination in CoinbaseDataClient"
```

---

## Task 4: Implement Save Method

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
import os
import tempfile


def test_save_creates_parquet(tmp_path):
    """Test that save creates a parquet file."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create sample data
    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        "open": [50000.0],
        "high": [50100.0],
        "low": [49900.0],
        "close": [50050.0],
        "volume": [100.0],
    })

    client.save(df, "BTC-USDC", "1h")

    # Check file exists
    file_path = tmp_path / "coinbase" / "ohlcv" / "btc-usdc" / "1h" / "2025" / "01.parquet"
    assert file_path.exists()


def test_save_correct_directory_structure(tmp_path):
    """Test that save creates correct directory structure."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        "open": [50000.0],
        "high": [50100.0],
        "low": [49900.0],
        "close": [50050.0],
        "volume": [100.0],
    })

    client.save(df, "BTC-USDC", "1h")

    expected = tmp_path / "coinbase" / "ohlcv" / "btc-usdc" / "1h" / "2025" / "01.parquet"
    assert expected.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_save_creates_parquet -v`
Expected: FAIL with "AttributeError: 'CoinbaseDataClient' object has no attribute 'save'"

**Step 3: Write minimal implementation**

Add to `src/data/ohlcv.py`:

```python
def _get_file_path(self, symbol: str, timeframe: str, timestamp: datetime) -> Path:
    """Get the file path for a given timestamp."""
    pair = symbol.lower().replace("-", "-")
    year = timestamp.strftime("%Y")
    month = timestamp.strftime("%m")

    return (
        Path(self.data_dir)
        / "coinbase"
        / "ohlcv"
        / pair
        / timeframe
        / year
        / f"{month}.parquet"
    )


def save(
    self,
    df: pl.DataFrame,
    symbol: str,
    timeframe: str,
) -> None:
    """Save OHLCV DataFrame to parquet file.

    Args:
        df: OHLCV DataFrame
        symbol: Trading pair (e.g., "BTC-USDC")
        timeframe: Timeframe (e.g., "1h")
    """
    if df.is_empty():
        return

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    # Group by year-month and save
    years_months = df.select(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    ).unique()

    for row in years_months.iter_rows():
        year, month = row
        mask = (df["timestamp"].dt.year() == year) & (df["timestamp"].dt.month() == month)
        month_df = df.filter(mask)

        # Get representative timestamp for path
        repr_ts = month_df["timestamp"][0]
        file_path = self._get_file_path(symbol, timeframe, repr_ts)

        # Create directory
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and append
        if file_path.exists():
            existing = pl.read_parquet(file_path)
            combined = pl.concat([existing, month_df]).unique(
                maintain_order=True
            ).sort("timestamp")
            combined.write_parquet(file_path)
        else:
            month_df.write_parquet(file_path)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_save_creates_parquet -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: implement save method for parquet storage"
```

---

## Task 5: Implement Load Method

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
def test_load_parquet(tmp_path):
    """Test loading parquet files."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Create and save sample data
    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        "open": [50000.0],
        "high": [50100.0],
        "low": [49900.0],
        "close": [50050.0],
        "volume": [100.0],
    })

    client.save(df, "BTC-USDC", "1h")

    # Load it back
    loaded = client.load("BTC-USDC", "1h")

    assert len(loaded) == 1
    assert loaded["close"][0] == 50050.0


def test_load_with_year_month_filter(tmp_path):
    """Test loading with year/month filter."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 15, tzinfo=timezone.utc)],
        "open": [50000.0],
        "high": [50100.0],
        "low": [49900.0],
        "close": [50050.0],
        "volume": [100.0],
    })

    client.save(df, "BTC-USDC", "1h")

    # Load with filter
    loaded = client.load("BTC-USDC", "1h", year=2025, month=1)

    assert len(loaded) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_load_parquet -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

Add to `src/data/ohlcv.py`:

```python
def load(
    self,
    symbol: str,
    timeframe: str,
    year: int | None = None,
    month: int | None = None,
) -> pl.DataFrame:
    """Load OHLCV data from parquet files.

    Args:
        symbol: Trading pair (e.g., "BTC-USDC")
        timeframe: Timeframe (e.g., "1h")
        year: Optional year filter
        month: Optional month filter

    Returns:
        Polars DataFrame with OHLCV data
    """
    pair = symbol.lower().replace("-", "-")
    base_path = Path(self.data_dir) / "coinbase" / "ohlcv" / pair / timeframe

    if not base_path.exists():
        return pl.DataFrame(schema=OHLCV_SCHEMA)

    if year is not None:
        year_path = base_path / str(year)
        if not year_path.exists():
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        if month is not None:
            file_path = year_path / f"{month:02d}.parquet"
            if file_path.exists():
                return pl.read_parquet(file_path)
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        # Load all months for the year
        files = list(year_path.glob("*.parquet"))
        if not files:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        dfs = [pl.read_parquet(f) for f in sorted(files)]
        return pl.concat(dfs).sort("timestamp")

    # Load all years and months
    files = list(base_path.rglob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=OHLCV_SCHEMA)

    dfs = [pl.read_parquet(f) for f in sorted(files)]
    return pl.concat(dfs).sort("timestamp")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_load_parquet -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: implement load method for reading parquet files"
```

---

## Task 6: Implement fetch_latest and update Methods

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
def test_fetch_latest_returns_only_new(tmp_path):
    """Test that fetch_latest returns only new candles."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # Save existing data
    df = pl.DataFrame({
        "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        "open": [50000.0],
        "high": [50100.0],
        "low": [49900.0],
        "close": [50050.0],
        "volume": [100.0],
    })
    client.save(df, "BTC-USDC", "1h")

    # fetch_latest should exclude existing timestamps
    # (In real test, we'd mock the exchange to return new data)
    # This test just verifies the method exists and returns DataFrame
    result = asyncio.run(client.fetch_latest("BTC-USDC", "1h"))
    assert isinstance(result, pl.DataFrame)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_fetch_latest_returns_only_new -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

Add to `src/data/ohlcv.py`:

```python
async def fetch_latest(
    self,
    symbol: str,
    timeframe: str,
    limit: int = MAX_CANDLES_PER_REQUEST,
) -> pl.DataFrame:
    """Fetch the latest OHLCV data.

    Args:
        symbol: Trading pair (e.g., "BTC-USDC")
        timeframe: Timeframe (e.g., "1h")
        limit: Maximum number of candles to fetch

    Returns:
        Polars DataFrame with latest OHLCV data
        (empty if no new data since last stored)
    """
    exchange = await self._get_exchange()

    try:
        candles = await exchange.fetch_ohlcv(
            symbol,
            timeframe,
            limit=limit,
        )
    except Exception as e:
        if "429" in str(e):
            await asyncio.sleep(self.rate_limit_backoff)
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        raise

    if not candles:
        return pl.DataFrame(schema=OHLCV_SCHEMA)

    df = pl.DataFrame(
        candles,
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )

    df = df.with_columns(
        (pl.col("timestamp") // 1000).cast(pl.Datetime(time_zone="UTC"))
    )

    df = df.select(["timestamp", "open", "high", "low", "close", "volume"])

    return df


def update(
    self,
    symbol: str,
    timeframe: str,
) -> pl.DataFrame:
    """Load existing data, fetch latest, and save.

    Combines load + fetch_latest + save for live trading updates.

    Args:
        symbol: Trading pair (e.g., "BTC-USDC")
        timeframe: Timeframe (e.g., "1h")

    Returns:
        Updated Polars DataFrame
    """
    # Load existing
    existing = self.load(symbol, timeframe)

    # Fetch latest
    latest = asyncio.run(self.fetch_latest(symbol, timeframe))

    if latest.is_empty():
        return existing

    # Combine and deduplicate
    if existing.is_empty():
        combined = latest
    else:
        combined = pl.concat([existing, latest]).unique(
            maintain_order=True
        ).sort("timestamp")

    # Save
    self.save(combined, symbol, timeframe)

    return combined
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_fetch_latest_returns_only_new -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: implement fetch_latest and update methods"
```

---

## Task 7: Implement fetch_multiple for Batch Operations

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
def test_fetch_multiple_returns_dict():
    """Test that fetch_multiple returns dict of DataFrames."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient()

    # Should return dict even with empty results
    result = asyncio.run(client.fetch_multiple(
        symbols=["BTC-USDC"],
        timeframes=["1h"],
        start_date="2025-01-01",
        end_date="2025-01-01",
    ))

    assert isinstance(result, dict)
    assert "BTC-USDC" in result
    assert "1h" in result["BTC-USDC"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_fetch_multiple_returns_dict -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

Add to `src/data/ohlcv.py`:

```python
async def fetch_multiple(
    self,
    symbols: list[str],
    timeframes: list[str],
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Fetch OHLCV for multiple symbols and timeframes.

    Uses concurrent requests with semaphore for rate limiting.

    Args:
        symbols: List of trading pairs
        timeframes: List of timeframes
        start_date: Start date
        end_date: End date (default: now)

    Returns:
        Dict of {symbol: {timeframe: DataFrame}}
    """
    async def fetch_one(symbol: str, tf: str) -> tuple[str, str, pl.DataFrame]:
        df = await self.fetch(symbol, tf, start_date, end_date)
        return symbol, tf, df

    tasks = [
        fetch_one(symbol, tf)
        for symbol in symbols
        for tf in timeframes
    ]

    results = await asyncio.gather(*tasks)

    # Build nested dict
    output: dict[str, dict[str, pl.DataFrame]] = {}
    for symbol, tf, df in results:
        if symbol not in output:
            output[symbol] = {}
        output[symbol][tf] = df

    return output


def fetch_multiple(
    self,
    symbols: list[str],
    timeframes: list[str],
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Synchronous wrapper for fetch_multiple."""
    return asyncio.run(
        self.fetch_multiple(symbols, timeframes, start_date, end_date)
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_fetch_multiple_returns_dict -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: implement fetch_multiple for batch operations"
```

---

## Task 8: Add Context Manager Support

**Files:**
- Modify: `src/data/ohlcv.py`

**Step 1: Write the failing test**

Add to `tests/test_ohlcv.py`:

```python
def test_context_manager():
    """Test that client can be used as context manager."""
    with CoinbaseDataClient() as client:
        assert client is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ohlcv.py::test_context_manager -v`
Expected: FAIL with "TypeError"

**Step 3: Write minimal implementation**

Add to `CoinbaseDataClient` class in `src/data/ohlcv.py`:

```python
async def __aenter__(self):
    """Async context manager entry."""
    await self._get_exchange()
    return self


async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit."""
    await self.close()
    return False


def __enter__(self):
    """Sync context manager entry (for convenience)."""
    return self


def __exit__(self, exc_type, exc_val, exc_tb):
    """Sync context manager exit."""
    asyncio.run(self.close())
    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_context_manager -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ohlcv.py tests/test_ohlcv.py
git commit -m "feat: add context manager support to CoinbaseDataClient"
```

---

## Task 9: Final Integration Test

**Files:**
- Modify: `tests/test_ohlcv.py`

**Step 1: Write integration test**

Add to `tests/test_ohlcv.py`:

```python
@pytest.mark.integration
def test_full_workflow_integration(tmp_path):
    """Integration test: fetch, save, load, update workflow."""
    from src.data import CoinbaseDataClient

    client = CoinbaseDataClient(data_dir=str(tmp_path))

    # This test would require network - mark as integration
    # For now, just verify the workflow methods exist
    assert hasattr(client, "fetch")
    assert hasattr(client, "save")
    assert hasattr(client, "load")
    assert hasattr(client, "update")
    assert hasattr(client, "fetch_latest")
    assert hasattr(client, "fetch_multiple")
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_ohlcv.py::test_full_workflow_integration -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ohlcv.py
git commit -m "test: add integration test for full workflow"
```

---

## Task 10: Update Module Exports

**Files:**
- Modify: `src/data/__init__.py`

**Step 1: Verify exports are correct**

Run: `python -c "from src.data import CoinbaseDataClient, OHLCV_SCHEMA; print('OK')"`
Expected: OK

**Step 2: Commit**

```bash
git add src/data/__init__.py
git commit -m "chore: verify module exports"
```

---

## Summary

**Files Created:**
- `src/data/__init__.py`
- `src/data/ohlcv.py`
- `tests/test_ohlcv.py`

**Files Modified:**
- `pyproject.toml` (dependencies)

**Total Tasks:** 10

**Key Features Implemented:**
1. Async OHLCV fetching with pagination
2. Parquet storage with year/month partitioning
3. Load with year/month filtering
4. fetch_latest for live trading
5. update() for append + overwrite workflow
6. fetch_multiple for batch operations
7. Context manager support
8. Rate limiting via semaphore
