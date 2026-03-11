# OHLCV Data Module Design

## Overview

A module for retrieving, storing, and loading OHLCV (candlestick) data from Coinbase via ccxt. Supports both historical data analysis and live trading use cases.

## Architecture

- **Class-based**: `CoinbaseDataClient` for connection management and state
- **Async-first**: Uses `ccxt.async_support` with proper rate limiting
- **Polars-native**: All data handled as Polars DataFrames

## File Organization

```
data/coinbase/ohlcv/
  {pair}/
    {timeframe}/
      {year}/
        {month}.parquet
```

Example: `data/coinbase/ohlcv/btc-usdc/1h/2025/01.parquet`

### Design Rationale

- **Month granularity**: Reduces file count (12/year vs 365/year) while staying append-friendly
- **Pair → timeframe → date**: Optimized for common access patterns (load all data for a pair/timeframe)
- **Polars predicate pushdown**: Efficient filtering on year/month partitions

## Core Components

### CoinbaseDataClient

Main client class for all operations.

```python
class CoinbaseDataClient:
    def __init__(
        self,
        data_dir: str = "data",
        max_concurrency: int = 10,
        rate_limit_backoff: float = 1.0,
    ): ...
```

### Methods

| Method | Purpose |
|--------|---------|
| `fetch(symbol, timeframe, start_date, end_date)` | Fetch historical OHLCV by date range |
| `fetch_latest(symbol, timeframe)` | Fetch only new candles since last stored |
| `save(df, symbol, timeframe)` | Save DataFrame to parquet |
| `load(symbol, timeframe, year, month)` | Load from parquet (supports filtering) |
| `update(symbol, timeframe)` | Load existing + fetch latest + overwrite |
| `fetch_multiple(symbols, timeframes, ...)` | Batch fetch with concurrency |

### Rate Limiting Strategy

- **Semaphore-based concurrency**: Limits simultaneous requests
- **Exponential backoff**: Automatic retry on 429 (rate limit) errors
- **Configurable**: Default 10 concurrent requests, adjustable per use case

### Data Schema

```python
pl.DataFrame({
    "timestamp": pl.Datetime,  # UTC timestamp (millisecond precision)
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
})
```

### Supported Timeframes

- `1m`, `5m`, `15m`, `30m` (minute)
- `1h`, `2h`, `6h` (hour)
- `1d` (day)

### Supported Symbols

All Coinbase USDC pairs (e.g., BTC-USDC, ETH-USDC, SOL-USDC)

## Pagination Logic

Coinbase Advanced limits to 300 candles per request. The fetch logic:

1. Calculate date range needed
2. Fetch in chunks (300 candles max per call)
3. Track last timestamp to set `since` for next call
4. Continue until reaching `end_date` or latest

## Error Handling

- **Rate limit (429)**: Exponential backoff + retry
- **Network errors**: Retry with backoff (max 3 attempts)
- **Invalid dates**: Validate start <= end
- **Missing data**: Return empty DataFrame with correct schema

## Use Cases

### Historical Data Analysis

```python
client = CoinbaseDataClient()
df = client.fetch("BTC-USDC", "1h", start_date="2025-01-01", end_date="2025-03-01")
client.save(df, "BTC-USDC", "1h")
```

### Live Trading

```python
client = CoinbaseDataClient()

# On timer/trigger
df = client.fetch_latest("BTC-USDC", "1h")  # Only new candles
client.update(df, "BTC-USDC", "1h")  # Append + overwrite
```

### Batch Loading for ML

```python
df = client.load("BTC-USDC", "1h")  # All available
df = client.load("BTC-USDC", "1h", year=2025, month=1)  # Filtered
```

## Future Considerations

- Support for other exchanges (extend base class)
- WebSocket streaming for live data
- Data validation (check for gaps, outliers)
- Automatic schema migrations for parquet files
