# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mad-money** is a high-performance crypto analysis library built with Polars for fast numerical computation. It provides statistical tools for analyzing cryptocurrency market data.

## Quick Start

```bash
uv sync          # Install all dependencies
pytest           # Run tests
```

## Common Commands

```bash
# Install dependencies (including dev)
uv sync --group dev

# Run a specific test file
uv run pytest tests/test_vif.py

# Run a specific test
uv run pytest tests/test_vif.py::test_vif_basic

# Lint, format, and type check code
uv run ruff check .
uv run ruff format .
uv run ty .
```

## Code Style

- **Linter/Formatter**: ruff (pyproject.toml configured for Python 3.13+)
- **Line length**: 88 characters
- **Docstring convention**: Google style
- **Imports**: All imports must be at the top of the file (no inline imports)

## Architecture

```
src/
├── __init__.py              # Package entry point, exports main APIs
├── main.py                 # CLI entry point (click-based)
├── settings.py             # Pydantic settings configuration
├── stats/
│   ├── __init__.py          # Stats module exports
│   └── vif.py              # Variance Inflation Factor implementation
├── data/
│   ├── __init__.py          # Data module exports
│   └── ohlcv.py            # OHLCV data retrieval from Coinbase
└── technical_indicators/
    ├── __init__.py          # Technical indicators module exports
    ├── adx.py              # ADX convenience function
    ├── core.py             # Core add_indicator logic
    ├── registry.py         # Indicator definitions
    └── timeframe.py        # Timeframe parsing utilities
```

The project is organized by domain:
- `src/stats/` - Statistical analysis tools (VIF currently implemented)
- `src/data/` - Data retrieval (OHLCV from Coinbase)
- `src/technical_indicators/` - Technical indicators with multi-timeframe support

### OHLCV Data Module (`src/data/ohlcv.py`)

Retrieves OHLCV (candlestick) data from Coinbase via ccxt.

| Method | Purpose |
|--------|---------|
| `fetch(symbol, timeframe, start_date, end_date)` | Fetch historical OHLCV by date range |
| `fetch_latest(symbol, timeframe)` | Fetch only new candles since last stored |
| `save(df, symbol, timeframe)` | Save DataFrame to parquet |
| `load(symbol, timeframe, year, month)` | Load from parquet |
| `update(symbol, timeframe)` | Load existing + fetch latest + overwrite |
| `fetch_multiple(symbols, timeframes, ...)` | Batch fetch with concurrency |

### VIF Module (`src/stats/vif.py`)

Three computation methods available:
- **matrix** (default): Fastest - uses correlation matrix inversion
- **parallel**: Multi-threaded OLS regression per column
- **streaming**: Chunked processing for large datasets

All methods return a Polars DataFrame with columns `["feature", "VIF"]`.

### Technical Indicators Module (`src/technical_indicators/`)

Provides a registry-based approach to adding technical indicators with support for synthetic higher timeframe computation.

**Core Functions:**
| Function | Purpose |
|----------|---------|
| `add_indicator(df, indicator, timeframe, base_timeframe, **params)` | Add a single indicator |
| `add_indicators(df, indicators, base_timeframe)` | Add multiple indicators at once |
| `add_adx(df, timeframe, base_timeframe, period)` | Convenience function for ADX |

**Supported Indicators:** adx, rsi, bbands, atr, stoch, macd, stochrsi, cci, willr, mom, roc

**Key Features:**
- **Multi-timeframe support**: Compute indicators on base timeframe or synthetic higher timeframes
- **Synthetic candles**: Higher timeframe indicators use rolling aggregation (max high, min low, current close)
- **Consistent naming**: Column names include indicator, params, and timeframe (e.g., `adx_14_5m`, `rsi_7_15m`)

**Example usage:**
```python
from src.technical_indicators import add_adx, add_indicators

# Single indicator
df = add_adx(df, timeframe="5m", base_timeframe="5m", period=14)

# Multiple indicators
indicators = [
    ("adx", "5m", {"timeperiod": 14}),
    ("adx", "15m", {"timeperiod": 5}),
    ("rsi", "5m", {"timeperiod": 7}),
]
df = add_indicators(df, indicators, base_timeframe="5m")
```

## Testing

Tests use pytest. Some tests use statsmodels for validation (optional - uses `pytest.importorskip`).

## Notes

- Python 3.13+ required (specified in pyproject.toml)
- Uses uv for dependency management
