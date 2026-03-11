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
pytest tests/test_vif.py

# Run a specific test
pytest tests/test_vif.py::test_vif_basic

# Lint, format, and type check code
ruff check .
ruff format .
uv run ty .
```

## Development Dependencies

Install via `uv` or pip with the dev group:
```bash
uv sync --group dev
```

## Code Style

- **Linter/Formatter**: ruff (pyproject.toml configured for Python 3.14+)
- **Line length**: 88 characters
- **Docstring convention**: Google style

## Architecture

```
src/
├── __init__.py           # Package entry point, exports main APIs
├── stats/
│   ├── __init__.py       # Stats module exports
│   └── vif.py           # Variance Inflation Factor implementation
└── data/
    ├── __init__.py       # Data module exports
    └── ohlcv.py         # OHLCV data retrieval from Coinbase (planned)
```

The project is organized by domain:
- `src/stats/` - Statistical analysis tools (VIF currently implemented)
- `src/data/` - Data retrieval (OHLCV from Coinbase - in development)

### OHLCV Data Module (`src/data/ohlcv.py`)

Planned module for retrieving OHLCV (candlestick) data from Coinbase via ccxt.
See `docs/plans/2026-03-11-ohlcv-design.md` for full design.

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

## Testing

Tests use pytest. Some tests use statsmodels for validation (optional - uses `pytest.importorskip`).

## Notes

- Python 3.14+ required (specified in pyproject.toml)
- Uses uv for dependency management
