# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mad-money** is a high-performance crypto analysis library built with Polars for fast numerical computation. It provides statistical tools for analyzing cryptocurrency market data.

## Common Commands

```bash
# Run all tests
pytest

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
mad_money/
├── __init__.py           # Package entry point, exports main APIs
└── stats/
    ├── __init__.py       # Stats module exports
    └── vif.py           # Variance Inflation Factor implementation
```

The project is organized by domain:
- `mad_money/stats/` - Statistical analysis tools (VIF currently implemented)

### VIF Module (`mad_money/stats/vif.py`)

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
