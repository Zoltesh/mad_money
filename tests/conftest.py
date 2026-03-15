"""Shared pytest fixtures for lean OHLCV test suites."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest


@pytest.fixture
def sample_df_factory() -> Callable[..., pl.DataFrame]:
    """Create a DataFrame builder for timestamp-indexed OHLCV rows."""

    def _build(*timestamps: datetime) -> pl.DataFrame:
        n = len(timestamps)
        return pl.DataFrame(
            {
                "timestamp": list(timestamps),
                "open": [float(i + 1) for i in range(n)],
                "high": [float(i + 1) for i in range(n)],
                "low": [float(i + 1) for i in range(n)],
                "close": [float(i + 1) for i in range(n)],
                "volume": [float(i + 1) for i in range(n)],
            }
        )

    return _build
