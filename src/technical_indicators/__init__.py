"""Technical indicators module for Mad Money.

Provides a registry-based approach to adding technical indicators to OHLCV DataFrames.
Supports both base timeframe computation and synthetic higher timeframe indicators.

Example usage:
    >>> import polars as pl
    >>> from src.technical_indicators import add_adx, add_indicators
    >>>
    >>> df = pl.read_parquet("data/coinbase/ohlcv/aave-usdc/5m/2025/01.parquet")
    >>>
    >>> # Single indicator
    >>> df_with_adx = add_adx(df, timeframe="5m", base_timeframe="5m", period=14)
    >>>
    >>> # Multiple indicators at once
    >>> indicators = [
    ...     ("adx", "5m", {"timeperiod": 14}),
    ...     ("adx", "15m", {"timeperiod": 5}),
    ...     ("rsi", "5m", {"timeperiod": 7}),
    ...     ("rsi", "15m", {"timeperiod": 14}),
    ... ]
    >>> df_with_features = add_indicators(df, indicators, base_timeframe="5m")
"""

from src.technical_indicators.core import add_indicator, add_indicators
from src.technical_indicators.registry import (
    INDICATOR_REGISTRY,
    IndicatorDef,
    get_indicator,
    validate_indicator_inputs,
)
from src.technical_indicators.timeframe import (
    is_base_timeframe,
    parse_timeframe,
    timeframe_ratio,
)

__all__ = [
    # Core functions
    "add_indicator",
    "add_indicators",
    # Registry
    "INDICATOR_REGISTRY",
    "IndicatorDef",
    "get_indicator",
    "validate_indicator_inputs",
    # Timeframe utilities
    "parse_timeframe",
    "timeframe_ratio",
    "is_base_timeframe",
]
