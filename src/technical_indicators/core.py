"""Core logic for adding technical indicators to DataFrames."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from src.technical_indicators.registry import (
    IndicatorDef,
    get_indicator,
    validate_indicator_inputs,
)
from src.technical_indicators.timeframe import parse_timeframe, timeframe_ratio


def _ensure_chronological(df: DataFrame) -> DataFrame:
    """Validate timestamp presence and guarantee ascending chronological order.

    Technical indicators and rolling synthetic candles are order-dependent.
    To avoid accidental leakage from unsorted inputs, all computations are
    performed on ascending `timestamp`.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame sorted by `timestamp` if needed.

    Raises:
        ValueError: If `timestamp` column is missing.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "technical_indicators requires a 'timestamp' column to ensure "
            "chronological (no-lookahead) calculations"
        )

    timestamp = df.get_column("timestamp")
    if timestamp.is_sorted():
        return df

    return df.sort("timestamp")


def _build_output_name(
    indicator_def: IndicatorDef,
    params: dict,
    timeframe: str,
) -> str:
    """Build the output column name from template, params, and timeframe.

    Args:
        indicator_def: Indicator definition.
        params: Actual parameter values.
        timeframe: The timeframe for the indicator.

    Returns:
        Column name string (e.g., "adx_14_5m").
    """
    name = indicator_def.output_template
    for param_name in indicator_def.param_names:
        value = params.get(param_name, indicator_def.defaults.get(param_name))
        # Convert to string, replacing dots with underscores for column name safety
        value_str = str(value).replace(".", "_")
        name = name.replace(f"{{{param_name}}}", value_str)
    # Append timeframe
    return f"{name}_{timeframe}"


def _synthetic_ohlc(
    df: DataFrame,
    window_size: int,
    prefix: str,
) -> DataFrame:
    """Build synthetic OHLC columns for higher timeframe indicators.

    Args:
        df: Input DataFrame with high, low, close columns.
        window_size: Number of base candles to aggregate.
        prefix: Prefix for new column names (e.g., "15m").

    Returns:
        DataFrame with additional synthetic columns.
    """
    return df.with_columns(
        pl.col("high").rolling_max(window_size).alias(f"high_{prefix}"),
        pl.col("low").rolling_min(window_size).alias(f"low_{prefix}"),
        pl.col("close").alias(f"close_{prefix}"),
    )


def _compute_indicator(
    df: DataFrame,
    indicator_def: IndicatorDef,
    params: dict,
    timeframe: str,
    use_synthetic: bool = False,
) -> DataFrame:
    """Compute an indicator on a DataFrame.

    Args:
        df: Input DataFrame.
        indicator_def: Indicator definition.
        params: Indicator parameters.
        timeframe: The timeframe for the indicator.
        use_synthetic: If True, use synthetic OHLC columns (e.g., high_15m).

    Returns:
        DataFrame with indicator column added.
    """
    output_name = _build_output_name(indicator_def, params, timeframe)

    # Build positional args for input columns
    input_cols = []
    for input_name in indicator_def.inputs:
        col_name = f"{input_name}_{timeframe}" if use_synthetic else input_name
        input_cols.append(pl.col(col_name))

    # Build keyword args for parameters
    kwargs: dict = {}
    for param_name in indicator_def.param_names:
        kwargs[param_name] = params.get(
            param_name, indicator_def.defaults.get(param_name)
        )

    # Call the indicator function with positional inputs and keyword params
    result_expr = indicator_def.func(*input_cols, **kwargs).alias(output_name)

    return df.with_columns(result_expr)


def add_indicator(
    df: DataFrame,
    indicator: str,
    timeframe: str,
    base_timeframe: str,
    **params: int | float,
) -> DataFrame:
    """Add a technical indicator to a DataFrame with multi-timeframe support.

    For base timeframe (timeframe == base_timeframe), computes the indicator
    directly on the input columns.

    For higher timeframes, constructs synthetic candles by aggregating
    base candles, then computes the indicator on the synthetic OHLC.

    Args:
        df: Input DataFrame with OHLCV columns.
        indicator: Indicator name (e.g., "adx", "rsi").
        timeframe: Timeframe for the indicator (e.g., "5m", "15m", "1h").
        base_timeframe: Base timeframe of the input data (e.g., "5m").
        **params: Indicator parameters (e.g., timeperiod=14).

    Returns:
        DataFrame with new indicator column(s).

    Raises:
        ValueError: If indicator not registered or missing required columns.
    """
    df = _ensure_chronological(df)
    indicator_def = get_indicator(indicator)
    validate_indicator_inputs(df, indicator_def)

    # Determine if we need synthetic candles
    timeframe_sec = parse_timeframe(timeframe)
    base_sec = parse_timeframe(base_timeframe)

    if timeframe_sec == base_sec:
        # Base timeframe - compute directly
        return _compute_indicator(
            df, indicator_def, params, timeframe, use_synthetic=False
        )
    elif timeframe_sec < base_sec:
        raise ValueError(
            f"Indicator timeframe ({timeframe}) cannot be smaller than "
            f"base timeframe ({base_timeframe})"
        )
    else:
        # Higher timeframe - construct synthetic candles
        window_size = timeframe_ratio(timeframe, base_timeframe)

        # Build synthetic OHLC columns (e.g., high_15m, low_15m, close_15m)
        df_with_synthetic = _synthetic_ohlc(df, window_size, timeframe)

        # Compute indicator on synthetic candles
        return _compute_indicator(
            df_with_synthetic, indicator_def, params, timeframe, use_synthetic=True
        )


def add_indicators(
    df: DataFrame,
    indicators: list[tuple[str, str, dict]],
    base_timeframe: str,
) -> DataFrame:
    """Add multiple indicators to a DataFrame in a single pass.

    Args:
        df: Input DataFrame with OHLCV columns.
        indicators: List of (indicator_name, timeframe, params) tuples.
            Example: [("adx", "5m", {"timeperiod": 14}), ("adx", "15m", {"timeperiod": 5})]
        base_timeframe: Base timeframe of the input data (e.g., "5m").

    Returns:
        DataFrame with all indicator columns added.
    """
    result_df = df

    for indicator_name, timeframe, params in indicators:
        result_df = add_indicator(
            result_df,
            indicator_name,
            timeframe,
            base_timeframe,
            **params,
        )

    return result_df
