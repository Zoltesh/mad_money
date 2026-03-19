"""ADX indicator convenience function."""

from __future__ import annotations

from polars import DataFrame

from src.technical_indicators.core import add_indicator


def add_adx(
    df: DataFrame,
    timeframe: str,
    base_timeframe: str,
    period: int = 14,
) -> DataFrame:
    """Add ADX (Average Directional Index) indicator to a DataFrame.

    Args:
        df: Input DataFrame with OHLCV columns.
        timeframe: Timeframe for the indicator (e.g., "5m", "15m", "1h").
        base_timeframe: Base timeframe of the input data (e.g., "5m").
        period: ADX period. Defaults to 14.

    Returns:
        DataFrame with new ADX column (e.g., "adx_14_5m").

    Examples:
        >>> df = pl.DataFrame({
        ...     "timestamp": [...] ,
        ...     "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]
        ... })
        >>> # Compute ADX with period 14 on 5m base data
        >>> df_with_adx = add_adx(df, timeframe="5m", base_timeframe="5m", period=14)
        >>> # Compute synthetic 15m ADX on 5m base data
        >>> df_with_adx = add_adx(df, timeframe="15m", base_timeframe="5m", period=5)
    """
    return add_indicator(
        df,
        indicator="adx",
        timeframe=timeframe,
        base_timeframe=base_timeframe,
        timeperiod=period,
    )
