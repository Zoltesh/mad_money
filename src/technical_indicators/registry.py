"""Indicator registry for technical indicators module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars_talib as ta
from polars import DataFrame, Expr


@dataclass(frozen=True)
class IndicatorDef:
    """Definition of a technical indicator's contract.

    Attributes:
        name: Unique identifier for the indicator (e.g., "adx").
        inputs: Required input columns (e.g., ["high", "low", "close"]).
        func: The polars_talib function to call.
        param_names: Ordered names of core parameters (e.g., ["period"] for ADX).
        defaults: Default values for parameters.
        output_template: Template for output column name using param values.
            Uses curly braces for param substitution (e.g., "adx_{period}").
        multi_output: Whether indicator produces multiple columns.
            If True, the indicator returns a struct/struct-like expression.
    """

    name: str
    inputs: list[str]
    func: Callable[..., Expr]
    param_names: tuple[str, ...]
    defaults: dict[str, int | float]
    output_template: str
    multi_output: bool = False


# Registry of all supported indicators
INDICATOR_REGISTRY: dict[str, IndicatorDef] = {
    "adx": IndicatorDef(
        name="adx",
        inputs=["high", "low", "close"],
        func=ta.adx,
        param_names=("timeperiod",),
        defaults={"timeperiod": 14},
        output_template="adx_{timeperiod}",
    ),
    "rsi": IndicatorDef(
        name="rsi",
        inputs=["close"],
        func=ta.rsi,
        param_names=("timeperiod",),
        defaults={"timeperiod": 14},
        output_template="rsi_{timeperiod}",
    ),
    "bbands": IndicatorDef(
        name="bbands",
        inputs=["close"],
        func=ta.bbands,
        param_names=("timeperiod", "nbdevup", "nbdevdn"),
        defaults={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
        output_template="bbands_{timeperiod}_{nbdevup}_{nbdevdn}",
    ),
    "atr": IndicatorDef(
        name="atr",
        inputs=["high", "low", "close"],
        func=ta.atr,
        param_names=("timeperiod",),
        defaults={"timeperiod": 14},
        output_template="atr_{timeperiod}",
    ),
    "stoch": IndicatorDef(
        name="stoch",
        inputs=["high", "low", "close"],
        func=ta.stoch,
        param_names=("fastk_period", "slowk_period", "slowd_period"),
        defaults={"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
        output_template="stoch_{fastk_period}_{slowk_period}_{slowd_period}",
    ),
    "macd": IndicatorDef(
        name="macd",
        inputs=["close"],
        func=ta.macd,
        param_names=("fastperiod", "slowperiod", "signalperiod"),
        defaults={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        output_template="macd_{fastperiod}_{slowperiod}_{signalperiod}",
    ),
    "stochrsi": IndicatorDef(
        name="stochrsi",
        inputs=["close"],
        func=ta.stochrsi,
        param_names=("timeperiod", "fastk_period", "fastd_period"),
        defaults={"timeperiod": 14, "fastk_period": 5, "fastd_period": 3},
        output_template="stochrsi_{timeperiod}_{fastk_period}_{fastd_period}",
    ),
    "cci": IndicatorDef(
        name="cci",
        inputs=["high", "low", "close"],
        func=ta.cci,
        param_names=("timeperiod",),
        defaults={"timeperiod": 14},
        output_template="cci_{timeperiod}",
    ),
    "willr": IndicatorDef(
        name="willr",
        inputs=["high", "low", "close"],
        func=ta.willr,
        param_names=("timeperiod",),
        defaults={"timeperiod": 14},
        output_template="willr_{timeperiod}",
    ),
    "mom": IndicatorDef(
        name="mom",
        inputs=["close"],
        func=ta.mom,
        param_names=("timeperiod",),
        defaults={"timeperiod": 10},
        output_template="mom_{timeperiod}",
    ),
    "roc": IndicatorDef(
        name="roc",
        inputs=["close"],
        func=ta.roc,
        param_names=("timeperiod",),
        defaults={"timeperiod": 10},
        output_template="roc_{timeperiod}",
    ),
}


def get_indicator(name: str) -> IndicatorDef:
    """Get an indicator definition by name.

    Args:
        name: Indicator name (e.g., "adx").

    Returns:
        The indicator definition.

    Raises:
        KeyError: If indicator is not registered.
    """
    if name not in INDICATOR_REGISTRY:
        raise KeyError(
            f"Unknown indicator: {name}. Available: {list(INDICATOR_REGISTRY.keys())}"
        )
    return INDICATOR_REGISTRY[name]


def validate_indicator_inputs(df: DataFrame, indicator_def: IndicatorDef) -> None:
    """Validate that a DataFrame has required columns for an indicator.

    Args:
        df: Input DataFrame.
        indicator_def: Indicator definition.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = set(indicator_def.inputs) - set(df.columns)
    if missing:
        raise ValueError(
            f"Indicator '{indicator_def.name}' requires columns {indicator_def.inputs}, "
            f"but DataFrame is missing: {sorted(missing)}"
        )
