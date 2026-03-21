"""Tests for technical indicators ordering and timestamp requirements."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.data.progress import TIMEFRAME_SECONDS
from src.technical_indicators.core import (
    _build_output_name,
    add_indicator,
    add_indicators,
)
from src.technical_indicators.registry import get_indicator, validate_indicator_inputs
from src.technical_indicators.timeframe import (
    is_base_timeframe,
    parse_timeframe,
    timeframe_ratio,
)


def _sample_ohlcv_df(n: int = 12) -> pl.DataFrame:
    """Build deterministic timestamped OHLCV test data."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


def test_add_indicator_requires_timestamp_column() -> None:
    """Reject inputs without timestamp to enforce chronological semantics."""
    df = pl.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
            "close": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="requires a 'timestamp' column"):
        add_indicator(df, "rsi", timeframe="5m", base_timeframe="5m", timeperiod=3)


def test_add_indicator_sorts_unsorted_input_before_computing() -> None:
    """Unsorted input should produce the same result as sorted input."""
    sorted_df = _sample_ohlcv_df()
    unsorted_df = sorted_df.reverse()

    out_from_sorted = add_indicator(
        sorted_df, "rsi", timeframe="15m", base_timeframe="5m", timeperiod=3
    )
    out_from_unsorted = add_indicator(
        unsorted_df, "rsi", timeframe="15m", base_timeframe="5m", timeperiod=3
    )

    # Function guarantees chronological calculations and returns chronological rows.
    assert_frame_equal(out_from_unsorted, out_from_sorted)


def test_add_indicator_synthetic_volume_indicator_mfi_raises_value_error() -> None:
    """Synthetic MFI request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported") as exc_info:
        add_indicator(df, "mfi", timeframe="15m", base_timeframe="5m", timeperiod=14)
    message = str(exc_info.value)
    assert "mfi" in message
    assert "15m" in message
    assert "5m" in message


def test_add_indicator_synthetic_volume_indicator_ad_raises_value_error() -> None:
    """Synthetic AD request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported") as exc_info:
        add_indicator(df, "ad", timeframe="15m", base_timeframe="5m")
    message = str(exc_info.value)
    assert "ad" in message
    assert "15m" in message
    assert "5m" in message


def test_add_indicator_synthetic_volume_indicator_adosc_raises_value_error() -> None:
    """Synthetic ADOSC request raises ValueError with clear unsupported message."""
    df = _sample_ohlcv_df(n=20)

    with pytest.raises(ValueError, match="requires volume.*not yet supported") as exc_info:
        add_indicator(
            df,
            "adosc",
            timeframe="15m",
            base_timeframe="5m",
            fastperiod=3,
            slowperiod=10,
        )
    message = str(exc_info.value)
    assert "adosc" in message
    assert "15m" in message
    assert "5m" in message


def test_add_indicator_base_timeframe_volume_indicator_succeeds() -> None:
    """Base-timeframe MFI still computes successfully."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(
        df, "mfi", timeframe="5m", base_timeframe="5m", timeperiod=14
    )

    assert "mfi_14_5m" in result.columns


def test_add_indicator_synthetic_timeframe_adx() -> None:
    """Synthetic ADX (OHLC-only) still works after volume guard."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(
        df, "adx", timeframe="15m", base_timeframe="5m", timeperiod=14
    )

    assert "adx_14_15m" in result.columns
    # Verify row count unchanged (invariant)
    assert len(result) == len(df)


def test_add_indicator_base_timeframe_adx() -> None:
    """Base-timeframe ADX computes directly and produces expected column name."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(
        df, "adx", timeframe="5m", base_timeframe="5m", timeperiod=14
    )

    assert "adx_14_5m" in result.columns
    # Verify original columns preserved and row count unchanged
    assert len(result) == len(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns


def test_add_indicator_base_timeframe_rsi() -> None:
    """Base-timeframe RSI computes directly and produces expected column name."""
    df = _sample_ohlcv_df(n=20)

    result = add_indicator(df, "rsi", timeframe="5m", base_timeframe="5m", timeperiod=7)

    assert "rsi_7_5m" in result.columns
    # Verify original columns preserved and row count unchanged
    assert len(result) == len(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns


def test_add_indicator_base_timeframe_period_variants() -> None:
    """Same indicator with different periods produces distinct names and values."""
    df = _sample_ohlcv_df(n=50)

    result_7 = add_indicator(
        df, "rsi", timeframe="5m", base_timeframe="5m", timeperiod=7
    )
    result_14 = add_indicator(
        df, "rsi", timeframe="5m", base_timeframe="5m", timeperiod=14
    )

    # Distinct column names
    assert "rsi_7_5m" in result_7.columns
    assert "rsi_14_5m" in result_14.columns
    assert "rsi_7_5m" not in result_14.columns
    assert "rsi_14_5m" not in result_7.columns

    # Distinct computed values - use drop_nulls() before comparing overlapping segments
    rsi_7_values = result_7["rsi_7_5m"].drop_nulls()
    rsi_14_values = result_14["rsi_14_5m"].drop_nulls()
    # Overlapping segment should have at least one different value
    min_len = min(len(rsi_7_values), len(rsi_14_values))
    overlapping_7 = rsi_7_values[:min_len]
    overlapping_14 = rsi_14_values[:min_len]
    # Check that at least one value differs between the two series
    assert not overlapping_7.equals(overlapping_14), (
        "RSI with timeperiod=7 and timeperiod=14 should produce different values"
    )


def test_add_indicator_base_timeframe_bbands() -> None:
    """Base-timeframe bbands produces deterministic multi-output column name and values."""
    df = _sample_ohlcv_df(n=80)

    result = add_indicator(df, "bbands", timeframe="5m", base_timeframe="5m")

    # Expected column name from _build_output_name with defaults: bbands_20_2_0_2_0_5m
    assert "bbands_20_2_0_2_0_5m" in result.columns
    # Verify original columns preserved and row count unchanged
    assert len(result) == len(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns
    # Verify expected struct fields and finite values after warmup period
    bbands_col = "bbands_20_2_0_2_0_5m"
    assert result.schema[bbands_col] == pl.Struct(
        {
            "upperband": pl.Float64,
            "middleband": pl.Float64,
            "lowerband": pl.Float64,
        }
    )
    middleband_finite_count = result.select(
        pl.col(bbands_col).struct.field("middleband").is_finite().sum()
    ).item()
    assert middleband_finite_count > 0


def test_add_indicator_synthetic_ohlc_values() -> None:
    """Higher-timeframe helper OHLC columns stay internal to indicator computation."""
    df = _sample_ohlcv_df(n=12)

    result = add_indicator(
        df, "rsi", timeframe="15m", base_timeframe="5m", timeperiod=3
    )

    # Synthetic helper columns should not leak into user-facing output.
    assert "high_15m" not in result.columns
    assert "low_15m" not in result.columns
    assert "close_15m" not in result.columns
    # Indicator column still computed.
    assert "rsi_3_15m" in result.columns
    # Invariant: row count preserved
    assert len(result) == len(df)


def test_add_indicator_smaller_timeframe_raises() -> None:
    """Target timeframe smaller than base raises ValueError with clear message."""
    df = _sample_ohlcv_df(n=12)

    with pytest.raises(ValueError, match="cannot be smaller than"):
        add_indicator(df, "rsi", timeframe="5m", base_timeframe="15m", timeperiod=14)


def test_add_indicators_batch_two_indicators() -> None:
    """Mixed-indicator batch adds both expected columns from adx and rsi."""
    df = _sample_ohlcv_df(n=20)

    indicators = [
        ("adx", "5m", {"timeperiod": 14}),
        ("rsi", "5m", {"timeperiod": 7}),
    ]
    result = add_indicators(df, indicators, base_timeframe="5m")

    assert "adx_14_5m" in result.columns
    assert "rsi_7_5m" in result.columns
    # Verify row count unchanged (invariant)
    assert len(result) == len(df)


def test_add_indicators_batch_different_timeframes() -> None:
    """Same indicator at different timeframes produces distinct namespaced columns."""
    df = _sample_ohlcv_df(n=30)

    indicators = [
        ("rsi", "5m", {"timeperiod": 7}),
        ("rsi", "15m", {"timeperiod": 7}),
    ]
    result = add_indicators(df, indicators, base_timeframe="5m")

    assert "rsi_7_5m" in result.columns
    assert "rsi_7_15m" in result.columns
    # Distinct column names prove the two are separate
    assert result["rsi_7_5m"].name != result["rsi_7_15m"].name
    # Verify row count unchanged (invariant)
    assert len(result) == len(df)


def test_add_indicators_close_only_higher_timeframe_differs_from_base() -> None:
    """Higher-timeframe close-only indicators should not duplicate base signal."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    close = [
        100.0,
        102.0,
        99.0,
        103.0,
        98.0,
        104.0,
        97.0,
        105.0,
        96.0,
        106.0,
        95.0,
        107.0,
        94.0,
        108.0,
        93.0,
        109.0,
        92.0,
        110.0,
    ]
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=5 * i) for i in range(len(close))],
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1000.0 + i for i in range(len(close))],
        }
    )

    result = add_indicators(
        df,
        [
            ("rsi", "5m", {"timeperiod": 3}),
            ("rsi", "15m", {"timeperiod": 3}),
        ],
        base_timeframe="5m",
    )

    comparable = result.filter(
        pl.col("rsi_3_5m").is_not_null() & pl.col("rsi_3_15m").is_not_null()
    )
    assert len(comparable) > 0

    has_difference = comparable.select(
        (pl.col("rsi_3_5m") - pl.col("rsi_3_15m")).abs().gt(1e-9).any()
    ).item()
    assert has_difference is True


def test_add_indicators_batch_with_synthetic_volume_indicator_raises() -> None:
    """Batch path propagates deterministic ValueError for synthetic volume indicators."""
    df = _sample_ohlcv_df(n=30)

    indicators = [
        ("rsi", "5m", {"timeperiod": 7}),
        ("mfi", "15m", {"timeperiod": 14}),
    ]

    with pytest.raises(ValueError, match="requires volume.*not yet supported") as exc_info:
        add_indicators(df, indicators, base_timeframe="5m")

    message = str(exc_info.value)
    assert "mfi" in message
    assert "15m" in message
    assert "5m" in message


def test_add_indicators_batch_different_params() -> None:
    """Same indicator with different periods produces distinct names and values."""
    df = _sample_ohlcv_df(n=50)

    indicators = [
        ("rsi", "5m", {"timeperiod": 7}),
        ("rsi", "5m", {"timeperiod": 14}),
    ]
    result = add_indicators(df, indicators, base_timeframe="5m")

    assert "rsi_7_5m" in result.columns
    assert "rsi_14_5m" in result.columns
    # Verify values differ for overlapping non-null values
    rsi_7 = result["rsi_7_5m"].drop_nulls()
    rsi_14 = result["rsi_14_5m"].drop_nulls()
    min_len = min(len(rsi_7), len(rsi_14))
    overlapping_7 = rsi_7[:min_len]
    overlapping_14 = rsi_14[:min_len]
    assert not overlapping_7.equals(overlapping_14)
    # Verify row count unchanged (invariant)
    assert len(result) == len(df)


def test_add_indicators_preserves_input_columns() -> None:
    """Batch call leaves original input frame columns unchanged."""
    df = _sample_ohlcv_df(n=20)
    original_columns = set(df.columns)

    indicators = [
        ("adx", "5m", {"timeperiod": 14}),
        ("rsi", "5m", {"timeperiod": 7}),
        ("bbands", "5m", {}),
    ]
    result = add_indicators(df, indicators, base_timeframe="5m")

    # Input columns unchanged
    assert set(df.columns) == original_columns
    # Output contains all original columns plus new indicator columns
    assert all(col in result.columns for col in df.columns)
    assert "adx_14_5m" in result.columns
    assert "rsi_7_5m" in result.columns
    assert "bbands_20_2_0_2_0_5m" in result.columns
    # Row count unchanged (invariant)
    assert len(result) == len(df)


def test_build_output_name_single_param() -> None:
    """Single-parameter indicator (adx) produces exact expected name."""
    adx_def = get_indicator("adx")
    params = {"timeperiod": 14}

    result = _build_output_name(adx_def, params, "5m")

    assert result == "adx_14_5m"


def test_build_output_name_multi_param() -> None:
    """Multi-parameter indicator (macd) produces exact expected name."""
    macd_def = get_indicator("macd")
    params = {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}

    result = _build_output_name(macd_def, params, "1h")

    assert result == "macd_12_26_9_1h"


def test_build_output_name_float_defaults() -> None:
    """Bbands with float defaults produces expected name with underscores replacing dots."""
    bbands_def = get_indicator("bbands")
    params = {}

    result = _build_output_name(bbands_def, params, "15m")

    assert result == "bbands_20_2_0_2_0_15m"
    # Dots replaced by underscores in float fragments
    assert "." not in result


def test_build_output_name_deterministic() -> None:
    """Repeated calls with identical inputs produce identical names."""
    macd_def = get_indicator("macd")
    params = {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}

    result1 = _build_output_name(macd_def, params, "1h")
    result2 = _build_output_name(macd_def, params, "1h")
    result3 = _build_output_name(macd_def, params, "1h")

    assert result1 == result2 == result3
    assert result1 == "macd_12_26_9_1h"


def test_get_indicator_known_definitions() -> None:
    """Known indicators return IndicatorDef with correct metadata fields."""
    adx_def = get_indicator("adx")
    assert adx_def.name == "adx"
    assert adx_def.inputs == ["high", "low", "close"]
    assert adx_def.param_names == ("timeperiod",)
    assert adx_def.defaults == {"timeperiod": 14}
    assert adx_def.output_template == "adx_{timeperiod}"
    assert adx_def.multi_output is False

    rsi_def = get_indicator("rsi")
    assert rsi_def.name == "rsi"
    assert rsi_def.inputs == ["close"]
    assert rsi_def.param_names == ("timeperiod",)
    assert rsi_def.defaults == {"timeperiod": 14}
    assert rsi_def.output_template == "rsi_{timeperiod}"
    assert rsi_def.multi_output is False


def test_get_indicator_unknown_raises_key_error() -> None:
    """Unknown indicator name raises KeyError with informative message."""
    with pytest.raises(KeyError) as exc_info:
        get_indicator("unknown_indicator")

    error_message = str(exc_info.value)
    assert "unknown_indicator" in error_message
    # Verify available indicator names appear in message (checking token membership)
    assert "adx" in error_message
    assert "rsi" in error_message


def test_validate_indicator_inputs_passes() -> None:
    """Validation passes when all required columns are present."""
    df = _sample_ohlcv_df(n=20)
    adx_def = get_indicator("adx")

    # Should not raise
    validate_indicator_inputs(df, adx_def)


def test_validate_indicator_inputs_reports_all_missing_columns() -> None:
    """Validation fail-case reports all missing columns, not just the first."""
    # Use adx which requires high, low, close - remove two of them
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)],
            "open": [100.0],
        }
    )
    adx_def = get_indicator("adx")

    with pytest.raises(ValueError) as exc_info:
        validate_indicator_inputs(df, adx_def)

    error_message = str(exc_info.value)
    # Both missing columns must be reported
    assert "high" in error_message
    assert "low" in error_message
    assert "close" in error_message


# ------------------------------------------------------------------
# Timeframe utility tests (FEAT-006)
# ------------------------------------------------------------------


@pytest.mark.parametrize("timeframe,expected", list(TIMEFRAME_SECONDS.items()))
def test_parse_timeframe_all_supported_values(timeframe: str, expected: int) -> None:
    """Every key in TIMEFRAME_SECONDS parses to its expected seconds."""
    result = parse_timeframe(timeframe)
    assert result == expected
    assert result > 0


def test_parse_timeframe_unknown_raises() -> None:
    """Unknown timeframe string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown timeframe"):
        parse_timeframe("7m")


@pytest.mark.parametrize(
    "target,base,expected",
    [
        ("15m", "5m", 3),
        ("1h", "5m", 12),
        ("5m", "5m", 1),
        ("30m", "1m", 30),
        ("1d", "1h", 24),
    ],
)
def test_timeframe_ratio_valid_values(target: str, base: str, expected: int) -> None:
    """Valid ratio pairs return correct integer ratio >= 1."""
    result = timeframe_ratio(target, base)
    assert result == expected
    assert isinstance(result, int)
    assert result >= 1


def test_timeframe_ratio_invalid_direction_raises() -> None:
    """Target smaller than base raises ValueError."""
    with pytest.raises(ValueError, match="not a multiple of"):
        timeframe_ratio("5m", "15m")


def test_timeframe_ratio_unknown_timeframe_raises() -> None:
    """Unknown timeframe input raises ValueError."""
    with pytest.raises(ValueError, match="Unknown timeframe"):
        timeframe_ratio("7m", "5m")


@pytest.mark.parametrize(
    "timeframe,candidate_base,expected",
    [
        ("15m", "5m", True),
        ("5m", "5m", True),
        ("1h", "5m", True),
        ("5m", "15m", False),
        ("5m", "1h", False),
    ],
)
def test_is_base_timeframe_cases(
    timeframe: str, candidate_base: str, expected: bool
) -> None:
    """is_base_timeframe returns correct bool for valid timeframe pairs."""
    result = is_base_timeframe(timeframe, candidate_base)
    assert result is expected
    assert isinstance(result, bool)


def test_is_base_timeframe_unknown_inputs_returns_false() -> None:
    """Unknown timeframe inputs return False instead of raising."""
    result = is_base_timeframe("7m", "5m")
    assert result is False
    result = is_base_timeframe("5m", "7m")
    assert result is False
