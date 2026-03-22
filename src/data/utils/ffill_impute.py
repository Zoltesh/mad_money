"""Forward-fill OHLC gaps and safely impute volume for OHLCV data."""

from typing import Any, cast

import polars as pl

OHLC_COLS: list[str] = ["open", "high", "low", "close"]
VALUE_COLS: list[str] = [*OHLC_COLS, "volume"]


def ffill_impute(lf: pl.LazyFrame, every: str) -> tuple[pl.LazyFrame, int]:
    """Fill missing timestamps without introducing lookahead bias.

    Missing rows are created over the full timestamp range using `every`.
    OHLC columns are forward-filled only, while missing volume is set to `0.0`.
    The function returns the imputed frame and the number of synthetic rows added.
    """
    base: pl.LazyFrame = (
        lf.select(["timestamp", *VALUE_COLS])
        .with_columns(
            pl.col(name="timestamp").cast(dtype=pl.Datetime(time_unit="ms")),
            *[pl.col(name=c).cast(dtype=pl.Float64) for c in VALUE_COLS],
        )
        .unique(subset="timestamp", keep="first")
        .sort(by="timestamp")
    )

    bounds: pl.DataFrame = cast(
        typ=pl.DataFrame,
        val=base.select(
            pl.col(name="timestamp").min().alias(name="start"),
            pl.col(name="timestamp").max().alias(name="end"),
        ).collect(),
    )

    start: Any = bounds.item(row=0, column="start")
    end: Any = bounds.item(row=0, column="end")

    if start is None or end is None:
        empty: pl.LazyFrame = pl.DataFrame(
            schema={
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        ).lazy()
        return empty, 0

    full_index: pl.Series = pl.datetime_range(
        start=start, end=end, interval=every, time_unit="ms", eager=True
    )

    skeleton: pl.LazyFrame = pl.DataFrame(data={"timestamp": full_index}).lazy()
    base_with_flag: pl.LazyFrame = base.with_columns(pl.lit(value=True).alias(name="_has_observation"))

    joined: pl.LazyFrame = skeleton.join(
        other=base_with_flag, on="timestamp", how="left"
    )

    imputed_count = int(
        cast(
            typ=pl.DataFrame,
            val=joined.select(pl.col(name="_has_observation").is_null().sum()).collect(),
        ).item()
    )

    out: pl.LazyFrame = (
        joined.with_columns(
            pl.col(name=OHLC_COLS).fill_null(strategy="forward"),
            pl.col(name="volume").fill_null(value=0.0),
        )
        .with_columns(
            pl.col(name="timestamp")
            .dt.timestamp(time_unit="ms")
            .cast(dtype=pl.Int64())
        )
        .select(["timestamp", *VALUE_COLS])
    )

    return out, imputed_count
