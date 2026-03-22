"""Local OHLCV range query helpers for partitioned parquet data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl

from src.data.ohlcv_storage import symbol_to_path
from src.data.progress import TIMEFRAME_SECONDS

OHLCV_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def _empty_ohlcv_frame() -> pl.DataFrame:
    """Return an empty OHLCV-shaped DataFrame."""
    return pl.DataFrame(schema=OHLCV_SCHEMA)


def _empty_ohlcv_lazyframe() -> pl.LazyFrame:
    """Return an empty OHLCV-shaped LazyFrame."""
    return _empty_ohlcv_frame().lazy()


def _normalize_symbol(symbol: str) -> str:
    """Normalize user symbol input to path-safe format."""
    return symbol_to_path(symbol.replace("-", "/"))


def _parse_datetime(value: str | datetime, *, end_of_day: bool = False) -> datetime:
    """Parse a datetime-like input into UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
            ]
            parsed: datetime | None = None
            for fmt in formats:
                try:
                    parsed = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                raise ValueError(f"Unable to parse date: {value}") from None
            dt = parsed

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)

    if (
        end_of_day
        and dt.hour == 0
        and dt.minute == 0
        and dt.second == 0
        and dt.microsecond == 0
    ):
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    return dt


def _floor_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """Floor a datetime to the candle open for a timeframe."""
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {sorted(TIMEFRAME_SECONDS.keys())}"
        )
    frame_seconds = TIMEFRAME_SECONDS[timeframe]
    ts = int(dt.timestamp())
    floored_ts = (ts // frame_seconds) * frame_seconds
    return datetime.fromtimestamp(floored_ts, tz=UTC)


def _list_partition_files(base_path: Path) -> list[Path]:
    """List valid parquet partitions sorted by year/month."""
    files: list[Path] = []
    for year_dir in base_path.iterdir():
        if not year_dir.is_dir():
            continue
        try:
            int(year_dir.name)
        except ValueError:
            continue
        for file_path in year_dir.glob("*.parquet"):
            stem = file_path.stem
            if len(stem) == 2 and stem.isdigit():
                month = int(stem)
                if 1 <= month <= 12:
                    files.append(file_path)
    return sorted(files, key=lambda p: (int(p.parent.name), int(p.stem)))


def _file_month_key(file_path: Path) -> tuple[int, int]:
    """Return (year, month) key for a partition file."""
    return int(file_path.parent.name), int(file_path.stem)


def _resolve_ohlcv_range_files(
    data_dir: str,
    symbol: str,
    timeframe: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> tuple[list[Path], datetime | None, datetime | None]:
    """Resolve parquet files and aligned datetime bounds for a range query."""
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {sorted(TIMEFRAME_SECONDS.keys())}"
        )

    start_dt = _parse_datetime(start) if start is not None else None
    end_dt = (
        _floor_to_timeframe(_parse_datetime(end, end_of_day=True), timeframe)
        if end is not None
        else None
    )

    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError("start must be less than or equal to end after alignment")

    pair_path = _normalize_symbol(symbol)
    base_path = Path(data_dir) / "coinbase" / "ohlcv" / pair_path / timeframe
    if not base_path.exists():
        return [], start_dt, end_dt

    partition_files = _list_partition_files(base_path)
    if not partition_files:
        return [], start_dt, end_dt

    if start_dt is not None:
        start_key = (start_dt.year, start_dt.month)
    else:
        start_key = _file_month_key(partition_files[0])

    if end_dt is not None:
        end_key = (end_dt.year, end_dt.month)
    else:
        end_key = _file_month_key(partition_files[-1])

    selected_files = [
        path
        for path in partition_files
        if start_key <= _file_month_key(path) <= end_key
    ]
    return selected_files, start_dt, end_dt


def lazy_load_ohlcv_range(
    data_dir: str,
    symbol: str,
    timeframe: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pl.LazyFrame:
    """Lazily load locally stored OHLCV rows for a symbol/timeframe range."""
    selected_files, start_dt, end_dt = _resolve_ohlcv_range_files(
        data_dir=data_dir, symbol=symbol, timeframe=timeframe, start=start, end=end
    )
    if not selected_files:
        return _empty_ohlcv_lazyframe()

    lazy = pl.scan_parquet([str(path) for path in selected_files]).sort("timestamp")
    lazy = lazy.unique(subset=["timestamp"], keep="last").sort("timestamp")

    if start_dt is not None:
        lazy = lazy.filter(pl.col("timestamp") >= pl.lit(start_dt))
    if end_dt is not None:
        lazy = lazy.filter(pl.col("timestamp") <= pl.lit(end_dt))
    return lazy


def load_ohlcv_range(
    data_dir: str,
    symbol: str,
    timeframe: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pl.DataFrame:
    """Load locally stored OHLCV rows for a symbol/timeframe datetime range.

    Args:
        data_dir: Root data directory containing `coinbase/ohlcv` partitions.
        symbol: Trading pair symbol (for example `AAVE/USDC` or `aave-usdc`).
        timeframe: Candle timeframe (for example `5m`, `1h`, `1d`).
        start: Optional inclusive lower bound.
        end: Optional inclusive upper bound. This bound is floored to the nearest
            candle open at-or-before the provided time.

    Returns:
        Polars DataFrame containing sorted, de-duplicated OHLCV rows.

    Example:
        >>> df = load_ohlcv_range(
        ...     data_dir="./data",
        ...     symbol="AAVE/USDC",
        ...     timeframe="5m",
        ...     start="2024-01-01T00:00:00",
        ...     end="2026-02-28T23:59:59",
        ... )
    """
    return cast(
        pl.DataFrame,
        lazy_load_ohlcv_range(
            data_dir=data_dir, symbol=symbol, timeframe=timeframe, start=start, end=end
        ).collect(),
    )
