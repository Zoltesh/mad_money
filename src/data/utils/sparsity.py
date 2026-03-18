"""OHLCV sparsity and gap report utilities."""

from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

VALID_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "6h": 360,
    "1d": 1_440,
}

_ASSET_PATTERN = re.compile(r"^[a-z0-9]+-[a-z0-9]+$")
_ASSET_PATTERN_CASE_INSENSITIVE = re.compile(r"^[A-Za-z0-9]+-[A-Za-z0-9]+$")
_YEAR_PATTERN = re.compile(r"^\d{4}$")
_MONTH_FILE_PATTERN = re.compile(r"^(0[1-9]|1[0-2])\.parquet$")

_REPORT_SCHEMA = {
    "asset": pl.String,
    "timeframe": pl.String,
    "range_start": pl.Datetime(time_unit="us", time_zone="UTC"),
    "range_end": pl.Datetime(time_unit="us", time_zone="UTC"),
    "files_processed": pl.Int64,
    "actual": pl.Int64,
    "expected": pl.Int64,
    "missing": pl.Int64,
    "availability": pl.Float64,
    "sparsity": pl.Float64,
    "gap_count": pl.Int64,
    "avg_gap_missing": pl.Float64,
    "shortest_gap_missing": pl.Int64,
    "largest_gap_missing": pl.Int64,
    "avg_extra_gap_seconds": pl.Float64,
}


def timeframe_to_minutes(timeframe: str) -> int:
    """Return timeframe size in minutes."""
    try:
        return VALID_TIMEFRAME_MINUTES[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def build_ohlcv_sparsity_report(
    base_path: Path | str,
    assets: list[str] | None = None,
) -> pl.DataFrame:
    """Build a machine-friendly sparsity report for OHLCV parquet data."""
    root = Path(base_path)
    if not root.exists():
        logger.warning("base path '%s' does not exist", root)
        return pl.DataFrame(schema=_REPORT_SCHEMA)

    discovered_assets = _discover_valid_assets(root)
    requested_assets = _normalize_requested_assets(assets)
    if requested_assets:
        for asset in sorted(requested_assets - discovered_assets):
            logger.warning("requested asset '%s' not found", asset)
        assets_to_process = sorted(discovered_assets & requested_assets)
    else:
        assets_to_process = sorted(discovered_assets)

    rows: list[dict[str, object]] = []
    for asset in assets_to_process:
        rows.extend(_build_asset_rows(root / asset, asset))

    if not rows:
        return pl.DataFrame(schema=_REPORT_SCHEMA)
    return pl.DataFrame(rows, schema=_REPORT_SCHEMA)


def _normalize_requested_assets(assets: list[str] | None) -> set[str]:
    if not assets:
        return set()
    return {asset.strip().lower() for asset in assets if asset.strip()}


def _discover_valid_assets(root: Path) -> set[str]:
    valid_assets: set[str] = set()
    for asset_dir in _iter_dirs(root):
        name = asset_dir.name
        if _ASSET_PATTERN.fullmatch(name):
            valid_assets.add(name)
            continue

        if _ASSET_PATTERN_CASE_INSENSITIVE.fullmatch(name):
            logger.warning(
                "invalid asset '%s' skipped: use lowercase <base-quote>", name
            )
        else:
            logger.warning("invalid asset '%s' skipped", name)
    return valid_assets


def _build_asset_rows(asset_path: Path, asset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for timeframe_dir in _iter_dirs(asset_path):
        timeframe = timeframe_dir.name
        if timeframe in VALID_TIMEFRAME_MINUTES:
            row = _build_timeframe_row(asset, timeframe_dir, timeframe)
            if row is not None:
                rows.append(row)
            continue

        if timeframe.lower() in VALID_TIMEFRAME_MINUTES:
            logger.warning(
                "invalid timeframe '%s' skipped for asset '%s': use lowercase",
                timeframe,
                asset,
            )
        else:
            logger.warning(
                "invalid timeframe '%s' skipped for asset '%s'",
                timeframe,
                asset,
            )
    return rows


def _build_timeframe_row(
    asset: str, timeframe_path: Path, timeframe: str
) -> dict[str, object] | None:
    month_files: list[Path] = []
    for year_dir in _iter_dirs(timeframe_path):
        year_name = year_dir.name
        if not _YEAR_PATTERN.fullmatch(year_name):
            logger.warning(
                "invalid year '%s' skipped for %s/%s",
                year_name,
                asset,
                timeframe,
            )
            continue

        for month_file in sorted(year_dir.iterdir()):
            if not month_file.is_file():
                continue
            if not month_file.name.endswith(".parquet"):
                logger.warning(
                    "invalid month file '%s' skipped for %s/%s/%s",
                    month_file.name,
                    asset,
                    timeframe,
                    year_name,
                )
                continue
            if not _MONTH_FILE_PATTERN.fullmatch(month_file.name):
                if month_file.stem.isdigit() and 1 <= int(month_file.stem) <= 12:
                    logger.warning(
                        "invalid month file '%s' skipped for %s/%s/%s: "
                        "use leading zero",
                        month_file.name,
                        asset,
                        timeframe,
                        year_name,
                    )
                else:
                    logger.warning(
                        "invalid month file '%s' skipped for %s/%s/%s",
                        month_file.name,
                        asset,
                        timeframe,
                        year_name,
                    )
                continue

            month_files.append(month_file)

    combined, files_processed = _read_timestamp_epoch_us_many(month_files)

    if combined is None or combined.len() == 0:
        if files_processed > 0:
            logger.warning(
                "no valid timestamps found for %s/%s (processed %d files)",
                asset,
                timeframe,
                files_processed,
            )
        return None

    unique = combined.unique().sort()
    actual = int(unique.len())
    raw_count = int(combined.len())
    duplicate_count = raw_count - actual
    if duplicate_count > 0:
        logger.warning(
            "duplicate timestamps detected for %s/%s: %d duplicates ignored",
            asset,
            timeframe,
            duplicate_count,
        )

    start_us = int(unique[0])
    end_us = int(unique[-1])
    cadence_us = timeframe_to_minutes(timeframe) * 60 * 1_000_000

    expected = ((end_us - start_us) // cadence_us) + 1
    missing = max(expected - actual, 0)
    availability = _floor_2(_safe_div(actual, expected))
    sparsity = _floor_2(_safe_div(missing, expected))

    deltas = unique.diff().drop_nulls()
    gap_deltas = deltas.filter(deltas > cadence_us)
    gap_count = int(gap_deltas.len())

    if gap_count == 0:
        avg_gap_missing = 0.0
        shortest_gap_missing: int | None = None
        largest_gap_missing: int | None = None
        avg_extra_gap_seconds = 0.0
    else:
        gap_missing = ((gap_deltas // cadence_us) - 1).cast(pl.Int64)
        gap_missing_mean = gap_missing.mean()
        avg_gap_missing = _floor_2(_coerce_float(gap_missing_mean))
        gap_missing_min = gap_missing.min()
        gap_missing_max = gap_missing.max()
        shortest_gap_missing = _coerce_int(gap_missing_min)
        largest_gap_missing = _coerce_int(gap_missing_max)
        extra_gap_seconds = ((gap_deltas - cadence_us) / 1_000_000).cast(pl.Float64)
        extra_gap_mean = extra_gap_seconds.mean()
        avg_extra_gap_seconds = _floor_2(_coerce_float(extra_gap_mean))

    return {
        "asset": asset,
        "timeframe": timeframe,
        "range_start": datetime.fromtimestamp(start_us / 1_000_000, tz=UTC),
        "range_end": datetime.fromtimestamp(end_us / 1_000_000, tz=UTC),
        "files_processed": files_processed,
        "actual": actual,
        "expected": expected,
        "missing": missing,
        "availability": availability,
        "sparsity": sparsity,
        "gap_count": gap_count,
        "avg_gap_missing": avg_gap_missing,
        "shortest_gap_missing": shortest_gap_missing,
        "largest_gap_missing": largest_gap_missing,
        "avg_extra_gap_seconds": avg_extra_gap_seconds,
    }


def _read_timestamp_epoch_us_many(parquet_files: list[Path]) -> tuple[pl.Series | None, int]:
    if not parquet_files:
        return None, 0

    try:
        frame = pl.read_parquet(parquet_files, columns=["timestamp"])
    except Exception:
        return _read_timestamp_epoch_us_many_fallback(parquet_files)

    if "timestamp" not in frame.columns:
        return _read_timestamp_epoch_us_many_fallback(parquet_files)

    raw_series = frame.get_column("timestamp")
    non_null_before = raw_series.len() - raw_series.null_count()
    epoch_series = _timestamp_to_epoch_us(raw_series, parquet_files[0])
    if epoch_series is None:
        return _read_timestamp_epoch_us_many_fallback(parquet_files)

    non_null_after = epoch_series.len() - epoch_series.null_count()
    invalid_values = max(non_null_before - non_null_after, 0)
    if invalid_values > 0:
        logger.warning(
            "invalid timestamp values across %d files: %d values ignored",
            len(parquet_files),
            invalid_values,
        )
    return epoch_series.drop_nulls().cast(pl.Int64), len(parquet_files)


def _read_timestamp_epoch_us_many_fallback(
    parquet_files: list[Path],
) -> tuple[pl.Series | None, int]:
    epoch_chunks: list[pl.Series] = []
    files_processed = 0
    for parquet_file in parquet_files:
        epoch_series = _read_timestamp_epoch_us(parquet_file)
        if epoch_series is None:
            continue
        files_processed += 1
        if epoch_series.len() > 0:
            epoch_chunks.append(epoch_series)

    if not epoch_chunks:
        return pl.Series(values=[], dtype=pl.Int64), files_processed
    return pl.concat(epoch_chunks), files_processed


def _read_timestamp_epoch_us(parquet_file: Path) -> pl.Series | None:
    try:
        frame = pl.read_parquet(parquet_file, columns=["timestamp"])
    except Exception as exc:  # pragma: no cover - backend-specific read failures
        logger.warning("failed reading '%s': %s", parquet_file, exc)
        return None

    if "timestamp" not in frame.columns:
        logger.warning("missing timestamp column in '%s'", parquet_file)
        return None

    raw_series = frame.get_column("timestamp")
    non_null_before = raw_series.len() - raw_series.null_count()
    epoch_series = _timestamp_to_epoch_us(raw_series, parquet_file)
    if epoch_series is None:
        return None

    non_null_after = epoch_series.len() - epoch_series.null_count()
    invalid_values = max(non_null_before - non_null_after, 0)
    if invalid_values > 0:
        logger.warning(
            "invalid timestamp values in '%s': %d values ignored",
            parquet_file,
            invalid_values,
        )
    return epoch_series.drop_nulls().cast(pl.Int64)


def _timestamp_to_epoch_us(series: pl.Series, parquet_file: Path) -> pl.Series | None:
    dtype_name = str(series.dtype)
    if dtype_name.startswith("Datetime("):
        return series.dt.epoch("us")
    if dtype_name == "Date":
        return series.cast(pl.Datetime(time_unit="us")).dt.epoch("us")
    if dtype_name == "String":
        parsed = series.str.strptime(
            pl.Datetime(time_unit="us", time_zone="UTC"),
            strict=False,
        )
        return parsed.dt.epoch("us")

    logger.warning(
        "unsupported timestamp dtype '%s' in '%s'",
        series.dtype,
        parquet_file,
    )
    return None


def _iter_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_dir()), key=lambda p: p.name
    )


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _coerce_float(value: object | None) -> float:
    if isinstance(value, Real):
        return float(value)
    return 0.0


def _coerce_int(value: object | None) -> int:
    if isinstance(value, Real):
        return int(float(value))
    return 0


def _floor_2(value: float) -> float:
    return math.floor(value * 100) / 100


__all__ = [
    "build_ohlcv_sparsity_report",
    "timeframe_to_minutes",
]
