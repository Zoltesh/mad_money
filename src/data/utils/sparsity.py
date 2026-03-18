"""Polars-native sparsity utilities."""

from __future__ import annotations

import calendar
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl


@dataclass(frozen=True, slots=True)
class SparsitySeriesStats:
    """Sparsity statistics for a single series."""

    name: str
    total_count: int
    null_count: int
    nan_count: int
    sparse_count: int
    sparse_pct: float

    def as_dict(self) -> dict[str, Any]:
        """Return stats as a plain dictionary."""
        return {
            "name": self.name,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "nan_count": self.nan_count,
            "sparse_count": self.sparse_count,
            "sparse_pct": self.sparse_pct,
        }


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _is_float_dtype(dtype: pl.DataType) -> bool:
    return dtype in (pl.Float32, pl.Float64)


def _sparse_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if _is_float_dtype(dtype):
        return col.is_null() | col.is_nan()
    return col.is_null()


def sparsity_for_series(series: pl.Series) -> SparsitySeriesStats:
    """Return sparsity stats for a single series."""
    name = series.name or "series"
    total_count = series.len()
    null_count = series.null_count()
    if _is_float_dtype(series.dtype):
        nan_count = int(series.is_nan().sum())
    else:
        nan_count = 0
    sparse_count = null_count + nan_count
    sparse_pct = _safe_div(sparse_count, total_count)
    return SparsitySeriesStats(
        name=name,
        total_count=total_count,
        null_count=null_count,
        nan_count=nan_count,
        sparse_count=sparse_count,
        sparse_pct=sparse_pct,
    )


def sparsity_for_column(df: pl.DataFrame, column: str) -> SparsitySeriesStats:
    """Return sparsity stats for a DataFrame column."""
    return sparsity_for_series(df.get_column(column))


def sparsity_report(df: pl.DataFrame) -> dict[str, Any]:
    """Return a comprehensive sparsity report for a DataFrame."""
    total_rows = df.height
    total_columns = df.width
    total_cells = total_rows * total_columns

    if total_columns == 0:
        return {
            "overall": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "total_cells": total_cells,
                "null_cells": 0,
                "nan_cells": 0,
                "sparse_cells": 0,
                "sparse_pct": 0.0,
            },
            "rows": {
                "any_sparse_count": 0,
                "any_sparse_pct": 0.0,
                "all_sparse_count": 0,
                "all_sparse_pct": 0.0,
            },
            "columns": {
                "any_sparse_count": 0,
                "any_sparse_pct": 0.0,
                "all_sparse_count": 0,
                "all_sparse_pct": 0.0,
                "sparse_pct_mean": 0.0,
                "sparse_pct_std": 0.0,
            },
            "per_column": pl.DataFrame(
                {
                    "column": [],
                    "total_count": [],
                    "null_count": [],
                    "nan_count": [],
                    "sparse_count": [],
                    "sparse_pct": [],
                }
            ),
        }

    schema = df.schema
    sparse_exprs = [_sparse_expr(name, dtype) for name, dtype in schema.items()]

    count_exprs: list[pl.Expr] = []
    for name, dtype in schema.items():
        col = pl.col(name)
        null_expr = col.is_null().sum().alias(f"{name}__null_count")
        if _is_float_dtype(dtype):
            nan_expr = col.is_nan().sum().alias(f"{name}__nan_count")
            sparse_expr = (
                (col.is_null() | col.is_nan()).sum().alias(f"{name}__sparse_count")
            )
        else:
            nan_expr = pl.lit(0).alias(f"{name}__nan_count")
            sparse_expr = col.is_null().sum().alias(f"{name}__sparse_count")
        count_exprs.extend([null_expr, nan_expr, sparse_expr])

    rows_exprs = [
        pl.any_horizontal(sparse_exprs).sum().alias("__rows_any_sparse"),
        pl.all_horizontal(sparse_exprs).sum().alias("__rows_all_sparse"),
    ]

    counts_row = df.select(count_exprs + rows_exprs).row(0)
    counts_cols = count_exprs + rows_exprs
    counts = dict(zip([expr.meta.output_name() for expr in counts_cols], counts_row))

    per_column_rows = []
    sparse_cells = 0
    null_cells = 0
    nan_cells = 0

    for name in schema:
        total_count = total_rows
        null_count = int(counts[f"{name}__null_count"])
        nan_count = int(counts[f"{name}__nan_count"])
        sparse_count = int(counts[f"{name}__sparse_count"])
        sparse_pct = _safe_div(sparse_count, total_count)
        per_column_rows.append(
            {
                "column": name,
                "total_count": total_count,
                "null_count": null_count,
                "nan_count": nan_count,
                "sparse_count": sparse_count,
                "sparse_pct": sparse_pct,
            }
        )
        sparse_cells += sparse_count
        null_cells += null_count
        nan_cells += nan_count

    per_column = pl.DataFrame(per_column_rows)

    row_any_sparse = int(counts["__rows_any_sparse"])
    row_all_sparse = int(counts["__rows_all_sparse"])

    mean_value = per_column["sparse_pct"].mean() if total_columns else None
    std_value = per_column["sparse_pct"].std() if total_columns else None
    sparse_pct_mean = cast(float, mean_value) if mean_value is not None else 0.0
    sparse_pct_std = cast(float, std_value) if std_value is not None else 0.0

    any_sparse_cols = int((per_column["sparse_count"] > 0).sum())
    all_sparse_cols = int((per_column["sparse_count"] == total_rows).sum())

    return {
        "overall": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_cells": total_cells,
            "null_cells": null_cells,
            "nan_cells": nan_cells,
            "sparse_cells": sparse_cells,
            "sparse_pct": _safe_div(sparse_cells, total_cells),
        },
        "rows": {
            "any_sparse_count": row_any_sparse,
            "any_sparse_pct": _safe_div(row_any_sparse, total_rows),
            "all_sparse_count": row_all_sparse,
            "all_sparse_pct": _safe_div(row_all_sparse, total_rows),
        },
        "columns": {
            "any_sparse_count": any_sparse_cols,
            "any_sparse_pct": _safe_div(any_sparse_cols, total_columns),
            "all_sparse_count": all_sparse_cols,
            "all_sparse_pct": _safe_div(all_sparse_cols, total_columns),
            "sparse_pct_mean": sparse_pct_mean,
            "sparse_pct_std": sparse_pct_std,
        },
        "per_column": per_column,
    }


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert a timeframe string (e.g., 1m, 5m, 1h) to minutes."""
    raw = timeframe.strip().lower()
    if len(raw) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    value_part = raw[:-1]
    unit = raw[-1]
    if not value_part.isdigit():
        raise ValueError(f"Invalid timeframe: {timeframe}")

    value = int(value_part)
    if value <= 0:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    if unit == "w":
        return value * 7 * 1440

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def expected_rows_for_month(year: int, month: int, timeframe: str) -> int:
    """Return the max possible rows for a month/timeframe combination."""
    minutes = timeframe_to_minutes(timeframe)
    minutes_per_day = 24 * 60
    if minutes_per_day % minutes != 0:
        raise ValueError(f"Timeframe {timeframe} does not align to full days")

    days_in_month = calendar.monthrange(year, month)[1]
    candles_per_day = minutes_per_day // minutes
    return days_in_month * candles_per_day


def month_to_int(month: int | str) -> int:
    """Normalize a month value into its integer form (1-12)."""
    if isinstance(month, int):
        month_int = month
    else:
        raw = month.strip().lower()
        if raw.isdigit():
            month_int = int(raw)
        else:
            name_map = {
                name.lower(): index
                for index, name in enumerate(calendar.month_name)
                if name
            }
            abbr_map = {
                name.lower(): index
                for index, name in enumerate(calendar.month_abbr)
                if name
            }
            if raw in name_map:
                month_int = name_map[raw]
            elif raw in abbr_map:
                month_int = abbr_map[raw]
            else:
                raise ValueError(f"Invalid month: {month}")

    if not 1 <= month_int <= 12:
        raise ValueError(f"Invalid month: {month}")
    return month_int


def _month_label(month: int) -> str:
    return calendar.month_abbr[month].lower()


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pl.read_parquet(path, columns=["timestamp"]).height)


_SYMBOL_PATTERN = re.compile(r"^[a-z0-9]+-[a-z0-9]+$")


def _iter_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(item.name for item in path.iterdir() if item.is_dir())


def _is_symbol_name(name: str) -> bool:
    return bool(_SYMBOL_PATTERN.fullmatch(name))


def _discover_symbols(base_path: Path, symbols: list[str]) -> list[str]:
    if symbols:
        return symbols
    return [name for name in _iter_dirs(base_path) if _is_symbol_name(name)]


def _discover_timeframes(symbol_path: Path, timeframes: list[str]) -> list[str]:
    if timeframes:
        return timeframes
    return _iter_dirs(symbol_path)


def _discover_years(timeframe_path: Path, years: list[int | str]) -> list[str]:
    if years:
        return [str(year) for year in years]
    return _iter_dirs(timeframe_path)


def _discover_months(year_path: Path, months: list[int | str]) -> list[int]:
    if months:
        return [month_to_int(month) for month in months]
    if not year_path.exists():
        return []
    month_numbers: set[int] = set()
    for parquet_path in year_path.glob("*.parquet"):
        if parquet_path.stem.isdigit():
            month_numbers.add(int(parquet_path.stem))
    return sorted(month_numbers)


def parquet_coverage_report(
    base_path: Path,
    symbols: Iterable[str] | None = None,
    years: Iterable[int | str] | None = None,
    months: Iterable[int | str] | None = None,
    timeframes: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return a DataFrame with parquet row coverage by month."""
    rows: list[dict[str, object]] = []

    symbol_list = list(symbols or [])
    timeframe_list = list(timeframes or [])
    year_list = list(years or [])
    month_list = list(months or [])

    for symbol in _discover_symbols(Path(base_path), symbol_list):
        symbol_path = Path(base_path) / symbol
        for timeframe in _discover_timeframes(symbol_path, timeframe_list):
            timeframe_path = symbol_path / timeframe
            for year_str in _discover_years(timeframe_path, year_list):
                year_path = timeframe_path / year_str
                year_int = int(year_str)
                for month_int in _discover_months(year_path, month_list):
                    month_name = _month_label(month_int)
                    file_path = year_path / f"{month_int:02d}.parquet"
                    row_count = _parquet_row_count(file_path)
                    max_possible = expected_rows_for_month(
                        year_int, month_int, timeframe
                    )
                    ratio_raw = _safe_div(row_count, max_possible)
                    ratio = math.floor(ratio_raw * 100) / 100
                    rows.append(
                        {
                            "symbol": symbol,
                            "year": year_str,
                            "month": month_name,
                            "timeframe": timeframe,
                            "row_count": row_count,
                            "max_possible": max_possible,
                            "ratio": ratio,
                        }
                    )

    return pl.DataFrame(rows)


__all__ = [
    "SparsitySeriesStats",
    "expected_rows_for_month",
    "month_to_int",
    "parquet_coverage_report",
    "sparsity_for_column",
    "sparsity_for_series",
    "sparsity_report",
    "timeframe_to_minutes",
]
