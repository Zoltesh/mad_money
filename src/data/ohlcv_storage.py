"""Storage helpers for OHLCV parquet persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl


def symbol_to_path(symbol: str) -> str:
    """Convert symbol to a path-friendly format."""
    return symbol.replace("/", "-").lower()


def _atomic_write_parquet(df: pl.DataFrame, file_path: Path) -> None:
    """Write parquet atomically via temp file then replace."""
    temp_path = file_path.with_name(f".{file_path.name}.{uuid4().hex}.tmp")
    df.write_parquet(temp_path)
    temp_path.replace(file_path)


def save_partitions(
    *,
    data_dir: str,
    df: pl.DataFrame,
    symbol: str,
    timeframe: str,
) -> None:
    """Save OHLCV data to partitioned parquet files."""
    if df.is_empty():
        return

    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    df_cols = set(df.columns)
    if not required_cols.issubset(df_cols):
        missing = required_cols - df_cols
        raise ValueError(f"DataFrame missing required columns: {missing}")

    pair_path = symbol_to_path(symbol)
    df_with_partition = df.with_columns(
        [
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
        ]
    )

    for (year, month), partition_df in df_with_partition.group_by(["year", "month"]):
        dir_path = (
            Path(data_dir) / "coinbase" / "ohlcv" / pair_path / timeframe / str(year)
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{month:02d}.parquet"

        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            existing_df = existing_df.filter(
                (pl.col("timestamp").dt.year() == int(year))
                & (pl.col("timestamp").dt.month() == int(month))
            )
            combined_df = pl.concat([existing_df, partition_df.drop(["year", "month"])])
            combined_df = combined_df.unique(subset=["timestamp"], keep="last")
            combined_df = combined_df.sort("timestamp")
            _atomic_write_parquet(combined_df, file_path)
        else:
            _atomic_write_parquet(
                partition_df.drop(["year", "month"]).sort("timestamp"),
                file_path,
            )


def load_partitions(
    *,
    data_dir: str,
    ohlcv_schema: dict[str, Any],
    symbol: str,
    timeframe: str,
    year: int | None = None,
    month: int | None = None,
) -> pl.DataFrame:
    """Load OHLCV data from partitioned parquet files."""
    if month is not None and year is None:
        raise ValueError("year must be specified when month is provided")

    if month is not None and (month < 1 or month > 12):
        raise ValueError(f"month must be between 1 and 12, got {month}")

    pair_path = symbol_to_path(symbol)
    base_path = Path(data_dir) / "coinbase" / "ohlcv" / pair_path / timeframe

    if not base_path.exists():
        return pl.DataFrame(schema=ohlcv_schema)

    parquet_files: list[Path] = []
    if year is not None:
        year_path = base_path / str(year)
        if year_path.exists():
            parquet_files = list(year_path.glob("*.parquet"))
    else:
        for year_dir in base_path.iterdir():
            if year_dir.is_dir():
                parquet_files.extend(year_dir.glob("*.parquet"))

    if not parquet_files:
        return pl.DataFrame(schema=ohlcv_schema)

    if month is not None:
        month_str = f"{month:02d}.parquet"
        parquet_files = [f for f in parquet_files if f.name == month_str]
        if not parquet_files:
            return pl.DataFrame(schema=ohlcv_schema)

    dfs = [pl.read_parquet(file_path) for file_path in sorted(parquet_files)]
    if not dfs:
        return pl.DataFrame(schema=ohlcv_schema)

    combined_df = pl.concat(dfs)
    combined_df = combined_df.sort("timestamp")
    combined_df = combined_df.unique(subset=["timestamp"], keep="last")
    return combined_df
