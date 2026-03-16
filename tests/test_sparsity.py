"""Tests for sparsity utilities."""

import math
from pathlib import Path

import polars as pl

from src.data.utils.sparsity import (
    SparsitySeriesStats,
    expected_rows_for_month,
    month_to_int,
    parquet_coverage_report,
    sparsity_for_column,
    sparsity_for_series,
    sparsity_report,
    timeframe_to_minutes,
)


def test_sparsity_for_series_basic() -> None:
    """Ensure series sparsity counts nan/null correctly."""
    series = pl.Series("values", [1.0, float("nan"), None])
    stats = sparsity_for_series(series)
    assert isinstance(stats, SparsitySeriesStats)
    assert stats.name == "values"
    assert stats.total_count == 3
    assert stats.null_count == 1
    assert stats.nan_count == 1
    assert stats.sparse_count == 2
    assert math.isclose(stats.sparse_pct, 2 / 3)


def test_sparsity_report_mixed_types() -> None:
    """Validate report metrics on mixed-type DataFrame."""
    df = pl.DataFrame(
        {
            "a": [1.0, float("nan"), None],
            "b": [1, None, 3],
            "c": [None, None, None],
        }
    )

    report = sparsity_report(df)

    assert report["overall"]["total_rows"] == 3
    assert report["overall"]["total_columns"] == 3
    assert report["overall"]["total_cells"] == 9
    assert report["overall"]["null_cells"] == 5
    assert report["overall"]["nan_cells"] == 1
    assert report["overall"]["sparse_cells"] == 6
    assert math.isclose(report["overall"]["sparse_pct"], 6 / 9)

    assert report["rows"]["any_sparse_count"] == 3
    assert math.isclose(report["rows"]["any_sparse_pct"], 1.0)
    assert report["rows"]["all_sparse_count"] == 1
    assert math.isclose(report["rows"]["all_sparse_pct"], 1 / 3)

    assert report["columns"]["any_sparse_count"] == 3
    assert report["columns"]["all_sparse_count"] == 1

    per_column = report["per_column"]
    assert per_column.shape == (3, 6)
    assert sparsity_for_column(df, "a").sparse_count == 2


def test_timeframe_to_minutes() -> None:
    """Ensure timeframes map to expected minute counts."""
    assert timeframe_to_minutes("1m") == 1
    assert timeframe_to_minutes("5m") == 5
    assert timeframe_to_minutes("2h") == 120
    assert timeframe_to_minutes("1d") == 1440
    assert timeframe_to_minutes("1w") == 10080


def test_expected_rows_for_month() -> None:
    """Validate expected row counts for key months."""
    assert expected_rows_for_month(2026, 1, "1m") == 44640
    assert expected_rows_for_month(2026, 2, "1m") == 40320
    assert expected_rows_for_month(2024, 2, "1m") == 41760
    assert expected_rows_for_month(2026, 1, "5m") == 8928


def test_month_to_int() -> None:
    """Normalize month inputs into integer form."""
    assert month_to_int(1) == 1
    assert month_to_int("01") == 1
    assert month_to_int("1") == 1
    assert month_to_int("jan") == 1
    assert month_to_int("January") == 1


def test_parquet_coverage_report(tmp_path: Path) -> None:
    """Create parquet data and confirm coverage output."""
    base_path = tmp_path / "ohlcv"
    symbol = "eth-usdc"
    timeframe = "1m"
    year = "2026"

    month_path = base_path / symbol / timeframe / year
    month_path.mkdir(parents=True)
    data = pl.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"] * 10,
            "open": [1.0] * 10,
            "high": [1.0] * 10,
            "low": [1.0] * 10,
            "close": [1.0] * 10,
            "volume": [1.0] * 10,
        }
    )
    data.write_parquet(month_path / "01.parquet")

    report = parquet_coverage_report(
        base_path=base_path,
        symbols=[symbol],
        years=[year],
        months=[1, 2],
        timeframes=[timeframe],
    )

    assert report.shape == (2, 7)
    jan_row = report.filter(pl.col("month") == "jan").row(0, named=True)
    feb_row = report.filter(pl.col("month") == "feb").row(0, named=True)

    assert jan_row["symbol"] == symbol
    assert jan_row["year"] == year
    assert jan_row["month"] == "jan"
    assert jan_row["timeframe"] == timeframe
    assert jan_row["row_count"] == 10
    assert jan_row["max_possible"] == 44640
    assert jan_row["ratio"] == 0.0

    assert feb_row["month"] == "feb"
    assert feb_row["row_count"] == 0
    assert feb_row["max_possible"] == 40320
    assert feb_row["ratio"] == 0.0


def test_parquet_coverage_report_rounds_down(tmp_path: Path) -> None:
    """Ensure ratio floors to nearest hundredth."""
    base_path = tmp_path / "ohlcv"
    symbol = "eth-usdc"
    timeframe = "5m"
    year = "2026"

    month_path = base_path / symbol / timeframe / year
    month_path.mkdir(parents=True)
    data = pl.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"] * 8063,
            "open": [1.0] * 8063,
            "high": [1.0] * 8063,
            "low": [1.0] * 8063,
            "close": [1.0] * 8063,
            "volume": [1.0] * 8063,
        }
    )
    data.write_parquet(month_path / "02.parquet")

    report = parquet_coverage_report(
        base_path=base_path,
        symbols=[symbol],
        years=[year],
        months=[2],
        timeframes=[timeframe],
    )
    feb_row = report.filter(pl.col("month") == "feb").row(0, named=True)
    assert feb_row["ratio"] == 0.99
