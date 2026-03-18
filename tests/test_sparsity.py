"""Tests for OHLCV sparsity report utility."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from src.data.utils.sparsity import build_ohlcv_sparsity_report


def _write_ts_parquet(path: Path, timestamps: Sequence[datetime | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": timestamps}).write_parquet(path)


def test_build_ohlcv_sparsity_report_computes_gap_metrics(tmp_path: Path) -> None:
    """Compute expected metrics from one asset/timeframe with gaps and duplicates."""
    base_path = tmp_path / "ohlcv"
    asset_path = base_path / "aave-usdc" / "1m" / "2026"

    timestamps = [
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
    ]
    _write_ts_parquet(asset_path / "01.parquet", timestamps)

    report = build_ohlcv_sparsity_report(base_path)

    assert report.shape == (1, 15)
    row = report.row(0, named=True)
    assert row["asset"] == "aave-usdc"
    assert row["timeframe"] == "1m"
    assert row["files_processed"] == 1
    assert row["actual"] == 3
    assert row["expected"] == 4
    assert row["missing"] == 1
    assert row["availability"] == 0.75
    assert row["sparsity"] == 0.25
    assert row["gap_count"] == 1
    assert row["avg_gap_missing"] == 1.0
    assert row["shortest_gap_missing"] == 1
    assert row["largest_gap_missing"] == 1
    assert row["avg_extra_gap_seconds"] == 60.0


def test_build_ohlcv_sparsity_report_normalizes_requested_assets(
    tmp_path: Path,
) -> None:
    """Normalize optional asset filter to lowercase and process only requested."""
    base_path = tmp_path / "ohlcv"
    _write_ts_parquet(
        base_path / "aave-usdc" / "1m" / "2026" / "01.parquet",
        [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
    )
    _write_ts_parquet(
        base_path / "btc-usdc" / "1m" / "2026" / "01.parquet",
        [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
    )

    report = build_ohlcv_sparsity_report(base_path, assets=["AAVE-USDC"])
    assert report.shape == (1, 15)
    assert report.row(0, named=True)["asset"] == "aave-usdc"


def test_build_ohlcv_sparsity_report_skips_invalid_structure_and_warns(
    tmp_path: Path, caplog
) -> None:
    """Warn and continue when encountering invalid assets/timeframes/years/files."""
    base_path = tmp_path / "ohlcv"

    _write_ts_parquet(
        base_path / "aave-usdc" / "1m" / "2026" / "01.parquet",
        [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
    )
    _write_ts_parquet(
        base_path / "aave-usdc" / "1m" / "2026" / "1.parquet",
        [datetime(2026, 1, 1, 0, 1, tzinfo=UTC)],
    )
    _write_ts_parquet(
        base_path / "aave-usdc" / "1m" / "26" / "01.parquet",
        [datetime(2026, 1, 1, 0, 2, tzinfo=UTC)],
    )
    _write_ts_parquet(
        base_path / "aave-usdc" / "1M" / "2026" / "01.parquet",
        [datetime(2026, 1, 1, 0, 3, tzinfo=UTC)],
    )
    _write_ts_parquet(
        base_path / "aave-usdc" / "1m" / "2026" / "02.parquet",
        ["bad-ts", "2026-01-01T00:01:00Z"],
    )
    _write_ts_parquet(
        base_path / "AAVE-USDC" / "1m" / "2026" / "01.parquet",
        [datetime(2026, 1, 1, 0, 4, tzinfo=UTC)],
    )

    caplog.set_level("WARNING")
    report = build_ohlcv_sparsity_report(base_path)

    assert report.shape == (1, 15)
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "invalid asset 'AAVE-USDC' skipped" in messages
    assert "invalid timeframe '1M' skipped" in messages
    assert "invalid year '26' skipped" in messages
    assert "invalid month file '1.parquet' skipped" in messages
    assert "invalid timestamp values in" in messages


def test_build_ohlcv_sparsity_report_returns_empty_when_no_valid_data(
    tmp_path: Path,
) -> None:
    """Return empty report when no valid asset data is available."""
    base_path = tmp_path / "ohlcv"
    (base_path / "invalid-asset-name").mkdir(parents=True)

    report = build_ohlcv_sparsity_report(base_path)

    assert report.is_empty()
    assert report.columns == [
        "asset",
        "timeframe",
        "range_start",
        "range_end",
        "files_processed",
        "actual",
        "expected",
        "missing",
        "availability",
        "sparsity",
        "gap_count",
        "avg_gap_missing",
        "shortest_gap_missing",
        "largest_gap_missing",
        "avg_extra_gap_seconds",
    ]


def test_build_ohlcv_sparsity_report_floors_float_metrics(tmp_path: Path) -> None:
    """Floor float metrics to 2 decimals instead of rounding."""
    base_path = tmp_path / "ohlcv"
    asset_path = base_path / "eth-usdc" / "1m" / "2026"

    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=minute) for minute in range(1000)]
    timestamps.pop(500)
    _write_ts_parquet(asset_path / "01.parquet", timestamps)

    report = build_ohlcv_sparsity_report(base_path)
    row = report.row(0, named=True)

    assert row["actual"] == 999
    assert row["expected"] == 1000
    assert row["availability"] == 0.99
    assert row["sparsity"] == 0.0
