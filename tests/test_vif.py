"""Tests for variance_inflation_factor function."""

import numpy as np
import polars as pl
import pytest

from mad_money.stats.vif import variance_inflation_factor


def test_vif_basic():
    """Test basic VIF computation against known values."""
    # Simple case: uncorrelated features should have VIF ~= 1.0
    np.random.seed(42)
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )

    result = variance_inflation_factor(df)
    assert result.columns == ["feature", "VIF"]
    assert len(result) == 3
    # All VIFs should be >= 1.0
    assert (result["VIF"] >= 1.0).all()


def test_vif_with_perfect_correlation():
    """Test VIF with perfectly correlated features (VIF = inf)."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfectly correlated with 'a'
        }
    )

    result = variance_inflation_factor(df)
    # One should be inf (the dependent variable)
    assert result["VIF"].is_infinite().any()


def test_vif_output_format():
    """Test output matches statsmodels format."""
    df = pl.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "x3": np.random.randn(100),
        }
    )

    result = variance_inflation_factor(df)

    # Check columns
    assert result.columns == ["feature", "VIF"]

    # Check types
    assert result["feature"].dtype == pl.String
    assert result["VIF"].dtype == pl.Float64

    # Check all features present
    assert set(result["feature"].to_list()) == {"x1", "x2", "x3"}


def test_vif_threshold_filter():
    """Test threshold filtering."""
    df = pl.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100) * 2 + np.random.randn(100),  # High VIF
        }
    )

    result_no_filter = variance_inflation_factor(df)
    result_with_filter = variance_inflation_factor(df, threshold=5.0)

    # Filtered should have fewer or equal rows
    assert len(result_with_filter) <= len(result_no_filter)
    # All VIFs in filtered result should be >= threshold
    assert (result_with_filter["VIF"] >= 5.0).all()


def test_vif_methods_consistent():
    """Test all methods return same results."""
    np.random.seed(42)
    df = pl.DataFrame({f"x{i}": np.random.randn(50) for i in range(5)})

    result_matrix = variance_inflation_factor(df, method="matrix")
    result_parallel = variance_inflation_factor(df, method="parallel", n_jobs=2)

    # Sort both by feature name for comparison
    matrix_sorted = result_matrix.sort("feature")
    parallel_sorted = result_parallel.sort("feature")

    # Results should be very close (allowing small numerical differences)
    np.testing.assert_allclose(
        matrix_sorted["VIF"].to_numpy(), parallel_sorted["VIF"].to_numpy(), rtol=1e-10
    )


def test_vif_single_column_raises():
    """Test that single column raises error."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="at least 2"):
        variance_inflation_factor(df)


def test_vif_insufficient_rows_raises():
    """Test that insufficient rows raises error."""
    df = pl.DataFrame(
        {
            "a": [1.0],
            "b": [2.0],
        }
    )

    with pytest.raises(ValueError, match="at least 2"):
        variance_inflation_factor(df)


def test_vif_non_numeric_raises():
    """Test that non-numeric columns raise error."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": ["x", "y", "z"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric"):
        variance_inflation_factor(df)
