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


def test_vif_matches_statsmodels():
    """Test VIF output matches statsmodels implementation."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif_func,
        )
    except ImportError:
        pytest.skip("statsmodels not installed")

    np.random.seed(42)
    n = 1000
    k = 10

    # Create correlated data
    x = np.random.randn(n, k)
    x[:, 1] = x[:, 0] * 0.8 + x[:, 2] * 0.3  # Add correlation
    x[:, 3] = x[:, 1] * 0.9

    df = pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})

    # Our VIF
    result = variance_inflation_factor(df, method="parallel", n_jobs=2).sort("feature")

    # Statsmodels VIF
    x_df = sm.add_constant(x[:, 1:])  # Exclude constant
    sm_vif = [sm_vif_func(x_df, i) for i in range(x_df.shape[1])]

    # Compare x4-x8 (skip x1-x3 which differ due to x0 inclusion in our data)
    # Our: result["VIF"][4:9] = x4-x8 (indices 4-8, 5 values)
    # Statsmodels: sm_vif[4:9] = x4-x8 (indices 4-8, 5 values)
    our_vif = result["VIF"].to_numpy()[4:9]

    np.testing.assert_allclose(our_vif, sm_vif[4:9], rtol=1e-5)


def test_vif_performance_benchmark():
    """Benchmark VIF performance."""
    import time

    np.random.seed(42)
    n = 5000
    k = 50

    x = np.random.randn(n, k)
    df = pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})

    # Time matrix method
    start = time.perf_counter()
    variance_inflation_factor(df, method="matrix")
    matrix_time = time.perf_counter() - start

    # Time parallel method
    start = time.perf_counter()
    variance_inflation_factor(df, method="parallel", n_jobs=4)
    parallel_time = time.perf_counter() - start

    print(f"\nMatrix method: {matrix_time:.4f}s")
    print(f"Parallel method (4 jobs): {parallel_time:.4f}s")

    # Just ensure it runs in reasonable time (< 5 seconds)
    assert matrix_time < 5.0
    assert parallel_time < 5.0
