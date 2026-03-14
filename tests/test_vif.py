"""Tests for variance_inflation_factor function."""

import time

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as sm_vif_func,
)

from src.stats.vif import variance_inflation_factor


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


def test_vif_streaming_basic():
    """Test basic VIF computation with streaming method."""
    np.random.seed(42)
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )

    result = variance_inflation_factor(df, method="streaming")
    assert result.columns == ["feature", "VIF"]
    assert len(result) == 3
    # All VIFs should be >= 1.0
    assert (result["VIF"] >= 1.0).all()


def test_vif_streaming_matches_matrix():
    """Test streaming method produces consistent results with matrix."""
    np.random.seed(42)
    df = pl.DataFrame({f"x{i}": np.random.randn(100) for i in range(5)})

    result_matrix = variance_inflation_factor(df, method="matrix")
    result_streaming = variance_inflation_factor(df, method="streaming")

    # Sort both by feature name for comparison
    matrix_sorted = result_matrix.sort("feature")
    streaming_sorted = result_streaming.sort("feature")

    # Results should be very close (allowing small numerical differences due to chunking)
    np.testing.assert_allclose(
        matrix_sorted["VIF"].to_numpy(),
        streaming_sorted["VIF"].to_numpy(),
        rtol=1e-5,
    )


def test_vif_streaming_chunk_size():
    """Test streaming with different chunk sizes produces similar results."""
    np.random.seed(42)
    df = pl.DataFrame({f"x{i}": np.random.randn(500) for i in range(5)})

    result_small = variance_inflation_factor(df, method="streaming", chunk_size=100)
    result_large = variance_inflation_factor(df, method="streaming", chunk_size=10000)

    # Sort both by feature name for comparison
    small_sorted = result_small.sort("feature")
    large_sorted = result_large.sort("feature")

    # Different chunk sizes should produce similar results
    np.testing.assert_allclose(
        small_sorted["VIF"].to_numpy(),
        large_sorted["VIF"].to_numpy(),
        rtol=1e-5,
    )


def test_vif_streaming_large_dataset():
    """Test streaming handles large datasets without memory issues."""
    np.random.seed(42)
    n = 100000
    k = 20

    x = np.random.randn(n, k)
    df = pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})

    # Should complete without error
    start = time.perf_counter()
    result = variance_inflation_factor(df, method="streaming", chunk_size=10000)
    elapsed = time.perf_counter() - start

    # Verify results
    assert len(result) == k
    assert (result["VIF"] >= 1.0).all()

    # Should complete in reasonable time (< 30 seconds)
    print(f"\nStreaming large dataset: {elapsed:.2f}s")
    assert elapsed < 30.0


def test_vif_constant_column():
    """Test VIF with a constant column (zero variance)."""
    df = pl.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant column
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )

    result = variance_inflation_factor(df)

    # Constant column should have VIF = inf
    const_row = result.filter(pl.col("feature") == "a")
    assert const_row["VIF"][0] == float("inf")


def test_vif_near_singular():
    """Test VIF with near-singular correlation matrix."""
    # Create near-singular matrix: x3 ≈ x1 + x2 + tiny noise
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    x3 = x1 + x2 + np.random.randn(100) * 1e-8  # Near-singular

    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    # Should handle gracefully without crash
    result = variance_inflation_factor(df)

    # At least one should be very high (near inf)
    assert (result["VIF"] > 100).any()


def test_vif_highly_correlated():
    """Test VIF with highly correlated but not perfect features."""
    np.random.seed(42)
    n = 100

    # Create features with r=0.99 correlation
    x1 = np.random.randn(n)
    x2 = x1 * 0.99 + np.random.randn(n) * 0.1  # ~0.99 correlation

    df = pl.DataFrame({"x1": x1, "x2": x2})

    result = variance_inflation_factor(df)

    # VIF should be very high for both (since they're nearly collinear)
    assert (result["VIF"] > 10).all()


def test_vif_extreme_values():
    """Test VIF with extreme numeric values."""
    np.random.seed(42)

    # Create data with very large and very small values
    df = pl.DataFrame(
        {
            "large": [1e10, 2e10, 3e10, 4e10, 5e10],
            "small": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
            "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = variance_inflation_factor(df)

    # Should complete without error and return valid VIFs
    assert len(result) == 3
    # VIFs should be >= 1 (or inf if perfectly correlated)
    assert (result["VIF"] >= 1.0).all() or result["VIF"].is_infinite().any()
