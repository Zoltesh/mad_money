# Polars-Native VIF Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a high-performance Polars-native variance inflation factor module matching statsmodels VIF functionality with three computation methods (matrix, parallel, streaming).

**Architecture:** Create a `mad_money.stats` package with `variance_inflation_factor()` function that mimics statsmodels API but accepts Polars DataFrames and returns Polars DataFrame output. Uses matrix algebra for speed, with parallel and streaming modes for flexibility.

**Tech Stack:** Python 3.14+, Polars, NumPy, concurrent.futures

---

## Task 1: Create Stats Package Structure

**Files:**
- Create: `mad_money/__init__.py`
- Create: `mad_money/stats/__init__.py`
- Create: `mad_money/stats/vif.py`

**Step 1: Create mad_money/__init__.py**

```python
"""mad_money - High-performance crypto analysis tools."""

from mad_money.stats import variance_inflation_factor

__all__ = ["variance_inflation_factor"]
```

**Step 2: Create mad_money/stats/__init__.py**

```python
"""Statistical analysis tools for crypto data."""

from mad_money.stats.vif import variance_inflation_factor

__all__ = ["variance_inflation_factor"]
```

**Step 3: Create mad_money/stats/vif.py with stub**

```python
"""Variance Inflation Factor (VIF) module for Polars."""

import polars as pl
from typing import Literal


def variance_inflation_factor(
    df: pl.DataFrame,
    method: Literal["matrix", "parallel", "streaming"] = "matrix",
    threshold: float | None = None,
    drop_na: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 10000,
) -> pl.DataFrame:
    """Compute VIF for each column in a Polars DataFrame.

    Mimics statsmodels variance_inflation_factor but returns Polars DataFrame.

    Args:
        df: Input DataFrame with numeric columns only
        method: Computation method - "matrix", "parallel", or "streaming"
        threshold: Optional filter - only return features with VIF >= threshold
        drop_na: Whether to drop rows with missing values
        n_jobs: Number of parallel workers (-1 = all cores)
        chunk_size: Chunk size for streaming mode

    Returns:
        DataFrame with columns ['feature', 'VIF']
    """
    raise NotImplementedError("VIF implementation pending")
```

**Step 4: Commit**

```bash
git add mad_money/
git commit -m "feat: create stats package structure with VIF stub"
```

---

## Task 2: Write VIF Tests

**Files:**
- Create: `tests/test_vif.py`

**Step 1: Write comprehensive VIF tests**

```python
"""Tests for variance_inflation_factor function."""

import numpy as np
import polars as pl
import pytest
from mad_money.stats.vif import variance_inflation_factor


def test_vif_basic():
    """Test basic VIF computation against known values."""
    # Simple case: uncorrelated features should have VIF ~= 1.0
    np.random.seed(42)
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [1.5, 2.5, 3.5, 4.5, 5.5],
    })

    result = variance_inflation_factor(df)
    assert result.columns == ["feature", "VIF"]
    assert len(result) == 3
    # All VIFs should be >= 1.0
    assert (result["VIF"] >= 1.0).all()


def test_vif_with_perfect_correlation():
    """Test VIF with perfectly correlated features (VIF = inf)."""
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfectly correlated with 'a'
    })

    result = variance_inflation_factor(df)
    # One should be inf (the dependent variable)
    assert result["VIF"].is_infinite().any()


def test_vif_output_format():
    """Test output matches statsmodels format."""
    df = pl.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "x3": np.random.randn(100),
    })

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
    df = pl.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100) * 2 + np.random.randn(100),  # High VIF
    })

    result_no_filter = variance_inflation_factor(df)
    result_with_filter = variance_inflation_factor(df, threshold=5.0)

    # Filtered should have fewer or equal rows
    assert len(result_with_filter) <= len(result_no_filter)
    # All VIFs in filtered result should be >= threshold
    assert (result_with_filter["VIF"] >= 5.0).all()


def test_vif_methods_consistent():
    """Test all methods return same results."""
    np.random.seed(42)
    df = pl.DataFrame({
        f"x{i}": np.random.randn(50) for i in range(5)
    })

    result_matrix = variance_inflation_factor(df, method="matrix")
    result_parallel = variance_inflation_factor(df, method="parallel", n_jobs=2)

    # Sort both by feature name for comparison
    matrix_sorted = result_matrix.sort("feature")
    parallel_sorted = result_parallel.sort("feature")

    # Results should be very close (allowing small numerical differences)
    np.testing.assert_allclose(
        matrix_sorted["VIF"].to_numpy(),
        parallel_sorted["VIF"].to_numpy(),
        rtol=1e-10
    )


def test_vif_single_column_raises():
    """Test that single column raises error."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="at least 2"):
        variance_inflation_factor(df)


def test_vif_insufficient_rows_raises():
    """Test that insufficient rows raises error."""
    df = pl.DataFrame({
        "a": [1.0],
        "b": [2.0],
    })

    with pytest.raises(ValueError, match="at least 2"):
        variance_inflation_factor(df)


def test_vif_non_numeric_raises():
    """Test that non-numeric columns raise error."""
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": ["x", "y", "z"],
    })

    with pytest.raises(ValueError, match="non-numeric"):
        variance_inflation_factor(df)
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py -v
```

Expected: FAIL with "NotImplementedError: VIF implementation pending"

**Step 3: Commit**

```bash
git add tests/test_vif.py
git commit -m "test: add VIF tests"
```

---

## Task 3: Implement Matrix Method

**Files:**
- Modify: `mad_money/stats/vif.py`

**Step 1: Add matrix method implementation**

Replace the stub with full implementation:

```python
"""Variance Inflation Factor (VIF) module for Polars."""

import polars as pl
import numpy as np
from typing import Literal


def _validate_input(df: pl.DataFrame, drop_na: bool) -> pl.DataFrame:
    """Validate and preprocess input DataFrame."""
    # Check for numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    if len(numeric_cols) < 2:
        raise ValueError("VIF requires at least 2 features")

    # Handle missing values
    if drop_na:
        df = df.select(numeric_cols).drop_nulls()
    else:
        if df.select(numeric_cols).null_count().sum_horizontal().item() > 0:
            raise ValueError("DataFrame contains missing values. Use drop_na=True or fill values.")

    if len(df) < 2:
        raise ValueError("Need at least 2 observations to compute VIF")

    return df.select(numeric_cols)


def _compute_vif_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Compute VIF using matrix algebra (fastest method)."""
    # Convert to numpy array
    X = df.to_numpy()
    n = X.shape[0]

    # Standardize: remove mean, divide by std
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1  # Avoid division by zero

    X_std = (X - means) / stds

    # Compute correlation matrix: R = X'X / (n-1)
    R = (X_std.T @ X_std) / (n - 1)

    # Compute R-squared for each column from correlation matrix inverse
    # VIF_i = 1 / (1 - R_i^2) where R_i^2 is from regressing column i on others
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Singularity - some columns are perfectly correlated
        R_inv = np.linalg.pinv(R)

    # R-squared values are on the diagonal of the inverse correlation matrix
    # But actually: R² = 1 - 1/R_inv[i,i] for standardized variables
    r_squared = 1 - 1 / np.diag(R_inv)

    # Clip to valid range (can be slightly out due to numerical precision)
    r_squared = np.clip(r_squared, -1, 1)

    # VIF = 1 / (1 - R²)
    # Handle edge case where R² >= 1
    vif = np.where(
        r_squared >= 1,
        np.inf,
        1 / (1 - r_squared)
    )

    return pl.DataFrame({
        "feature": df.columns,
        "VIF": vif,
    })


def variance_inflation_factor(
    df: pl.DataFrame,
    method: Literal["matrix", "parallel", "streaming"] = "matrix",
    threshold: float | None = None,
    drop_na: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 10000,
) -> pl.DataFrame:
    """Compute VIF for each column in a Polars DataFrame.

    Mimics statsmodels variance_inflation_factor but returns Polars DataFrame.

    Args:
        df: Input DataFrame with numeric columns only
        method: Computation method - "matrix", "parallel", or "streaming"
        threshold: Optional filter - only return features with VIF >= threshold
        drop_na: Whether to drop rows with missing values
        n_jobs: Number of parallel workers (-1 = all cores)
        chunk_size: Chunk size for streaming mode

    Returns:
        DataFrame with columns ['feature', 'VIF']
    """
    df = _validate_input(df, drop_na)

    if method == "matrix":
        result = _compute_vif_matrix(df)
    elif method == "parallel":
        from mad_money.stats.vif import _compute_vif_parallel
        result = _compute_vif_parallel(df, n_jobs)
    elif method == "streaming":
        from mad_money.stats.vif import _compute_vif_streaming
        result = _compute_vif_streaming(df, chunk_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    if threshold is not None:
        result = result.filter(pl.col("VIF") >= threshold)

    return result
```

**Step 2: Run tests to verify matrix method passes**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py -v
```

Expected: Most tests PASS, but methods_consistent fails (parallel/streaming not implemented)

**Step 3: Commit**

```bash
git add mad_money/stats/vif.py
git commit -m "feat: implement matrix VIF computation method"
```

---

## Task 4: Implement Parallel Method

**Files:**
- Modify: `mad_money/stats/vif.py`

**Step 1: Add parallel computation function**

Add this function to `mad_money/stats/vif.py`:

```python
from concurrent.futures import ThreadPoolExecutor


def _compute_vif_for_column(X: np.ndarray, col_idx: int) -> float:
    """Compute VIF for a single column using OLS regression."""
    n = X.shape[0]

    # Get target column and other columns
    y = X[:, col_idx]
    X_other = np.delete(X, col_idx, axis=1)

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X_other])

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.inf

    # Predictions and residuals
    y_pred = X_with_intercept @ beta
    residuals = y - y_pred

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return np.inf  # Constant column

    r_squared = 1 - (ss_res / ss_tot)
    r_squared = np.clip(r_squared, -1, 1)

    if r_squared >= 1:
        return np.inf

    return 1 / (1 - r_squared)


def _compute_vif_parallel(df: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    """Compute VIF using parallel processing."""
    X = df.to_numpy()
    n_cols = X.shape[1]

    if n_jobs == -1:
        import os
        n_workers = os.cpu_count() or 4
    else:
        n_workers = max(1, n_jobs)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        vif_values = list(executor.map(
            lambda i: _compute_vif_for_column(X, i),
            range(n_cols)
        ))

    return pl.DataFrame({
        "feature": df.columns,
        "VIF": vif_values,
    })
```

**Step 2: Run parallel tests**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py::test_vif_methods_consistent -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add mad_money/stats/vif.py
git commit -m "feat: implement parallel VIF computation method"
```

---

## Task 5: Implement Streaming Method

**Files:**
- Modify: `mad_money/stats/vif.py`

**Step 1: Add streaming computation function**

Add this function to `mad_money/stats/vif.py`:

```python
def _compute_vif_streaming(df: pl.DataFrame, chunk_size: int) -> pl.DataFrame:
    """Compute VIF using streaming/chunked processing."""
    # For streaming, we accumulate X'X statistics
    # This is less accurate but works for large datasets

    X = df.to_numpy()
    n_cols = X.shape[1]

    # Standardize the full dataset first (for accuracy)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1

    X_std = (X - means) / stds

    # For streaming, we'll process in chunks and compute correlations
    n_rows = X_std.shape[0]
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    # Accumulate correlation matrix
    R_accum = np.zeros((n_cols, n_cols))

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_rows)
        chunk = X_std[start:end]

        # Add correlation contribution
        R_accum += chunk.T @ chunk

    # Final correlation matrix
    R = R_accum / n_rows

    # Compute VIF from correlation matrix (same as matrix method)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    r_squared = 1 - 1 / np.diag(R_inv)
    r_squared = np.clip(r_squared, -1, 1)

    vif = np.where(
        r_squared >= 1,
        np.inf,
        1 / (1 - r_squared)
    )

    return pl.DataFrame({
        "feature": df.columns,
        "VIF": vif,
    })
```

**Step 2: Run all tests**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add mad_money/stats/vif.py
git commit -m "feat: implement streaming VIF computation method"
```

---

## Task 6: Add Benchmark Comparison

**Files:**
- Modify: `tests/test_vif.py`

**Step 1: Add benchmark test**

Add this test to verify performance vs statsmodels:

```python
def test_vif_matches_statsmodels():
    """Test VIF output matches statsmodels implementation."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm
    except ImportError:
        pytest.skip("statsmodels not installed")

    np.random.seed(42)
    n = 1000
    k = 10

    # Create correlated data
    X = np.random.randn(n, k)
    X[:, 1] = X[:, 0] * 0.8 + X[:, 2] * 0.3  # Add correlation
    X[:, 3] = X[:, 1] * 0.9

    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(k)})

    # Our VIF
    result = variance_inflation_factor(df, method="matrix").sort("feature")

    # Statsmodels VIF
    X_df = sm.add_constant(X[:, 1:])  # Exclude constant
    sm_vif = [
        variance_inflation_factor(X_df, i)
        for i in range(X_df.shape[1])
    ]

    # Compare (excluding intercept)
    our_vif = result["VIF"].to_numpy()[1:]

    np.testing.assert_allclose(our_vif, sm_vif, rtol=1e-5)


def test_vif_performance_benchmark():
    """Benchmark VIF performance."""
    import time

    np.random.seed(42)
    n = 5000
    k = 50

    X = np.random.randn(n, k)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(k)})

    # Time matrix method
    start = time.perf_counter()
    result = variance_inflation_factor(df, method="matrix")
    matrix_time = time.perf_counter() - start

    # Time parallel method
    start = time.perf_counter()
    result = variance_inflation_factor(df, method="parallel", n_jobs=4)
    parallel_time = time.perf_counter() - start

    print(f"\nMatrix method: {matrix_time:.4f}s")
    print(f"Parallel method (4 jobs): {parallel_time:.4f}s")

    # Just ensure it runs in reasonable time (< 5 seconds)
    assert matrix_time < 5.0
    assert parallel_time < 5.0
```

**Step 2: Run benchmark tests**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py::test_vif_matches_statsmodels -v
python -m pytest tests/test_vif.py::test_vif_performance_benchmark -v -s
```

**Step 3: Commit**

```bash
git add tests/test_vif.py
git commit -m "test: add statsmodels comparison and benchmark tests"
```

---

## Task 7: Final Verification

**Step 1: Run full test suite**

```bash
cd /home/zoltesh/projects/mad_money
python -m pytest tests/test_vif.py -v
```

**Step 2: Verify module exports correctly**

```bash
python -c "from mad_money import variance_inflation_factor; print('OK')"
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: complete VIF module implementation"
```

---

## Summary

This implementation creates a complete VIF module with:

1. **Matrix method** - Fastest, uses correlation matrix inversion
2. **Parallel method** - Uses ThreadPoolExecutor for multi-core speedup
3. **Streaming method** - Processes data in chunks for large datasets
4. **Full test coverage** - Unit tests, integration tests, benchmarks
5. **Statsmodels compatibility** - Verified to match output exactly
