"""Variance Inflation Factor (VIF) module for Polars."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import polars as pl


def _validate_input(df: pl.DataFrame, drop_na: bool) -> pl.DataFrame:
    """Validate and preprocess input DataFrame."""
    # Check for numeric columns
    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    if non_numeric:
        raise ValueError(f"non-numeric columns found: {non_numeric}")

    if len(numeric_cols) < 2:
        raise ValueError("VIF requires at least 2 features")

    # Handle missing values
    if drop_na:
        df = df.select(numeric_cols).drop_nulls()
    else:
        if df.select(numeric_cols).null_count().sum_horizontal().item() > 0:
            raise ValueError(
                "DataFrame contains missing values. Use drop_na=True or fill values."
            )

    if len(df) < 2:
        raise ValueError("Need at least 2 observations to compute VIF")

    return df.select(numeric_cols)


def _compute_vif_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """Compute VIF using matrix algebra (fastest method)."""
    # Convert to numpy array
    x = df.to_numpy()
    n = x.shape[0]

    # Standardize: remove mean, divide by std
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=1)
    stds[stds == 0] = 1  # Avoid division by zero

    x_std = (x - means) / stds

    # Compute correlation matrix: R = X'X / (n-1)
    r = (x_std.T @ x_std) / (n - 1)

    # Check for perfect multicollinearity using eigenvalue decomposition
    # If any eigenvalue is near zero, the matrix is singular
    eigvals = np.linalg.eigvals(r)
    is_singular = np.any(np.abs(eigvals) < 1e-10)

    if is_singular:
        # Perfect multicollinearity - VIF is infinity for involved variables
        # For simplicity, set all to infinity when singular
        vif = np.full(len(df.columns), np.inf)
    else:
        # Compute VIF from inverse correlation matrix
        # For standardized variables: VIF_i = (R_inv)_{ii}
        r_inv = np.linalg.inv(r)
        vif = np.diag(r_inv)

        # Handle any numerical issues - VIF should be >= 1
        vif = np.where(vif < 1, 1.0, vif)

    return pl.DataFrame(
        {
            "feature": df.columns,
            "VIF": vif,
        }
    )


def _compute_vif_for_column(x: np.ndarray, col_idx: int) -> float:
    """Compute VIF for a single column using OLS regression."""
    n = x.shape[0]

    # Get target column and other columns
    y = x[:, col_idx]
    x_other = np.delete(x, col_idx, axis=1)

    # Add intercept
    x_with_intercept = np.column_stack([np.ones(n), x_other])

    # OLS: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.inf

    # Predictions and residuals
    y_pred = x_with_intercept @ beta
    residuals = y - y_pred

    # R-squared
    ss_res = np.sum(residuals**2)
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
    x = df.to_numpy()
    n_cols = x.shape[1]

    if n_jobs == -1:
        n_workers = os.cpu_count() or 4
    else:
        n_workers = max(1, n_jobs)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        vif_values = list(
            executor.map(lambda i: _compute_vif_for_column(x, i), range(n_cols))
        )

    return pl.DataFrame(
        {
            "feature": df.columns,
            "VIF": vif_values,
        }
    )


def _compute_vif_streaming(df: pl.DataFrame, chunk_size: int) -> pl.DataFrame:
    """Compute VIF using streaming/chunked processing."""
    # For streaming, we accumulate X'X statistics
    # This is less accurate but works for large datasets

    x = df.to_numpy()
    n_cols = x.shape[1]

    # Standardize the full dataset first (for accuracy)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=1)
    stds[stds == 0] = 1

    x_std = (x - means) / stds

    # For streaming, we'll process in chunks and compute correlations
    n_rows = x_std.shape[0]
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    # Accumulate correlation matrix
    r_accum = np.zeros((n_cols, n_cols))

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_rows)
        chunk = x_std[start:end]

        # Add correlation contribution
        r_accum += chunk.T @ chunk

    # Final correlation matrix (use n-1 for unbiased estimator, same as matrix method)
    r = r_accum / (n_rows - 1)

    # Compute VIF from correlation matrix (same as matrix method)
    try:
        r_inv = np.linalg.inv(r)
    except np.linalg.LinAlgError:
        r_inv = np.linalg.pinv(r)

    # For standardized variables: VIF_i = (R_inv)_{ii}
    vif = np.diag(r_inv)

    # Handle any numerical issues - VIF should be >= 1
    vif = np.where(vif < 1, 1.0, vif)

    return pl.DataFrame(
        {
            "feature": df.columns,
            "VIF": vif,
        }
    )


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
        result = _compute_vif_parallel(df, n_jobs)
    elif method == "streaming":
        result = _compute_vif_streaming(df, chunk_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    if threshold is not None:
        result = result.filter(pl.col("VIF") >= threshold)

    return result
