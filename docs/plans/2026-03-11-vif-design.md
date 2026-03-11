# Polars-Native Variance Inflation Factor (VIF) Module

**Date:** 2026-03-11
**Status:** Approved

## Overview

Create a high-performance Polars-native VIF module that provides the same functionality as statsmodels' `variance_inflation_factor` but with significantly improved speed and native Polars DataFrame support. This is the first module in a planned stats package for a crypto analysis pipeline.

## Goals

1. Match statsmodels VIF output format and accuracy
2. Leverage Polars for maximum speed and efficiency
3. Provide flexible computation modes for different use cases
4. Integrate seamlessly into the broader mad_money stats package

## Function Signature

```python
def variance_inflation_factor(
    df: pl.DataFrame,
    method: Literal["matrix", "parallel", "streaming"] = "matrix",
    threshold: float | None = None,
    drop_na: bool = True,
    n_jobs: int = -1,
    chunk_size: int = 10000,
) -> pl.DataFrame
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| df | pl.DataFrame | required | Input DataFrame with numeric columns only |
| method | str | "matrix" | Computation method: "matrix", "parallel", or "streaming" |
| threshold | float \| None | None | Optional filter: only return features with VIF >= threshold |
| drop_na | bool | True | Whether to drop rows with missing values before computation |
| n_jobs | int | -1 | Number of parallel workers (-1 = all cores). Only used in "parallel" mode |
| chunk_size | int | 10000 | Chunk size for streaming mode. Only used in "streaming" mode |

## Return Value

Returns a Polars DataFrame with two columns:

| Column | Type | Description |
|--------|------|-------------|
| feature | str | Column name from input DataFrame |
| VIF | f64 | Variance Inflation Factor value |

**Note:** Features with VIF = infinity (constant columns) are included with value `inf`.

## Computation Methods

### 1. Matrix Method (Default)

Uses matrix algebra to compute all VIFs in a single operation:

1. Standardize the data (remove mean, divide by std)
2. Compute correlation matrix: R = X'X / (n-1)
3. Compute R² for each column from the inverse of R
4. Calculate VIF = 1 / (1 - R²)

**Complexity:** O(n³) for matrix operations, but single pass through data.

### 2. Parallel Method

Uses `concurrent.futures.ThreadPoolExecutor` to compute VIFs in parallel:

1. Spawns worker threads (default: all CPU cores)
2. Each worker computes VIF for a subset of features independently
3. Results are aggregated into final DataFrame

**Best for:** Many features (50+), multi-core systems.

### 3. Streaming Method

Processes data in chunks for memory efficiency:

1. Loads data in chunks of `chunk_size` rows
2. Accumulates sufficient statistics (X'X sums)
3. Computes VIF from aggregated statistics

**Best for:** Datasets larger than available RAM.

## Error Handling

| Error Condition | Behavior |
|----------------|----------|
| Non-numeric columns | Raise `ValueError` listing non-numeric column names |
| All constant columns | Raise `ValueError` (cannot compute VIF) |
| Insufficient data (< 2 rows) | Raise `ValueError` (need at least 2 observations) |
| Single column | Raise `ValueError` (VIF requires 2+ features) |
| Missing values | Drop rows if `drop_na=True`, else raise error |

## Performance Targets

- **Matrix method:** 10-50x faster than statsmodels for 100-1000 features
- **Memory:** < 3x input DataFrame size for matrix/parallel modes
- **Accuracy:** Matches statsmodels VIF to 1e-10 relative error

## Testing Strategy

1. **Unit tests:** Compare output against statsmodels for random matrices
2. **Edge cases:** Constant columns, NaN handling, single column
3. **Performance benchmarks:** Compare runtime against statsmodels implementation
4. **Integration tests:** Ensure compatibility with Polars DataFrame operations

## File Structure

```
mad_money/
├── docs/
│   └── plans/
│       └── 2026-03-11-vif-design.md
├── mad_money/
│   ├── __init__.py
│   └── stats/
│       ├── __init__.py
│       └── vif.py
└── tests/
    └── test_vif.py
```

## Future Considerations

This module may be extended with:
- Sequential VIF (iteratively remove highest VIF feature)
- Column selection utilities
- Integration with other stats modules (correlation, regression)
