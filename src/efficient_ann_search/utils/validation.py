from __future__ import annotations

import numpy as np


def validate_vector(vector: np.ndarray, *, expected_dim: int, dtype: np.dtype) -> np.ndarray:
    v = np.asarray(vector)
    if v.ndim != 1:
        raise ValueError("vector must be 1D")
    if v.shape[0] != expected_dim:
        raise ValueError(f"vector dimension mismatch: expected {expected_dim}, got {v.shape[0]}")
    if not np.issubdtype(v.dtype, np.number):
        raise TypeError("vector dtype must be numeric")
    return v.astype(dtype, copy=False)


def validate_query(query: np.ndarray, *, expected_dim: int, dtype: np.dtype) -> np.ndarray:
    # Same validation rules as vectors.
    return validate_vector(query, expected_dim=expected_dim, dtype=dtype)
