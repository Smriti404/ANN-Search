from __future__ import annotations

import numpy as np


def euclidean_distance_sq(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def pairwise_euclidean_distance_sq(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances from each row in X to q."""
    # ||x-q||^2 = ||x||^2 + ||q||^2 - 2 x·q
    X = np.asarray(X)
    q = np.asarray(q)
    q_norm = float(np.dot(q, q))
    X_norm = np.einsum("ij,ij->i", X, X)
    cross = 2.0 * (X @ q)
    return (X_norm + q_norm - cross).astype(np.float64, copy=False)
