from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import time

import numpy as np

from ..core.distance import pairwise_euclidean_distance_sq
from ..core.vector_store import VectorStore
from ..utils.validation import validate_query
from ..indexes.base import Neighbor


@dataclass(frozen=True, slots=True)
class LatencyStats:
    mean_ms: float
    p95_ms: float


class SearchEngine:
    """Implements brute-force search and evaluation utilities."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @property
    def store(self) -> VectorStore:
        return self._store

    def brute_force_knn(self, query: np.ndarray, k: int) -> List[Neighbor]:
        if k <= 0:
            raise ValueError("k must be > 0")
        q = validate_query(query, expected_dim=self.store.dimension, dtype=self.store.vectors_matrix().dtype)

        rows = self.store.active_rows()
        if rows.size == 0:
            return []
        X = self.store.vectors_matrix()[rows]
        d2 = pairwise_euclidean_distance_sq(X, q)
        k_eff = min(k, rows.size)
        topk_idx = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
        sorted_idx = topk_idx[np.argsort(d2[topk_idx])]
        result: List[Neighbor] = []
        for i in sorted_idx:
            row = int(rows[i])
            result.append(Neighbor(vector_id=self.store.get_id(row), distance=float(np.sqrt(d2[i]))))
        return result

    @staticmethod
    def recall_at_k(approx: Sequence[Neighbor], exact: Sequence[Neighbor]) -> float:
        if not exact:
            return 1.0
        approx_ids = {n.vector_id for n in approx}
        exact_ids = {n.vector_id for n in exact}
        return len(approx_ids & exact_ids) / max(1, len(exact_ids))

    @staticmethod
    def latency_stats(latencies_s: Sequence[float]) -> LatencyStats:
        if not latencies_s:
            return LatencyStats(mean_ms=0.0, p95_ms=0.0)
        arr = np.array(latencies_s, dtype=np.float64)
        return LatencyStats(
            mean_ms=float(arr.mean() * 1000.0),
            p95_ms=float(np.percentile(arr, 95) * 1000.0),
        )

    def time_queries(self, fn, queries: np.ndarray) -> List[float]:
        latencies: List[float] = []
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            latencies.append(time.perf_counter() - t0)
        return latencies
