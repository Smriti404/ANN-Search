from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Optional, Tuple

import numpy as np

from ..core.distance import pairwise_euclidean_distance_sq
from ..utils.validation import validate_query
from .base import BaseTreeIndex, Neighbor


@dataclass(slots=True)
class _KDNode:
    axis: int | None = None
    threshold: float | None = None
    left: Optional["_KDNode"] = None
    right: Optional["_KDNode"] = None
    indices: Optional[np.ndarray] = None  # leaf: 1D array of row indices

    def is_leaf(self) -> bool:
        return self.indices is not None


class KDTreeIndex(BaseTreeIndex):
    """KD-Tree index with approximate search budget.

    This implementation is optimized for clarity and correctness:
    - The tree is rebuilt lazily after dynamic updates.
    - Approximation is controlled by `max_visits` during querying.
    """

    def __init__(
        self,
        store,
        *,
        leaf_size: int = 32,
        auto_rebuild: bool = True,
    ) -> None:
        super().__init__(store, leaf_size=leaf_size, auto_rebuild=auto_rebuild)
        self._root: _KDNode | None = None

    def build(self) -> None:
        rows = self.store.active_rows()
        self._root = self._build_recursive(rows)
        self._dirty = False

    def _build_recursive(self, rows: np.ndarray) -> _KDNode:
        if rows.size == 0:
            return _KDNode(indices=np.empty((0,), dtype=np.int64))
        if rows.size <= self.leaf_size:
            return _KDNode(indices=rows.astype(np.int64, copy=False))

        X = self.store.vectors_matrix()[rows]
        variances = X.var(axis=0)
        axis = int(np.argmax(variances))
        values = X[:, axis]
        median = float(np.median(values))

        left_rows = rows[values <= median]
        right_rows = rows[values > median]

        # Guard against pathological splits (all points equal on axis).
        if left_rows.size == 0 or right_rows.size == 0:
            order = np.argsort(values)
            mid = rows.size // 2
            left_rows = rows[order[:mid]]
            right_rows = rows[order[mid:]]
            median = float(values[order[mid - 1]]) if mid > 0 else float(values[order[0]])

        node = _KDNode(axis=axis, threshold=median)
        node.left = self._build_recursive(left_rows)
        node.right = self._build_recursive(right_rows)
        return node

    def knn_search(self, query: np.ndarray, k: int, *, max_visits: int | None = 2048) -> List[Neighbor]:
        self.ensure_built()
        if self._root is None:
            return []
        if k <= 0:
            raise ValueError("k must be > 0")
        q = validate_query(query, expected_dim=self.store.dimension, dtype=self.store.vectors_matrix().dtype)

        best: List[Tuple[float, int]] = []  # max-heap via negative dist

        def push_candidate(dist_sq: float, row: int) -> None:
            if not self.store.is_active(row):
                return
            if len(best) < k:
                heapq.heappush(best, (-dist_sq, row))
            else:
                if dist_sq < -best[0][0]:
                    heapq.heapreplace(best, (-dist_sq, row))

        # Best-first traversal with backtracking; priority = distance to splitting plane.
        heap: List[Tuple[float, _KDNode]] = []
        heapq.heappush(heap, (0.0, self._root))
        visits = 0

        while heap:
            if max_visits is not None and visits >= max_visits:
                break
            _, node = heapq.heappop(heap)
            visits += 1

            if node.is_leaf():
                idx = node.indices
                if idx is None or idx.size == 0:
                    continue
                X = self.store.vectors_matrix()[idx]
                d2 = pairwise_euclidean_distance_sq(X, q)
                for i, row in enumerate(idx):
                    push_candidate(float(d2[i]), int(row))
                continue

            axis = int(node.axis)
            thr = float(node.threshold)
            val = float(q[axis])
            near, far = (node.left, node.right) if val <= thr else (node.right, node.left)
            if near is not None:
                heapq.heappush(heap, (0.0, near))
            if far is not None:
                plane_dist = (val - thr) ** 2
                # If we're exact (max_visits=None) we always explore. Otherwise use the budget.
                heapq.heappush(heap, (plane_dist, far))

        # Convert to sorted ascending by distance.
        best_sorted = sorted(((-d2, row) for d2, row in best), key=lambda t: t[0])
        return [Neighbor(vector_id=self.store.get_id(row), distance=float(np.sqrt(d2))) for d2, row in best_sorted]
