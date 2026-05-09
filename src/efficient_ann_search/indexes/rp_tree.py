from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Optional, Tuple

import numpy as np

from ..core.distance import pairwise_euclidean_distance_sq
from ..utils.validation import validate_query
from .base import BaseTreeIndex, Neighbor


@dataclass(slots=True)
class _RPNode:
    direction: Optional[np.ndarray] = None  # 1D
    threshold: float | None = None
    left: Optional["_RPNode"] = None
    right: Optional["_RPNode"] = None
    indices: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        return self.indices is not None


class RPTreeIndex(BaseTreeIndex):
    """Random Projection Tree (RP-Tree) index.

    Splits are performed by projecting points onto a random direction vector and
    splitting at the median projection.
    """

    def __init__(
        self,
        store,
        *,
        leaf_size: int = 32,
        auto_rebuild: bool = True,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(store, leaf_size=leaf_size, auto_rebuild=auto_rebuild)
        self._root: _RPNode | None = None
        self._rng = np.random.default_rng(random_state)

    def build(self) -> None:
        rows = self.store.active_rows()
        self._root = self._build_recursive(rows)
        self._dirty = False

    def _build_recursive(self, rows: np.ndarray) -> _RPNode:
        if rows.size == 0:
            return _RPNode(indices=np.empty((0,), dtype=np.int64))
        if rows.size <= self.leaf_size:
            return _RPNode(indices=rows.astype(np.int64, copy=False))

        d = self.store.dimension
        direction = self._rng.normal(size=(d,)).astype(self.store.vectors_matrix().dtype, copy=False)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            direction[0] = 1.0
            norm = 1.0
        direction = direction / norm

        X = self.store.vectors_matrix()[rows]
        proj = X @ direction
        thr = float(np.median(proj))
        left_rows = rows[proj <= thr]
        right_rows = rows[proj > thr]
        if left_rows.size == 0 or right_rows.size == 0:
            order = np.argsort(proj)
            mid = rows.size // 2
            left_rows = rows[order[:mid]]
            right_rows = rows[order[mid:]]
            thr = float(proj[order[mid - 1]]) if mid > 0 else float(proj[order[0]])

        node = _RPNode(direction=direction, threshold=thr)
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

        heap: List[Tuple[float, _RPNode]] = []
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

            direction = node.direction
            thr = float(node.threshold)
            assert direction is not None

            val = float(np.dot(q, direction))
            near, far = (node.left, node.right) if val <= thr else (node.right, node.left)
            if near is not None:
                heapq.heappush(heap, (0.0, near))
            if far is not None:
                # distance to splitting hyperplane along projection axis
                plane_dist = (val - thr) ** 2
                heapq.heappush(heap, (plane_dist, far))

        best_sorted = sorted(((-d2, row) for d2, row in best), key=lambda t: t[0])
        return [Neighbor(vector_id=self.store.get_id(row), distance=float(np.sqrt(d2))) for d2, row in best_sorted]
