from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..core.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class Neighbor:
    vector_id: str
    distance: float


class BaseTreeIndex(ABC):
    """Abstract base class for ANN indexes."""

    def __init__(
        self,
        store: VectorStore,
        *,
        leaf_size: int = 32,
        auto_rebuild: bool = True,
    ) -> None:
        if leaf_size <= 0:
            raise ValueError("leaf_size must be > 0")
        self._store = store
        self._leaf_size = int(leaf_size)
        self._auto_rebuild = bool(auto_rebuild)
        self._dirty = True

    @property
    def leaf_size(self) -> int:
        return self._leaf_size

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def ensure_built(self) -> None:
        if self._dirty and self._auto_rebuild:
            self.build()

    @abstractmethod
    def build(self) -> None:
        """(Re)build the index from the store's active vectors."""

    def insert(self, vector_id: str) -> None:
        """Notify the index that a vector was inserted into the shared store."""
        self.mark_dirty()

    def delete(self, vector_id: str) -> None:
        """Notify the index that a vector was deleted from the shared store."""
        self.mark_dirty()

    @abstractmethod
    def knn_search(
        self,
        query: np.ndarray,
        k: int,
        *,
        max_visits: int | None = 2048,
    ) -> List[Neighbor]:
        """Return k nearest neighbors.

        Args:
            query: 1D query vector.
            k: number of neighbors.
            max_visits: maximum number of tree nodes/leaf expansions. If None, search is exact.
        """
