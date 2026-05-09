from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from .core.vector_store import VectorStore
from .engine.search_engine import SearchEngine
from .indexes.base import BaseTreeIndex, Neighbor
from .indexes.kd_tree import KDTreeIndex
from .indexes.rp_tree import RPTreeIndex
from .utils.validation import validate_vector


class Treendex:
    """Singleton manager for vector storage and tree indexes.

    Responsibilities:
        - Own a shared `VectorStore`
        - Create/manage multiple indexes over the same store
        - Provide a unified API for insert/delete/search
    """

    _instance: "Treendex | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dimension: int = 0) -> None:
        # Initialize-once fields.
        if not hasattr(self, "_initialized"):
            self._initialized = False
            self._store = None
            self._engine = None
            self._indexes = {}
            self._dimension = 0

        if not self._initialized:
            self._initialized = True

        dim = int(dimension)
        if self._store is None:
            if dim <= 0:
                return
            self._dimension = dim
            self._init_store(self._dimension)
            return

        # If a store already exists, only allow the same dimension.
        if dim > 0 and dim != self.store.dimension:
            raise ValueError(
                f"Treendex already initialized with dimension={self.store.dimension}; cannot reinitialize with dimension={dim}"
            )

    @classmethod
    def instance(cls) -> "Treendex":
        return cls()

    def _init_store(self, dimension: int) -> None:
        self._store = VectorStore(dimension=dimension)
        self._engine = SearchEngine(self._store)

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            raise RuntimeError("Treendex store is not initialized; pass dimension=... on first construction")
        return self._store

    @property
    def engine(self) -> SearchEngine:
        if self._engine is None:
            raise RuntimeError("Treendex engine is not initialized")
        return self._engine

    def create_index(self, name: str, kind: str, **kwargs) -> BaseTreeIndex:
        if not name:
            raise ValueError("index name must be non-empty")
        if name in self._indexes:
            raise ValueError(f"index already exists: {name}")
        kind = kind.lower().strip()
        if kind == "kd":
            idx = KDTreeIndex(self.store, **kwargs)
        elif kind in {"rp", "rptree"}:
            idx = RPTreeIndex(self.store, **kwargs)
        else:
            raise ValueError("kind must be 'kd' or 'rp'")
        self._indexes[name] = idx
        return idx

    def get_index(self, name: str) -> BaseTreeIndex:
        try:
            return self._indexes[name]
        except KeyError as e:
            raise KeyError(f"unknown index: {name}") from e

    def list_indexes(self) -> List[str]:
        return sorted(self._indexes.keys())

    def build_index(self, name: str) -> None:
        self.get_index(name).build()

    def insert_vector(self, vector_id: str, vector: np.ndarray, payload: Mapping | None = None) -> None:
        v = validate_vector(vector, expected_dim=self.store.dimension, dtype=self.store.vectors_matrix().dtype)
        self.store.add(vector_id, v, payload=payload)
        for idx in self._indexes.values():
            idx.insert(vector_id)

    def delete_vector(self, vector_id: str) -> None:
        self.store.deactivate(vector_id)
        for idx in self._indexes.values():
            idx.delete(vector_id)

    def knn_search(
        self,
        index_name: str,
        query: np.ndarray,
        k: int,
        *,
        max_visits: int | None = 2048,
    ) -> List[Neighbor]:
        idx = self.get_index(index_name)
        return idx.knn_search(query, k, max_visits=max_visits)

    def brute_force_search(self, query: np.ndarray, k: int) -> List[Neighbor]:
        return self.engine.brute_force_knn(query, k)

    def reset(self) -> None:
        """Reset singleton state (useful for tests)."""
        self._store = None
        self._engine = None
        self._indexes = {}
        self._dimension = 0
        self._initialized = False

