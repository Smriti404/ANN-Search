from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from ..utils.validation import validate_vector


@dataclass(slots=True)
class VectorStoreStats:
    total_vectors: int
    active_vectors: int
    dimension: int


class VectorStore:
    """A shared storage for vectors used by all indexes.

    Storage model:
        - vectors are appended to a NumPy array
        - deletion uses an active mask (tombstones)
        - ids are unique and mapped to row indices

    This keeps inserts O(1) amortized and allows efficient bulk distance computation.
    """

    def __init__(self, dimension: int, dtype: np.dtype = np.float32) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = int(dimension)
        self._dtype = np.dtype(dtype)

        self._capacity = 0
        self._size = 0

        self._vectors = np.empty((0, self._dimension), dtype=self._dtype)
        self._ids: List[str | None] = []
        self._active = np.empty((0,), dtype=bool)
        self._payloads: List[Mapping[str, Any] | None] = []

        self._id_to_row: Dict[str, int] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def stats(self) -> VectorStoreStats:
        return VectorStoreStats(
            total_vectors=self._size,
            active_vectors=int(self._active[: self._size].sum()),
            dimension=self._dimension,
        )

    def __len__(self) -> int:
        return self._size

    def _ensure_capacity(self, min_capacity: int) -> None:
        if self._capacity >= min_capacity:
            return
        new_capacity = max(64, self._capacity)
        while new_capacity < min_capacity:
            new_capacity *= 2

        new_vectors = np.empty((new_capacity, self._dimension), dtype=self._dtype)
        new_active = np.zeros((new_capacity,), dtype=bool)

        if self._size > 0:
            new_vectors[: self._size] = self._vectors[: self._size]
            new_active[: self._size] = self._active[: self._size]

        # Extend id/payload arrays.
        if len(self._ids) < new_capacity:
            self._ids.extend([None] * (new_capacity - len(self._ids)))
        if len(self._payloads) < new_capacity:
            self._payloads.extend([None] * (new_capacity - len(self._payloads)))

        self._vectors = new_vectors
        self._active = new_active
        self._capacity = new_capacity

    def has_id(self, vector_id: str) -> bool:
        return vector_id in self._id_to_row

    def add(self, vector_id: str, vector: np.ndarray, payload: Mapping[str, Any] | None = None) -> int:
        if not vector_id:
            raise ValueError("vector_id must be a non-empty string")
        if self.has_id(vector_id):
            raise ValueError(f"vector_id already exists: {vector_id}")
        v = validate_vector(vector, expected_dim=self._dimension, dtype=self._dtype)

        row = self._size
        self._ensure_capacity(row + 1)

        self._id_to_row[vector_id] = row
        self._ids[row] = vector_id
        self._payloads[row] = payload

        self._vectors[row] = v
        self._active[row] = True
        self._size += 1
        return row

    def deactivate(self, vector_id: str) -> None:
        row = self._id_to_row.get(vector_id)
        if row is None:
            raise KeyError(f"unknown vector_id: {vector_id}")
        self._active[row] = False

    def reactivate(self, vector_id: str) -> None:
        row = self._id_to_row.get(vector_id)
        if row is None:
            raise KeyError(f"unknown vector_id: {vector_id}")
        self._active[row] = True

    def is_active(self, row: int) -> bool:
        return bool(self._active[row])

    def get_row(self, vector_id: str) -> int:
        row = self._id_to_row.get(vector_id)
        if row is None:
            raise KeyError(f"unknown vector_id: {vector_id}")
        return row

    def get_id(self, row: int) -> str:
        if row < 0 or row >= self._size:
            raise IndexError("row out of range")
        vid = self._ids[row]
        assert vid is not None
        return vid

    def get_vector_by_row(self, row: int) -> np.ndarray:
        if row < 0 or row >= self._size:
            raise IndexError("row out of range")
        return self._vectors[row]

    def get_payload_by_row(self, row: int) -> Mapping[str, Any] | None:
        if row < 0 or row >= self._size:
            raise IndexError("row out of range")
        return self._payloads[row]

    def active_rows(self) -> np.ndarray:
        """Return a 1D array of active row indices."""
        return np.flatnonzero(self._active[: self._size])

    def active_vectors(self) -> np.ndarray:
        """Return a compact matrix of only active vectors."""
        rows = self.active_rows()
        return self._vectors[rows]

    def vectors_matrix(self) -> np.ndarray:
        """Return the full (including inactive) vectors matrix."""
        return self._vectors[: self._size]

    def active_mask(self) -> np.ndarray:
        return self._active[: self._size]

    def iter_active(self) -> Iterable[Tuple[str, np.ndarray]]:
        for row in self.active_rows():
            yield self.get_id(int(row)), self._vectors[int(row)]

