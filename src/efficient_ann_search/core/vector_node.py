from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorNode:
    """A single vector record.

    Notes:
        The tree indexes in this project store vectors in a contiguous NumPy matrix
        for efficiency, but `VectorNode` is kept as a public, user-facing record type.
    """

    vector_id: str
    vector: np.ndarray
    payload: Mapping[str, Any] | None = None
