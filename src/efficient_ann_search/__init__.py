"""Efficient ANN Search.

Public API exports:
- Treendex (singleton manager)
- KDTreeIndex, RPTreeIndex
- VectorNode
- SearchEngine
"""

from .treendex import Treendex
from .core.vector_node import VectorNode
from .indexes.kd_tree import KDTreeIndex
from .indexes.rp_tree import RPTreeIndex
from .engine.search_engine import SearchEngine

__all__ = [
    "Treendex",
    "VectorNode",
    "KDTreeIndex",
    "RPTreeIndex",
    "SearchEngine",
]
