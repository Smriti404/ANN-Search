from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import time

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficient_ann_search import Treendex
from efficient_ann_search.engine.search_engine import LatencyStats


@dataclass(frozen=True, slots=True)
class IndexResult:
    name: str
    build_s: float
    mean_ms: float
    p95_ms: float
    recall: float


def benchmark_once(
    *,
    n: int,
    d: int,
    q: int,
    k: int,
    max_visits: int,
    seed: int = 0,
) -> Tuple[List[IndexResult], LatencyStats]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    queries = rng.normal(size=(q, d)).astype(np.float32)
    ids = [f"v{i}" for i in range(n)]

    # Treendex is a singleton, so ensure each benchmark run starts clean.
    treendex = Treendex.instance()
    treendex.reset()
    treendex = Treendex(dimension=d)
    treendex.create_index("kd", "kd", leaf_size=32)
    treendex.create_index("rp", "rp", leaf_size=32, random_state=seed)

    for vector_id, v in zip(ids, X):
        treendex.insert_vector(vector_id, v)

    engine = treendex.engine

    # brute force baseline latency
    bf_lat = engine.time_queries(lambda qq: engine.brute_force_knn(qq, k), queries)
    bf_stats = engine.latency_stats(bf_lat)

    results: List[IndexResult] = []
    for idx_name in ["kd", "rp"]:
        t0 = time.perf_counter()
        treendex.build_index(idx_name)
        build_s = time.perf_counter() - t0

        def run_query(qq: np.ndarray):
            return treendex.knn_search(idx_name, qq, k, max_visits=max_visits)

        lat = engine.time_queries(run_query, queries)
        stats = engine.latency_stats(lat)

        recalls: List[float] = []
        for qq in queries:
            approx = treendex.knn_search(idx_name, qq, k, max_visits=max_visits)
            exact = engine.brute_force_knn(qq, k)
            recalls.append(engine.recall_at_k(approx, exact))
        recall = float(np.mean(recalls))
        results.append(IndexResult(idx_name, build_s, stats.mean_ms, stats.p95_ms, recall))

    return results, bf_stats


def main() -> None:
    n, d, q, k = 20000, 64, 200, 10
    max_visits = 2048
    results, brute = benchmark_once(n=n, d=d, q=q, k=k, max_visits=max_visits)

    print(f"Dataset: n={n}, d={d}, queries={q}, k={k}, max_visits={max_visits}")
    for r in results:
        print(
            f"{r.name.upper():9s}: build={r.build_s:.3f}s, mean={r.mean_ms:.3f}ms, p95={r.p95_ms:.3f}ms, recall@{k}={r.recall:.3f}"
        )
    print(f"BruteForce: mean={brute.mean_ms:.3f}ms, p95={brute.p95_ms:.3f}ms")


if __name__ == "__main__":
    main()

