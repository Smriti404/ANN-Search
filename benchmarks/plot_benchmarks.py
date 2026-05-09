from __future__ import annotations

"""Optional plotting helper.

This script is intentionally simple; it runs a few configurations and plots
recall vs latency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.run_benchmarks import benchmark_once


@dataclass(frozen=True, slots=True)
class Point:
    label: str
    mean_ms: float
    recall: float


def main() -> None:
    n, d, q, k = 30000, 64, 200, 10
    budgets = [128, 256, 512, 1024, 2048]
    kd_points: List[Point] = []
    rp_points: List[Point] = []

    for b in budgets:
        results, _ = benchmark_once(n=n, d=d, q=q, k=k, max_visits=b)
        for r in results:
            p = Point(label=f"{r.name}:{b}", mean_ms=r.mean_ms, recall=r.recall)
            if r.name == "kd":
                kd_points.append(p)
            else:
                rp_points.append(p)

    plt.figure(figsize=(7, 5))
    plt.plot([p.mean_ms for p in kd_points], [p.recall for p in kd_points], marker="o", label="KD-Tree")
    plt.plot([p.mean_ms for p in rp_points], [p.recall for p in rp_points], marker="o", label="RP-Tree")
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Recall@k")
    plt.title("ANN tradeoff: latency vs recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

