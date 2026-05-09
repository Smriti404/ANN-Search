from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# Allow running from repo root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficient_ann_search import Treendex
from efficient_ann_search.utils.dataset import load_csv_dataset


def main() -> None:
    dataset_path = ROOT / "datasets" / "sample_vectors.csv"
    ids, X = load_csv_dataset(dataset_path)

    treendex = Treendex.instance()
    treendex.reset()
    treendex = Treendex(dimension=X.shape[1])
    treendex.create_index("kd", "kd", leaf_size=16)

    for vector_id, v in zip(ids, X):
        treendex.insert_vector(vector_id, v)
    treendex.build_index("kd")

    q = X[0] + 0.01 * np.random.default_rng(0).normal(size=X.shape[1]).astype(np.float32)
    ann = treendex.knn_search("kd", q, k=5, max_visits=256)
    exact = treendex.brute_force_search(q, k=5)

    print("Query: nearest neighbors (ANN KD-Tree)")
    for n in ann:
        print(f"  {n.vector_id:>8s}  dist={n.distance:.6f}")
    print("\nQuery: nearest neighbors (Brute force)")
    for n in exact:
        print(f"  {n.vector_id:>8s}  dist={n.distance:.6f}")


if __name__ == "__main__":
    main()

