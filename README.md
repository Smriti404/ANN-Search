# Efficient ANN Search

Efficient ANN Search is a modular, object-oriented Python 3 project implementing **Approximate Nearest Neighbor (ANN)** search using **tree-based indexing** techniques:

- **KD-Tree** (`KDTreeIndex`) — axis-aligned splits
- **Random Projection Tree (RP-Tree)** (`RPTreeIndex`) — random hyperplane / projection splits

It supports dynamic vector insert/delete, $k$-nearest-neighbor querying, and benchmarking against brute-force search.

## What is ANN?

Given a query vector $q$ and a dataset $X = \{x_1, \dots, x_n\}$, **nearest neighbor search** finds the closest vector(s) under a distance metric (here: **Euclidean distance**).

Exact search via brute force computes all distances: $O(nd)$ per query for $d$-dimensional vectors.

**Approximate Nearest Neighbor (ANN)** algorithms trade a small amount of accuracy for faster queries by using an index that avoids scanning every vector.

## KD-Tree vs RP-Tree (high-level)

- **KD-Tree**
  - Splits along a single coordinate axis at each node.
  - Works well for low-to-moderate dimensions; performance degrades as dimensionality increases.
  - Deterministic splits (based on variance + median).

- **RP-Tree**
  - Splits using random projection directions.
  - Often more robust in higher dimensions than axis-aligned splits.
  - Randomized structure; repeated builds can vary.

Both implementations in this repo support:

- Build from dataset
- Insert/delete vectors (dynamic updates)
- $k$NN queries
- Approximate search via a **budget** (max node visits)

## Project structure

```
src/
  efficient_ann_search/
    treendex.py
    core/
    indexes/
    engine/
    utils/
tests/
datasets/
benchmarks/
README.md
requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Usage examples

### 1) Minimal example (KD-Tree)

```powershell
python -m benchmarks.example_usage
```

### 2) Run benchmarks (KD vs RP vs brute-force)

```powershell
python -m benchmarks.run_benchmarks
```

### 3) Run tests

```powershell
pytest -q
```

## Benchmarks (how to interpret)

The benchmark script reports:

- **Build time** for each index
- **Query latency** (mean / p95)
- **Recall@k** vs brute force (accuracy)

Results depend heavily on hardware, dataset size, dimensionality, and the approximation budget.

Example output format (your numbers will vary):

```
Dataset: n=20000, d=64, queries=200, k=10
KDTreeIndex: build=0.42s, mean=1.3ms, p95=2.4ms, recall@10=0.86
RPTreeIndex: build=0.31s, mean=1.1ms, p95=2.0ms, recall@10=0.88
BruteForce : mean=9.8ms, p95=12.5ms
```

## Notes on dynamic updates

Tree structures are rebuilt lazily after insert/delete operations.

- Inserts/deletes update a shared vector store.
- Indexes are marked **dirty** and will auto-rebuild on the next query (configurable).

This keeps the code production-friendly (simple invariants, predictable correctness) while still supporting dynamic data.
