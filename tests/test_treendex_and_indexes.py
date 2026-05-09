from __future__ import annotations

import numpy as np

from efficient_ann_search import Treendex


def _fresh_treendex(d: int) -> Treendex:
    t = Treendex.instance()
    t.reset()
    return Treendex(dimension=d)


def test_singleton_identity() -> None:
    a = Treendex.instance()
    b = Treendex.instance()
    assert a is b


def test_kdtree_exact_matches_bruteforce_when_unbounded() -> None:
    rng = np.random.default_rng(0)
    d = 16
    X = rng.normal(size=(500, d)).astype(np.float32)
    q = rng.normal(size=(d,)).astype(np.float32)

    t = _fresh_treendex(d)
    t.create_index("kd", "kd", leaf_size=16)
    for i, v in enumerate(X):
        t.insert_vector(f"v{i}", v)
    t.build_index("kd")

    exact = t.brute_force_search(q, k=10)
    kd = t.knn_search("kd", q, k=10, max_visits=None)
    assert [n.vector_id for n in kd] == [n.vector_id for n in exact]


def test_rptree_reasonable_recall() -> None:
    rng = np.random.default_rng(1)
    d = 32
    X = rng.normal(size=(2000, d)).astype(np.float32)
    queries = rng.normal(size=(20, d)).astype(np.float32)

    t = _fresh_treendex(d)
    t.create_index("rp", "rp", leaf_size=32, random_state=123)
    for i, v in enumerate(X):
        t.insert_vector(f"v{i}", v)
    t.build_index("rp")

    recalls = []
    for q in queries:
        approx = t.knn_search("rp", q, k=10, max_visits=256)
        exact = t.brute_force_search(q, k=10)
        approx_ids = {n.vector_id for n in approx}
        exact_ids = {n.vector_id for n in exact}
        recalls.append(len(approx_ids & exact_ids) / 10)
    assert float(np.mean(recalls)) >= 0.4


def test_dynamic_insert_delete_marks_dirty_and_updates_results() -> None:
    rng = np.random.default_rng(2)
    d = 8
    t = _fresh_treendex(d)
    t.create_index("kd", "kd", leaf_size=8)

    base = rng.normal(size=(200, d)).astype(np.float32)
    for i, v in enumerate(base):
        t.insert_vector(f"v{i}", v)
    t.build_index("kd")

    # Insert a vector very close to query.
    q = np.zeros((d,), dtype=np.float32)
    t.insert_vector("special", 1e-6 * np.ones((d,), dtype=np.float32))
    ann = t.knn_search("kd", q, k=1, max_visits=None)
    assert ann[0].vector_id == "special"

    # Delete it and ensure it no longer appears.
    t.delete_vector("special")
    ann2 = t.knn_search("kd", q, k=5, max_visits=None)
    assert "special" not in {n.vector_id for n in ann2}
