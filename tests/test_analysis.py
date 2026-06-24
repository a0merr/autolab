from __future__ import annotations

from autolab import analysis
from autolab.store import RunStore


def _populate(store_dir, scores):
    store = RunStore(store_dir)
    parent = None
    for s in scores:
        run = store.add(
            task="t:T",
            objective="score",
            direction="max",
            config={"v": s},
            seed=0,
            metrics={"score": s},
            env={},
            parent=parent,
        )
        parent = run.run_id
    return store


def test_summarize_progression_is_monotone_max(store_dir):
    store = _populate(store_dir, [1.0, 3.0, 2.0, 5.0, 4.0])
    s = analysis.summarize(store.list(), "score", "max")
    assert s.best_score == 5.0
    assert s.n_runs == 5
    assert s.progression == [1.0, 3.0, 3.0, 5.0, 5.0]  # running best


def test_summarize_empty(store_dir):
    store = RunStore(store_dir)
    s = analysis.summarize(store.list(), "score", "max")
    assert s.n_runs == 0
    assert s.best_score is None


def test_rank_orders_by_direction(store_dir):
    store = _populate(store_dir, [1.0, 3.0, 2.0])
    top = [r.metrics["score"] for r in analysis.rank(store.list(), "score", "max")]
    assert top == [3.0, 2.0, 1.0]
    bottom = [r.metrics["score"] for r in analysis.rank(store.list(), "score", "min")]
    assert bottom == [1.0, 2.0, 3.0]


def test_search_tree_is_chained(store_dir):
    store = _populate(store_dir, [1.0, 2.0, 3.0])
    tree = analysis.search_tree(store.list(newest_first=False))
    lines = tree.splitlines()
    # Each run derives from the previous, so indentation increases by step.
    assert lines[0].startswith("0000")
    assert lines[1].startswith("  0001")
    assert lines[2].startswith("    0002")
