from __future__ import annotations

from autolab import analysis
from autolab.store import Run, RunStore


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
    assert s.n_scored == 5
    assert s.progression == [1.0, 3.0, 3.0, 5.0, 5.0]  # running best


def test_summarize_empty(store_dir):
    store = RunStore(store_dir)
    s = analysis.summarize(store.list(), "score", "max")
    assert s.n_runs == 0
    assert s.n_scored == 0
    assert s.best_score is None


def test_unscored_runs_are_counted_but_not_measured(store_dir):
    store = _populate(store_dir, [1.0, 3.0])
    store.add(
        task="t:T",
        objective="score",
        direction="max",
        config={},
        seed=0,
        metrics={},
        env={},
        error="RuntimeError: boom",
    )
    s = analysis.summarize(store.list(), "score", "max")
    # Reporting only the scored count made a store of nothing but crashes
    # print "runs: 0" while `runs list` showed them.
    assert (s.n_runs, s.n_scored) == (3, 2)
    assert s.best_score == 3.0


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


def _chain(length):
    return [
        Run(
            run_id=f"{i:04d}",
            task="t:T",
            objective="score",
            direction="max",
            config={},
            seed=0,
            metrics={"score": float(i)},
            env={},
            created_at=f"2026-01-01T00:00:{i:05d}",
            parent=None if i == 0 else f"{i - 1:04d}",
        )
        for i in range(length)
    ]


def test_search_tree_survives_a_long_improving_search():
    # Each proposal derives from the best so far, so a search that keeps
    # improving is one chain as deep as it is long. Recursion died here at
    # ~1000 runs: report() failed on exactly the searches that went well.
    tree = analysis.search_tree(_chain(1500))
    assert len(tree.splitlines()) == 1500


def test_search_tree_stops_indenting_past_the_cap():
    lines = analysis.search_tree(_chain(20), max_indent=4).splitlines()
    assert lines[4].startswith("        0004")  # 4 levels, still indented
    # Deeper runs carry their depth instead of marching off the terminal.
    assert lines[9].startswith("        +9  0009")


def test_search_tree_orders_siblings_by_run_id(store_dir):
    store = RunStore(store_dir)
    root = store.add(
        task="t:T",
        objective="score",
        direction="max",
        config={},
        seed=0,
        metrics={"score": 1.0},
        env={},
    )
    for _ in range(3):
        store.add(
            task="t:T",
            objective="score",
            direction="max",
            config={},
            seed=0,
            metrics={"score": 2.0},
            env={},
            parent=root.run_id,
        )
    ids = [line.split()[0] for line in analysis.search_tree(store.list()).splitlines()]
    assert ids == ["0000", "0001", "0002", "0003"]
