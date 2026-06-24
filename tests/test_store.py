from __future__ import annotations

from autolab.store import RunStore


def _add(store, score, **kw):
    return store.add(
        task="tasks.quadratic:Quadratic",
        objective="score",
        direction="max",
        config={"x": score},
        seed=0,
        metrics={"score": score},
        env={},
        **kw,
    )


def test_add_persists_and_roundtrips(store_dir):
    store = RunStore(store_dir)
    run = _add(store, 1.0)
    reloaded = RunStore(store_dir).get(run.run_id)
    assert reloaded.metrics["score"] == 1.0
    assert reloaded.config == {"x": 1.0}
    assert reloaded.run_id == run.run_id


def test_run_ids_are_unique_and_sequential(store_dir):
    store = RunStore(store_dir)
    ids = [_add(store, float(i)).run_id for i in range(5)]
    assert ids == ["0000", "0001", "0002", "0003", "0004"]
    assert len(store) == 5


def test_best_max_and_min(store_dir):
    store = RunStore(store_dir)
    for s in (1.0, 5.0, 3.0):
        _add(store, s)
    assert store.best("score", "max").metrics["score"] == 5.0
    assert store.best("score", "min").metrics["score"] == 1.0


def test_best_empty_is_none(store_dir):
    assert RunStore(store_dir).best("score", "max") is None


def test_list_newest_first(store_dir):
    store = RunStore(store_dir)
    for s in range(3):
        _add(store, float(s))
    ids = [r.run_id for r in store.list(newest_first=True)]
    assert ids == ["0002", "0001", "0000"]
