from __future__ import annotations

import json
import math

import pytest

from autolab.store import RunStore


def _add(store, score, *, x=None, **kw):
    return store.add(
        task="tasks.quadratic:Quadratic",
        objective="score",
        direction="max",
        # The config is always finite — space.coerce guarantees it — so it is
        # kept separate from the metric, which may legitimately diverge.
        config={"x": 0.0 if x is None else x},
        seed=0,
        metrics={"score": score},
        env={},
        **kw,
    )


def test_add_persists_and_roundtrips(store_dir):
    store = RunStore(store_dir)
    run = _add(store, 1.0, x=1.0)
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


# -- non-finite metrics ----------------------------------------------------


@pytest.mark.parametrize(
    "value,name",
    [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
    ],
)
def test_non_finite_metrics_are_written_as_valid_json(store_dir, value, name):
    store = RunStore(store_dir)
    run = _add(store, value)
    raw = store._path(run.run_id).read_text(encoding="utf-8")
    # Python's json emits a bare NaN token by default, which no other parser
    # accepts. parse_constant fires on exactly those tokens.
    assert json.loads(raw, parse_constant=_reject)["metrics"]["score"] == name


def _reject(token):
    raise AssertionError(f"non-standard JSON constant on disk: {token}")


def test_non_finite_config_is_rejected_loudly(store_dir):
    # coerce guarantees finite configs, so this is a programming error and
    # should surface rather than write an unparseable file.
    with pytest.raises(ValueError, match="not JSON compliant"):
        _add(RunStore(store_dir), 1.0, x=float("nan"))


def test_non_finite_metrics_round_trip_back_to_floats(store_dir):
    store = RunStore(store_dir)
    run = store.add(
        task="tasks.quadratic:Quadratic",
        objective="score",
        direction="max",
        config={"x": 1.0},
        seed=0,
        metrics={"score": float("nan"), "loss": float("inf"), "acc": 0.5},
        env={},
    )
    reloaded = RunStore(store_dir).get(run.run_id)
    assert math.isnan(reloaded.metrics["score"])
    assert reloaded.metrics["loss"] == math.inf
    assert reloaded.metrics["acc"] == 0.5


def test_diverged_run_cannot_win_best(store_dir):
    store = RunStore(store_dir)
    _add(store, float("nan"))  # written first — would win max() unguarded
    _add(store, 1.0)
    _add(store, 5.0)
    assert store.best("score", "max").metrics["score"] == 5.0
    assert store.best("score", "min").metrics["score"] == 1.0


def test_best_is_none_when_every_run_diverged(store_dir):
    store = RunStore(store_dir)
    _add(store, float("nan"))
    _add(store, float("inf"))
    assert store.best("score", "max") is None


# -- notes -----------------------------------------------------------------


def test_notes_round_trip_and_stay_out_of_the_run_listing(store_dir):
    store = RunStore(store_dir)
    _add(store, 1.0)
    store.write_note("breaker", {"reason": "diverged"})
    _add(store, 2.0)

    assert store.read_note("breaker") == {"reason": "diverged"}
    assert len(store) == 2
    assert [r.run_id for r in store.list(newest_first=False)] == ["0000", "0001"]


def test_missing_note_is_none(store_dir):
    assert RunStore(store_dir).read_note("breaker") is None
