from __future__ import annotations

import math
import os

import pytest

from autolab import Lab, RandomAgent
from autolab.replay import _matches, load_task, replay
from autolab.seeding import seed_everything
from autolab.store import RunStore
from tasks.quadratic import Quadratic


def test_replay_reproduces_quadratic(store_dir):
    lab = Lab(
        Quadratic(),
        objective="score",
        budget=5,
        agent=RandomAgent(seed=2),
        store=RunStore(store_dir),
    )
    lab.run()
    for run in lab.store.list():
        result = replay(run)
        assert result.reproduced, str(result)
        assert result.metrics["score"] == run.metrics["score"]


def test_load_task_by_qualified_name():
    task = load_task("tasks.quadratic:Quadratic")
    assert isinstance(task, Quadratic)


def test_replay_detects_mismatch(store_dir):
    store = RunStore(store_dir)
    run = store.add(
        task="tasks.quadratic:Quadratic",
        objective="score",
        direction="max",
        config={"x": 2.0, "y": -3.0},
        seed=0,
        metrics={"score": 999.0},  # deliberately wrong
        env={},
    )
    result = replay(run, task=Quadratic())
    assert not result.reproduced
    assert "score" in result.diffs


# -- metric comparison -----------------------------------------------------


def test_a_divergence_that_stopped_happening_is_a_mismatch():
    # abs(nan - 0.9) > tol is False, so the old comparison called this
    # reproduced — the most informative replay outcome, reported as clean.
    assert not _matches(float("nan"), 0.9, 1e-9)


def test_nan_reproduces_only_nan():
    assert _matches(float("nan"), float("nan"), 1e-9)
    assert not _matches(0.9, float("nan"), 1e-9)


def test_large_metrics_tolerate_float_noise():
    # An absolute 1e-9 tolerance fails on ordinary rounding at this magnitude.
    assert _matches(1.2345e12, 1.2345e12 + 1e-3, 1e-9)


def test_small_metrics_stay_strict():
    assert not _matches(1e-6, 1.1e-6, 1e-9)


def test_replay_flags_a_run_that_no_longer_diverges(store_dir):
    store = RunStore(store_dir)
    run = store.add(
        task="tasks.quadratic:Quadratic",
        objective="score",
        direction="max",
        config={"x": 2.0, "y": -3.0},
        seed=0,
        metrics={"score": float("nan")},  # recorded as diverged
        env={},
    )
    result = replay(run, task=Quadratic())
    assert not result.reproduced
    original, replayed = result.diffs["score"]
    assert math.isnan(original)
    assert not math.isnan(replayed)


# -- seeding ---------------------------------------------------------------


def test_seeding_does_not_claim_to_fix_hash_randomization(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seed_everything(7)
    # Setting it here would have been a no-op — the interpreter reads it at
    # startup — so it is no longer set at all rather than implying otherwise.
    assert "PYTHONHASHSEED" not in os.environ


@pytest.mark.parametrize("seed", [0, 7, 99])
def test_seeding_is_reproducible(seed):
    import random as pyrandom

    seed_everything(seed)
    first = [pyrandom.random() for _ in range(5)]
    seed_everything(seed)
    assert [pyrandom.random() for _ in range(5)] == first
