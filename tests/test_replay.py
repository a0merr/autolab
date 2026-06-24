from __future__ import annotations

from autolab import Lab, RandomAgent
from autolab.replay import load_task, replay
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
