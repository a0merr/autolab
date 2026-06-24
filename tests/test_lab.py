from __future__ import annotations

import pytest

from autolab import Lab, RandomAgent, Task
from autolab.store import RunStore


def test_search_finds_a_run_and_records_budget(counting_task, store_dir):
    lab = Lab(
        counting_task,
        objective="score",
        budget=40,
        agent=RandomAgent(seed=7),
        store=RunStore(store_dir),
        seed=100,
    )
    best = lab.run()
    assert len(lab.store) == 40
    # Random search over 40 draws should beat a single average draw comfortably.
    assert best.metrics["score"] > -20
    assert best.objective == "score"


def test_seeds_are_offset_per_experiment(counting_task, store_dir):
    lab = Lab(counting_task, objective="score", budget=3, store=store_dir, seed=50)
    lab.run()
    seeds = sorted(r.seed for r in lab.store.list())
    assert seeds == [50, 51, 52]


def test_parent_points_at_best_so_far(counting_task, store_dir):
    lab = Lab(
        counting_task,
        objective="score",
        budget=10,
        agent=RandomAgent(seed=1),
        store=store_dir,
    )
    lab.run()
    runs = lab.store.list(newest_first=False)
    assert runs[0].parent is None  # first run derives from nothing
    # Every later parent is a real, earlier run id.
    ids = {r.run_id for r in runs}
    for r in runs[1:]:
        assert r.parent in ids


def test_direction_min(counting_task, store_dir):
    lab = Lab(
        counting_task,
        objective="score",
        budget=15,
        direction="min",
        agent=RandomAgent(seed=3),
        store=store_dir,
    )
    best = lab.run()
    all_scores = [r.metrics["score"] for r in lab.store.list()]
    assert best.metrics["score"] == min(all_scores)


class BadTask(Task):
    def propose_space(self):
        return {"x": (0.0, 1.0)}

    def run(self, config, seed):
        return {"not_the_objective": 1.0}


def test_missing_objective_raises(store_dir):
    lab = Lab(BadTask(), objective="score", budget=1, store=store_dir)
    with pytest.raises(KeyError):
        lab.run()


def test_invalid_args():
    from tests.conftest import CountingTask

    with pytest.raises(ValueError):
        Lab(CountingTask(), objective="score", budget=0)
    with pytest.raises(ValueError):
        Lab(CountingTask(), objective="score", budget=5, direction="sideways")


def test_report_runs(counting_task, store_dir, capsys):
    lab = Lab(counting_task, objective="score", budget=5, store=store_dir)
    lab.run()
    text = lab.report()
    out = capsys.readouterr().out
    assert "autolab report" in out
    assert "best:" in text
