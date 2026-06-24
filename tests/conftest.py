"""Make the repo root importable so `autolab` and `tasks` resolve without an
editable install, and provide shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolab import Task  # noqa: E402


class CountingTask(Task):
    """Deterministic task with a single optimum, no dependencies.

    ``score`` peaks at the declared center; the agent should climb toward it.
    """

    name = "CountingTask"

    def propose_space(self):
        return {
            "a": (-10.0, 10.0),
            "b": [1, 2, 4, 8],
            "c": (0, 5),  # int range
        }

    def run(self, config, seed):
        score = -((config["a"] - 3.0) ** 2) + config["b"] - abs(config["c"] - 2)
        return {"score": score, "b_used": float(config["b"])}


class CrashTask(Task):
    """A task whose run() always raises — for testing failure capture."""

    name = "CrashTask"

    def propose_space(self):
        return {"x": (0.0, 1.0)}

    def run(self, config, seed):
        raise RuntimeError("boom")


@pytest.fixture
def counting_task():
    return CountingTask()


@pytest.fixture
def store_dir(tmp_path):
    return tmp_path / "runs"
