"""Re-execute a stored run exactly and check it reproduces.

Replay is the payoff of capturing config + seed: given a run id, reconstruct
the task, re-run it with the recorded config and seed, and compare metrics.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .seeding import seed_everything
from .store import Run
from .task import Task


def load_task(qualified_name: str) -> Task:
    """Instantiate a task from its ``module:Class`` path (no-arg constructor)."""
    if ":" not in qualified_name:
        raise ValueError(f"expected 'module:Class', got {qualified_name!r}")
    module_path, _, class_path = qualified_name.partition(":")
    module = importlib.import_module(module_path)
    obj: Any = module
    for part in class_path.split("."):
        obj = getattr(obj, part)
    instance = obj()
    if not isinstance(instance, Task):
        raise TypeError(f"{qualified_name} is not a Task subclass")
    return instance


@dataclass
class ReplayResult:
    run: Run
    metrics: dict[str, float]
    reproduced: bool
    diffs: dict[str, tuple[float, float]]  # metric -> (original, replayed)

    def __str__(self) -> str:
        status = "reproduced" if self.reproduced else "MISMATCH"
        head = f"replay {self.run.run_id}: {status}"
        if self.reproduced:
            return head
        rows = "\n".join(
            f"  {name}: original={a:.6g} replayed={b:.6g}"
            for name, (a, b) in self.diffs.items()
        )
        return f"{head}\n{rows}"


def replay(run: Run, *, task: Task | None = None, tol: float = 1e-9) -> ReplayResult:
    """Re-run *run* and compare metrics to the stored ones."""
    task = task or load_task(run.task)
    seed_everything(run.seed)
    fresh = {k: float(v) for k, v in task.run(dict(run.config), run.seed).items()}

    diffs: dict[str, tuple[float, float]] = {}
    for name, original in run.metrics.items():
        replayed = fresh.get(name)
        if replayed is None or abs(float(original) - replayed) > tol:
            diffs[name] = (
                float(original),
                float(replayed) if replayed is not None else float("nan"),
            )

    return ReplayResult(run=run, metrics=fresh, reproduced=not diffs, diffs=diffs)
