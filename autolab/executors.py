"""Execution backends for running experiments.

An :class:`Executor` runs a batch of ``(config, seed)`` jobs for one task and
returns their results **in submission order** — never completion order. That
ordering is what keeps the run store deterministic regardless of how a parallel
scheduler interleaves jobs: same agent + same seeds produce an identical store.

Workers receive only the task's ``module:Class`` path plus the config and seed;
they reconstruct the task by import (never by unpickling a live instance), seed
the RNGs *inside the worker*, and run. A job that raises is captured as a
:class:`Result` with an ``error`` rather than crashing the batch.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

from .replay import load_task
from .seeding import seed_everything

# A job is a (config, seed) pair; a batch is a list of jobs for one task.
Job = tuple[dict, int]


@dataclass
class Result:
    """Outcome of one job: metrics on success, or an error message."""

    metrics: dict[str, float] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _run_job(task_name: str, config: dict, seed: int) -> Result:
    """Worker entrypoint (module-level so it is picklable across processes)."""
    try:
        task = load_task(task_name)
        seed_everything(seed)
        metrics = task.run(dict(config), seed)
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Task.run must return a dict of metrics, got {type(metrics).__name__}"
            )
        return Result(metrics={k: float(v) for k, v in metrics.items()})
    except Exception as exc:  # captured, not raised — one job must not kill the batch
        return Result(metrics=None, error=f"{type(exc).__name__}: {exc}")


class Executor(ABC):
    """Runs a batch of jobs for a task and returns results in submission order."""

    @abstractmethod
    def run_batch(self, task_name: str, jobs: list[Job]) -> list[Result]: ...


class SerialExecutor(Executor):
    """Runs jobs one at a time in the current process. The default backend.

    With ``concurrency=1`` this reproduces the original sequential loop exactly.
    """

    def run_batch(self, task_name: str, jobs: list[Job]) -> list[Result]:
        return [_run_job(task_name, config, seed) for config, seed in jobs]


def _init_worker(path: list[str]) -> None:
    """Restore the parent's import paths in a freshly spawned worker."""
    for entry in reversed(path):
        if entry not in sys.path:
            sys.path.insert(0, entry)


class ProcessExecutor(Executor):
    """Runs jobs across a process pool (spawn context, CUDA-safe).

    Results are reassembled in submission order, so the store stays
    deterministic even though jobs may finish out of order.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run_batch(self, task_name: str, jobs: list[Job]) -> list[Result]:
        if not jobs:
            return []
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(list(sys.path),),
        ) as pool:
            # pool.map preserves input order — exactly the invariant we need.
            return list(
                pool.map(
                    _run_job,
                    [task_name] * len(jobs),
                    [c for c, _ in jobs],
                    [s for _, s in jobs],
                )
            )
