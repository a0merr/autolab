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
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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

    The pool is created once and reused across batches. Spawning one per batch
    cost a fresh interpreter per worker every round — measured at ~1.7s per
    batch on a task that runs in microseconds, which a long search pays on
    every round. Call :meth:`close` when done, or use the executor as a
    context manager; it is also closed on garbage collection.

    :param timeout: seconds a single batch may take before its unfinished jobs
        are abandoned and recorded as errors. ``None`` (the default) waits
        forever. This is what puts a floor under a hung experiment: a job stuck
        in a wedged kernel or a socket with no timeout of its own cannot be
        interrupted by a circuit breaker, because breakers are only checked
        between experiments. Whatever survives the timeout is killed with the
        pool, and the next batch starts from a fresh one.
    """

    def __init__(self, max_workers: int = 4, *, timeout: float | None = None) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self.max_workers = max_workers
        self.timeout = timeout
        self._pool: ProcessPoolExecutor | None = None

    # -- pool lifecycle ----------------------------------------------------

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(list(sys.path),),
            )
        return self._pool

    def close(self) -> None:
        """Shut the pool down. Safe to call more than once."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _discard_pool(self) -> None:
        """Drop the pool and kill its workers, without waiting for them.

        Used after a timeout, where waiting is precisely what we are refusing
        to do: the point is that a worker is not coming back. ``shutdown`` has
        no public way to kill running children, so the worker handles are
        reached through the private attribute and terminated directly — the
        fallback below is what happens if a future CPython stops exposing it.
        """
        pool, self._pool = self._pool, None
        if pool is None:
            return
        for process in list(getattr(pool, "_processes", {}).values()):
            try:
                process.kill()
            except (OSError, ValueError):  # already dead
                pass
        pool.shutdown(wait=False)

    def __enter__(self) -> "ProcessExecutor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass

    # -- execution ---------------------------------------------------------

    def run_batch(self, task_name: str, jobs: list[Job]) -> list[Result]:
        if not jobs:
            return []
        pool = self._get_pool()
        futures = [
            pool.submit(_run_job, task_name, config, seed) for config, seed in jobs
        ]

        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        results: list[Result] = []
        timed_out = False
        for future in futures:
            if timed_out:
                # The pool is gone, but a job that had already finished still
                # holds a real result — keep it rather than throwing away work
                # a sibling's hang did not actually invalidate.
                if future.done() and future.exception() is None:
                    results.append(future.result())
                else:
                    results.append(_timeout_result(self.timeout))
                continue

            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                results.append(future.result(timeout=remaining))
            except FutureTimeoutError:
                timed_out = True
                self._discard_pool()
                results.append(_timeout_result(self.timeout))
            except Exception as exc:
                # A worker that died outright — OOM kill, segfault — surfaces
                # here rather than inside _run_job's own handler.
                results.append(
                    Result(metrics=None, error=f"{type(exc).__name__}: {exc}")
                )
        # One result appended per future, in submission order.
        return results


def _timeout_result(timeout: float | None) -> Result:
    """A job abandoned at the batch deadline, recorded like any other failure.

    Recorded rather than dropped so the run store still accounts for the
    budget, and so ``max_consecutive_errors`` sees a hung task the same way it
    sees a crashing one.
    """
    return Result(
        metrics=None,
        error=f"TimeoutError: batch exceeded {timeout:g}s and was abandoned",
    )
