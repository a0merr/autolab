from __future__ import annotations

import time

import pytest

from autolab import (
    BreakerTripped,
    CircuitBreaker,
    Lab,
    ProcessExecutor,
    RandomAgent,
    SerialExecutor,
)
from autolab.executors import Result, _run_job
from tasks.quadratic import Quadratic


def _run(executor, concurrency, store_dir, budget=12, seed=3):
    lab = Lab(
        Quadratic(),
        objective="score",
        budget=budget,
        agent=RandomAgent(seed=seed),
        store=store_dir,
        concurrency=concurrency,
        executor=executor,
    )
    lab.run()
    return [(r.run_id, r.config, r.metrics) for r in lab.store.list(newest_first=False)]


def test_concurrency_one_serial_matches_legacy_behavior(tmp_path):
    # Budget honored exactly, runs scored.
    runs = _run(SerialExecutor(), 1, tmp_path / "a")
    assert len(runs) == 12
    assert all("score" in m for _, _, m in runs)


def test_serial_batched_is_deterministic(tmp_path):
    # Same agent + seeds → identical store regardless of batch size.
    a = _run(SerialExecutor(), 1, tmp_path / "c1")
    b = _run(SerialExecutor(), 4, tmp_path / "c4")
    assert [c for _, c, _ in a] == [c for _, c, _ in b]
    assert [m for _, _, m in a] == [m for _, _, m in b]


def test_process_executor_matches_serial(tmp_path):
    # The whole point: parallel scheduling must not change the store.
    serial = _run(SerialExecutor(), 1, tmp_path / "s")
    parallel = _run(ProcessExecutor(max_workers=4), 4, tmp_path / "p")
    assert [c for _, c, _ in serial] == [c for _, c, _ in parallel]
    assert [m for _, _, m in serial] == [m for _, _, m in parallel]


def test_failed_job_is_recorded_not_raised(tmp_path):
    lab = Lab(
        # CrashTask lives in conftest; reconstructed by its qualified name.
        __import__("tests.conftest", fromlist=["CrashTask"]).CrashTask(),
        objective="score",
        budget=4,
        agent=RandomAgent(seed=0),
        store=tmp_path / "f",
        concurrency=2,
        # This test is about the executor capturing failures, not about the
        # breaker stopping them — see test_guard.py for that.
        breaker=CircuitBreaker.off(),
    )
    # All experiments fail → no best run, but the loop completes and records them.
    try:
        lab.run()
        raised = False
    except RuntimeError:
        raised = True
    assert raised  # "no successful runs"
    runs = lab.store.list()
    assert len(runs) == 4
    assert all(r.failed and "RuntimeError: boom" in r.error for r in runs)


def test_run_job_captures_exception():
    result = _run_job("tests.conftest:CrashTask", {"x": 0.5}, 0)
    assert isinstance(result, Result)
    assert not result.ok
    assert "RuntimeError" in result.error


def test_budget_respected_with_uneven_batches(tmp_path):
    # budget=12, concurrency=5 → rounds of 5, 5, 2.
    runs = _run(SerialExecutor(), 5, tmp_path / "u", budget=12)
    assert len(runs) == 12


# -- pool lifecycle --------------------------------------------------------

_QUADRATIC = "tasks.quadratic:Quadratic"
_JOBS = [({"x": 0.1, "y": 0.2}, 0), ({"x": 0.3, "y": 0.4}, 1)]


def test_pool_is_reused_across_batches():
    # A pool per batch cost a fresh interpreter per worker every round —
    # ~1.7s measured on a task that runs in microseconds.
    with ProcessExecutor(max_workers=2) as ex:
        ex.run_batch(_QUADRATIC, _JOBS)
        first = ex._pool
        ex.run_batch(_QUADRATIC, _JOBS)
        assert ex._pool is first


def test_close_is_idempotent():
    ex = ProcessExecutor(max_workers=2)
    ex.run_batch(_QUADRATIC, _JOBS)
    ex.close()
    ex.close()
    assert ex._pool is None
    # Still usable afterwards: the next batch builds a fresh pool.
    assert len(ex.run_batch(_QUADRATIC, _JOBS)) == 2
    ex.close()


def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout must be positive"):
        ProcessExecutor(timeout=0)


# -- timeout ---------------------------------------------------------------

_HANG = "tests.conftest:HangTask"


def test_a_hung_job_is_abandoned_and_recorded():
    with ProcessExecutor(max_workers=2, timeout=3.0) as ex:
        ex.run_batch(_QUADRATIC, _JOBS)  # warm the pool first
        started = time.monotonic()
        results = ex.run_batch(_HANG, [({"x": 0.9}, 0)])
        elapsed = time.monotonic() - started

    assert not results[0].ok
    assert "TimeoutError" in results[0].error
    # Abandoned near the deadline, not waited out.
    assert elapsed < 30


def test_a_finished_sibling_survives_a_timeout():
    with ProcessExecutor(max_workers=2, timeout=3.0) as ex:
        ex.run_batch(_QUADRATIC, _JOBS)  # warm the pool first
        # Hanging job first, so the fast one has already finished when the
        # deadline fires: its result is real, and killing the pool around it
        # must not throw away work the hang did not invalidate.
        results = ex.run_batch(_HANG, [({"x": 0.9}, 0), ({"x": 0.1}, 1)])

    assert not results[0].ok
    assert "TimeoutError" in results[0].error
    assert results[1].ok
    assert results[1].metrics["score"] == 0.1


def test_the_search_continues_after_a_timeout():
    with ProcessExecutor(max_workers=2, timeout=3.0) as ex:
        assert not ex.run_batch(_HANG, [({"x": 0.9}, 0)])[0].ok
        # The pool was killed; the next batch must get a working one.
        assert all(r.ok for r in ex.run_batch(_QUADRATIC, _JOBS))


def test_a_hung_task_trips_the_breaker(tmp_path):
    # The point of the timeout: max_seconds cannot fire mid-experiment, so
    # without this a single wedged job parks an unattended search forever.
    with ProcessExecutor(max_workers=1, timeout=3.0) as ex:
        lab = Lab(
            __import__("tests.conftest", fromlist=["HangTask"]).HangTask(),
            objective="score",
            budget=50,
            agent=RandomAgent(seed=0),
            store=tmp_path / "hang",
            executor=ex,
            breaker=CircuitBreaker(max_consecutive_errors=1),
        )
        with pytest.raises(BreakerTripped, match="failed in a row"):
            lab.run()
    assert "TimeoutError" in lab.status()["reason"]
