from __future__ import annotations

from autolab import CircuitBreaker, Lab, ProcessExecutor, RandomAgent, SerialExecutor
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
