from __future__ import annotations

import pytest

from autolab import (
    BreakerTripped,
    CircuitBreaker,
    Lab,
    RandomAgent,
    SerialExecutor,
    TokenUsage,
)
from autolab.guard import CACHE_READ_MULTIPLIER
from autolab.store import SEARCH_NOTE, Run


def _crash_task():
    return __import__("tests.conftest", fromlist=["CrashTask"]).CrashTask()


def _run(metrics=None, error=None):
    return Run(
        run_id="0000",
        task="t:T",
        objective="score",
        direction="max",
        config={},
        seed=0,
        metrics=metrics or {},
        env={},
        created_at="2026-01-01T00:00:00+00:00",
        error=error,
    )


# -- token accounting ------------------------------------------------------


def test_cost_uses_per_model_rates():
    usage = TokenUsage()
    usage.add(input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_usd("claude-opus-5") == pytest.approx(30.0)
    assert usage.cost_usd("claude-sonnet-5") == pytest.approx(18.0)
    assert usage.cost_usd("claude-haiku-4-5") == pytest.approx(6.0)


def test_cached_input_is_discounted():
    usage = TokenUsage()
    usage.add(cache_read_tokens=1_000_000)
    assert usage.cost_usd("claude-opus-5") == pytest.approx(5.0 * CACHE_READ_MULTIPLIER)


def test_unknown_model_prices_at_the_top_rate():
    usage = TokenUsage()
    usage.add(input_tokens=1_000_000)
    # Errs high so an unrecognized model trips the cost cap early, not late.
    assert usage.cost_usd("some-future-model") == pytest.approx(10.0)


def test_unknown_model_takes_the_top_rate_of_each_component():
    usage = TokenUsage()
    usage.add(output_tokens=1_000_000)
    # max() over the tuples would rank by input rate and pick (9.0, 1.0),
    # pricing an unknown model below one already in the table.
    table = {"cheap-in": (1.0, 99.0), "cheap-out": (9.0, 1.0)}
    assert usage.cost_usd("some-future-model", table) == pytest.approx(99.0)


def test_pricing_can_be_overridden_per_call():
    usage = TokenUsage()
    usage.add(input_tokens=1_000_000, output_tokens=1_000_000)
    negotiated = {"claude-opus-5": (1.0, 2.0)}
    assert usage.cost_usd("claude-opus-5", negotiated) == pytest.approx(3.0)


def test_breaker_uses_the_supplied_pricing_table():
    class _Agent:
        model = "claude-opus-5"

        def __init__(self):
            self.usage = TokenUsage()
            self.usage.add(input_tokens=1_000_000)

    agent = _Agent()
    # $5.00 at list price would trip; $0.10 at the negotiated rate does not.
    CircuitBreaker(
        max_cost_usd=1.0, pricing={"claude-opus-5": (0.1, 0.5)}
    ).before_batch(agent)


# -- breaker limits --------------------------------------------------------


def test_consecutive_errors_trip():
    breaker = CircuitBreaker(max_consecutive_errors=3)
    breaker.start()
    breaker.observe(_run(error="RuntimeError: boom"), "score")
    breaker.observe(_run(error="RuntimeError: boom"), "score")
    with pytest.raises(BreakerTripped, match="failed in a row"):
        breaker.observe(_run(error="RuntimeError: boom"), "score")


def test_a_success_resets_the_error_streak():
    breaker = CircuitBreaker(max_consecutive_errors=2)
    breaker.start()
    breaker.observe(_run(error="boom"), "score")
    breaker.observe(_run({"score": 1.0}), "score")
    breaker.observe(_run(error="boom"), "score")  # streak restarted, no trip


def test_diverging_objective_trips():
    breaker = CircuitBreaker(max_consecutive_nonfinite=2)
    breaker.start()
    breaker.observe(_run({"score": float("nan")}), "score")
    with pytest.raises(BreakerTripped, match="diverged"):
        breaker.observe(_run({"score": float("inf")}), "score")


def test_agent_stuck_on_fallback_trips():
    class _Degraded:
        consecutive_failures = 3
        last_error = "APIConnectionError: no route to host"

    breaker = CircuitBreaker(max_agent_failures=3)
    breaker.start()
    with pytest.raises(BreakerTripped, match="no route to host"):
        breaker.before_batch(_Degraded())


def test_cost_cap_trips():
    class _Spendy:
        model = "claude-opus-5"

        def __init__(self):
            self.usage = TokenUsage()
            self.usage.add(input_tokens=1_000_000)  # $5.00

    breaker = CircuitBreaker(max_cost_usd=1.0)
    breaker.start()
    with pytest.raises(BreakerTripped, match=r"exceeds the \$1"):
        breaker.before_batch(_Spendy())


def test_cost_cap_ignores_agents_without_usage():
    breaker = CircuitBreaker(max_cost_usd=0.0)
    breaker.start()
    breaker.before_batch(RandomAgent(seed=0))  # no usage attribute, no trip


def test_wall_clock_cap_trips():
    breaker = CircuitBreaker(max_seconds=0.0)
    breaker.start()
    with pytest.raises(BreakerTripped, match="wall-clock"):
        breaker.before_batch(RandomAgent(seed=0))


def test_wall_clock_cap_also_trips_on_a_finished_experiment():
    # Checked per experiment as well as per round, so one long-running job
    # stops the search when it lands rather than a whole round later.
    breaker = CircuitBreaker(max_seconds=0.0)
    breaker.start()
    with pytest.raises(BreakerTripped, match="wall-clock"):
        breaker.observe(_run({"score": 1.0}), "score")


def test_off_disables_every_limit():
    breaker = CircuitBreaker.off()
    breaker.start()
    for _ in range(50):
        breaker.observe(_run(error="boom"), "score")
        breaker.observe(_run({"score": float("nan")}), "score")


def test_tripped_error_reports_progress():
    breaker = CircuitBreaker(max_consecutive_errors=1)
    breaker.start()
    with pytest.raises(BreakerTripped) as excinfo:
        breaker.observe(_run(error="boom"), "score")
    assert excinfo.value.runs_observed == 1


# -- integration with the loop --------------------------------------------


def test_lab_stops_a_crashing_task_early(tmp_path):
    lab = Lab(
        __import__("tests.conftest", fromlist=["CrashTask"]).CrashTask(),
        objective="score",
        budget=100,
        agent=RandomAgent(seed=0),
        store=tmp_path / "runs",
        breaker=CircuitBreaker(max_consecutive_errors=3),
    )
    with pytest.raises(BreakerTripped):
        lab.run()
    # Stopped after the third failure instead of burning all 100.
    assert len(lab.store) == 3
    assert all(r.failed for r in lab.store.list())


def test_lab_runs_full_budget_with_breaker_off(tmp_path):
    lab = Lab(
        __import__("tests.conftest", fromlist=["CrashTask"]).CrashTask(),
        objective="score",
        budget=4,
        agent=RandomAgent(seed=0),
        store=tmp_path / "runs",
        breaker=CircuitBreaker.off(),
    )
    with pytest.raises(RuntimeError):  # "no successful runs"
        lab.run()
    assert len(lab.store) == 4


def test_healthy_search_is_unaffected_by_default_breaker(counting_task, store_dir):
    lab = Lab(counting_task, objective="score", budget=20, store=store_dir)
    best = lab.run()
    assert len(lab.store) == 20
    note = lab.status()
    assert note["status"] == "completed"
    assert note["reason"] is None
    assert note["best_run_id"] == best.run_id


def test_trip_reason_is_written_to_the_store(tmp_path):
    lab = Lab(
        _crash_task(),
        objective="score",
        budget=100,
        agent=RandomAgent(seed=0),
        store=tmp_path / "runs",
        breaker=CircuitBreaker(max_consecutive_errors=2),
    )
    with pytest.raises(BreakerTripped):
        lab.run()

    # A cron job has nowhere to raise to — the reason must survive the process.
    note = lab.status()
    assert note["status"] == "tripped"
    assert "failed in a row" in note["reason"]
    assert note["runs_observed"] == 2
    assert note["budget"] == 100
    assert note["objective"] == "score"
    assert "CrashTask" in note["task"]
    assert note["started_at"].startswith("20")
    assert note["stopped_at"].startswith("20")


def test_a_clean_search_clears_an_earlier_trip(counting_task, store_dir):
    tripped = Lab(
        _crash_task(),
        objective="score",
        budget=100,
        store=store_dir,
        breaker=CircuitBreaker(max_consecutive_errors=1),
    )
    with pytest.raises(BreakerTripped):
        tripped.run()
    assert tripped.status()["status"] == "tripped"

    # Same store, second search, finishes cleanly. Yesterday's reason must not
    # still be sitting there to be read as today's.
    lab = Lab(counting_task, objective="score", budget=3, store=store_dir)
    lab.run()
    assert lab.status()["status"] == "completed"
    assert lab.status()["reason"] is None


def test_a_search_in_flight_says_so(counting_task, store_dir):
    seen: dict = {}

    class _PeekingExecutor(SerialExecutor):
        def run_batch(self, task_name, jobs):
            seen.update(lab.status() or {})
            return super().run_batch(task_name, jobs)

    lab = Lab(
        counting_task,
        objective="score",
        budget=2,
        store=store_dir,
        executor=_PeekingExecutor(),
    )
    lab.run()
    # Written before the first experiment, so a killed process leaves behind
    # "running" rather than nothing at all.
    assert seen["status"] == "running"
    assert seen["stopped_at"] is None


def test_a_search_that_produces_nothing_usable_is_recorded_as_crashed(tmp_path):
    lab = Lab(
        _crash_task(),
        objective="score",
        budget=2,
        store=tmp_path / "runs",
        breaker=CircuitBreaker.off(),
    )
    with pytest.raises(RuntimeError):
        lab.run()
    note = lab.status()
    assert note["status"] == "crashed"
    assert "no successful runs" in note["reason"]


def test_trip_note_prices_with_the_breakers_own_table(tmp_path):
    class _PricedAgent(RandomAgent):
        model = "claude-opus-5"

        def __init__(self):
            super().__init__(seed=0)
            self.usage = TokenUsage()
            self.usage.add(input_tokens=1_000_000)  # $2.00 at the rate below

    lab = Lab(
        _crash_task(),
        objective="score",
        budget=10,
        agent=_PricedAgent(),
        store=tmp_path / "runs",
        breaker=CircuitBreaker(max_cost_usd=0.5, pricing={"claude-opus-5": (2.0, 4.0)}),
    )
    with pytest.raises(BreakerTripped):
        lab.run()
    # The cap fired on $2.00; recording $5.00 from the list-price table would
    # make the postmortem contradict the limit that produced it.
    assert lab.status()["agent"]["estimated_cost_usd"] == pytest.approx(2.0)


def test_a_trip_mid_batch_still_records_the_whole_batch(tmp_path):
    lab = Lab(
        _crash_task(),
        objective="score",
        budget=10,
        agent=RandomAgent(seed=0),
        store=tmp_path / "runs",
        concurrency=3,
        breaker=CircuitBreaker(max_consecutive_errors=1),
    )
    with pytest.raises(BreakerTripped) as excinfo:
        lab.run()
    # The breaker stops after the first failure it sees, but all three
    # experiments had already run — discarding two records would lose work the
    # search has already paid for.
    assert excinfo.value.runs_observed == 1
    assert len(lab.store) == 3


def test_note_does_not_pollute_the_run_listing(tmp_path):
    lab = Lab(
        _crash_task(),
        objective="score",
        budget=100,
        agent=RandomAgent(seed=0),
        store=tmp_path / "runs",
        breaker=CircuitBreaker(max_consecutive_errors=2),
    )
    with pytest.raises(BreakerTripped):
        lab.run()
    # Notes live beside the runs; they must not be counted, iterated, or
    # allowed to shift the next run id.
    assert lab.store.read_note(SEARCH_NOTE) is not None
    assert len(lab.store) == 2
    assert [r.run_id for r in lab.store.list(newest_first=False)] == ["0000", "0001"]
    assert lab.store._next_index() == 2
