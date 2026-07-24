from __future__ import annotations

import pytest

from autolab import BreakerTripped, CircuitBreaker, Lab, RandomAgent, TokenUsage
from autolab.guard import CACHE_READ_MULTIPLIER
from autolab.store import Run


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
    lab.run()
    assert len(lab.store) == 20
