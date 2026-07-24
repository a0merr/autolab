"""Circuit breakers: stop a runaway search before it burns money.

An autonomous loop left running overnight has three ways to go wrong that the
loop itself cannot notice:

* every experiment crashes, and the search spends its whole budget recording
  identical stack traces;
* the objective diverges to NaN/inf, and the "best" run becomes meaningless;
* the LLM agent keeps getting called — and billed — while producing nothing
  useful.

:class:`CircuitBreaker` watches for all three and aborts the loop with a
:class:`BreakerTripped` describing which limit fired. Everything already
written to the run store is kept: tripping is a stop, not a rollback.

The defaults are deliberately conservative — a breaker that never fires is
indistinguishable from not having one. Use :meth:`CircuitBreaker.off` to
disable it entirely.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

#: Anthropic list prices in USD per million tokens, as ``(input, output)``.
#:
#: Cached 2026-06-24, and only used for the spend estimate behind
#: ``max_cost_usd`` — it is an estimate, not a billing record. List prices
#: change and negotiated rates differ, so this is public and mutable::
#:
#:     from autolab.guard import PRICING_USD_PER_MTOK
#:     PRICING_USD_PER_MTOK["claude-opus-5"] = (4.0, 20.0)
#:
#: For a one-off override without touching the module, pass ``pricing=`` to
#: :meth:`TokenUsage.cost_usd` or to :class:`CircuitBreaker`.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Cache reads bill at ~0.1x the input rate; 5-minute cache writes at ~1.25x.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def _rate_for(
    model: str, pricing: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Look up ``(input, output)`` rates, falling back conservatively.

    An unrecognized model is priced at the most expensive rate in the table, so
    a model this version has never heard of makes the cost breaker fire early
    rather than never.
    """
    if model in pricing:
        return pricing[model]
    if not pricing:
        return (0.0, 0.0)
    # Componentwise max, not max() over the tuples: tuple comparison ranks by
    # input rate and only consults the output rate to break ties, so a table
    # containing a cheap-in/expensive-out model would price an unknown one
    # below a model already in the table. The table is public and mutable, so
    # that shape is a caller's edit away.
    return (
        max(rate_in for rate_in, _ in pricing.values()),
        max(rate_out for _, rate_out in pricing.values()),
    )


@dataclass
class TokenUsage:
    """Running token totals for one agent, and their estimated dollar cost."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cache_read_tokens += int(cache_read_tokens or 0)
        self.cache_write_tokens += int(cache_write_tokens or 0)

    def cost_usd(
        self, model: str, pricing: dict[str, tuple[float, float]] | None = None
    ) -> float:
        """Estimated spend so far for *model*, in USD.

        Pass *pricing* to price against a table other than
        :data:`PRICING_USD_PER_MTOK` — negotiated rates, or a model this
        version predates.
        """
        rate_in, rate_out = _rate_for(model, pricing or PRICING_USD_PER_MTOK)
        billable_in = (
            self.input_tokens
            + self.cache_read_tokens * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * CACHE_WRITE_MULTIPLIER
        )
        return (billable_in * rate_in + self.output_tokens * rate_out) / 1_000_000


class BreakerTripped(RuntimeError):
    """Raised when a :class:`CircuitBreaker` limit is exceeded.

    Runs recorded before the trip remain in the store — inspect them with
    ``autolab runs list`` to see what the search had found when it stopped.
    """

    def __init__(self, reason: str, *, runs_observed: int) -> None:
        super().__init__(
            f"circuit breaker tripped after {runs_observed} run(s): {reason}"
        )
        self.reason = reason
        self.runs_observed = runs_observed


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class CircuitBreaker:
    """Aborts a search when it looks like it has stopped making progress.

    Every limit may be set to ``None`` to disable that check individually.

    Limits are evaluated at experiment boundaries, never mid-experiment: a
    running job cannot be preempted. So a search overshoots ``max_seconds`` by
    at most the duration of one experiment, and ``max_cost_usd`` by at most one
    round's worth of agent calls. Size the caps accordingly.

    :param max_consecutive_errors: trip after this many failed experiments in
        a row (a crashing task, a bad environment).
    :param max_consecutive_nonfinite: trip after this many consecutive runs
        whose objective is NaN or inf (a diverging loss).
    :param max_agent_failures: trip after this many consecutive agent calls
        that fell back to random search (network down, unparseable output).
    :param max_cost_usd: trip once the agent's estimated API spend exceeds
        this. Ignored for agents that make no API calls.
    :param max_seconds: trip once the search has run this long in wall-clock.
    :param pricing: per-model ``(input, output)`` USD-per-million-token rates
        used for ``max_cost_usd``. Defaults to :data:`PRICING_USD_PER_MTOK`.
    """

    max_consecutive_errors: int | None = 3
    max_consecutive_nonfinite: int | None = 3
    max_agent_failures: int | None = 3
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    pricing: dict[str, tuple[float, float]] | None = None

    _consecutive_errors: int = field(default=0, init=False, repr=False)
    _consecutive_nonfinite: int = field(default=0, init=False, repr=False)
    _runs_observed: int = field(default=0, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def off(cls) -> "CircuitBreaker":
        """A breaker with every limit disabled. Use when you want the loop to
        run to its full budget no matter what happens."""
        return cls(
            max_consecutive_errors=None,
            max_consecutive_nonfinite=None,
            max_agent_failures=None,
            max_cost_usd=None,
            max_seconds=None,
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def runs_observed(self) -> int:
        """Experiments seen since :meth:`start`."""
        return self._runs_observed

    def start(self) -> None:
        """Reset counters and start the clock. Called once per search."""
        self._consecutive_errors = 0
        self._consecutive_nonfinite = 0
        self._runs_observed = 0
        self._started_at = time.monotonic()

    def _check_clock(self) -> None:
        if self.max_seconds is None or self._started_at is None:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed > self.max_seconds:
            self._trip(f"wall-clock limit of {self.max_seconds:g}s exceeded")

    def before_batch(self, agent: Any) -> None:
        """Check the limits that do not depend on an experiment's result.

        Runs before each round of proposals, so a search that is only burning
        API credit stops before it spends another round's worth.
        """
        self._check_clock()

        if self.max_agent_failures is not None:
            failures = getattr(agent, "consecutive_failures", 0)
            if failures >= self.max_agent_failures:
                last = getattr(agent, "last_error", None)
                detail = f" (last error: {last})" if last else ""
                self._trip(
                    f"{failures} consecutive agent call(s) fell back to random "
                    f"search{detail}"
                )

        if self.max_cost_usd is not None:
            usage = getattr(agent, "usage", None)
            model = getattr(agent, "model", None)
            if usage is not None and model is not None:
                spent = usage.cost_usd(model, self.pricing)
                if spent > self.max_cost_usd:
                    self._trip(
                        f"estimated API spend ${spent:.4f} exceeds the "
                        f"${self.max_cost_usd:.4f} cap"
                    )

    def observe(self, run: Any, objective: str) -> None:
        """Record one finished experiment and check the result-based limits.

        The clock is re-checked here as well as in :meth:`before_batch`, so a
        single long experiment stops the search when it finishes rather than
        after the whole next round has been proposed and run.
        """
        self._runs_observed += 1
        self._check_clock()

        if getattr(run, "failed", False):
            self._consecutive_errors += 1
            self._consecutive_nonfinite = 0
            if (
                self.max_consecutive_errors is not None
                and self._consecutive_errors >= self.max_consecutive_errors
            ):
                self._trip(
                    f"{self._consecutive_errors} experiment(s) failed in a row; "
                    f"last error: {run.error}"
                )
            return

        self._consecutive_errors = 0

        score = run.metrics.get(objective)
        if score is not None and not _is_finite(score):
            self._consecutive_nonfinite += 1
            if (
                self.max_consecutive_nonfinite is not None
                and self._consecutive_nonfinite >= self.max_consecutive_nonfinite
            ):
                self._trip(
                    f"objective {objective!r} was non-finite ({score!r}) on "
                    f"{self._consecutive_nonfinite} run(s) in a row — the search "
                    f"has diverged"
                )
        else:
            self._consecutive_nonfinite = 0

    def _trip(self, reason: str) -> None:
        raise BreakerTripped(reason, runs_observed=self._runs_observed)
