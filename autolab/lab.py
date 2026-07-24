"""The propose → run → analyze loop.

:class:`Lab` ties the pieces together: it asks the agent for a config, runs the
task under a fixed seed, records the result in the store, and repeats until the
budget is spent. It then returns the best run found.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import analysis, env, space as space_mod
from .agent import Agent, RandomAgent
from .executors import Executor, SerialExecutor
from .guard import BreakerTripped, CircuitBreaker
from .store import DEFAULT_STORE_DIR, Run, RunStore
from .task import Task


class Lab:
    """Runs an agent-driven search over a :class:`Task`."""

    def __init__(
        self,
        task: Task,
        objective: str,
        budget: int,
        *,
        direction: str = "max",
        agent: Agent | None = None,
        store: RunStore | str | Path | None = None,
        seed: int = 0,
        concurrency: int = 1,
        executor: Executor | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        if direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")

        self.task = task
        self.objective = objective
        self.budget = budget
        self.direction = direction
        self.agent = agent or RandomAgent(seed=seed)
        self.seed = seed
        self.concurrency = concurrency
        self.executor = executor or SerialExecutor()
        # Defaults on: an unattended loop should stop itself. Pass
        # CircuitBreaker.off() to run the full budget regardless.
        self.breaker = breaker if breaker is not None else CircuitBreaker()
        # Draws replacement values when an agent proposal is unusable. Seeded
        # off the lab seed so repairs are reproducible too.
        self._repair_rng = random.Random(seed)

        if isinstance(store, RunStore):
            self.store = store
        else:
            self.store = RunStore(store or DEFAULT_STORE_DIR)

        self._space = task.propose_space()
        space_mod.validate_space(self._space)

    # -- the loop ----------------------------------------------------------

    def run(self) -> Run:
        """Execute the search and return the best run found.

        Experiments run in rounds of up to ``concurrency`` jobs. Within a round
        the agent proposes the whole batch from the current history; results are
        written to the store in submission order, so the store is identical
        regardless of how jobs are scheduled.

        Raises :class:`~autolab.guard.BreakerTripped` if a circuit breaker
        limit fires. Runs recorded before that point stay in the store, and the
        reason is written to the store as a ``breaker`` note so an unattended
        run explains itself after the fact.
        """
        env_snapshot = env.capture()
        task_name = self.task.qualified_name()
        self.breaker.start()

        done = 0
        try:
            while done < self.budget:
                k = min(self.concurrency, self.budget - done)
                history = self.store.list(newest_first=False)
                parent = self._parent_for(history)

                self.breaker.before_batch(self.agent)
                raw_batch = self.agent.propose_batch(
                    self._space, history, self.objective, self.direction, k
                )
                configs = [
                    space_mod.coerce(raw, self._space, self._repair_rng)
                    for raw in raw_batch
                ]
                jobs = [(cfg, self.seed + done + j) for j, cfg in enumerate(configs)]

                results = self.executor.run_batch(task_name, jobs)

                for (config, seed), result in zip(jobs, results):
                    run = self._record(
                        config, seed, result, env_snapshot, parent, task_name
                    )
                    self.breaker.observe(run, self.objective)
                done += k
        except BreakerTripped as exc:
            self._record_trip(exc, task_name)
            raise

        best = self.best()
        if best is None:
            raise RuntimeError(
                "search produced no successful runs (all experiments failed)"
            )
        return best

    def _record(self, config, seed, result, env_snapshot, parent, task_name) -> Run:
        """Persist one job's outcome — metrics on success, an error otherwise."""
        if result.ok and self.objective not in result.metrics:
            # A contract violation (task ran but didn't report the objective)
            # is a programming error, not a flaky run — surface it loudly.
            raise KeyError(
                f"Task.run did not report the objective {self.objective!r}; "
                f"got metrics {sorted(result.metrics)}"
            )
        return self.store.add(
            task=task_name,
            objective=self.objective,
            direction=self.direction,
            config=config,
            seed=seed,
            metrics=result.metrics or {},
            env=env_snapshot,
            parent=parent,
            error=result.error,
        )

    def _record_trip(self, exc: BreakerTripped, task_name: str) -> None:
        """Leave a durable record of why the search stopped.

        A cron-driven run that trips at 3am has nowhere to raise to, so the
        reason has to survive the process.
        """
        note: dict[str, Any] = {
            "reason": exc.reason,
            "runs_observed": exc.runs_observed,
            "task": task_name,
            "objective": self.objective,
            "direction": self.direction,
            "budget": self.budget,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        }
        usage = getattr(self.agent, "usage", None)
        model = getattr(self.agent, "model", None)
        if usage is not None and model is not None:
            note["agent"] = {
                "model": model,
                "calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "estimated_cost_usd": round(usage.cost_usd(model), 6),
            }
        self.store.write_note("breaker", note)

    def _parent_for(self, history: list[Run]) -> str | None:
        """The run a new proposal is derived from: the best so far."""
        best = analysis.best(history, self.objective, self.direction)
        return best.run_id if best else None

    # -- results -----------------------------------------------------------

    def best(self) -> Run | None:
        return self.store.best(self.objective, self.direction)

    def summary(self) -> analysis.Summary:
        return analysis.summarize(self.store.list(), self.objective, self.direction)

    def report(self) -> str:
        """Build and print a human-readable summary of the search."""
        s = self.summary()
        lines = [
            f"autolab report — task={self.task.task_name} "
            f"objective={self.objective} ({self.direction})",
            f"runs: {s.n_runs}",
        ]
        if s.best_score is not None:
            lines += [
                f"best: run {s.best_run_id}  {self.objective}={s.best_score:.6g}",
                f"best config: {s.best_config}",
                f"mean {self.objective}: {s.mean_score:.6g}  "
                f"(std {s.std_score:.6g})",
            ]
            tree = analysis.search_tree(self.store.list(newest_first=False))
            if tree:
                lines.append("search tree:")
                lines.append(tree)
        else:
            lines.append("(no scored runs)")
        text = "\n".join(lines)
        print(text)
        return text
