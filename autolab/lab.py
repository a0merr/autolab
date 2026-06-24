"""The propose → run → analyze loop.

:class:`Lab` ties the pieces together: it asks the agent for a config, runs the
task under a fixed seed, records the result in the store, and repeats until the
budget is spent. It then returns the best run found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import analysis, env, space as space_mod
from .agent import Agent, RandomAgent
from .seeding import seed_everything
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
    ) -> None:
        if direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")

        self.task = task
        self.objective = objective
        self.budget = budget
        self.direction = direction
        self.agent = agent or RandomAgent(seed=seed)
        self.seed = seed

        if isinstance(store, RunStore):
            self.store = store
        else:
            self.store = RunStore(store or DEFAULT_STORE_DIR)

        self._space = task.propose_space()
        space_mod.validate_space(self._space)

    # -- the loop ----------------------------------------------------------

    def run(self) -> Run:
        """Execute the search and return the best run found."""
        env_snapshot = env.capture()

        for i in range(self.budget):
            history = self.store.list(newest_first=False)
            parent = self._parent_for(history)

            raw = self.agent.propose(
                self._space, history, self.objective, self.direction
            )
            config = space_mod.clip(raw, self._space)

            seed = self.seed + i
            seed_everything(seed)
            metrics = self._run_one(config, seed)

            self.store.add(
                task=self.task.qualified_name(),
                objective=self.objective,
                direction=self.direction,
                config=config,
                seed=seed,
                metrics=metrics,
                env=env_snapshot,
                parent=parent,
            )

        best = self.best()
        if best is None:
            raise RuntimeError("search produced no runs")
        return best

    def _run_one(self, config: dict[str, Any], seed: int) -> dict[str, float]:
        metrics = self.task.run(config, seed)
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Task.run must return a dict of metrics, got {type(metrics).__name__}"
            )
        if self.objective not in metrics:
            raise KeyError(
                f"Task.run did not report the objective {self.objective!r}; "
                f"got metrics {sorted(metrics)}"
            )
        return {k: float(v) for k, v in metrics.items()}

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
