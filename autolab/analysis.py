"""Cross-run analysis: ranking, summary statistics, and the search tree.

Pure functions over a list of :class:`~autolab.store.Run` objects. No I/O, so
they are easy to test and to reuse from both the lab and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from .store import Run, is_scored


def scored(runs: list[Run], objective: str) -> list[Run]:
    """Runs that reported a usable *objective*, in store order.

    A diverged run reporting NaN is excluded: NaN compares false against
    everything, so leaving it in lets it win ``max()`` by arriving first and
    poisons the mean and standard deviation.
    """
    return [r for r in runs if is_scored(r, objective)]


def rank(runs: list[Run], objective: str, direction: str) -> list[Run]:
    """Runs sorted best objective first."""
    rs = scored(runs, objective)
    return sorted(
        rs,
        key=lambda r: float(r.metrics[objective]),
        reverse=(direction == "max"),
    )


def best(runs: list[Run], objective: str, direction: str) -> Run | None:
    ranked = rank(runs, objective, direction)
    return ranked[0] if ranked else None


@dataclass
class Summary:
    objective: str
    direction: str
    n_runs: int
    best_score: float | None
    best_run_id: str | None
    best_config: dict[str, Any] | None
    mean_score: float | None
    std_score: float | None
    # Best score seen after each run, in chronological order. Shows whether the
    # search actually improved over time.
    progression: list[float]


def summarize(runs: list[Run], objective: str, direction: str) -> Summary:
    """Compute a :class:`Summary` of a search over *objective*."""
    rs = scored(runs, objective)
    if not rs:
        return Summary(objective, direction, 0, None, None, None, None, None, [])

    chronological = sorted(rs, key=lambda r: (r.created_at, r.run_id))
    scores = [float(r.metrics[objective]) for r in chronological]
    better = max if direction == "max" else min

    progression: list[float] = []
    running = scores[0]
    for s in scores:
        running = better(running, s)
        progression.append(running)

    top = best(rs, objective, direction)
    assert top is not None
    return Summary(
        objective=objective,
        direction=direction,
        n_runs=len(rs),
        best_score=top.score,
        best_run_id=top.run_id,
        best_config=top.config,
        mean_score=mean(scores),
        std_score=pstdev(scores) if len(scores) > 1 else 0.0,
        progression=progression,
    )


def search_tree(runs: list[Run]) -> str:
    """Render the parent→child derivation of runs as an indented tree."""
    by_id = {r.run_id: r for r in runs}
    children: dict[str | None, list[Run]] = {}
    for r in sorted(runs, key=lambda r: r.run_id):
        parent = r.parent if r.parent in by_id else None
        children.setdefault(parent, []).append(r)

    lines: list[str] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for run in children.get(parent_id, []):
            metric = (
                f"{run.objective}={run.metrics[run.objective]:.4g}"
                if is_scored(run, run.objective)
                else "(no objective)"
            )
            lines.append(f"{'  ' * depth}{run.run_id}  {metric}")
            walk(run.run_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)
