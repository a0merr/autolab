"""Versioned, on-disk run store.

Every experiment is persisted as one immutable JSON record under the store
directory (default ``.autolab/runs``). Records are append-only: the store is
the audit trail of the whole search, and any run can be re-read or replayed
later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_STORE_DIR = ".autolab/runs"


@dataclass(frozen=True)
class Run:
    """An immutable record of a single experiment."""

    run_id: str
    task: str  # qualified module:Class path
    objective: str
    direction: str  # "max" or "min"
    config: dict[str, Any]
    seed: int
    metrics: dict[str, float]
    env: dict[str, Any]
    created_at: str
    parent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """The objective metric for this run."""
        return float(self.metrics[self.objective])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Append-only collection of :class:`Run` records on disk."""

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------

    def _next_index(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))

    def add(
        self,
        *,
        task: str,
        objective: str,
        direction: str,
        config: dict[str, Any],
        seed: int,
        metrics: dict[str, float],
        env: dict[str, Any],
        parent: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Run:
        """Create, persist, and return a new run record."""
        run_id = f"{self._next_index():04d}"
        run = Run(
            run_id=run_id,
            task=task,
            objective=objective,
            direction=direction,
            config=config,
            seed=seed,
            metrics=metrics,
            env=env,
            created_at=_now_iso(),
            parent=parent,
            extra=extra or {},
        )
        self._path(run_id).write_text(
            json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return run

    # -- reading -----------------------------------------------------------

    def _path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def get(self, run_id: str) -> Run:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(f"no run {run_id!r} in {self.directory}")
        return Run.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, newest_first: bool = True) -> list[Run]:
        """All runs, sorted by creation time."""
        runs = sorted(self, key=lambda r: (r.created_at, r.run_id))
        return list(reversed(runs)) if newest_first else runs

    def __iter__(self) -> Iterator[Run]:
        for path in self.directory.glob("*.json"):
            yield Run.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))

    def best(self, objective: str, direction: str) -> Run | None:
        """The run with the best objective value, or ``None`` if empty."""
        candidates = [r for r in self if objective in r.metrics]
        if not candidates:
            return None
        pick = max if direction == "max" else min
        return pick(candidates, key=lambda r: float(r.metrics[objective]))
