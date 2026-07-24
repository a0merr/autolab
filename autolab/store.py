"""Versioned, on-disk run store.

Every experiment is persisted as one immutable JSON record under the store
directory (default ``.autolab/runs``). Records are append-only: the store is
the audit trail of the whole search, and any run can be re-read or replayed
later.

Files are standard JSON — readable by any parser, not just Python's. A metric
that diverged to ``NaN`` or ``inf`` is stored as a string (``"NaN"``,
``"Infinity"``, ``"-Infinity"``) and decoded back to a float on read, because
JSON has no literal for them and Python's default output is non-standard. A
non-finite value anywhere else is a programming error — configs come from
:func:`autolab.space.coerce`, which guarantees finite values — and raises on
write rather than producing an unparseable file.

Alongside the runs, :meth:`RunStore.write_note` records facts about the search
as a whole rather than one experiment, such as why a circuit breaker stopped
it. Notes are excluded from iteration, ``len()``, and run-id assignment.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_STORE_DIR = ".autolab/runs"

# Run files are named by a zero-padded index (``0000.json``). Matching that
# shape rather than ``*.json`` keeps sidecar notes out of the run listing —
# and out of the index counter that assigns the next run id.
_RUN_GLOB = "[0-9][0-9][0-9][0-9]*.json"

# JSON has no literal for these. Python's json module emits bare ``NaN`` and
# ``Infinity`` tokens, which it reads back happily and every other parser
# rejects. Store them as strings so the files stay portable.
_NON_FINITE_NAMES = ("NaN", "Infinity", "-Infinity")


def _encode_metric(value: Any) -> Any:
    """Render one metric as something standard JSON can represent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number):
        return "NaN"
    if number == math.inf:
        return "Infinity"
    if number == -math.inf:
        return "-Infinity"
    return number


def _decode_metric(value: Any) -> Any:
    """Inverse of :func:`_encode_metric`."""
    if isinstance(value, str) and value in _NON_FINITE_NAMES:
        return float(value)
    return value


def is_scored(run: "Run", objective: str) -> bool:
    """True if *run* reported *objective* as a finite number.

    A run that diverged to NaN or inf reported *something*, but not a value
    any comparison can rank — treat it as unscored everywhere rankings are
    computed.
    """
    if objective not in run.metrics:
        return False
    try:
        return math.isfinite(float(run.metrics[objective]))
    except (TypeError, ValueError):
        return False


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
    error: str | None = None  # set when the experiment failed to run

    @property
    def score(self) -> float:
        """The objective metric for this run."""
        return float(self.metrics[self.objective])

    @property
    def failed(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = {k: _encode_metric(v) for k, v in self.metrics.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        fields = {k: v for k, v in data.items() if k in known}
        metrics = fields.get("metrics")
        if isinstance(metrics, dict):
            fields["metrics"] = {k: _decode_metric(v) for k, v in metrics.items()}
        return cls(**fields)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Append-only collection of :class:`Run` records on disk."""

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------

    def _next_index(self) -> int:
        return sum(1 for _ in self.directory.glob(_RUN_GLOB))

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
        error: str | None = None,
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
            error=error,
        )
        # allow_nan=False turns a non-finite value that escaped encoding into a
        # loud failure instead of an unparseable file on disk.
        self._path(run_id).write_text(
            json.dumps(run.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return run

    # -- notes -------------------------------------------------------------

    def _note_path(self, name: str) -> Path:
        return self.directory / f"{name}.note.json"

    def write_note(self, name: str, payload: dict[str, Any]) -> Path:
        """Record a sidecar fact about the search, next to the runs.

        Notes are not runs: they are excluded from iteration, ``len()``, and
        run-id assignment. Used for things that describe the search as a whole
        rather than one experiment — why it stopped, for instance.
        """
        path = self._note_path(name)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def read_note(self, name: str) -> dict[str, Any] | None:
        """Read a note written by :meth:`write_note`, or ``None`` if absent."""
        path = self._note_path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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
        for path in self.directory.glob(_RUN_GLOB):
            yield Run.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob(_RUN_GLOB))

    def best(self, objective: str, direction: str) -> Run | None:
        """The run with the best objective value, or ``None`` if empty."""
        candidates = [r for r in self if is_scored(r, objective)]
        if not candidates:
            return None
        pick = max if direction == "max" else min
        return pick(candidates, key=lambda r: float(r.metrics[objective]))
