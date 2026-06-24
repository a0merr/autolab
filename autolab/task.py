"""The :class:`Task` plugin interface.

A task is the only thing a user must write. Implement two methods and the
core handles the agent, the run store, reproducibility, and analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .space import Space


class Task(ABC):
    """Base class for an experiment a :class:`~autolab.lab.Lab` can optimize.

    Subclasses implement :meth:`propose_space` (what may vary) and :meth:`run`
    (execute one experiment, return metrics). Everything else is provided.
    """

    #: Optional human-readable name; defaults to the class name.
    name: str | None = None

    @abstractmethod
    def propose_space(self) -> Space:
        """Return the parameter space the agent is allowed to vary.

        See :mod:`autolab.space` for the spec format.
        """

    @abstractmethod
    def run(self, config: dict[str, Any], seed: int) -> dict[str, float]:
        """Run one experiment with *config* under *seed*; return metrics.

        The returned dict maps metric names to numeric values. It must contain
        the lab's objective. ``run`` should be deterministic given
        ``(config, seed)`` so the experiment can be replayed exactly.
        """

    @property
    def task_name(self) -> str:
        return self.name or type(self).__name__

    def qualified_name(self) -> str:
        """Dotted ``module:Class`` path, used by the CLI to replay runs."""
        cls = type(self)
        return f"{cls.__module__}:{cls.__qualname__}"
