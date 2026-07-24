"""autolab — autonomous ML experimentation.

An agent proposes hypotheses, runs experiments against any model or task, and
analyzes the results — with versioned runs, full reproducibility, and a plugin
interface for custom objectives.
"""

from __future__ import annotations

from .agent import Agent, AnthropicAgent, RandomAgent
from .executors import Executor, ProcessExecutor, Result, SerialExecutor
from .guard import (
    PRICING_USD_PER_MTOK,
    BreakerTripped,
    CircuitBreaker,
    TokenUsage,
)
from .lab import Lab
from .replay import ReplayResult, replay
from .store import Run, RunStore
from .task import Task

__version__ = "0.3.0"

__all__ = [
    "Lab",
    "Task",
    "Agent",
    "RandomAgent",
    "AnthropicAgent",
    "Executor",
    "SerialExecutor",
    "ProcessExecutor",
    "Result",
    "Run",
    "RunStore",
    "CircuitBreaker",
    "BreakerTripped",
    "TokenUsage",
    "PRICING_USD_PER_MTOK",
    "replay",
    "ReplayResult",
    "__version__",
]
