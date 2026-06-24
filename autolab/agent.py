"""Agents that propose the next experiment.

The agent is the smallest, most swappable part of the system. It sees the
parameter space and the history of past runs and returns the next config to
try. Two backends ship in the box:

* :class:`RandomAgent` — samples uniformly from the space. No API, fully
  deterministic given a seed. The default, and what the tests run against.
* :class:`AnthropicAgent` — asks an LLM to read the run history and propose
  the next config. Requires the ``anthropic`` extra and an API key.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from . import space as space_mod
from .store import Run

# A run-history entry as seen by an agent: just config + metrics.
History = Sequence[Run]


class Agent(ABC):
    """Proposes the next config given the space and past runs."""

    @abstractmethod
    def propose(
        self,
        space: space_mod.Space,
        history: History,
        objective: str,
        direction: str,
    ) -> dict[str, Any]:
        """Return the next config to try.

        The returned config is clipped to *space* by the lab before running,
        so an agent may be approximate; it must not crash the loop.
        """


class RandomAgent(Agent):
    """Uniform random search. Deterministic for a given seed."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(self, space, history, objective, direction):
        return space_mod.sample(space, self._rng)


_SYSTEM_PROMPT = (
    "You are an ML research agent driving a hyperparameter/experiment search. "
    "Given the parameter space and the history of past runs (each with its "
    "config and resulting metrics), propose the single most promising next "
    "config to try. Reason about what the history implies, then exploit and "
    "explore. Respond with ONLY a JSON object mapping every parameter name to "
    "a value inside its allowed range or choice set. No prose, no code fences."
)


class AnthropicAgent(Agent):
    """Proposes configs with the Anthropic API.

    Falls back to a random sample if the API response cannot be parsed, so a
    flaky response never breaks the search loop.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        max_tokens: int = 1024,
        history_window: int = 30,
        api_key: str | None = None,
        seed: int = 0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(
                "AnthropicAgent needs the 'anthropic' package. "
                "Install with: pip install 'autolab[anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._history_window = history_window
        self._fallback = RandomAgent(seed=seed)

    def propose(self, space, history, objective, direction):
        space_mod.validate_space(space)
        prompt = self._build_prompt(space, history, objective, direction)
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in message.content if block.type == "text"
            )
            return _extract_json(text)
        except Exception:  # pragma: no cover - network / parse fallbacks
            # Any failure (network, parse, bad JSON) degrades to random search
            # rather than aborting the loop.
            return self._fallback.propose(space, history, objective, direction)

    def _build_prompt(self, space, history, objective, direction) -> str:
        recent = list(history)[-self._history_window :]
        rows = [{"config": r.config, "metrics": r.metrics} for r in recent]
        goal = "maximize" if direction == "max" else "minimize"
        return (
            f"Objective: {goal} the metric '{objective}'.\n\n"
            f"Parameter space (name -> allowed values):\n"
            f"{json.dumps(_describe_space(space), indent=2)}\n\n"
            f"History of {len(rows)} past run(s), oldest first:\n"
            f"{json.dumps(rows, indent=2)}\n\n"
            "Propose the next config as a JSON object."
        )


def _describe_space(space: space_mod.Space) -> dict[str, Any]:
    """Render a space spec as JSON-friendly hints for the agent."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, tuple):
            lo, hi = spec
            kind = "int" if isinstance(lo, int) and isinstance(hi, int) else "float"
            out[name] = {"type": kind, "range": [lo, hi]}
        else:
            out[name] = {"type": "choice", "options": list(spec)}
    return out


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM response, tolerant of stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")
