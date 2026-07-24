"""Agents that propose the next experiment.

The agent is the smallest, most swappable part of the system. It sees the
parameter space and the history of past runs and returns the next config to
try. Two backends ship in the box:

* :class:`RandomAgent` — samples uniformly from the space. No API, fully
  deterministic given a seed. The default, and what the tests run against.
* :class:`AnthropicAgent` — asks an LLM to read the run history and propose
  the next config. Requires the ``anthropic`` extra and an API key.

:class:`AnthropicAgent` constrains the model's output with the API's
structured-outputs feature: the schema is generated from the space itself, so
a response is guaranteed to be valid JSON containing every parameter with the
right type. Bounds are still enforced downstream by
:func:`autolab.space.coerce` — structured outputs do not support numeric
``minimum``/``maximum``.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from . import space as space_mod
from .guard import TokenUsage
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

        The returned config is coerced into *space* by the lab before running,
        so an agent may be approximate; it must not crash the loop.
        """

    def propose_batch(
        self,
        space: space_mod.Space,
        history: History,
        objective: str,
        direction: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """Return *k* configs to try in parallel.

        The default asks :meth:`propose` ``k`` times. Agents that can reason
        about a whole batch at once (e.g. to keep proposals diverse) should
        override this.
        """
        return [self.propose(space, history, objective, direction) for _ in range(k)]


class RandomAgent(Agent):
    """Uniform random search. Deterministic for a given seed."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(self, space, history, objective, direction):
        return space_mod.sample(space, self._rng)


_SYSTEM_PROMPT = (
    "You are an ML research agent driving a hyperparameter/experiment search. "
    "Given the parameter space and the history of past runs (each with its "
    "config and resulting metrics), propose the most promising next config to "
    "try. Reason about what the history implies, then exploit and explore. "
    "Every parameter value must lie inside its allowed range or choice set."
)

# HTTP statuses that mean the request itself is wrong — a bad model id, a
# missing key, a schema the API rejected. These are programming errors and
# must surface, not degrade quietly into random search for the whole run.
_CONFIG_ERROR_STATUSES = frozenset({400, 401, 403, 404})


class AnthropicAgent(Agent):
    """Proposes configs with the Anthropic API.

    Transient failures (network, rate limits, unparseable output) degrade to a
    random sample so a flaky response never breaks the search loop, and are
    counted in :attr:`consecutive_failures` so a
    :class:`~autolab.guard.CircuitBreaker` can stop a search that has silently
    become random search. Configuration errors — bad API key, unknown model —
    are raised instead of swallowed.

    :param effort: how hard the model thinks per proposal. ``"low"`` is the
        default: proposal quality is dominated by the run history, not by
        reasoning depth, and low effort keeps an overnight search cheap and
        its outputs stable. Note that ``temperature`` is not available on
        current models — effort is the knob.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        max_tokens: int = 2048,
        history_window: int = 30,
        effort: str = "low",
        structured_output: bool = True,
        api_key: str | None = None,
        client: Any | None = None,
        seed: int = 0,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on extra
                raise ImportError(
                    "AnthropicAgent needs the 'anthropic' package. "
                    "Install with: pip install 'autolab[anthropic]'"
                ) from exc
            client = anthropic.Anthropic(api_key=api_key)

        self._client = client
        self.model = model
        self.effort = effort
        self.structured_output = structured_output
        self.usage = TokenUsage()
        #: Consecutive API calls that fell back to random search.
        self.consecutive_failures = 0
        #: Message from the most recent fallback, for breaker diagnostics.
        self.last_error: str | None = None

        self._max_tokens = max_tokens
        self._history_window = history_window
        self._fallback = RandomAgent(seed=seed)

    def propose(self, space, history, objective, direction):
        return self.propose_batch(space, history, objective, direction, 1)[0]

    def propose_batch(self, space, history, objective, direction, k):
        space_mod.validate_space(space)
        try:
            configs = self._ask(space, history, objective, direction, k)
        except Exception as exc:
            if _is_config_error(exc):
                # A 400/401/403/404 will recur on every call. Falling back
                # would turn a broken key into an entire night of random
                # search that looks like it worked.
                raise
            self.consecutive_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            configs = []
        else:
            self.consecutive_failures = 0
            self.last_error = None

        # Top up a short or empty response rather than discarding what came
        # back: a partial batch is still better than none.
        while len(configs) < k:
            configs.append(self._fallback.propose(space, history, objective, direction))
        return configs[:k]

    # -- API call ----------------------------------------------------------

    def _ask(self, space, history, objective, direction, k) -> list[dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_prompt(
                        space, history, objective, direction, k
                    ),
                }
            ],
        }
        output_config: dict[str, Any] = {"effort": self.effort}
        if self.structured_output:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _batch_schema(space),
            }
        request["output_config"] = output_config

        message = self._client.messages.create(**request)
        self._record_usage(message)

        text = "".join(block.text for block in message.content if block.type == "text")
        return _extract_configs(text)

    def _record_usage(self, message: Any) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        self.usage.add(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def cost_usd(self, pricing: dict[str, tuple[float, float]] | None = None) -> float:
        """Estimated API spend so far, in USD."""
        return self.usage.cost_usd(self.model, pricing)

    def _build_prompt(self, space, history, objective, direction, k) -> str:
        recent = list(history)[-self._history_window :]
        rows = [{"config": r.config, "metrics": r.metrics} for r in recent]
        goal = "maximize" if direction == "max" else "minimize"
        ask = (
            f"Propose {k} DIVERSE next configs (explore different regions)."
            if k > 1
            else "Propose the single most promising next config."
        )
        return (
            f"Objective: {goal} the metric '{objective}'.\n\n"
            f"Parameter space (name -> allowed values):\n"
            f"{json.dumps(_describe_space(space), indent=2)}\n\n"
            f"History of {len(rows)} past run(s), oldest first:\n"
            f"{json.dumps(rows, indent=2, default=str)}\n\n"
            f'{ask} Return them under the "configs" key.'
        )


def _is_config_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _CONFIG_ERROR_STATUSES


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


def _config_schema(space: space_mod.Space) -> dict[str, Any]:
    """JSON Schema for a single config drawn from *space*.

    Guarantees presence and type of every parameter. Bounds are deliberately
    absent: structured outputs do not support ``minimum``/``maximum``, so
    range enforcement stays with :func:`autolab.space.coerce`.
    """
    properties: dict[str, Any] = {}
    for name, info in _describe_space(space).items():
        if info["type"] == "choice":
            properties[name] = {"enum": list(info["options"])}
        elif info["type"] == "int":
            properties[name] = {"type": "integer"}
        else:
            properties[name] = {"type": "number"}
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _batch_schema(space: space_mod.Space) -> dict[str, Any]:
    """Schema for a batch of configs. Always a list, even for a batch of one,
    so the response shape does not depend on the concurrency setting."""
    return {
        "type": "object",
        "properties": {"configs": {"type": "array", "items": _config_schema(space)}},
        "required": ["configs"],
        "additionalProperties": False,
    }


def _extract_configs(text: str) -> list[dict[str, Any]]:
    """Pull the config list out of a response.

    With structured outputs on, *text* is already a ``{"configs": [...]}``
    object. The looser paths below keep the agent working when the caller has
    turned structured output off or is pointed at a model that lacks it.
    """
    data = _extract_json(text)
    if isinstance(data, dict) and isinstance(data.get("configs"), list):
        candidates = data["configs"]
    elif isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        candidates = [data]  # a bare single config
    else:
        raise ValueError(f"unexpected response shape: {type(data).__name__}")
    return [c for c in candidates if isinstance(c, dict)]


def _extract_json(text: str) -> Any:
    """Parse JSON out of an LLM response, tolerant of stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"no JSON found in response: {text[:200]!r}")
