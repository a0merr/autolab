"""Parameter-space description, sampling, and validation.

A *space* is a plain ``dict`` returned by ``Task.propose_space``. Each value
describes the set of values one parameter may take:

================  ===========================================  =================
Spec             Meaning                                       Example
================  ===========================================  =================
``(lo, hi)``      continuous float range (both bounds float)    ``(1e-5, 1e-1)``
``(lo, hi)``      integer range (both bounds int)               ``(1, 8)``
``[a, b, c]``     categorical choice                            ``[64, 128, 256]``
================  ===========================================  =================

Keeping the spec this small is deliberate: it is the entire surface a task
author has to learn, and it is trivial for an LLM agent to read and write.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

Space = Mapping[str, Any]


class SpaceError(ValueError):
    """Raised when a space spec or a config is malformed."""


def _kind(spec: Any) -> str:
    """Classify a single parameter spec. Returns 'int', 'float', or 'choice'."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise SpaceError(f"range spec must have exactly 2 bounds, got {spec!r}")
        lo, hi = spec
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise SpaceError(f"range bounds must be numbers, got {spec!r}")
        if isinstance(lo, int) and isinstance(hi, int):
            kind = "int"
        elif isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            kind = "float"
        else:
            raise SpaceError(f"range bounds must be numbers, got {spec!r}")
        if lo > hi:
            raise SpaceError(f"range lower bound exceeds upper: {spec!r}")
        return kind
    if isinstance(spec, list):
        if not spec:
            raise SpaceError("categorical spec must list at least one choice")
        return "choice"
    raise SpaceError(
        f"unrecognized parameter spec {spec!r}; use a (lo, hi) tuple or a list"
    )


def validate_space(space: Space) -> None:
    """Raise :class:`SpaceError` if *space* is not a well-formed spec."""
    if not isinstance(space, Mapping):
        raise SpaceError(f"space must be a dict, got {type(space).__name__}")
    if not space:
        raise SpaceError("space is empty; nothing for the agent to vary")
    for name, spec in space.items():
        if not isinstance(name, str):
            raise SpaceError(f"parameter names must be strings, got {name!r}")
        _kind(spec)


def sample(space: Space, rng: random.Random) -> dict[str, Any]:
    """Draw one config uniformly at random from *space*."""
    validate_space(space)
    config: dict[str, Any] = {}
    for name, spec in space.items():
        kind = _kind(spec)
        if kind == "choice":
            config[name] = rng.choice(spec)
        elif kind == "int":
            config[name] = rng.randint(spec[0], spec[1])
        else:  # float
            config[name] = rng.uniform(spec[0], spec[1])
    return config


def clip(config: Mapping[str, Any], space: Space) -> dict[str, Any]:
    """Coerce *config* into *space*: clamp ranges, snap categoricals.

    Used to make an agent's proposal safe to run even if it strays slightly
    outside the declared bounds or emits the wrong numeric type.
    """
    validate_space(space)
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = _kind(spec)
        if name not in config:
            raise SpaceError(f"config is missing parameter {name!r}")
        value = config[name]
        if kind == "choice":
            out[name] = value if value in spec else _nearest_choice(value, spec)
        elif kind == "int":
            out[name] = int(round(_as_number(value, name)))
            out[name] = max(spec[0], min(spec[1], out[name]))
        else:  # float
            out[name] = max(spec[0], min(spec[1], float(_as_number(value, name))))
    return out


def _as_number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SpaceError(f"parameter {name!r} expected a number, got {value!r}")


def _nearest_choice(value: Any, choices: list[Any]) -> Any:
    """Pick the closest numeric choice, else the first choice."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = [c for c in choices if isinstance(c, (int, float))]
        if numeric:
            return min(numeric, key=lambda c: abs(c - value))
    return choices[0]
