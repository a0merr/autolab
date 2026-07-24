"""Parameter-space description, sampling, and validation.

A *space* is a plain ``dict`` returned by ``Task.propose_space``. Each value
describes the set of values one parameter may take:

==================  =========================================  =================
Spec               Meaning                                     Example
==================  =========================================  =================
``(lo, hi)``        continuous float range (both bounds float)  ``(0.0, 1.0)``
``(lo, hi)``        integer range (both bounds int)             ``(1, 8)``
``LogRange(lo,hi)`` float range sampled by order of magnitude   ``LogRange(1e-5, 1e-1)``
``[a, b, c]``       categorical choice                          ``[64, 128, 256]``
==================  =========================================  =================

Keeping the spec this small is deliberate: it is the entire surface a task
author has to learn, and it is trivial for an LLM agent to read and write.

:class:`LogRange` earns its place because a plain range does the wrong thing
for the parameters people actually tune. Sampled uniformly, ``(1e-5, 1e-1)``
puts ninety percent of its draws in the top decade and effectively never tries
a learning rate below ``1e-2`` — the search looks like it covered four orders
of magnitude while covering one. ``LogRange`` samples the exponent instead, so
each decade gets equal weight.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

Space = Mapping[str, Any]


class SpaceError(ValueError):
    """Raised when a space spec or a config is malformed."""


@dataclass(frozen=True)
class LogRange:
    """A float range sampled uniformly in log space. Both bounds must be > 0.

    Deliberately not a tuple subclass: a ``NamedTuple`` here would satisfy the
    ``isinstance(spec, tuple)`` test above it and be sampled linearly — the
    exact bug it exists to prevent, and a silent one.
    """

    lo: float
    hi: float


def _kind(spec: Any) -> str:
    """Classify a spec: 'int', 'float', 'logfloat', or 'choice'."""
    if isinstance(spec, LogRange):
        if not (math.isfinite(spec.lo) and math.isfinite(spec.hi)):
            raise SpaceError(f"log range bounds must be finite, got {spec!r}")
        if spec.lo <= 0 or spec.hi <= 0:
            raise SpaceError(f"log range bounds must be positive, got {spec!r}")
        if spec.lo > spec.hi:
            raise SpaceError(f"range lower bound exceeds upper: {spec!r}")
        return "logfloat"
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
        f"unrecognized parameter spec {spec!r}; use a (lo, hi) tuple, a "
        f"LogRange(lo, hi), or a list of choices"
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
        elif kind == "logfloat":
            config[name] = math.exp(rng.uniform(math.log(spec.lo), math.log(spec.hi)))
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
        elif kind == "logfloat":
            # Clamped in linear space: the bounds are the same numbers either
            # way, and only *sampling* cares about the scale.
            out[name] = max(spec.lo, min(spec.hi, float(_as_number(value, name))))
        else:  # float
            out[name] = max(spec[0], min(spec[1], float(_as_number(value, name))))
    return out


def _as_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SpaceError(f"parameter {name!r} expected a number, got {value!r}")
    # NaN and inf survive json.loads and defeat min/max clamping silently
    # (max(lo, min(hi, nan)) returns hi), so reject them here instead.
    if not math.isfinite(number):
        raise SpaceError(f"parameter {name!r} must be finite, got {value!r}")
    return number


def _nearest_choice(value: Any, choices: list[Any]) -> Any:
    """Pick the closest numeric choice, else the first choice."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return choices[0]
        numeric = [c for c in choices if isinstance(c, (int, float))]
        if numeric:
            return min(numeric, key=lambda c: abs(c - value))
    return choices[0]


def coerce(
    config: Mapping[str, Any], space: Space, rng: random.Random
) -> dict[str, Any]:
    """Force *config* into *space*, substituting a random draw where it can't.

    This is :func:`clip` with the sharp edges removed. ``clip`` raises when a
    parameter is missing or non-numeric, which is the right behaviour for a
    config a human wrote — but an agent's proposal is untrusted input, and a
    malformed one must not take down an overnight search. Here a parameter
    that cannot be salvaged is replaced by a fresh sample from its own spec,
    so the returned config is always valid and always complete.

    Per-parameter rather than all-or-nothing: one bad entry costs one
    resampled value, not the agent's whole proposal.
    """
    validate_space(space)
    out: dict[str, Any] = {}
    for name, spec in space.items():
        single = {name: spec}
        try:
            if name not in config:
                raise SpaceError(f"config is missing parameter {name!r}")
            out[name] = clip({name: config[name]}, single)[name]
        except SpaceError:
            out[name] = sample(single, rng)[name]
    return out
