from __future__ import annotations

import random

import pytest

from autolab import space as sp


def test_sample_respects_bounds_and_choices():
    space = {"lr": (1e-4, 1e-1), "dim": [64, 128, 256], "layers": (1, 4)}
    rng = random.Random(0)
    for _ in range(200):
        cfg = sp.sample(space, rng)
        assert 1e-4 <= cfg["lr"] <= 1e-1
        assert cfg["dim"] in (64, 128, 256)
        assert cfg["layers"] in (1, 2, 3, 4)
        assert isinstance(cfg["layers"], int)


def test_sample_is_deterministic_for_seed():
    space = {"x": (0.0, 1.0)}
    a = sp.sample(space, random.Random(42))
    b = sp.sample(space, random.Random(42))
    assert a == b


def test_int_vs_float_range_detection():
    rng = random.Random(1)
    assert isinstance(sp.sample({"x": (0, 10)}, rng)["x"], int)
    assert isinstance(sp.sample({"x": (0.0, 10.0)}, rng)["x"], float)


@pytest.mark.parametrize(
    "bad",
    [
        {},  # empty
        {"x": (1,)},  # wrong arity
        {"x": (5, 1)},  # inverted bounds
        {"x": []},  # empty choices
        {"x": "nope"},  # bad spec
        {"x": (True, False)},  # bools are not numbers
    ],
)
def test_validate_rejects_malformed(bad):
    with pytest.raises(sp.SpaceError):
        sp.validate_space(bad)


def test_clip_clamps_and_snaps():
    space = {"lr": (0.0, 1.0), "dim": [64, 128, 256], "layers": (1, 4)}
    out = sp.clip({"lr": 5.0, "dim": 130, "layers": 99}, space)
    assert out["lr"] == 1.0
    assert out["dim"] == 128  # nearest choice
    assert out["layers"] == 4


def test_clip_missing_param_raises():
    with pytest.raises(sp.SpaceError):
        sp.clip({"lr": 0.5}, {"lr": (0.0, 1.0), "dim": [1, 2]})
