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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_clip_rejects_non_finite(bad):
    # json.loads happily parses NaN/Infinity, and max(lo, min(hi, nan)) would
    # silently return hi. Reject instead of pretending to clamp.
    with pytest.raises(sp.SpaceError):
        sp.clip({"lr": bad}, {"lr": (0.0, 1.0)})


def test_clip_rejects_non_numeric():
    with pytest.raises(sp.SpaceError):
        sp.clip({"lr": "fast"}, {"lr": (0.0, 1.0)})


# -- coerce: the never-raising variant the loop uses -----------------------


def test_coerce_fills_a_missing_parameter():
    space = {"lr": (0.0, 1.0), "dim": [64, 128]}
    out = sp.coerce({"lr": 0.5}, space, random.Random(0))
    assert out["lr"] == 0.5  # good value kept
    assert out["dim"] in (64, 128)  # missing one resampled


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), "fast", None, {"a": 1}, [1, 2]]
)
def test_coerce_replaces_unusable_values(value):
    out = sp.coerce({"lr": value}, {"lr": (0.0, 1.0)}, random.Random(0))
    assert 0.0 <= out["lr"] <= 1.0


def test_coerce_still_clamps_and_snaps():
    space = {"lr": (0.0, 1.0), "dim": [64, 128, 256], "layers": (1, 4)}
    out = sp.coerce({"lr": 5.0, "dim": 130, "layers": 99}, space, random.Random(0))
    assert out == {"lr": 1.0, "dim": 128, "layers": 4}


def test_coerce_ignores_unknown_keys():
    out = sp.coerce({"lr": 0.5, "bogus": 1}, {"lr": (0.0, 1.0)}, random.Random(0))
    assert out == {"lr": 0.5}


def test_coerce_is_deterministic_for_seed():
    space = {"lr": (0.0, 1.0), "dim": [64, 128]}
    a = sp.coerce({}, space, random.Random(7))
    b = sp.coerce({}, space, random.Random(7))
    assert a == b
