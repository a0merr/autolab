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


# -- log-scale ranges ------------------------------------------------------


def test_log_range_gives_each_decade_equal_weight():
    rng = random.Random(0)
    space = {"lr": sp.LogRange(1e-5, 1e-1)}
    draws = [sp.sample(space, rng)["lr"] for _ in range(4000)]

    assert all(1e-5 <= v <= 1e-1 for v in draws)
    decades = [
        sum(1 for v in draws if 10.0**-e > v >= 10.0 ** -(e + 1)) for e in (1, 2, 3, 4)
    ]
    # Four decades, ~1000 each. Uniform sampling would put ~90% in the top one
    # and effectively never try a learning rate below 1e-2.
    assert all(800 < count < 1200 for count in decades), decades


def test_uniform_sampling_is_what_log_range_exists_to_avoid():
    rng = random.Random(0)
    draws = [sp.sample({"lr": (1e-5, 1e-1)}, rng)["lr"] for _ in range(1000)]
    assert sum(1 for v in draws if v < 1e-2) < 150  # ~90% land in the top decade


def test_log_range_is_not_mistaken_for_a_plain_range():
    # A NamedTuple here would pass isinstance(spec, tuple) and be sampled
    # linearly — the exact bug LogRange exists to prevent.
    assert not isinstance(sp.LogRange(1e-5, 1e-1), tuple)
    assert sp._kind(sp.LogRange(1e-5, 1e-1)) == "logfloat"


@pytest.mark.parametrize(
    "bad",
    [
        sp.LogRange(0.0, 1.0),  # log(0) undefined
        sp.LogRange(-1.0, 1.0),  # negative bound
        sp.LogRange(1e-1, 1e-5),  # inverted
        sp.LogRange(1e-5, float("inf")),  # non-finite
    ],
)
def test_log_range_rejects_impossible_bounds(bad):
    with pytest.raises(sp.SpaceError):
        sp.validate_space({"lr": bad})


def test_log_range_clips_in_linear_space():
    space = {"lr": sp.LogRange(1e-5, 1e-1)}
    assert sp.clip({"lr": 10.0}, space)["lr"] == 1e-1
    assert sp.clip({"lr": 1e-9}, space)["lr"] == 1e-5
    assert sp.clip({"lr": 3e-3}, space)["lr"] == 3e-3


def test_coerce_resamples_a_bad_log_value():
    out = sp.coerce({"lr": "fast"}, {"lr": sp.LogRange(1e-5, 1e-1)}, random.Random(0))
    assert 1e-5 <= out["lr"] <= 1e-1


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
