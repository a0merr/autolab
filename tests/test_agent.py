from __future__ import annotations

import pytest

from autolab import LogRange, RandomAgent
from autolab import agent as agent_mod
from autolab.store import Run


def test_random_agent_deterministic():
    space = {"x": (0.0, 1.0), "k": [1, 2, 3]}
    a = RandomAgent(seed=11)
    b = RandomAgent(seed=11)
    assert [a.propose(space, [], "s", "max") for _ in range(5)] == [
        b.propose(space, [], "s", "max") for _ in range(5)
    ]


def test_extract_json_tolerates_prose_and_fences():
    assert agent_mod._extract_json('{"x": 1}') == {"x": 1}
    assert agent_mod._extract_json('here you go:\n{"x": 2, "y": 3}\nthanks') == {
        "x": 2,
        "y": 3,
    }


def test_describe_space_shapes():
    desc = agent_mod._describe_space({"lr": (1e-3, 1.0), "n": (1, 8), "k": [1, 2]})
    assert desc["lr"]["type"] == "float"
    assert desc["n"]["type"] == "int"
    assert desc["k"] == {"type": "choice", "options": [1, 2]}


def test_describe_space_flags_a_log_range_as_log():
    desc = agent_mod._describe_space({"lr": LogRange(1e-5, 1e-1)})
    # A model shown a bare [1e-5, 1e-1] proposes evenly spaced values and
    # never tries the small decades — the same mistake uniform sampling makes.
    assert desc["lr"]["type"] == "float"
    assert desc["lr"]["scale"] == "log"
    assert desc["lr"]["range"] == [1e-5, 1e-1]


def test_log_range_is_still_a_plain_number_in_the_schema():
    schema = agent_mod._config_schema({"lr": LogRange(1e-5, 1e-1)})
    assert schema["properties"]["lr"] == {"type": "number"}


def test_history_row_carries_the_error_of_a_failed_run():
    failed = Run(
        run_id="0000",
        task="t:T",
        objective="score",
        direction="max",
        config={"x": 1.0},
        seed=0,
        metrics={},
        env={},
        created_at="2026-01-01T00:00:00+00:00",
        error="RuntimeError: CUDA out of memory",
    )
    row = agent_mod._history_row(failed)
    # Sent as bare empty metrics, the model could not tell "bad config" from
    # "crashed the trainer", and kept proposing into the region that crashes.
    assert row["failed"] is True
    assert "CUDA out of memory" in row["error"]


def test_history_row_of_a_good_run_is_unchanged():
    ok = Run(
        run_id="0001",
        task="t:T",
        objective="score",
        direction="max",
        config={"x": 1.0},
        seed=0,
        metrics={"score": 0.5},
        env={},
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert agent_mod._history_row(ok) == {
        "config": {"x": 1.0},
        "metrics": {"score": 0.5},
    }


def test_long_errors_are_truncated_in_the_history():
    noisy = Run(
        run_id="0002",
        task="t:T",
        objective="score",
        direction="max",
        config={},
        seed=0,
        metrics={},
        env={},
        created_at="2026-01-01T00:00:00+00:00",
        error="Traceback " + "x" * 5000,
    )
    # One stack trace must not crowd the rest of the history out of the prompt.
    assert len(agent_mod._history_row(noisy)["error"]) == agent_mod._ERROR_EXCERPT


def test_config_schema_pins_types_and_requires_every_param():
    schema = agent_mod._config_schema({"lr": (1e-3, 1.0), "n": (1, 8), "k": [1, 2]})
    assert schema["properties"]["lr"] == {"type": "number"}
    assert schema["properties"]["n"] == {"type": "integer"}
    assert schema["properties"]["k"] == {"enum": [1, 2]}
    assert schema["required"] == ["k", "lr", "n"]
    assert schema["additionalProperties"] is False


def test_batch_schema_wraps_configs_in_an_array():
    schema = agent_mod._batch_schema({"x": (0.0, 1.0)})
    assert schema["required"] == ["configs"]
    assert schema["properties"]["configs"]["type"] == "array"


def test_extract_configs_accepts_batch_array_and_bare_object():
    assert agent_mod._extract_configs('{"configs": [{"x": 1}, {"x": 2}]}') == [
        {"x": 1},
        {"x": 2},
    ]
    assert agent_mod._extract_configs('[{"x": 1}]') == [{"x": 1}]
    assert agent_mod._extract_configs('{"x": 1}') == [{"x": 1}]


# -- fake client -----------------------------------------------------------


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _FakeMessage:
    def __init__(self, text, usage):
        self.content = [_FakeBlock(text)]
        self.usage = usage


class _FakeMessages:
    """Returns *text*, or raises *error*, recording every request."""

    def __init__(self, text=None, error=None, usage=(100, 20)):
        self._text = text
        self._error = error
        self._usage = usage
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeMessage(self._text, _FakeUsage(*self._usage))


class _FakeClient:
    def __init__(self, messages):
        self.messages = messages


class _ApiError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


def _agent(text=None, error=None, **kwargs):
    return agent_mod.AnthropicAgent(
        client=_FakeClient(_FakeMessages(text=text, error=error)), **kwargs
    )


def test_anthropic_agent_parses_response():
    agent = _agent('{"configs": [{"x": 0.5, "k": 2}]}')
    out = agent.propose({"x": (0.0, 1.0), "k": [1, 2, 3]}, [], "score", "max")
    assert out == {"x": 0.5, "k": 2}
    assert agent.consecutive_failures == 0


def test_request_carries_schema_and_effort():
    agent = _agent('{"configs": [{"x": 0.5}]}', effort="low")
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    request = agent._client.messages.requests[0]
    output_config = request["output_config"]
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["format"]["schema"]["required"] == ["configs"]
    # temperature is rejected by current models — it must never be sent.
    assert "temperature" not in request


def test_structured_output_can_be_disabled():
    agent = _agent('{"configs": [{"x": 0.5}]}', structured_output=False)
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    assert "format" not in agent._client.messages.requests[0]["output_config"]


def test_anthropic_agent_falls_back_on_garbage():
    agent = _agent("sorry, no JSON here")
    out = agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    # Fell back to a random sample inside the space rather than crashing.
    assert 0.0 <= out["x"] <= 1.0
    assert agent.consecutive_failures == 1
    assert "ValueError" in agent.last_error


def test_transient_error_falls_back_and_counts():
    agent = _agent(error=_ApiError(529, "overloaded"))
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    assert agent.consecutive_failures == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_config_errors_raise_instead_of_degrading(status):
    agent = _agent(error=_ApiError(status))
    with pytest.raises(_ApiError):
        agent.propose({"x": (0.0, 1.0)}, [], "score", "max")


def test_short_batch_is_topped_up_not_discarded():
    agent = _agent('{"configs": [{"x": 0.25}]}')
    out = agent.propose_batch({"x": (0.0, 1.0)}, [], "score", "max", 3)
    assert len(out) == 3
    assert out[0] == {"x": 0.25}  # the model's proposal is kept
    assert all(0.0 <= c["x"] <= 1.0 for c in out)


def test_usage_and_cost_accumulate():
    agent = _agent('{"configs": [{"x": 0.5}]}', model="claude-opus-5")
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    assert agent.usage.calls == 2
    assert agent.usage.input_tokens == 200
    assert agent.usage.output_tokens == 40
    # 200 in @ $5/MTok + 40 out @ $25/MTok
    assert agent.cost_usd() == pytest.approx((200 * 5 + 40 * 25) / 1_000_000)


def test_failed_call_records_no_usage():
    agent = _agent(error=_ApiError(500))
    agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    assert agent.usage.calls == 0
    assert agent.cost_usd() == 0.0
