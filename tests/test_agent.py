from __future__ import annotations

import random

from autolab import RandomAgent
from autolab import agent as agent_mod


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


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeMessage(self._text)


def _make_anthropic_agent(text):
    """Build an AnthropicAgent without importing the real SDK."""
    agent = agent_mod.AnthropicAgent.__new__(agent_mod.AnthropicAgent)
    agent._client = type("C", (), {"messages": _FakeMessages(text)})()
    agent._model = "test"
    agent._max_tokens = 64
    agent._history_window = 10
    agent._fallback = RandomAgent(seed=0)
    return agent


def test_anthropic_agent_parses_response():
    agent = _make_anthropic_agent('{"x": 0.5, "k": 2}')
    out = agent.propose({"x": (0.0, 1.0), "k": [1, 2, 3]}, [], "score", "max")
    assert out == {"x": 0.5, "k": 2}


def test_anthropic_agent_falls_back_on_garbage():
    agent = _make_anthropic_agent("sorry, no JSON here")
    out = agent.propose({"x": (0.0, 1.0)}, [], "score", "max")
    # Fell back to a random sample inside the space rather than crashing.
    assert 0.0 <= out["x"] <= 1.0
