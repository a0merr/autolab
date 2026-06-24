from __future__ import annotations

import tempfile

from autolab import Lab, RandomAgent, replay
from tasks.prompt_optimization import SentimentPipeline


def test_search_finds_a_strong_config():
    best = Lab(
        SentimentPipeline(),
        objective="accuracy",
        budget=60,
        agent=RandomAgent(seed=1),
        store=tempfile.mkdtemp(),
    ).run()
    # The dataset is solvable; a 60-draw search should reach perfect accuracy.
    assert best.metrics["accuracy"] == 1.0


def test_knobs_actually_matter():
    task = SentimentPipeline()
    good = {
        "lowercase": True,
        "negation_handling": True,
        "intensifier_weight": 1.3,
        "bias": 0.5,
    }
    naive = {
        "lowercase": False,
        "negation_handling": False,
        "intensifier_weight": 0.0,
        "bias": 0.5,
    }
    assert task.run(good, 0)["accuracy"] > task.run(naive, 0)["accuracy"]


def test_run_is_deterministic_and_replayable():
    lab = Lab(
        SentimentPipeline(),
        objective="accuracy",
        budget=5,
        agent=RandomAgent(seed=0),
        store=tempfile.mkdtemp(),
    )
    lab.run()
    for run in lab.store.list():
        assert replay(run).reproduced
