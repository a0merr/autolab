from __future__ import annotations

import tempfile

from autolab import Lab, RandomAgent, replay
from tasks.feature_selection import (
    INFORMATIVE,
    N_FEATURES,
    FeatureSelection,
    _accuracy,
)


def test_informative_subset_beats_select_all():
    informative = _accuracy(list(INFORMATIVE))
    all_on = _accuracy(list(range(N_FEATURES)))
    assert informative > all_on  # noise features only hurt


def test_search_recovers_informative_features():
    best = Lab(
        FeatureSelection(),
        objective="accuracy",
        budget=80,
        agent=RandomAgent(seed=2),
        store=tempfile.mkdtemp(),
    ).run()
    selected = {int(k.split("_")[1]) for k, v in best.config.items() if v}
    # The search must actually recover the informative features.
    assert set(INFORMATIVE).issubset(selected)
    assert best.metrics["accuracy"] > 0.9


def test_empty_selection_is_chance_level():
    blind = _accuracy([])
    assert 0.4 <= blind <= 0.6


def test_runs_replay_exactly():
    lab = Lab(
        FeatureSelection(),
        objective="accuracy",
        budget=5,
        agent=RandomAgent(seed=0),
        store=tempfile.mkdtemp(),
    )
    lab.run()
    for run in lab.store.list():
        assert replay(run).reproduced
