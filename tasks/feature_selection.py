"""Select an informative feature subset — same loop, a combinatorial task.

The agent flips one boolean per candidate feature; the objective is the
accuracy of a simple linear classifier that uses only the selected features.
The synthetic dataset is built so that a handful of features carry the signal
and the rest are pure noise — so the search has a real target: keep the
informative features, drop the noise (which only adds variance and *lowers*
accuracy).

Like the other examples it is deterministic and dependency-free, so it runs in
CI with no API key. It shows autolab driving a *combinatorial* search (subset
selection), not just continuous/categorical hyperparameters — same `Task`
interface, nothing else changes.

    from autolab import Lab
    from tasks.feature_selection import FeatureSelection

    best = Lab(FeatureSelection(), objective="accuracy", budget=60).run()
    print(best.config, best.metrics)   # selects the informative features
"""

from __future__ import annotations

import random

from autolab import Task

N_FEATURES = 8
N_SAMPLES = 240
INFORMATIVE = (0, 1, 2)  # the only features correlated with the label


def _build_dataset() -> tuple[list[list[float]], list[int]]:
    """Deterministically synthesize (X, y).

    Informative features correlate with the label; the rest are zero-mean
    noise. Built once at import with a fixed seed so every run sees identical
    data — the task's randomness comes from feature *selection*, not the data.
    """
    rng = random.Random(0)
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(N_SAMPLES):
        label = 1 if i % 2 == 0 else -1  # balanced
        row = []
        for f in range(N_FEATURES):
            if f in INFORMATIVE:
                row.append(label * 1.0 + rng.gauss(0.0, 0.6))
            else:
                row.append(rng.gauss(0.0, 1.0))
        X.append(row)
        y.append(label)
    return X, y


_X, _Y = _build_dataset()


def _selected_indices(config: dict) -> list[int]:
    return [f for f in range(N_FEATURES) if config[f"feature_{f}"]]


def _accuracy(selected: list[int]) -> float:
    if not selected:
        # No features → the classifier is blind; predicts the +1 class.
        return sum(1 for label in _Y if label == 1) / len(_Y)
    correct = 0
    for row, label in zip(_X, _Y):
        score = sum(row[f] for f in selected)
        prediction = 1 if score >= 0.0 else -1
        correct += prediction == label
    return correct / len(_Y)


class FeatureSelection(Task):
    """Choose the feature subset that maximizes classifier accuracy."""

    def propose_space(self):
        # One on/off switch per candidate feature.
        return {f"feature_{f}": [False, True] for f in range(N_FEATURES)}

    def run(self, config, seed):
        selected = _selected_indices(config)
        return {
            "accuracy": _accuracy(selected),
            "n_features": float(len(selected)),
        }
