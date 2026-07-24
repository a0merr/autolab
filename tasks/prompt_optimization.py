"""Optimize a text-classification *pipeline* — the same loop, a non-tuning task.

This shows autolab driving something other than model hyperparameters: here the
agent tunes the knobs of a small sentiment classifier (case handling, negation
handling, intensifier weight, decision bias) to maximize accuracy on a fixed
labeled set.

It is deliberately dependency-free and deterministic so it runs in CI with no
API key — the "model" is a transparent lexicon classifier whose behavior really
does depend on the config, so the search has a genuine optimum to find.

To turn this into a real *prompt* optimization, keep the exact same structure
and replace ``_classify`` with a call to your LLM, mapping the config to prompt
knobs (number of few-shot examples, instruction wording, reasoning effort). The
Task interface does not change — only the body of ``run``.

    from autolab import Lab
    from tasks.prompt_optimization import SentimentPipeline

    best = Lab(SentimentPipeline(), objective="accuracy", budget=40).run()
    print(best.config, best.metrics)   # accuracy approaches 1.0
"""

from __future__ import annotations

import re

from autolab import Task

_POSITIVE = {"good", "great", "love", "excellent", "happy", "best", "wonderful"}
_NEGATIVE = {"bad", "terrible", "hate", "awful", "sad", "worst", "broken"}
_INTENSIFIERS = {"very", "really", "so"}
_NEGATORS = {"not", "no", "never", "nt"}

# (text, label) with label True = positive. Mix of plain, negated, intensified,
# and mixed-case cases so every knob matters for at least some examples.
_DATASET: list[tuple[str, bool]] = [
    ("This is good", True),
    ("This is great and I love it", True),
    ("The best, truly wonderful", True),
    ("very good experience", True),
    ("really excellent work", True),
    ("This is bad", False),
    ("absolutely terrible, I hate it", False),
    ("the worst, totally broken", False),
    ("not good at all", False),  # needs negation handling
    ("this is not bad", True),  # needs negation handling
    ("GREAT product", True),  # needs lowercasing
    ("very bad and awful", False),
]

_TOKEN = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str, lowercase: bool) -> list[str]:
    if lowercase:
        text = text.lower()
    return [t.replace("'", "") for t in _TOKEN.findall(text)]


def _classify(text: str, config: dict) -> bool:
    """A transparent lexicon classifier parameterized by the config."""
    tokens = _tokenize(text, config["lowercase"])
    score = 0.0
    negate = False
    intensify = 1.0
    for tok in tokens:
        low = tok.lower()
        polarity = 0.0
        if low in _POSITIVE:
            polarity = 1.0
        elif low in _NEGATIVE:
            polarity = -1.0

        if polarity != 0.0:
            if config["negation_handling"] and negate:
                polarity = -polarity
            score += polarity * intensify
            negate = False
            intensify = 1.0
            continue

        if config["negation_handling"] and low in _NEGATORS:
            negate = True
        if low in _INTENSIFIERS:
            intensify = 1.0 + config["intensifier_weight"]

    return score > config["bias"]


class SentimentPipeline(Task):
    """Tune a lexicon sentiment classifier to maximize labeling accuracy."""

    def propose_space(self):
        return {
            "lowercase": [False, True],
            "negation_handling": [False, True],
            "intensifier_weight": (0.0, 2.0),
            "bias": (-1.0, 1.0),
        }

    def run(self, config, seed):
        correct = sum(_classify(text, config) == label for text, label in _DATASET)
        accuracy = correct / len(_DATASET)
        return {"accuracy": accuracy, "n_correct": float(correct)}
