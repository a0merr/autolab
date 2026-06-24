"""Scripted demo for recording the README GIF — paced, captioned, one take.

Run it and just let it play; it prints captions and pauses between beats so a
screen recorder captures a readable sequence. No API key required.

    python scripts/demo.py

Beats:
    1. agent-driven search over a real-ish task
    2. the report: best run, mean, and the derivation tree
    3. replay: the same config + seed reproduces the metric exactly

Set DEMO_FAST=1 to remove the pauses (for a CI smoke test).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autolab import Lab, RandomAgent, replay  # noqa: E402
from tasks.prompt_optimization import SentimentPipeline  # noqa: E402

PAUSE = 0.0 if os.environ.get("DEMO_FAST") else 1.6


def beat(text: str) -> None:
    print(f"\n\033[1;36m# {text}\033[0m")
    time.sleep(PAUSE)


def main() -> None:
    beat("autolab: an agent tunes a text-classification pipeline (no API key)")
    lab = Lab(
        task=SentimentPipeline(),
        objective="accuracy",
        budget=30,
        agent=RandomAgent(seed=1),
        store=tempfile.mkdtemp(),
    )
    best = lab.run()
    time.sleep(PAUSE)

    beat("the search report — best run, mean, and the derivation tree")
    lab.report()
    time.sleep(PAUSE)

    beat("reproducibility: replay the best run from its config + seed")
    result = replay(best)
    status = "reproduced exactly" if result.reproduced else "MISMATCH"
    print(f"replay {best.run_id}: \033[1;32m{status}\033[0m")
    print(f"  accuracy = {best.metrics['accuracy']:.3f}")
    time.sleep(PAUSE)


if __name__ == "__main__":
    main()
