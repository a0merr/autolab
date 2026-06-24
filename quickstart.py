"""Zero-config demo of the autolab loop — no API key, no training, no GPU.

Run it straight from a fresh clone::

    python quickstart.py

It drives the toy :class:`Quadratic` task with the built-in ``RandomAgent``,
prints the search report (including the derivation tree), then proves the best
run replays to the exact same metric. This is the 30-second "it works" tour
before you wire in a real task or the Anthropic agent.
"""

from __future__ import annotations

import tempfile

from autolab import Lab, RandomAgent, replay
from tasks.quadratic import Quadratic


def main() -> None:
    # Use a throwaway store so the demo leaves nothing behind.
    with tempfile.TemporaryDirectory() as store_dir:
        lab = Lab(
            task=Quadratic(),
            objective="score",  # maximize; true peak is 0 at (x=2, y=-3)
            budget=25,
            agent=RandomAgent(seed=0),  # deterministic, no API key needed
            store=store_dir,
        )

        best = lab.run()
        print()
        lab.report()

        # Reproducibility is the whole point: re-run the best config + seed.
        result = replay(best)
        print()
        print(
            f"replay of best run {best.run_id}: "
            f"{'reproduced exactly' if result.reproduced else 'MISMATCH'}"
        )


if __name__ == "__main__":
    main()
