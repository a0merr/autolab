"""A pure-Python toy task: find the peak of a quadratic surface.

No dependencies, no training, fully deterministic — the simplest possible
demonstration of the propose → run → analyze loop, and a good smoke test for
the store and replay.

    from autolab import Lab
    from tasks.quadratic import Quadratic

    lab = Lab(Quadratic(), objective="score", budget=30)
    best = lab.run()
    print(best.config, best.metrics)   # config near {"x": 2, "y": -3}
"""

from __future__ import annotations

from autolab import Task

# The (hidden) optimum the search should rediscover.
_PEAK = {"x": 2.0, "y": -3.0}


class Quadratic(Task):
    """Maximize ``score = -((x-2)^2 + (y+3)^2)``; peak is 0 at (2, -3)."""

    def propose_space(self):
        return {
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        }

    def run(self, config, seed):
        dx = config["x"] - _PEAK["x"]
        dy = config["y"] - _PEAK["y"]
        return {"score": -(dx * dx + dy * dy)}
