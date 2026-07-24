"""Best-effort global seeding for reproducible experiments.

Seeds Python's ``random`` always, and ``numpy`` / ``torch`` if they are
installed. Task authors who use other sources of randomness should seed them
inside ``Task.run`` using the ``seed`` they are handed.

Two limits worth knowing, because "reproduced exactly" is a claim this project
makes and neither of these is fixable from in here:

* **Hash randomization is fixed at interpreter startup.** Setting
  ``PYTHONHASHSEED`` from running code does nothing — this module used to do
  exactly that, which read as a determinism guarantee it was not providing. If
  your task's results depend on ``set`` or ``dict`` iteration order over
  strings, set ``PYTHONHASHSEED`` in the environment *before* launching.
* **GPU determinism needs more than a seed.** cuDNN autotuning is disabled
  below, but some CUDA kernels are non-deterministic by construction. For a
  hard guarantee add ``torch.use_deterministic_algorithms(True)`` and
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in your own task — not done here because
  it makes several common ops raise instead of running.
"""

from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Seed all RNGs we know about with *seed*."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Autotuning picks a different algorithm depending on what else the
        # machine is doing, so the same run can produce different numbers on
        # the same hardware. Off by default here: replay is the point.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
