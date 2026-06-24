"""Best-effort global seeding for reproducible experiments.

Seeds Python's ``random`` always, and ``numpy`` / ``torch`` if they are
installed. Task authors who use other sources of randomness should seed them
inside ``Task.run`` using the ``seed`` they are handed.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed all RNGs we know about with *seed*."""
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    except ImportError:
        pass
