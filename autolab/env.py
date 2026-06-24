"""Capture a snapshot of the execution environment for honest comparison.

Two runs are only comparable if they ran under comparable conditions. We
record the Python version, platform, a handful of relevant library versions,
and the git commit if the working tree is a repo.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata

# Libraries worth recording when present. Kept short on purpose — the point is
# honest comparison, not an exhaustive freeze.
_TRACKED = ("numpy", "torch", "scikit-learn", "transformers", "anthropic")


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def capture() -> dict[str, object]:
    """Return a JSON-serializable snapshot of the current environment."""
    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
        "libraries": _lib_versions(),
    }
    commit = _git_commit()
    if commit:
        snapshot["git_commit"] = commit
    return snapshot
