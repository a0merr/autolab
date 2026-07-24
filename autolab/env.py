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


def _git(*args: str) -> str | None:
    """Run a git command, or return ``None`` if git can't answer."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=2,
            # Detached from any console: a search started by cron or a service
            # manager may have no usable stdin to inherit, and on Windows
            # inheriting an invalid handle fails the spawn outright.
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _git_commit() -> str | None:
    out = _git("rev-parse", "HEAD")
    return out.strip() if out and out.strip() else None


def _git_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes, if git can tell.

    A commit hash on its own overstates what was captured: the code that ran
    is the commit *plus* whatever was uncommitted, and a run recorded against
    a clean-looking hash cannot be reconstructed from it. Recording the flag
    is cheap; discovering the omission from a failed reproduction is not.
    """
    out = _git("status", "--porcelain")
    return None if out is None else bool(out.strip())


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
        dirty = _git_dirty()
        if dirty is not None:
            snapshot["git_dirty"] = dirty
    return snapshot
