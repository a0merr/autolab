from __future__ import annotations

import subprocess

from autolab import env


def test_capture_has_the_basics():
    snapshot = env.capture()
    assert snapshot["python"]
    assert snapshot["platform"]
    assert isinstance(snapshot["libraries"], dict)


def test_dirty_tree_is_recorded_alongside_the_commit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("one", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "first")

    monkeypatch.chdir(repo)
    clean = env.capture()
    assert clean["git_commit"]
    assert clean["git_dirty"] is False

    (repo / "f.txt").write_text("two", encoding="utf-8")
    dirty = env.capture()
    # Same commit, different code. Recording only the hash claims a
    # reproducibility the run does not have.
    assert dirty["git_commit"] == clean["git_commit"]
    assert dirty["git_dirty"] is True


def test_no_git_no_claim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    snapshot = env.capture()
    assert "git_commit" not in snapshot
    assert "git_dirty" not in snapshot


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
