from __future__ import annotations

from autolab import Lab, RandomAgent
from autolab.cli import main
from autolab.store import RunStore
from tasks.quadratic import Quadratic


def _seed_store(store_dir):
    lab = Lab(
        Quadratic(),
        objective="score",
        budget=6,
        agent=RandomAgent(seed=4),
        store=RunStore(store_dir),
    )
    lab.run()
    return str(store_dir)


def test_runs_list(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["runs", "list", "--store", path]) == 0
    out = capsys.readouterr().out
    assert out.count("score=") == 6


def test_runs_show(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["runs", "show", "0000", "--store", path]) == 0
    out = capsys.readouterr().out
    assert '"score"' in out
    assert '"config"' in out


def test_report(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["report", "--store", path]) == 0
    out = capsys.readouterr().out
    assert "best:" in out
    assert "search tree:" in out


def test_replay_exit_code_zero(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["replay", "0000", "--store", path]) == 0
    assert "reproduced" in capsys.readouterr().out


def test_list_empty(store_dir, capsys):
    assert main(["runs", "list", "--store", str(store_dir)]) == 0
    assert "(no runs)" in capsys.readouterr().out
