from __future__ import annotations

import pytest

from autolab import BreakerTripped, CircuitBreaker, Lab, RandomAgent
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


def _seed_tripped_store(store_dir):
    lab = Lab(
        __import__("tests.conftest", fromlist=["CrashTask"]).CrashTask(),
        objective="score",
        budget=50,
        agent=RandomAgent(seed=0),
        store=RunStore(store_dir),
        breaker=CircuitBreaker(max_consecutive_errors=2),
    )
    with pytest.raises(BreakerTripped):
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


# -- status ----------------------------------------------------------------


def test_status_after_a_clean_search(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["status", "--store", path]) == 0
    assert '"completed"' in capsys.readouterr().out


def test_status_exits_nonzero_when_the_search_tripped(store_dir, capsys):
    path = _seed_tripped_store(store_dir)
    # Non-zero so a cron wrapper can tell a finished overnight run from a
    # stopped one without parsing anything.
    assert main(["status", "--store", path]) == 1
    out = capsys.readouterr().out
    assert '"tripped"' in out
    assert "failed in a row" in out


def test_status_with_no_search_on_record(store_dir, capsys):
    assert main(["status", "--store", str(store_dir)]) == 1
    assert "no search on record" in capsys.readouterr().out


def test_report_warns_that_the_search_did_not_finish(store_dir, capsys):
    path = _seed_tripped_store(store_dir)
    assert main(["report", "--store", path]) == 0
    # Without the banner the report reads as a small but complete search.
    assert "!! search tripped" in capsys.readouterr().out


def test_report_is_quiet_after_a_clean_search(store_dir, capsys):
    path = _seed_store(store_dir)
    assert main(["report", "--store", path]) == 0
    assert "!!" not in capsys.readouterr().out
