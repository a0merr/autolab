"""Command-line interface: inspect, replay, and report on stored runs.

autolab runs list                 # every experiment, newest first
autolab runs show <run_id>        # full config + metrics for one run
autolab replay <run_id>           # re-execute it exactly
autolab report                    # summary of the search
autolab status                    # how the last search ended (exit 1 if not cleanly)
"""

from __future__ import annotations

import argparse
import json
import sys

from . import analysis
from .replay import replay as replay_run
from .store import DEFAULT_STORE_DIR, SEARCH_NOTE, RunStore


def _add_store_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--store",
        default=DEFAULT_STORE_DIR,
        help=f"run store directory (default: {DEFAULT_STORE_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autolab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("runs", help="inspect stored runs")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)

    runs_list = runs_sub.add_parser("list", help="list every run, newest first")
    _add_store_arg(runs_list)

    runs_show = runs_sub.add_parser("show", help="show one run in full")
    runs_show.add_argument("run_id")
    _add_store_arg(runs_show)

    replay_p = sub.add_parser("replay", help="re-execute a run exactly")
    replay_p.add_argument("run_id")
    _add_store_arg(replay_p)

    report_p = sub.add_parser("report", help="summarize the search")
    report_p.add_argument(
        "--objective",
        help="objective metric (default: inferred from the latest run)",
    )
    _add_store_arg(report_p)

    status_p = sub.add_parser(
        "status", help="how the last search over this store ended"
    )
    _add_store_arg(status_p)

    return parser


def _infer_objective(store: RunStore, override: str | None) -> tuple[str, str]:
    if override:
        runs = store.list()
        direction = runs[0].direction if runs else "max"
        return override, direction
    runs = store.list()
    if not runs:
        raise SystemExit("no runs in store; nothing to report")
    return runs[0].objective, runs[0].direction


def cmd_runs_list(args: argparse.Namespace) -> int:
    store = RunStore(args.store)
    runs = store.list()
    if not runs:
        print("(no runs)")
        return 0
    for r in runs:
        score = r.metrics.get(r.objective)
        score_str = f"{r.objective}={score:.6g}" if score is not None else "-"
        print(f"{r.run_id}  {r.created_at}  {score_str}  {r.task}")
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    store = RunStore(args.store)
    try:
        run = store.get(args.run_id)
    except KeyError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    store = RunStore(args.store)
    try:
        run = store.get(args.run_id)
    except KeyError as exc:
        raise SystemExit(str(exc))
    result = replay_run(run)
    print(result)
    return 0 if result.reproduced else 1


def _print_status_banner(store: RunStore) -> None:
    """Warn before the summary if the search did not finish cleanly.

    A tripped search still produces a plausible-looking report — fewer runs
    than the budget, a best config, no hint that it stopped early. The banner
    is what stops that being read as a finished search.
    """
    note = store.read_note(SEARCH_NOTE)
    if not note or note.get("status") == "completed":
        return
    line = f"!! search {note.get('status', 'unknown')}"
    if note.get("reason"):
        line += f": {note['reason']}"
    print(line)
    print("   (autolab status for the full record)")


def cmd_status(args: argparse.Namespace) -> int:
    """Print the search note. Exit code is 0 only for a clean finish, so cron
    can tell a completed overnight run from a tripped one."""
    store = RunStore(args.store)
    note = store.read_note(SEARCH_NOTE)
    if note is None:
        print("(no search on record in this store)")
        return 1
    print(json.dumps(note, indent=2, sort_keys=True))
    return 0 if note.get("status") == "completed" else 1


def cmd_report(args: argparse.Namespace) -> int:
    store = RunStore(args.store)
    _print_status_banner(store)
    objective, direction = _infer_objective(store, args.objective)
    runs = store.list(newest_first=False)
    s = analysis.summarize(runs, objective, direction)
    print(
        f"runs: {s.n_runs} ({s.n_scored} scored)  "
        f"objective: {objective} ({direction})"
    )
    if s.best_score is None:
        print("(no scored runs)")
        return 0
    print(f"best: run {s.best_run_id}  {objective}={s.best_score:.6g}")
    print(f"best config: {json.dumps(s.best_config)}")
    print(f"mean: {s.mean_score:.6g}  std: {s.std_score:.6g}")
    tree = analysis.search_tree(runs)
    if tree:
        print("search tree:")
        print(tree)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "runs":
        if args.runs_command == "list":
            return cmd_runs_list(args)
        if args.runs_command == "show":
            return cmd_runs_show(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "status":
        return cmd_status(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
