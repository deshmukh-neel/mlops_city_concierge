#!/usr/bin/env python3
"""Stale-baseline CI lint (Plan 03-07 / EVAL-07 / P9 mitigation).

Hard-fails PRs that touch ``app/agent/`` without refreshing
``configs/eval_baselines/*.json`` — unless the latest commit message carries
an explicit ``[skip-baseline]`` bypass token.

The script intentionally uses only the standard library
(``subprocess``, ``sys``, ``pathlib``, ``argparse``, ``os``, ``re``) so the
``lint-baselines`` GitHub Actions job can invoke it BEFORE installing
Python dependencies — the lint step's dependency surface is zero.

CI invocation::

    poetry run python scripts/check_baselines_fresh.py ${{ github.event.pull_request.base.sha }}

Local invocation (defaults to ``origin/main`` as the diff base)::

    python scripts/check_baselines_fresh.py
    python scripts/check_baselines_fresh.py origin/main
    python scripts/check_baselines_fresh.py --merge-base origin/main

Truth table (per plan 03-07 task 1 <behavior>):

    | agent_changed | baselines_changed | [skip-baseline] | exit |
    |     T         |        F          |       F         |  1   |  ← stale
    |     T         |        T          |       F         |  0   |  ← updated
    |     F         |        *          |       *         |  0   |  ← no agent change
    |     T         |        F          |       T         |  0   |  ← explicit bypass
"""

from __future__ import annotations

import argparse
import subprocess  # noqa: S404 - this is the script; subprocess is the whole point
import sys
from collections.abc import Sequence

AGENT_PREFIX = "app/agent/"
BASELINES_PREFIX = "configs/eval_baselines/"
BASELINE_SUFFIX = ".json"
SKIP_BASELINE_TOKEN = "[skip-baseline]"  # noqa: S105 - bypass marker, not a credential
DEFAULT_BASE = "origin/main"


def _run_git(args: list[str]) -> str:
    """Run ``git`` with ``args`` and return stdout. Tests monkeypatch this.

    The single seam every test patches — keeping it tiny means we don't need
    to mock ``subprocess.run`` directly (which would also intercept the
    test runner's own subprocess calls). Returns an empty string on failure
    so a missing/shallow ``origin/main`` ref doesn't crash the gate; a
    follow-on dev-ergonomics improvement could surface git errors loudly.
    """
    # cmd is a closed allowlist (literal "git" + caller-supplied diff/log
    # argv we construct ourselves below) — no shell, no untrusted input.
    cmd = ["git", *args]
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def _changed_paths(base_sha: str) -> set[str]:
    """Return the set of paths that changed between ``base_sha`` and HEAD.

    Uses the three-dot ``BASE...HEAD`` syntax (merge-base diff) so the
    output reflects only what the PR introduced, not unrelated mainline
    commits that landed since the branch forked.
    """
    raw = _run_git(["diff", "--name-only", f"{base_sha}...HEAD"])
    return {line for line in raw.splitlines() if line}


def _last_commit_message() -> str:
    """Return the full body (subject + body) of HEAD."""
    return _run_git(["log", "-1", "--format=%B"])


def _agent_changed(paths: set[str]) -> list[str]:
    """Return changed paths under ``app/agent/`` (sorted for deterministic msg)."""
    return sorted(p for p in paths if p.startswith(AGENT_PREFIX))


def _baselines_changed(paths: set[str]) -> list[str]:
    """Return changed ``configs/eval_baselines/*.json`` paths.

    Non-``.json`` files under the baselines dir (e.g. a stray README) do
    NOT count — only the per-scenario baseline JSONs carry the scorer
    medians the Phase 4-6 merge rule diffs against.
    """
    return sorted(
        p for p in paths if p.startswith(BASELINES_PREFIX) and p.endswith(BASELINE_SUFFIX)
    )


def _format_stale_error(agent_paths: list[str]) -> str:
    """Render the actionable error message for the stale-baseline gate."""
    lines = [
        "ERROR: lint-baselines gate (Plan 03-07 / EVAL-07 / P9):",
        "",
        "  The following app/agent/ files changed in this PR but no",
        "  configs/eval_baselines/*.json was updated to match:",
        "",
    ]
    for path in agent_paths:
        lines.append(f"    - {path}")
    lines += [
        "",
        "  Remediation (one of):",
        "",
        "    1. Re-run the matrix locally and commit the refreshed baselines:",
        "         APP_ENV=eval make eval-matrix RUNS=3",
        "         # post-process eval_reports/{ts}/summary.json into:",
        "         #   configs/eval_baselines/omakase_mission_open_ended.json",
        "         #   configs/eval_baselines/refinement_cheaper.json",
        "         #   configs/eval_baselines/late_night_closure_cascade.json",
        "",
        "    2. Add the explicit bypass to the latest commit message if this",
        "       PR genuinely does not warrant a baseline refresh (rare):",
        "         [skip-baseline]",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse argv. Both positional BASE_SHA and ``--merge-base`` are accepted."""
    parser = argparse.ArgumentParser(
        prog="check_baselines_fresh",
        description=(
            "Stale-baseline CI lint (Plan 03-07). Exits 1 when app/agent/ "
            "changed without a configs/eval_baselines/*.json refresh, unless "
            "the latest commit message carries the [skip-baseline] bypass."
        ),
    )
    parser.add_argument(
        "base_sha",
        nargs="?",
        default=None,
        help=("Diff base (defaults to origin/main; CI passes github.event.pull_request.base.sha)."),
    )
    parser.add_argument(
        "--merge-base",
        dest="merge_base",
        default=None,
        help="Alternative to positional BASE_SHA; useful in scripted contexts.",
    )
    return parser.parse_args(list(argv))


def _resolve_base(args: argparse.Namespace) -> str:
    """Pick the effective diff base from positional/flag/default precedence."""
    if args.merge_base:
        return str(args.merge_base)
    if args.base_sha:
        return str(args.base_sha)
    return DEFAULT_BASE


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the script's exit code (0 = pass, 1 = stale)."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    base = _resolve_base(args)

    paths = _changed_paths(base)
    agent_paths = _agent_changed(paths)
    baseline_paths = _baselines_changed(paths)
    commit_msg = _last_commit_message()
    bypass_used = SKIP_BASELINE_TOKEN in commit_msg

    # Branch 3: no agent change at all → trivially pass.
    if not agent_paths:
        print(
            f"check_baselines_fresh: OK — no app/agent/ changes vs {base} "
            f"({len(paths)} paths changed total)"
        )
        return 0

    # Branch 4: explicit bypass — pass with a loud notice so reviewers see it.
    if bypass_used:
        print(
            "check_baselines_fresh: WARNING — [skip-baseline] bypass used. "
            "app/agent/ changed but baselines were NOT refreshed:"
        )
        for path in agent_paths:
            print(f"  - {path}")
        print(
            "  Reviewer: confirm the agent change is behavior-preserving "
            "(refactor/rename) or scoped narrowly enough to leave baselines untouched."
        )
        return 0

    # Branch 2: agent changed AND baseline refreshed → pass.
    if baseline_paths:
        print(
            f"check_baselines_fresh: OK — app/agent/ changed and "
            f"{len(baseline_paths)} baseline file(s) refreshed:"
        )
        for path in baseline_paths:
            print(f"  - {path}")
        return 0

    # Branch 1: agent changed, NO baseline refresh, NO bypass → HARD FAIL.
    sys.stderr.write(_format_stale_error(agent_paths))
    sys.stderr.write("\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
