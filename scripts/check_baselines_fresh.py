#!/usr/bin/env python3
"""Stale-baseline CI lint (Plan 03-07 / EVAL-07 / P9 mitigation).

Hard-fails PRs that touch any file in the watch-set without refreshing
``configs/eval_baselines/*.json`` — unless the latest commit message carries
an explicit ``[skip-baseline]`` bypass token.

Watch-set (D-11-21 / BASE-04): ``app/agent/``, ``app/llm_factory.py``,
and ``configs/eval_matrix`` (bare prefix matching both
``configs/eval_matrix.yaml`` and ``configs/eval_matrix_refinement.yaml``).

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

    | watch_set_changed | baselines_changed | [skip-baseline] | exit |
    |        T          |        F          |       F         |  1   |  ← stale
    |        T          |        T          |       F         |  0   |  ← updated
    |        F          |        *          |       *         |  0   |  ← no watch-set change
    |        T          |        F          |       T         |  0   |  ← explicit bypass

Exit code conventions (Plan 03-10 / WR-02 hardening):

    | rc | meaning                                                          |
    |  0 | gate passed (one of the four truth-table pass branches above)    |
    |  1 | gate failed: stale baselines (rule-1 violation)                  |
    |  2 | infrastructure failure (RuntimeError from run_git: rc != 0,     |
    |    | missing git binary, or explicit empty-string BASE_SHA)           |
"""

from __future__ import annotations

import argparse
import re
import subprocess  # noqa: S404 - this is the script; subprocess is the whole point
import sys
from collections.abc import Sequence

# D-11-21 / BASE-04: extended watch-set.
# - app/agent/         : agent graph, critique scorers, state, tools
# - app/llm_factory.py : provider branches, thinking policies, temperature clamps
# - configs/eval_matrix: bare prefix matches both eval_matrix.yaml and
#                        eval_matrix_refinement.yaml (entries directly change
#                        which models are measured)
WATCH_PREFIXES = [
    "app/agent/",
    "app/llm_factory.py",
    "configs/eval_matrix",
]
BASELINES_PREFIX = "configs/eval_baselines/"
BASELINE_SUFFIX = ".json"
SKIP_BASELINE_TOKEN = "[skip-baseline]"  # noqa: S105 - bypass marker, not a credential
# IN-04: tighten the bypass match so a documentation PR that quotes the token
# mid-sentence (e.g. "docs: explain the [skip-baseline] bypass token") cannot
# accidentally trip the gate. The token must appear at a line boundary (start
# of message or after a newline), optionally preceded by whitespace, and must
# be followed by whitespace or end-of-string — i.e. trailer-style placement.
SKIP_BASELINE_RE = re.compile(r"(^|\n)\s*\[skip-baseline\](\s|$)")
DEFAULT_BASE = "origin/main"


def run_git(args: list[str]) -> str:
    """Run ``git`` with ``args`` and return stdout.

    Raises ``RuntimeError`` on non-zero git exit or a missing ``git`` binary
    (Plan 03-10 / WR-02). The truth-table tests monkeypatch this function
    wholesale (see ``tests/unit/test_check_baselines_fresh.py``) so the new
    raise paths only affect production callers; the WR-02 regression tests
    patch ``subprocess.run`` one level deeper to exercise the raise paths
    themselves.
    """
    # cmd is a closed allowlist (literal "git" + caller-supplied diff/log
    # argv we construct ourselves below) — no shell, no untrusted input.
    cmd = ["git", *args]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        # WR-02: missing git binary used to bubble a confusing FileNotFoundError
        # traceback. Convert to an actionable RuntimeError so an operator sees
        # "install git" rather than a Python-internal trace.
        raise RuntimeError(
            f"git binary not found on PATH; install git or fix the CI image (original error: {exc})"
        ) from exc

    if result.returncode != 0:
        # WR-02: rc != 0 used to be swallowed (we'd return empty stdout and
        # silently pass the gate when origin/main was missing or BASE_SHA was
        # bogus). Loud-fail with rc + argv + stderr so the operator can diagnose.
        argv_str = " ".join(["git", *args])
        stderr_clean = result.stderr.strip()
        raise RuntimeError(f"{argv_str} failed (rc={result.returncode}): {stderr_clean}")

    return result.stdout


def changed_paths(base_sha: str) -> set[str]:
    """Return the set of paths that changed between ``base_sha`` and HEAD.

    Uses the three-dot ``BASE...HEAD`` syntax (merge-base diff) so the
    output reflects only what the PR introduced, not unrelated mainline
    commits that landed since the branch forked.
    """
    raw = run_git(["diff", "--name-only", f"{base_sha}...HEAD"])
    return {line for line in raw.splitlines() if line}


def last_commit_message() -> str:
    """Return the full body (subject + body) of HEAD."""
    return run_git(["log", "-1", "--format=%B"])


def agent_changed(paths: set[str]) -> list[str]:
    """Return changed paths under any watch-set prefix (sorted for determinism).

    Watch-set (D-11-21 / BASE-04): ``app/agent/``, ``app/llm_factory.py``,
    and ``configs/eval_matrix`` (bare prefix matching both eval_matrix*.yaml).
    """
    return sorted(p for p in paths if any(p.startswith(prefix) for prefix in WATCH_PREFIXES))


def baselines_changed(paths: set[str]) -> list[str]:
    """Return changed ``configs/eval_baselines/*.json`` paths.

    Non-``.json`` files under the baselines dir (e.g. a stray README) do
    NOT count — only the per-scenario baseline JSONs carry the scorer
    medians the Phase 4-6 merge rule diffs against.
    """
    return sorted(
        p for p in paths if p.startswith(BASELINES_PREFIX) and p.endswith(BASELINE_SUFFIX)
    )


def format_stale_error(agent_paths: list[str]) -> str:
    """Render the actionable error message for the stale-baseline gate."""
    lines = [
        "ERROR: lint-baselines gate (Plan 03-07 / EVAL-07 / P9 / D-11-21):",
        "",
        "  The following watch-set files changed in this PR but no",
        "  configs/eval_baselines/*.json was updated to match:",
        "  (watch-set: app/agent/, app/llm_factory.py, configs/eval_matrix*)",
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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse argv. Both positional BASE_SHA and ``--merge-base`` are accepted."""
    parser = argparse.ArgumentParser(
        prog="check_baselines_fresh",
        description=(
            "Stale-baseline CI lint (Plan 03-07 / D-11-21). Exits 1 when any "
            "watch-set file (app/agent/, app/llm_factory.py, configs/eval_matrix*) "
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


def resolve_base(args: argparse.Namespace) -> str:
    """Pick the effective diff base from positional/flag/default precedence.

    WR-02: empty-string positional BASE_SHA used to be falsy and silently
    fell back to ``origin/main``. That accidental robustness hid CI workflow
    misconfiguration (e.g. ``${{ github.event.pull_request.base.sha }}``
    emitting "" on a ``workflow_dispatch`` event where pull_request context
    is absent). Now an explicit empty string raises so the workflow is
    surfaced as broken instead of trivially passing.
    """
    if args.merge_base is not None:
        if args.merge_base == "":
            raise RuntimeError(
                "--merge-base was the empty string; pass a real ref or omit "
                "the flag to use origin/main"
            )
        return str(args.merge_base)
    if args.base_sha is not None:
        if args.base_sha == "":
            raise RuntimeError(
                "BASE_SHA positional argument was the empty string; pass a real "
                "SHA or omit the argument to use origin/main"
            )
        return str(args.base_sha)
    return DEFAULT_BASE


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the script's exit code.

    Exit codes:
        0 = gate passed (one of the four truth-table pass branches)
        1 = gate failed: stale baselines (rule-1 violation)
        2 = infrastructure failure (RuntimeError from run_git / resolve_base)
    """
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # WR-02: surface infrastructure failures (missing git, rc != 0 from git,
    # empty BASE_SHA) as rc=2 with an actionable stderr message — distinct
    # from rc=1 stale-baseline failures so CI signal is unambiguous.
    try:
        base = resolve_base(args)
        paths = changed_paths(base)
        commit_msg = last_commit_message()
    except RuntimeError as exc:
        sys.stderr.write(f"check_baselines_fresh: {exc}\n")
        return 2

    agent_paths = agent_changed(paths)
    baseline_paths = baselines_changed(paths)
    bypass_used = bool(SKIP_BASELINE_RE.search(commit_msg))

    # Branch 3: no watch-set change at all → trivially pass.
    if not agent_paths:
        print(
            f"check_baselines_fresh: OK — no watch-set changes vs {base} "
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
    sys.stderr.write(format_stale_error(agent_paths))
    sys.stderr.write("\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
