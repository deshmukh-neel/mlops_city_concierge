"""Unit tests for scripts/check_baselines_fresh.py.

Plan 03-07 / EVAL-07. The lint script enforces that PRs touching app/agent/
also refresh configs/eval_baselines/*.json — unless the latest commit message
carries the explicit [skip-baseline] bypass.

The four branches of the truth table (per plan 03-07 task 1 <behavior>):

    | agent_changed | baselines_changed | [skip-baseline] | exit |
    |     T         |        F          |       F         |  1   |  ← stale
    |     T         |        T          |       F         |  0   |  ← updated
    |     F         |        *          |       *         |  0   |  ← no agent change
    |     T         |        F          |       T         |  0   |  ← explicit bypass

Tests use monkeypatching of `run_git` (the thin subprocess wrapper inside
the script) so they don't need a real git repo state. This mirrors the
testing pattern used elsewhere in tests/unit/ (e.g. test_eval_matrix.py)
and keeps the test suite hermetic.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_baselines_fresh.py"


def load_script() -> ModuleType:
    """Load scripts/check_baselines_fresh.py as a module.

    The script lives under scripts/ which isn't a package, so we load it
    via importlib.util to keep tests hermetic (no sys.path mutation needed
    at import time for the script itself).
    """
    spec = importlib.util.spec_from_file_location("check_baselines_fresh", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_baselines_fresh"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    """The check_baselines_fresh module under test."""
    return load_script()


def stub_git(
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    *,
    changed_paths: list[str],
    last_commit_message: str,
) -> None:
    """Replace `run_git` with a deterministic stub.

    `run_git(args)` is the thin subprocess wrapper inside the script. It
    must return whatever stdout `git` would have produced — newline-joined
    for `git diff --name-only`, and the raw commit message body for
    `git log -1 --format=%B`. The stub routes by the first arg after `git`
    so we can drive both branches from a single fixture.
    """

    def fake_run_git(args: list[str]) -> str:
        # `args` is the argv tail (e.g. ["diff", "--name-only", "SHA...HEAD"]).
        if not args:
            return ""
        subcmd = args[0]
        if subcmd == "diff":
            return "\n".join(changed_paths)
        if subcmd == "log":
            return last_commit_message
        return ""

    monkeypatch.setattr(script, "run_git", fake_run_git)


# ---------------------------------------------------------------------------
# Truth table branches
# ---------------------------------------------------------------------------


def test_agent_changed_and_no_baseline_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """RULE 1 (the gate): agent/ touched, baseline NOT touched, no bypass → exit 1."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/agent/graph.py", "app/agent/state.py"],
        last_commit_message="feat(agent): tweak graph wiring\n\n- no baseline updated",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, "agent changed without baseline refresh must exit 1"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # Actionable error must name at least one of the changed agent paths and
    # the remediation command (make eval-matrix RUNS=3).
    assert "app/agent/graph.py" in combined or "app/agent/state.py" in combined
    assert "make eval-matrix" in combined
    assert "[skip-baseline]" in combined


def test_agent_changed_with_baseline_passes(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """RULE 2: agent/ touched AND configs/eval_baselines/*.json touched → exit 0."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=[
            "app/agent/graph.py",
            "configs/eval_baselines/omakase_mission_open_ended.json",
        ],
        last_commit_message="feat(agent): tweak graph wiring + refresh baseline",
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "agent + baseline change must exit 0"


def test_no_agent_change_passes(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> None:
    """RULE 3: nothing under app/agent/ changed → exit 0 regardless of bypass."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=[
            "README.md",
            "scripts/eval_agent.py",
            "tests/unit/test_eval_agent.py",
        ],
        last_commit_message="docs: clarify README",
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "non-agent changes must exit 0"


def test_skip_baseline_bypass_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """RULE 4: agent/ touched, baseline NOT touched, BUT [skip-baseline] → exit 0.

    The bypass token must appear at a line boundary (start of message or
    after a newline), optionally preceded by whitespace — trailer-style.
    Mid-sentence mentions in commit prose are NOT a bypass (see
    test_incidental_skip_baseline_mention_does_not_bypass).
    """
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/agent/graph.py"],
        last_commit_message=(
            "refactor(agent): rename internal helper\n\n"
            "[skip-baseline]\n"
            "No behavior change; baseline refresh not warranted."
        ),
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "[skip-baseline] at a line boundary must bypass the gate"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # The bypass path must announce itself so reviewers see it in CI logs.
    assert "skip-baseline" in combined.lower()


def test_skip_baseline_at_subject_line_start_bypasses(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """IN-04: trailer-style token at the very start of the message bypasses."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/agent/graph.py"],
        last_commit_message="[skip-baseline] refactor(agent): cosmetic rename",
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "[skip-baseline] at message start must bypass the gate"


def test_incidental_skip_baseline_mention_does_not_bypass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """IN-04: a documentation PR that quotes the token mid-sentence (e.g.
    docs explaining how the bypass works) must NOT trip the gate. The
    previous substring match would silently let such a PR through.
    """
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/agent/graph.py"],
        last_commit_message=(
            "docs(agent): explain the [skip-baseline] bypass token used by "
            "scripts/check_baselines_fresh.py"
        ),
    )
    rc = script.main(["origin/main"])
    assert rc == 1, (
        "incidental [skip-baseline] mid-sentence mention must NOT bypass the stale-baseline gate"
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # The stale-baseline error path fired, so the gate emitted its remediation.
    assert "make eval-matrix" in combined


# ---------------------------------------------------------------------------
# Argv / flag plumbing
# ---------------------------------------------------------------------------


def test_main_accepts_positional_base_sha(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """Positional `BASE_SHA` argv shape is what the CI job invokes."""
    captured_args: list[list[str]] = []

    def fake_run_git(args: list[str]) -> str:
        captured_args.append(list(args))
        if args and args[0] == "diff":
            return ""  # no changes
        if args and args[0] == "log":
            return "chore: no-op"
        return ""

    monkeypatch.setattr(script, "run_git", fake_run_git)
    rc = script.main(["abc1234"])
    assert rc == 0
    # The first git invocation must be `git diff --name-only abc1234...HEAD`.
    diff_invocations = [a for a in captured_args if a and a[0] == "diff"]
    assert diff_invocations, "must invoke git diff"
    assert any("abc1234...HEAD" in arg for arg in diff_invocations[0])


def test_main_accepts_merge_base_flag(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> None:
    """`--merge-base SHA` is equivalent to passing SHA positionally."""
    captured_args: list[list[str]] = []

    def fake_run_git(args: list[str]) -> str:
        captured_args.append(list(args))
        if args and args[0] == "diff":
            return ""
        if args and args[0] == "log":
            return ""
        return ""

    monkeypatch.setattr(script, "run_git", fake_run_git)
    rc = script.main(["--merge-base", "deadbeef"])
    assert rc == 0
    diff_invocations = [a for a in captured_args if a and a[0] == "diff"]
    assert any("deadbeef...HEAD" in arg for arg in diff_invocations[0])


def test_main_defaults_to_origin_main_when_no_args(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """Calling without argv defaults to `origin/main` as the diff base."""
    captured_args: list[list[str]] = []

    def fake_run_git(args: list[str]) -> str:
        captured_args.append(list(args))
        if args and args[0] == "diff":
            return ""
        if args and args[0] == "log":
            return ""
        return ""

    monkeypatch.setattr(script, "run_git", fake_run_git)
    rc = script.main([])
    assert rc == 0
    diff_invocations = [a for a in captured_args if a and a[0] == "diff"]
    assert any("origin/main...HEAD" in arg for arg in diff_invocations[0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_only_baseline_changed_is_not_a_gate_violation(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """Baseline-only refresh PRs (no agent/ change) are always allowed."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=[
            "configs/eval_baselines/refinement_cheaper.json",
            "configs/eval_baselines/omakase_mission_open_ended.json",
        ],
        last_commit_message="chore(baselines): refresh after Phase 4 sign-off",
    )
    rc = script.main(["origin/main"])
    assert rc == 0


def test_non_baseline_json_under_eval_baselines_does_not_satisfy_gate(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """Only `.json` files under configs/eval_baselines/ count as a baseline refresh.

    A stray README.md or .gitkeep under that dir must not satisfy the gate
    because it doesn't carry the scorer numbers Phase 4-6 merge rules need.
    """
    stub_git(
        monkeypatch,
        script,
        changed_paths=[
            "app/agent/graph.py",
            "configs/eval_baselines/README.md",  # NOT a .json
        ],
        last_commit_message="feat(agent): tweak",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, "non-.json file under eval_baselines must not satisfy the gate"


# ─── WR-02 loud-fail regression tests (Plan 03-10) ────────────────────────
#
# Unlike the truth-table tests above (which monkeypatch `run_git` wholesale
# to drive the four pass/fail branches), the tests in this section exercise
# the real `run_git` itself by patching one level deeper — `subprocess.run`
# inside the script module's namespace. This pins the loud-fail contract:
#
#   - rc != 0 from git           → RuntimeError naming the argv + stderr
#   - missing git binary         → RuntimeError naming `git`
#   - explicit empty BASE_SHA    → non-zero exit (no silent origin/main fallback)
#   - rc == 0 happy path         → stdout returned unchanged (backward compat pin)


def test_run_git_raises_on_nonzero_return_code(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """`run_git` must raise RuntimeError when git exits non-zero.

    Current behaviour silently swallows the rc and returns stdout (which is
    empty when git failed) — meaning a missing/shallow `origin/main` ref or
    a bogus BASE_SHA quietly passes the gate. The hard-gate posture is
    loud-fail: the error message must carry the rc, the argv, and stderr
    so an operator can diagnose without re-running locally.
    """
    fake_stderr = "fatal: bad revision 'origin/main'"

    def fake_subprocess_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=128, stdout="", stderr=fake_stderr)

    monkeypatch.setattr(script.subprocess, "run", fake_subprocess_run)
    with pytest.raises(RuntimeError) as excinfo:
        script.run_git(["diff", "--name-only", "abc...HEAD"])

    msg = str(excinfo.value)
    assert "128" in msg, f"rc not surfaced in error: {msg!r}"
    assert "git diff" in msg or "diff" in msg, f"argv not surfaced in error: {msg!r}"
    assert "fatal: bad revision" in msg, f"stderr not surfaced in error: {msg!r}"


def test_run_git_raises_actionable_error_when_git_binary_missing(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """`run_git` must convert FileNotFoundError into an actionable RuntimeError.

    On a CI image without git installed (or with a $PATH mangled by an env
    block), `subprocess.run(["git", ...])` raises FileNotFoundError. The
    current implementation lets that bubble untouched, which is a confusing
    Python traceback for a missing-tool failure. The contract: re-raise as
    RuntimeError naming `git` so an operator immediately knows to install it.
    """

    def fake_subprocess_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("[Errno 2] No such file or directory: 'git'")

    monkeypatch.setattr(script.subprocess, "run", fake_subprocess_run)
    with pytest.raises(RuntimeError) as excinfo:
        script.run_git(["status"])

    msg = str(excinfo.value)
    assert "git" in msg.lower(), f"git not named in error: {msg!r}"


def test_main_exits_non_zero_when_base_sha_is_empty_string(
    capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """Explicit empty-string BASE_SHA must NOT silently fall back to origin/main.

    A misconfigured CI workflow could conceivably emit the empty string from
    `${{ github.event.pull_request.base.sha }}` (e.g. on workflow_dispatch
    where pull_request context is absent). Current behaviour: the empty
    string is falsy so `resolve_base` silently returns `origin/main`. New
    contract: empty-string positional is rejected loudly with a non-zero
    exit code (no subprocess invocation needed — should fail before the
    first git call).
    """
    rc = script.main([""])
    assert rc != 0, "empty-string positional BASE_SHA must NOT silently pass"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "BASE_SHA" in combined or "base_sha" in combined or "empty" in combined.lower(), (
        f"empty-BASE_SHA error not surfaced to operator: out={captured.out!r} err={captured.err!r}"
    )


def test_run_git_still_returns_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """Backward-compat pin: rc == 0 still returns stdout verbatim.

    The 9 existing tests monkeypatch `run_git` wholesale (they never reach
    the real `subprocess.run` path), so without this test the rc == 0 happy
    path inside `run_git` itself is untested after the WR-02 refactor.
    """

    def fake_subprocess_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=0, stdout="path1.py\npath2.py\n", stderr="")

    monkeypatch.setattr(script.subprocess, "run", fake_subprocess_run)
    out = script.run_git(["diff", "--name-only", "abc...HEAD"])
    assert out == "path1.py\npath2.py\n"


# Keep `subprocess` import referenced even if a future refactor removes the
# patch-target indirection above (currently subprocess.run is patched via
# `script.subprocess.run` which is the script module's own `subprocess`
# binding — the top-level `subprocess` import here documents intent and
# stays available for any future test that wants to patch the stdlib module
# directly).
_ = subprocess


# ---------------------------------------------------------------------------
# D-11-21: Watch-set extension tests (BASE-04)
# ---------------------------------------------------------------------------


def test_llm_factory_change_triggers_stale_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """D-11-21: a diff touching app/llm_factory.py without a baseline refresh exits 1.

    Provider branches, thinking policies, and temperature clamps live in
    llm_factory.py — these affect measured behavior just as much as agent/ files.
    """
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/llm_factory.py"],
        last_commit_message="fix: update provider temperature clamp",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, "app/llm_factory.py change without baseline refresh must exit 1"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "make eval-matrix" in combined


def test_eval_matrix_yaml_change_triggers_stale_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], script: ModuleType
) -> None:
    """D-11-21: a diff touching configs/eval_matrix.yaml without a baseline refresh exits 1."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=["configs/eval_matrix.yaml"],
        last_commit_message="feat: add gpt-5-mini entry",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, "configs/eval_matrix.yaml change without baseline refresh must exit 1"


def test_eval_matrix_refinement_yaml_change_triggers_stale_gate(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """D-11-21: configs/eval_matrix_refinement.yaml is also in the watch-set.

    The 'configs/eval_matrix' bare prefix matches both *.yaml files.
    """
    stub_git(
        monkeypatch,
        script,
        changed_paths=["configs/eval_matrix_refinement.yaml"],
        last_commit_message="feat: add deepseek-reasoner to refinement matrix",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, (
        "configs/eval_matrix_refinement.yaml change without baseline refresh must exit 1"
    )


def test_agent_file_change_still_triggers_stale_gate_no_regression(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """D-11-21: app/agent/ prefix still triggers the gate (no regression on existing behavior)."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=["app/agent/agent.py"],
        last_commit_message="refactor: tweak agent internals",
    )
    rc = script.main(["origin/main"])
    assert rc == 1, "app/agent/ change without baseline refresh must still exit 1 (no regression)"


def test_llm_factory_change_with_baseline_refresh_passes(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType
) -> None:
    """D-11-21: app/llm_factory.py change WITH a baseline refresh exits 0."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=[
            "app/llm_factory.py",
            "configs/eval_baselines/omakase_mission_open_ended.json",
        ],
        last_commit_message="fix: update factory + refresh baselines",
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "llm_factory.py + baseline refresh must exit 0"


def test_unrelated_change_still_passes(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> None:
    """D-11-21: an unrelated diff (e.g. README.md only) still exits 0."""
    stub_git(
        monkeypatch,
        script,
        changed_paths=["README.md", "docs/some_doc.md"],
        last_commit_message="docs: update README",
    )
    rc = script.main(["origin/main"])
    assert rc == 0, "unrelated file changes must exit 0"
