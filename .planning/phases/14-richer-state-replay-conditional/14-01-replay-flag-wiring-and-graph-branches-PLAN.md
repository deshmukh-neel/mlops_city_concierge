---
phase: 14-richer-state-replay-conditional
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - app/agent/adapters/__init__.py
  - app/agent/graph.py
  - scripts/eval_agent.py
  - tests/unit/test_agent_graph.py
autonomous: true
requirements: [REPLAY-01, REPLAY-02]
must_haves:
  truths:
    - "With both replay flags unset, graph.py plan() and _prune_for_llm produce byte-identical message lists to the Phase-13 plateau (flag-off default path unchanged)"
    - "REPLAY_MULTI_MESSAGE_ENABLED=1 routes plan() through adapter.replay_reasoning_state_multi instead of the single most-recent-only injection"
    - "REPLAY_CONTENT_BLOCKS_ENABLED=1 makes _prune_for_llm preserve pre-cutoff AIMessage content shape verbatim instead of collapsing to str()"
    - "run JSON arm_flags dict gains replay_multi_message and replay_content_blocks keys without removing the four Phase-13 keys"
    - "ProviderAdapter ABC has a generic replay_reasoning_state_multi default; the existing replay_reasoning_state abstract signature is untouched"
  artifacts:
    - path: "app/agent/adapters/__init__.py"
      provides: "replay_reasoning_state_multi generic ABC default + NoOpAdapter override"
      contains: "def replay_reasoning_state_multi"
    - path: "app/agent/graph.py"
      provides: "two build-time replay flag reads + flag-gated plan() replay branch + flag-gated _prune_for_llm preservation branch"
      contains: "REPLAY_MULTI_MESSAGE_ENABLED"
    - path: "scripts/eval_agent.py"
      provides: "arm_flags extended with two replay keys"
      contains: "replay_multi_message"
    - path: "tests/unit/test_agent_graph.py"
      provides: "flag-gated graph tests for both replay branches + greppable flag-name test"
      contains: "REPLAY_CONTENT_BLOCKS_ENABLED"
  key_links:
    - from: "app/agent/graph.py plan()"
      to: "adapter.replay_reasoning_state_multi"
      via: "if _replay_multi_message_enabled branch"
      pattern: "replay_reasoning_state_multi"
    - from: "app/agent/graph.py plan()"
      to: "_prune_for_llm preserve_content_blocks keyword"
      via: "_prune_for_llm(messages_in, preserve_content_blocks=_replay_content_blocks_enabled)"
      pattern: "preserve_content_blocks"
---

<objective>
Wire the two Phase-14 replay arms (REPLAY-01 multi-message reasoning-state replay, REPLAY-02 content-block preservation) into the agent loop behind env flags, extend the run-JSON arm_flags self-description, and add the generic multi-replay ABC method. Both flags default OFF and the flag-off path stays byte-identical to the Phase-13 plateau baseline.

Purpose: This is the implementation backbone of Phase 14. Every downstream task (adapter conformance tests, live A/B runs, verdict doc) depends on these flags existing and the flag-off path being provably unchanged so the A/B deltas are clean.
Output: Two new env-flag reads in build_agent_graph, a flag-gated plan() replay branch, a flag-gated _prune_for_llm preservation branch, the replay_reasoning_state_multi ABC default, the extended arm_flags dict, and graph-level flag-gated tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md
@.planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add replay_reasoning_state_multi generic default to the ProviderAdapter ABC</name>
  <files>app/agent/adapters/__init__.py</files>
  <read_first>
    - app/agent/adapters/__init__.py (the full ProviderAdapter ABC, NoOpAdapter, MockReasoningAdapter, ADAPTERS registry, __all__ — see current state before editing)
    - app/agent/adapters/openai_gpt5.py (the existing single-message replay_reasoning_state implementation — the per-message contract the new default delegates to)
    - app/agent/adapters/anthropic.py (asymmetric content-list replay — confirms the generic per-message delegation works for content-block adapters via the reverse-walk-finds-most-recent semantics)
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (section "app/agent/adapters/__init__.py — new replay_reasoning_state_multi ABC method": exact method body + NoOpAdapter override)
  </read_first>
  <action>
    Add a NON-abstract generic instance method `replay_reasoning_state_multi(self, outbound: list[BaseMessage]) -> list[BaseMessage]` to the `ProviderAdapter` ABC (D-14-03, REPLAY-01). The generic default iterates `outbound` with `enumerate`, and for each `AIMessage` whose `additional_kwargs` carries a non-None `_reasoning_state`, calls the existing single-message `self.replay_reasoning_state(outbound[:i + 1], per_msg_state)` so each in-window AIMessage receives its own stashed state. The reverse-walk in each adapter's single-message method targets the most-recent AIMessage of the passed sub-list, which is message `i` — confirm this targeting is correct for Anthropic's content-list path too. Return `outbound`. Do NOT make this method abstract and do NOT touch the existing `replay_reasoning_state` or `capture_reasoning_state` abstract signatures (the existing 9-test conformance harness MUST still pass). Add an explicit no-op override `replay_reasoning_state_multi` to `NoOpAdapter` that returns `outbound` unchanged. No changes to the `ADAPTERS` registry literal or the `__all__` list (the method lands on already-exported classes). Preserve the mutation-safety invariant: the default delegates to single-message methods that already do in-place additional_kwargs edits per D-08-06 — do not introduce new aliasing of message kwargs containers.
  </action>
  <verify>
    <automated>poetry run python -c "from app.agent.adapters import ProviderAdapter, NoOpAdapter; assert hasattr(ProviderAdapter, 'replay_reasoning_state_multi'); n=NoOpAdapter(); assert n.replay_reasoning_state_multi([]) == []"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "def replay_reasoning_state_multi" app/agent/adapters/__init__.py` returns 2 (ABC default + NoOpAdapter override)
    - `replay_reasoning_state_multi` is NOT decorated with `@abstractmethod` (source assertion: no `@abstractmethod` line immediately precedes the ABC default)
    - The existing `replay_reasoning_state` and `capture_reasoning_state` abstract method signatures in `__init__.py` are byte-unchanged (no diff on those two method definitions)
    - `poetry run pytest tests/unit/test_adapters.py -q` passes with the existing 9-test harness count unchanged (no existing test modified)
  </acceptance_criteria>
  <done>The ABC exposes a generic replay_reasoning_state_multi default, NoOpAdapter overrides it as a no-op, the abstract contract is untouched, and the existing adapter conformance harness passes unchanged.</done>
</task>

<task type="auto">
  <name>Task 2: Wire both replay flags into graph.py (build-time reads, plan() branch, _prune_for_llm preservation) and extend eval_agent arm_flags</name>
  <files>app/agent/graph.py, scripts/eval_agent.py</files>
  <read_first>
    - app/agent/graph.py lines 128-238 (_prune_for_llm and _RECENT_TOOL_EXCHANGES_KEPT) and lines 285-345 (Phase-13 flag-read block, plan() prune call, single-message replay site)
    - scripts/eval_agent.py lines 920-938 (the arm_flags dict assembly in EvalRunReport construction)
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (Patterns 1-3: exact flag-read lines, plan() branch shape, _prune_for_llm signature + branch, the arm_flags extension block)
    - app/config.py (env_flag truthy-parsing helper — already imported in graph.py at line 47 and used in eval_agent.py)
  </read_first>
  <action>
    Three coordinated edits, all flag-off byte-identical (REPLAY-01 + REPLAY-02, D-14-06).

    (1) In `build_agent_graph` (graph.py), immediately after the three Phase-13 DEC flag reads (~line 307) add two build-time reads using the already-imported `env_flag` helper: `_replay_multi_message_enabled: bool = env_flag("REPLAY_MULTI_MESSAGE_ENABLED")` and `_replay_content_blocks_enabled: bool = env_flag("REPLAY_CONTENT_BLOCKS_ENABLED")`. Add a comment block matching the Phase-13 precedent noting flag-off is byte-identical to the Phase-13 plateau and must be re-verified before merge.

    (2) Change `_prune_for_llm` to accept a keyword-only `preserve_content_blocks: bool = False` parameter. Inside the pre-cutoff `AIMessage with tool_calls` branch (~line 225-235), when `preserve_content_blocks` is True, construct `AIMessage(content=m.content, additional_kwargs=m.additional_kwargs)` preserving the original content shape verbatim (list or str); when False, keep the existing `content=m.content if isinstance(m.content, str) else str(m.content)` collapse. tool_calls are still stripped (the AIMessage constructor omits them by default — the unanswered-tool_call contract MUST hold). ToolMessages are still dropped. No block-type filtering. Update the plan() call site to `_prune_for_llm(messages_in, preserve_content_blocks=_replay_content_blocks_enabled)`.

    (3) In plan() at the single-message replay site (~line 336-341), gate it: when `_replay_multi_message_enabled` is True, call `messages_for_llm = adapter.replay_reasoning_state_multi(messages_for_llm)`; else fall through to the EXISTING reverse-walk single-message extraction + `adapter.replay_reasoning_state(...)` block unchanged. The else branch must be byte-identical to the current code.

    (4) In scripts/eval_agent.py, extend the `arm_flags` dict (~line 928) by ADDING two keys after the four existing Phase-13 keys: `"replay_multi_message": env_flag("REPLAY_MULTI_MESSAGE_ENABLED")` and `"replay_content_blocks": env_flag("REPLAY_CONTENT_BLOCKS_ENABLED")`. Do NOT remove or rename the four Phase-13 keys (viability_contract, forced_commit_step, parallel_tool, viability_threshold_override).
  </action>
  <verify>
    <automated>poetry run python -c "import inspect, app.agent.graph as g; s=inspect.getsource(g); assert 'REPLAY_MULTI_MESSAGE_ENABLED' in s and 'REPLAY_CONTENT_BLOCKS_ENABLED' in s and 'preserve_content_blocks' in s and 'replay_reasoning_state_multi' in s"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "preserve_content_blocks" app/agent/graph.py` returns at least 3 (param def, branch read, call-site keyword)
    - `_prune_for_llm` signature contains `*, preserve_content_blocks: bool = False` (source assertion)
    - The plan() prune call passes `preserve_content_blocks=_replay_content_blocks_enabled` (source assertion)
    - The flag-off else-branch in plan() still contains the reverse-walk `for m in reversed(messages_for_llm)` extracting `_reasoning_state` then `adapter.replay_reasoning_state(messages_for_llm, captured_state)` (source assertion — existing path preserved verbatim)
    - `grep -E "replay_multi_message|replay_content_blocks" scripts/eval_agent.py` returns 2 lines AND all four Phase-13 arm_flags keys (`viability_contract`, `forced_commit_step`, `parallel_tool`, `viability_threshold_override`) still present in the same dict
    - With both flags unset: `poetry run python -c "import os; [os.environ.pop(k,None) for k in ['REPLAY_MULTI_MESSAGE_ENABLED','REPLAY_CONTENT_BLOCKS_ENABLED']]; from app.agent.graph import _prune_for_llm; from langchain_core.messages import AIMessage; m=AIMessage(content=['block'], tool_calls=[{'name':'t','args':{},'id':'1'}]); m2=AIMessage(content=['b2'], tool_calls=[{'name':'t','args':{},'id':'2'}]); m3=AIMessage(content=['b3'], tool_calls=[{'name':'t','args':{},'id':'3'}]); out=_prune_for_llm([m,m2,m3]); assert isinstance(out[0].content, str), 'flag-off must collapse list content to str'"` exits 0 (flag-off str collapse preserved)
  </acceptance_criteria>
  <done>Both replay flags are read at build time, plan() routes through multi-replay when REPLAY_MULTI_MESSAGE_ENABLED is set, _prune_for_llm preserves content shape when REPLAY_CONTENT_BLOCKS_ENABLED is set, the flag-off path is byte-identical, and arm_flags carries both new keys alongside the Phase-13 keys.</done>
</task>

<task type="auto">
  <name>Task 3: Add flag-gated graph tests for both replay branches + greppable flag-name test</name>
  <files>tests/unit/test_agent_graph.py</files>
  <read_first>
    - tests/unit/test_agent_graph.py (current state — existing graph test structure and imports)
    - tests/unit/test_graph_forced_commit.py lines 219-234 and 366-369 (the monkeypatch.setenv flag-gated build pattern + the greppable `test_forced_commit_step_flag_reads_at_build_time` source-inspection test to mirror)
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (section "Flag-gated test pattern" and "Greppable flag check test")
  </read_first>
  <action>
    Add unit tests (REPLAY-01 + REPLAY-02) following the Phase-13 monkeypatch.setenv pattern. Every test that builds the graph must set ALL three Phase-13 DEC flags to OFF (FORCED_COMMIT_STEP=0, VIABILITY_CONTRACT_ENABLED=0, PARALLEL_TOOL_EXECUTION_ENABLED=0) plus the relevant replay flag, and build the graph AFTER the monkeypatch so the build-time reads see the patched values. Add: (a) `test_replay_flags_read_at_build_time` — inspect.getsource(app.agent.graph) asserts both "REPLAY_MULTI_MESSAGE_ENABLED" and "REPLAY_CONTENT_BLOCKS_ENABLED" appear (mirrors the forced-commit greppable test). (b) A REPLAY-02 prune test: with REPLAY_CONTENT_BLOCKS_ENABLED=1 call _prune_for_llm with preserve_content_blocks=True on a message list containing >2 tool-calling AIMessages whose content is a list, assert a pre-cutoff AIMessage retains list content (not str); and the flag-off counterpart asserts str collapse. (c) A REPLAY-01 plan-routing test: build the graph with REPLAY_MULTI_MESSAGE_ENABLED=1 and a recording adapter/spy (or assert via a NoOp/scripted provider that the multi path is reachable) — at minimum assert that the flag-on build does not raise and that _prune_for_llm + replay path executes; prefer asserting adapter.replay_reasoning_state_multi is invoked over replay_reasoning_state when the flag is on (monkeypatch the adapter instance method to a spy if needed). Do NOT modify existing tests. Mocked/unit level only — no live LLM, no DB.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_graph.py -q -k "replay"</automated>
  </verify>
  <acceptance_criteria>
    - `poetry run pytest tests/unit/test_agent_graph.py -q -k "replay"` collects and passes at least 3 tests
    - A test named like `test_replay_flags_read_at_build_time` exists and asserts both replay flag names appear in graph.py source
    - One test asserts REPLAY-02 flag-ON preserves list content through _prune_for_llm and a paired test asserts flag-OFF collapses to str
    - One test asserts REPLAY-01 flag-ON routes plan() through replay_reasoning_state_multi (spy/recording-adapter assertion or equivalent reachability assertion)
    - No pre-existing test in test_agent_graph.py is modified (git diff shows only additions)
  </acceptance_criteria>
  <done>Flag-gated graph tests cover both replay branches and the greppable flag-name contract, all passing at the unit level with no existing test changed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| env vars → build_agent_graph | REPLAY_MULTI_MESSAGE_ENABLED / REPLAY_CONTENT_BLOCKS_ENABLED parsed at graph-build time; operator-controlled, not end-user input |
| persisted reasoning state → next LLM call | _reasoning_state replayed onto outbound messages crosses back to the provider API |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-14-01 | Tampering | env-flag parsing in build_agent_graph | mitigate | env_flag uses a fixed truthy allowlist ("1"/"true"/"yes"/"on"); unrecognized values default to False (flag-off), so a malformed flag cannot silently enable the experimental path |
| T-14-02 | Information disclosure | multi-message replay re-injecting reasoning state | accept | Reasoning state is already round-tripped per-turn in the flag-off path (Phase 8/9 contract); multi-message replay re-applies state already stored on in-window AIMessages within the same request — no new data source, no cross-request leak (cross-request persistence is explicitly out of scope, Deferred Ideas) |
| T-14-03 | Denial of service | content-block preservation increasing payload size | accept | Pre-cutoff AIMessage count is bounded by the _RECENT_TOOL_EXCHANGES_KEPT prune window (D-14-06); any token/latency movement is measured via existing INST-04 llm_call_seconds telemetry, no new guardrail per anti-scope |
| T-14-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs in this plan (pure code change against existing langchain_core / app deps); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_adapters.py tests/unit/test_agent_graph.py -q` passes (existing harness unchanged + new replay tests green)
- `make test` passes (full suite — mandatory for app/agent/ changes per CLAUDE.md; targeted runs miss DB-pool contamination from real-graph tests)
- `make typecheck` passes (mypy app/) — new keyword param and ABC method are typed
- `make lint` passes (ruff)
- Flag-off byte-identity: with both replay flags unset, _prune_for_llm collapses list content to str and plan() uses the single-message replay path (verified by Task 2/Task 3 assertions)
</verification>

<success_criteria>
- Both replay flags exist, default OFF, and are read once at build time
- REPLAY_MULTI_MESSAGE_ENABLED routes plan() through replay_reasoning_state_multi
- REPLAY_CONTENT_BLOCKS_ENABLED makes _prune_for_llm preserve content shape
- arm_flags carries replay_multi_message + replay_content_blocks alongside the four Phase-13 keys
- The ProviderAdapter ABC exposes a generic replay_reasoning_state_multi default; the existing abstract contract and 9-test conformance harness are untouched
- All new tests pass; no existing test modified
</success_criteria>

<output>
Create `.planning/phases/14-richer-state-replay-conditional/14-01-SUMMARY.md` when done
</output>
