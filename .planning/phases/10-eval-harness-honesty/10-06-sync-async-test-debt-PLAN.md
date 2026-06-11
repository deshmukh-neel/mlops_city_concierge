---
phase: 10-eval-harness-honesty
plan: 06
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/unit/test_llm_factory.py
  - app/agent/critique/vibe.py
autonomous: true
requirements: [EVAL-06]
must_haves:
  truths:
    - "build_chat_model('openai', 'gpt-5-mini') routes through OpenAIReasoningChatModel with use_responses_api=True"
    - "build_chat_model('openai', 'gpt-4o-mini') stays on plain ChatOpenAI (v2.0 anchor untouched)"
    - "ScriptedChatModel is proven to work via ainvoke (the only call the graph makes)"
    - "vibe_check's sync judge_llm.invoke does not block the async event loop, documented with evidence"
  artifacts:
    - path: "tests/unit/test_llm_factory.py"
      provides: "gpt-5 dispatch tests + gpt-4o-mini regression guard + ScriptedChatModel ainvoke test"
      contains: "use_responses_api"
    - path: "app/agent/critique/vibe.py"
      provides: "doc-comment recording the LangGraph sync-node-in-executor finding (behavior-preserving)"
      contains: "executor"
  key_links:
    - from: "tests/unit/test_llm_factory.py"
      to: "app/llm_factory.py:350-362 gpt-5 dispatch branch"
      via: "mocker.patch on OpenAIReasoningChatModel + assert use_responses_api True"
      pattern: "use_responses_api"
    - from: "tests/unit/test_llm_factory.py"
      to: "ScriptedChatModel.ainvoke"
      via: "async test awaiting ainvoke and asserting content"
      pattern: "ainvoke"
---

<objective>
Close the sync/async test debt (EVAL-06). Three latent gaps:

1. The gpt-5 dispatch branch in `build_chat_model` (`app/llm_factory.py:350-362`,
   `use_responses_api=True`) has ZERO tests referencing `use_responses_api` or
   `_is_openai_reasoning_model` — a refactor could silently route gpt-5 onto plain ChatOpenAI
   (losing reasoning-state preservation) or, worse, route the gpt-4o-mini v2.0 anchor onto the
   reasoning path (D-10-15).
2. `ScriptedChatModel` is only exercised via the sync `_generate`; the graph only ever calls
   `ainvoke`. A test must prove the BaseChatModel async executor fallback works for it (D-10-16).
3. `vibe_check` (`app/agent/critique/vibe.py:78`) makes a sync `judge_llm.invoke` inside the sync
   `critique` node, which runs inside an otherwise-async graph. D-10-17 requires the planner to
   FIRST verify whether LangGraph runs sync nodes in an executor under `ainvoke` (in which case
   the event loop is not blocked and a doc-comment suffices) before making any code change.

**Planner finding (D-10-17 resolved):** verified against LangGraph 1.2.0 (this repo's pinned
version) that sync nodes run in a separate thread (ThreadPoolExecutor) under `ainvoke` — the
event loop is NOT blocked. A minimal repro (a sync node returning `threading.get_ident()`
through `app.ainvoke` returns a thread id distinct from the main thread's). Therefore the
`vibe_check` sync call does NOT block the loop and a doc-comment is the correct, behavior-
preserving resolution. NO agent-code behavior change is made — the only edit to vibe.py is a
clarifying comment (the one agent-code touch permitted in this phase, and it changes no
behavior).

Purpose: lock the gpt-5 reasoning dispatch and the v2.0 anchor path with regression tests before
Phase 11's cross-model regen.
Output: factory dispatch tests, a ScriptedChatModel ainvoke test, and an evidence-backed vibe.py comment.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@app/llm_factory.py
@app/agent/critique/vibe.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Factory-level tests for gpt-5 dispatch + gpt-4o-mini regression guard + ScriptedChatModel ainvoke</name>
  <files>tests/unit/test_llm_factory.py</files>
  <read_first>
    - tests/unit/test_llm_factory.py (read the existing test_build_chat_model_dispatches_per_provider :17-35 and the monkeypatch + get_settings.cache_clear() pattern — copy this exactly for the new tests)
    - app/llm_factory.py (read build_chat_model :330-362 — the openai branch with _is_openai_reasoning_model gating OpenAIReasoningChatModel(use_responses_api=True) vs plain ChatOpenAI; read ScriptedChatModel :270-304 — note it has _generate but NO _agenerate override, so ainvoke relies on the BaseChatModel executor fallback)
    - .planning/phases/10-eval-harness-honesty/10-PATTERNS.md (the three test skeletons: gpt5 dispatch, gpt4o-mini stays-plain regression guard, ScriptedChatModel ainvoke)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-15 gpt-5 returns OpenAIReasoningChatModel use_responses_api=True / gpt-4o-mini plain; D-10-16 ScriptedChatModel via ainvoke)
  </read_first>
  <behavior>
    - build_chat_model("openai", "gpt-5-mini", temperature=1.0) returns OpenAIReasoningChatModel and the constructor is called with model="gpt-5-mini" and use_responses_api=True.
    - build_chat_model("openai", "gpt-4o-mini", temperature=1.0) returns plain ChatOpenAI and OpenAIReasoningChatModel is NOT called (the v2.0 anchor path is byte-preserved).
    - ScriptedChatModel(scripted=[AIMessage(content="hello")]).ainvoke([HumanMessage(...)]) returns an AIMessage with content "hello" (the executor fallback for a model with only _generate works under the async path the graph uses).
  </behavior>
  <action>
    In tests/unit/test_llm_factory.py, add three tests using the existing monkeypatch + `get_settings.cache_clear()` pattern (the pattern is MANDATORY — cached settings survive monkeypatch). (1) `test_build_chat_model_gpt5_returns_openai_reasoning_chat_model`: monkeypatch OPENAI_API_KEY, clear settings cache, mocker.patch("app.llm_factory.OpenAIReasoningChatModel", return_value=sentinel), call build_chat_model("openai", "gpt-5-mini", temperature=1.0), assert the sentinel is returned, the patched class was called once, and call kwargs include model=="gpt-5-mini" and use_responses_api is True. (2) `test_build_chat_model_gpt4o_mini_stays_plain_chat_openai`: monkeypatch + cache clear, patch BOTH OpenAIReasoningChatModel and ChatOpenAI, call build_chat_model("openai", "gpt-4o-mini", temperature=1.0), assert plain ChatOpenAI returned and OpenAIReasoningChatModel.assert_not_called() (the v2.0 anchor regression guard). (3) `test_scripted_chat_model_ainvoke_works`: an async test (asyncio_mode="auto" is set in pyproject — no asyncio.run needed) constructing ScriptedChatModel(scripted=[AIMessage(content="hello")]) and awaiting `.ainvoke([HumanMessage(content="go")])`, asserting result.content == "hello" — proving the BaseChatModel async executor fallback works for a model that defines only _generate (D-10-16). No live calls; all mocked/scripted.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_llm_factory.py -q -k "gpt5 or gpt4o or gpt_4o or scripted or use_responses or ainvoke"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "use_responses_api" tests/unit/test_llm_factory.py` returns at least one match (previously zero in the test suite).
    - `test_build_chat_model_gpt5_returns_openai_reasoning_chat_model` asserts `kwargs["use_responses_api"] is True` and `kwargs["model"] == "gpt-5-mini"`.
    - `test_build_chat_model_gpt4o_mini_stays_plain_chat_openai` calls `OpenAIReasoningChatModel.assert_not_called()` (anchor regression guard).
    - `test_scripted_chat_model_ainvoke_works` awaits `ScriptedChatModel(...).ainvoke(...)` and asserts the returned content.
    - `poetry run pytest tests/unit/test_llm_factory.py -q` exits 0.
    - `poetry run ruff check tests/unit/test_llm_factory.py` passes.
  </acceptance_criteria>
  <done>The gpt-5 reasoning dispatch and the gpt-4o-mini anchor path are both locked by tests; ScriptedChatModel is proven to work via the ainvoke path the graph actually uses.</done>
</task>

<task type="auto">
  <name>Task 2: Document the vibe_check sync-node-in-executor finding (behavior-preserving)</name>
  <files>app/agent/critique/vibe.py</files>
  <read_first>
    - app/agent/critique/vibe.py (read vibe_check :47-86, specifically the sync `judge_llm.invoke([HumanMessage(...)])` at :78)
    - app/agent/graph.py (read the critique node :430-463 — it is `def critique` (sync) surrounded by async plan/act/retime nodes; it calls critique_final_with_stops -> vibe.vibe_check)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-17 — verify-first; doc-comment suffices if LangGraph runs sync nodes in an executor under ainvoke; behavior must be byte-identical)
  </read_first>
  <action>
    Add a doc-comment immediately above the `judge_llm.invoke([HumanMessage(content=prompt)])` call at app/agent/critique/vibe.py:78 recording the D-10-17 finding: the enclosing `critique` node is a SYNC LangGraph node; verified against the repo's pinned LangGraph (1.2.0) that sync nodes execute in a ThreadPoolExecutor thread under `graph.ainvoke`, so this blocking sync `judge_llm.invoke` does NOT block the asyncio event loop (a sync node returning threading.get_ident() through ainvoke returns a non-main thread id). Therefore no async refactor is required; this call stays sync by design. Reference D-10-17 and EVAL-06. Make NO behavior change — only the comment is added (this is the single permitted agent-code touch in the phase, and it is behavior-preserving). Do NOT convert vibe_check to async, do NOT change the call, do NOT touch the critique node.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/ -q -k "vibe or critique" && grep -n "executor\|D-10-17" app/agent/critique/vibe.py</automated>
  </verify>
  <acceptance_criteria>
    - `app/agent/critique/vibe.py` contains a comment near line 78 referencing the executor finding and D-10-17 (source assertion: `grep -n "executor" app/agent/critique/vibe.py` returns a match within vibe_check).
    - The `judge_llm.invoke([HumanMessage(content=prompt)])` call is byte-unchanged (still sync `.invoke`; no `await`, no `_agenerate`) — verifiable by `git diff app/agent/critique/vibe.py` showing only added comment lines, no logic-line deletions.
    - `poetry run pytest tests/unit/ -q -k "vibe or critique"` exits 0 (no behavior regression).
    - `poetry run ruff check app/agent/critique/vibe.py` passes.
  </acceptance_criteria>
  <done>The vibe_check sync call is documented as safe under LangGraph's sync-node executor with evidence and a D-ID; no behavior changes.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| factory dispatch logic → reasoning-state preservation | a mis-routed gpt-5 (onto plain ChatOpenAI) silently drops reasoning state; a mis-routed gpt-4o-mini (onto the reasoning path) changes the v2.0 prod anchor |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-06-01 | Tampering | build_chat_model gpt-5/gpt-4o-mini dispatch | mitigate | Regression tests assert gpt-5 -> OpenAIReasoningChatModel(use_responses_api=True) and gpt-4o-mini -> plain ChatOpenAI with OpenAIReasoningChatModel.assert_not_called(); a refactor that crosses the wires fails CI |
| T-10-06-02 | Denial of service | vibe_check sync invoke blocking the event loop | accept | Verified non-issue: LangGraph 1.2.0 runs sync nodes in a ThreadPoolExecutor under ainvoke (planner repro); the loop is not blocked. Documented; no code change. Re-evaluate only if LangGraph is upgraded |
| T-10-06-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs; tests use existing mocker/monkeypatch and the in-repo ScriptedChatModel |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_llm_factory.py -q` exits 0.
- `poetry run pytest tests/unit/ -q -k "vibe or critique"` exits 0.
- `git diff app/agent/critique/vibe.py` shows comment-only additions (no logic deletions).
- `poetry run ruff check tests/unit/test_llm_factory.py app/agent/critique/vibe.py` passes.
</verification>

<success_criteria>
- gpt-5 dispatch (use_responses_api=True) and the gpt-4o-mini anchor path are both test-locked (EVAL-06).
- ScriptedChatModel is proven to work via ainvoke.
- The vibe_check sync call is documented as non-blocking under LangGraph's sync-node executor with evidence; no behavior change.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-06-SUMMARY.md` when done.
</output>
