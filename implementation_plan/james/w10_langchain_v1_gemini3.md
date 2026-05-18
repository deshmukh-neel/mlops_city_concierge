# W10 — LangChain 1.x migration + native Gemini 3 thought signatures

**Branch:** `feature/agent-w10-langchain-v1` (not started)
**Depends on:** nothing code-wise, but should land **before** any decision to run
Gemini 3 in production (W-model-selection).
**Status:** Planned 2026-05-17. Root cause confirmed by debugging session.

## Why this exists (root cause, evidence-backed)

The agent intermittently-to-consistently fails to produce an itinerary when
running a **Gemini 3** model (`gemini-3.1-pro-preview`): it loops
`semantic_search`/`nearby`, never calls `commit_itinerary`, exhausts
`max_steps=8`, returns "I hit the planning step limit" with `stops=0` and an
empty reply. Measured success rate at temp=1.0: **~0/6** (one lucky pass in ~9
total observed runs). The user's *original* report (on deployed
`gpt-4o-mini`) was the same failure at lower frequency — **the bug predates any
model switch.**

**Root cause:** Gemini 3 uses **thought signatures** — opaque reasoning-state
tokens emitted with each function call and expected back in history on the next
turn to preserve chain-of-thought across tool calls. The pinned
`langchain-google-genai 2.1.12` has **zero** thought-signature support
(`thought_signature` appears 0× in its source). `app/gemini_compat.py`
monkeypatches a **static bypass sentinel** (`b"skip_thought_signature_
validator"`) so the API *accepts* requests — but Gemini never gets its real
reasoning state back. Across the plan→act→critique loop it loses its train of
thought every tool call, re-decides from scratch, re-searches, and never
converges to commit. `gemini_compat.py`'s own docstring names the real fix:
*"until the project can move to the newer LangChain 1.x-compatible adapter."*

This is **not** a bug in the agent loop, prompt, tools, or graph — those work
(they shipped on gpt-4o-mini and produced a correct 3-stop itinerary on the one
Gemini run that converged before losing the thread). It is a
**LangChain↔Gemini 3 integration gap**.

ADK (Gemini Agent Development Kit) is **not** the fix — the problem is
thought-signature round-tripping in the Gemini API contract, which a second
agent framework doesn't change. The fix is a correct adapter.

## What this delivers

`langchain-google-genai` 4.x (native Gemini 3 thought-signature handling),
which requires LangChain `0.3 → 1.x`. Delete the `gemini_compat.py` bypass.
Gemini 3 then preserves reasoning across tool calls and the agent converges.

## Scope

**Blast radius (measured):** 22 files import `langchain`; 21 use
`langchain_core` directly; 5 `langchain_google_genai`; 4 `langchain_openai`;
`langchain-community` **unused** (drop the dep). 50 test files to re-validate.

- Bump pins (`pyproject.toml` 25-28): `langchain`, `langchain-openai` →
  1.x; `langchain-google-genai` → ^4; remove unused `langchain-community`.
- Migrate `langchain 0.3 → 1.x` API churn across the 21 `langchain_core`
  files (message classes, `.bind_tools`, callbacks, `Runnable` surface — mostly
  import paths + message-API, not rewrites).
- **Delete `app/gemini_compat.py`** and its call site in `app/chain.py:43`
  once 4.x handles signatures natively. Verify no other callers.
- Re-validate the full suite (50 test files) + the agent convergence
  quantification harness (6× the Mission query — must go ~0/6 → passing;
  this retroactively confirms the diagnosis since the temp=0.0 confirmation
  test was skipped per user choice).
- Re-test the OpenAI path too (langchain-openai 1.x) — gpt-4o-mini must still
  work; this migration is provider-agnostic.

## Out of scope

- W8 map/routing work (independent; do not entangle).
- Model/temperature selection policy (separate concern; see notes below).
- Smaller "fix the bypass to round-trip real signatures on 2.1.12" approach —
  user chose the full migration as the durable fix.

## Pre-migration baseline (2026-05-17, clean W10 branch off main)

- Unit suite: **416 pass, 1 fail**. The 1 failure
  (`test_chat_functional.py::test_chat_runs_real_graph_with_tool_call`) is a
  **pre-existing test-ordering flake on `main`**, not a W10 regression: it
  PASSES in isolation, FAILS in the full run. Cause: `gemini_compat.py`
  monkeypatches `langchain_google_genai.chat_models` *module-globally* with a
  sticky guard flag (`_city_concierge_gemini3_patch`) — state leaks across
  tests. **W10 deleting this hack is expected to FIX this flake.** Treat
  "416 pass, this 1 known flake" as green baseline; any *other* failure during
  migration is a real regression.
- Convergence harness: gemini-3.1-pro-preview ~0/6 on the standard 3-stop
  Mission query (the bug). This is the Phase-4 regression oracle: W10 is
  correct only when this goes to consistently passing.

## CORRECTED baseline (2026-05-17, rebranched off post-W8 main)

> The baseline above was measured on a W10 branch that forked **before W8a/b/c**
> merged. That branch is 34 commits behind current `main`. W10 was rebranched
> off current `main` (incl. W8a/b/c) as `feature/agent-w10-langchain-v1-rebased`
> + the 2 W10 docs cherry-picked. The numbers below supersede the section above.

- **Full suite on current main: `451 passed, 39 skipped, 0 failed`** (pre-migration,
  `feature/agent-w10-langchain-v1-rebased`, no code change yet). This is the
  authoritative green-line. ANY failure after migration is a real regression.
- **The documented "known flake" no longer reproduces on current main.**
  `test_chat_functional.py::test_chat_runs_real_graph_with_tool_call` passes in
  the full run AND in the historically-contaminating order
  (`pytest test_gemini_compat.py test_chat_functional.py` → 7 passed). Reason:
  W8c commit `7cabea2` added `mocker.patch("app.agent.revision.itinerary_violations")`
  to that test (test-isolation fix), neutralizing the *symptom*. The
  `gemini_compat.py` global-monkeypatch state-leak (root cause of that flake)
  still exists in the code — W10-3 deleting the hack is what truly eliminates it.
  Revised green criterion: **451/0/39 must hold after migration**, and the
  global monkeypatch must be gone (verified by absence, not by that one test).
- Pre-migration installed: `langchain 0.3.28`, `langchain_core 0.3.84`,
  `langchain-google-genai` (pinned `>=1.0.0,<3.0.0`), `langchain-openai`
  (pinned `>=0.1.0,<1.0.0`), `langchain-community` (pinned, **unused** in code).
- **Blast radius correction:** the Scope section says "delete gemini_compat.py
  and its call site in `app/chain.py:43`". On current main there are **4 call
  sites**, not 1: `app/chain.py:12,43`; `app/agent/critique/vibe.py:25,117`;
  `scripts/eval_agent.py:44,207`; plus `tests/unit/test_gemini_compat.py` (delete
  whole file). `vibe.py`/`eval_agent.py` callers merged after W10 branched.

## Open risks / notes

- LangChain 1.x is a breaking major; `langchain-openai` 1.x may shift the RAG
  chain too. Re-test both providers.
- Temperature: failures were measured at **temp=1.0** (user's choice for the
  Gemini registry version). High temp amplifies wandering even with correct
  signatures. Revisit temp when validating the fix.
- The shared MLflow `production` alias is currently **v2 (Gemini, temp 1.0)** —
  see [[project_mlflow_prod_alias_gemini]]. It is at ~0/6 in prod-equivalent
  testing. Rolling it back to v1 (`make set-production-alias VERSION=1`) is the
  evidence-backed default until W10 lands; **user has chosen to keep Gemini** —
  decision pending, tracked separately, NOT silently actioned.

## Outcome (2026-05-17) — what W10 actually delivered

The original premise (Gemini-3 fails purely on thought signatures; lcgg 4.x
fixes it) was **disproven by evidence**. The investigation uncovered a layered
set of real issues; all the mechanical ones are fixed and committed.

**Shipped & verified (full suite 467 passed / 39 skipped / 0 failed):**
- LangChain 0.3→1.x migration; RAG chain ported off removed `langchain.chains`
  to an LCEL stuff-chain (`build_retrieval_qa`, same I/O contract).
- Resolver cascade: langgraph 1.x, google-genai 1.x, openai 2.x.
- `app/gemini_compat.py` static-bypass hack + all 4 call sites deleted.
- Default `EMBEDDING_TABLE` → `place_embeddings_v2` (v1 was noisy; v2 always
  intended; KG already v2-only). Retrieval SQL tests made default-independent.
- Prompt query-construction contract: every `semantic_search` query must
  carry cuisine/vibe + place type + neighborhood (filters refine, don't
  replace) — models were tanking cosine with bare `"lunch"` queries.
- Central `app/llm_factory.py:build_chat_model` — the single provider→LLM
  seam; all 5 prior duplicate construction sites route through it.
- Reasoning-model support via `langchain-deepseek` / `langchain-moonshot`,
  with **thinking disabled** (not a fragile reasoning_content shim):
  kimi `thinking=False`, deepseek `extra_body thinking.type=disabled`,
  gemini flash-lite `thinking_budget=0`, gemini pro-preview
  `thinking_level="low"` (hard floor; budget=0/minimal both 400).
- Per-model hard-constraint clamps (kimi-k2.6 temp forced 0.6).
- `_ToolLoopChatMoonshot`: backfills empty assistant content (Kimi rejects
  empty assistant messages on replay — both its own empty tool-call turns
  and the content-only reconstruction `_prune_for_llm` makes from them).
- Reusable convergence oracle: `scripts/w10_convergence_check.py`
  (`--provider openai|gemini|deepseek|kimi`).

**Provider results on the standard 3-stop Mission query (live API + prod DB):**

| Provider / model | Mechanically runs | Converges (commits itinerary) |
|---|---|---|
| openai / gpt-4o-mini | yes | 2/6 @ temp 1.0; 4/6 @ temp 0.0 (v2) — best |
| deepseek / deepseek-v4-pro | yes (was hard-400) | not yet (asks clarifying Q / step limit) |
| kimi / kimi-k2.6 | yes (was hard-400 ×2) | not yet (step limit) |
| gemini / gemini-3.1-pro-preview | yes (was hard-400) | 0/3 (step limit) |

**Bottom line:** every provider now runs the agent loop with **zero hard
errors** — the reasoning-state and serialization bugs are fixed. The
**remaining gap is convergence quality**: only gpt-4o-mini reliably commits
a full itinerary; the others run cleanly but tend to hit the step limit /
ask clarifying questions. That is a separate behavioral problem (see
[[project_gemini3_thin_query_root_cause]]) — agent-loop/prompt tuning, not
an integration bug — tracked as follow-up. **gpt-4o-mini is the supported
production / demo driver.**

**Status:** Migration + v2 + prompt + factory/DRY + reasoning-provider
support — implemented & verified on `feature/agent-w10-langchain-v1-rebased`.
Convergence-quality tuning for non-OpenAI providers deferred to follow-up.
