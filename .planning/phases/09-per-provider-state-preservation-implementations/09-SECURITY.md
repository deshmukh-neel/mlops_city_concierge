---
phase: 09
slug: per-provider-state-preservation-implementations
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-08
---

# Phase 9 — Security

> Per-phase security contract: threat register verification, accepted risks, and audit trail.
>
> Phase shipped 3 SHIPPED-WITH-GAP + 1 SHIPPED-STRUCTURAL + 1 PASS-WITH-FINDINGS plans. Code-review iteration applied 5 fixes (CR-01 BLOCKER + WR-01..WR-05 WARNINGs + IN-01 bundled); `09-REVIEW-FIX.md` reports `all_fixed`. WR-05 (`OpenAIReasoningChatModel._lift_reasoning_blocks` inner-dict copy) is the mutation-safety regression guard cross-referenced from T-09-01-T3.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Probe artifact ↔ committed repo (`09-PROV-01-PROBE.md`) | One-shot live probe of `gpt-5-mini` shape; persisted in committed planning notes. | `additional_kwargs` keys, sanitized `response_metadata`, raw content shape. MUST NOT cross with `OPENAI_API_KEY`, `Authorization`, or any `sk-*` secret. |
| Local-only matrix gates ↔ CI workflow files | Phase 9 PROV-01..04 merge gates run via `make eval-matrix-refinement` on a developer laptop with provider keys exported into the shell. | Provider API keys (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) stay in the developer's shell; CI only runs structural-check + `scripts/check_baselines_fresh.py`. |
| Adapter capture call ↔ inbound `AIMessage` | `capture_reasoning_state(message)` reads provider-specific reasoning state from each provider's incoming `AIMessage`. | Reasoning content (str), signed thinking blocks (list[dict]), thought signatures (bytes / dict). MUST NOT mutate input `additional_kwargs` or `content` (T3 mitigations). |
| Adapter replay call ↔ outbound message list | `replay_reasoning_state(outbound, state)` writes captured state back onto the next outbound `AIMessage` so the wire-level provider contract is satisfied. | For Anthropic the `signature` field MUST round-trip byte-identical or the API returns 400 (T-09-03-T6). |
| `langchain-anthropic` ↔ application code | New third-party dependency added in Plan 09-03 (pinned `>=0.3.0,<1.0.0`; lock pins `0.3.21`). | Library code path executes against `ANTHROPIC_API_KEY` at runtime; supply-chain risk surface (T-09-03-SC). |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-09-01-T1 | Information Disclosure | `scripts/probe_gpt5_capture.py` + `09-PROV-01-PROBE.md` | mitigate | Probe artifact grep `sk-[A-Za-z0-9]{20,}` returns **zero** matches. Probe script line 15-16 explicitly declares it logs only kwarg names + sanitized `response_metadata`. The `system_fingerprint` and `id` fields are visibly `<redacted-for-probe>` in the artifact. | closed |
| T-09-01-T2 | Information Disclosure | CI live-provider key leak (OpenAI gpt-5) | mitigate | `git log --since=2026-06-04 -- .github/workflows/` returns empty. No workflow files added or modified during Phase 9 (last workflow commit was `b7f5982` for Phase 6, 2026-06-03). CI continues with structural-check + `check_baselines_fresh.py` only; live gates run LOCAL-only (D-09-10). | closed |
| T-09-01-T3 | Tampering | `OpenAIReasoningAdapter.capture_reasoning_state` mutation safety | mitigate | `tests/unit/test_adapters.py::test_openai_reasoning_adapter_capture_does_not_mutate_input_message` (lines 95-117). Test mutates returned payload then asserts `dict(msg.additional_kwargs)` before/after equality. Adapter implementation `openai_gpt5.py:64` returns fresh dict `{"provider": ..., "reasoning_content": reasoning}`. **WR-05 reinforcement**: upstream `OpenAIReasoningChatModel._lift_reasoning_blocks` in `app/llm_factory.py:118-119` uses `[dict(block) for block in ...]` per-block copy (was list-only copy; fixed in commit `23f561f`) — additional defense against inner-dict aliasing between `msg.content` and `additional_kwargs["reasoning_content"]`. | closed |
| T-09-02-T2 | Information Disclosure | CI `DEEPSEEK_API_KEY` leak | mitigate | No workflow file added or modified during Phase 9. `grep DEEPSEEK_API_KEY` over `.github/workflows/*.yml` returns zero matches. PROV-02 gate (`refinement_minimal_edit` median ≥ 0.6) is local-only (D-09-10). | closed |
| T-09-02-T3 | Tampering | `DeepSeekReasonerAdapter.capture_reasoning_state` mutation safety | mitigate | `tests/unit/test_adapters.py::test_deepseek_reasoner_adapter_capture_does_not_mutate_input_message` (lines 196-218). Adapter implementation `deepseek.py:74` returns fresh dict — does not alias `message.additional_kwargs`. | closed |
| T-09-02-T4 | Tampering | Factory branch leaking thinking-enabled into `deepseek-chat` | mitigate | `tests/unit/test_llm_factory.py::test_deepseek_chat_keeps_thinking_disabled` (line 90). Factory carve-out at `app/llm_factory.py:197` uses `frozenset({"deepseek-reasoner"})` membership lookup (NOT `startswith`), so `deepseek-chat` and `deepseek-v4-pro` paths cannot accidentally enable thinking. The carve-out is checked at line 430 — `extra_body` defaults to thinking-disabled, only flipped when `chat_model in _DEEPSEEK_REASONER_THINKING_ENABLED`. | closed |
| T-09-03-T2 | Information Disclosure | CI `ANTHROPIC_API_KEY` leak | mitigate | No workflow file added or modified during Phase 9. `grep ANTHROPIC_API_KEY` in `.github/workflows/ci.yml` returns zero matches. The `ANTHROPIC_API_KEY` reference in `docker.yml:286` is the pre-existing Cloud Run **runtime deploy** secret (file last modified `2026-05-06` per `git log`), not a Phase 9 CI test-gate addition. | closed |
| T-09-03-T3 | Tampering | `AnthropicAdapter.capture_reasoning_state` mutation of `message.content` list | mitigate | `tests/unit/test_adapters.py::test_anthropic_adapter_capture_does_not_mutate_input_message_content` (lines 500-529). Test sets `payload["thinking_blocks"][0]["signature"] = "TAMPERED"` and asserts `msg.content[0]["signature"] == "abc"` — proves shallow-copied blocks. Adapter implementation `anthropic.py:97-100` uses `[dict(b) for b in thinking_blocks]`. | closed |
| T-09-03-T6 | Tampering — signature integrity | Anthropic signed `thinking_blocks` byte-identity contract | mitigate | **Triple-defended.** (1) Unit test `test_anthropic_adapter_replay_prepends_thinking_blocks_to_list_content` (line 304-305) asserts `msg_content[0]["signature"] == "abc"`. (2) Unit test `test_anthropic_adapter_replay_is_idempotent_when_thinking_blocks_already_present` (lines 308-364, signature check at 353-356) verifies the live-run idempotency fix (commit `38b567a`) — when target already has blocks, replay leaves the ORIGINAL wire-correct blocks untouched (Anthropic's `400` "blocks cannot be modified" contract makes this mandatory). (3) Integration test `test_reason_02_anthropic_real_adapter` (lines 488-528) drives the full `graph.ainvoke` round-trip and asserts `first_block.get("signature") == "abc"` end-to-end. Idempotency fix preserves byte-identity by skipping the prepend when blocks already exist (target wins; original signatures remain on the wire). | closed |
| T-09-03-SC | Supply Chain | `langchain-anthropic` new dep | mitigate | `pyproject.toml:29` pins `langchain-anthropic = ">=0.3.0,<1.0.0"`. `poetry.lock:1859-1860` resolves to `langchain-anthropic==0.3.21`. Package is published by `langchain-ai` (the official LangChain Anthropic integration); already referenced from `app/config.py:159` per planning notes. Pin range bounds future minor/patch upgrades without majors. | closed |
| T-09-04-T2 | Information Disclosure | CI `GEMINI_API_KEY` leak | mitigate | No workflow file added or modified during Phase 9. `grep GEMINI_API_KEY` in `.github/workflows/ci.yml` returns zero matches. (Reference in `docker.yml:253,258` is pre-existing Cloud Run runtime deploy secret documentation, not a CI test gate.) PROV-04 is EXPERIMENTAL — empirical median is logged-not-gated (D-09-08); no live keys reach CI. | closed |
| T-09-04-T3 | Tampering | `GeminiAdapter.capture_reasoning_state` mutation of `additional_kwargs` / `tool_calls` | mitigate | `tests/unit/test_adapters.py::test_gemini_adapter_capture_does_not_mutate_input_message` (line 668) and `::test_gemini_adapter_capture_does_not_mutate_real_lcgg_function_call_map` (line 769). Adapter `gemini.py:165` returns `dict(fc_signatures)` shallow-copy for the real lcgg-shape path; line 182 returns `bytes(signature)` defensive copy for the synthetic fixture path (bytes is immutable, so identity == equality, but the explicit copy documents intent for reviewers). | closed |
| T-09-04-T7 | Tampering | Bytes payload corruption across `add_messages` reducer | mitigate | `tests/unit/test_adapters.py::test_gemini_adapter_capture_to_replay_round_trip_preserves_bytes_byte_for_byte` (line 697). Integration test `test_reason_02_gemini_real_adapter` in `tests/integration/test_reasoning_state_roundtrip.py` (line 542+) drives the round-trip through `graph.ainvoke` end-to-end and asserts the bytes survive the reducer. PROV-04 acceptance per D-09-08. | closed |
| T-09-05-T8 | Tampering | Working-tree corruption from revert-then-abort | mitigate | `09-05-AUDIT.md:32` explicitly documents `git reset --hard HEAD` between per-sub-phase revert experiments. `09-05-AUDIT.md:16` confirms pre-audit working tree is clean (`git status --short` = no output). Audit is local-only, never pushed. | closed |
| T-09-05-T9 | Repudiation | Audit results unverifiable | mitigate | `09-05-AUDIT.md` pastes verbatim `make test` summary lines (e.g. line 148 `1051 passed, 7 skipped`, line 205 `1038 passed, 49 skipped, 8 deselected`, line 211 `1 failed, 1023 passed, ...`) and per-sub-phase commit SHAs (audit-head `218cf5da`, PROV-04 range `10e88b9..17e9187`, PROV-03 range `8850371..92c92b6`, PROV-02 commit `3800737`). Audit is the permanent committed record. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Unregistered Flags

None. No `## Threat Flags` section appeared in any of `09-01-SUMMARY.md` through `09-05-SUMMARY.md`. No new attack surface was flagged by the executor during implementation — all surface area was covered by the plan-time threat register (13/13 mapped).

---

## Accepted Risks Log

No accepted risks.

(Phase shipped 3 SHIPPED-WITH-GAP + 1 SHIPPED-STRUCTURAL + 1 PASS-WITH-FINDINGS verdicts at the eval/empirical gate level, but those are gate-miss product-level findings — NOT security threats. The security register's 13 threats all closed cleanly.)

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-08 | 13 | 13 | 0 | gsd-secure-phase (Claude Opus 4.7) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log (none)
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-08
