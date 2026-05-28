---
phase: 05
slug: rationale-stop-alignment-fix
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-27
register_authored_at_plan_time: false
security_audit_mode: retroactive-stride
---

# Phase 05 - Security

Per-phase security contract: retroactive STRIDE register, accepted risks, and
audit trail for Phase 05, Rationale-Stop Alignment Fix.

---

## Input State

State B: no pre-existing `05-SECURITY.md`; `05-01-PLAN.md`, `05-02-PLAN.md`,
`05-01-SUMMARY.md`, and `05-02-SUMMARY.md` exist.

No plan file contained a parseable `<threat_model>` block, so this audit ran in
retroactive-STRIDE mode. No `## Threat Flags` section was present in either
summary file.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Google Places/retrieval to backend | `PlaceHit` candidate names, primary types, addresses, ratings, coordinates, and distances are converted into committed `Stop` models during closure swaps. | Third-party place metadata; user-visible text fields. |
| Prior conversation state to swap node | Existing `Stop.rationale` values can be inherited when a closed stop is replaced. | LLM-authored or previously synthesized rationale text. |
| Backend final reply to frontend | `summarize_stops()` emits the final itinerary text using stop names and rationales. | User-visible chat reply containing external place metadata and rationale text. |
| Frontend API adapter to chat renderer | The Vite/React client converts plain backend reply text into HTML with line breaks before `ChatMessage` renders it. | Escaped HTML string rendered via `dangerouslySetInnerHTML`. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-05-01 | Tampering | `app/agent/swap.py::_candidates_to_matches` | mitigate | Candidate rationales are generated for the replacement stop, not the closed stop. The code defaults to a deterministic fallback containing `c.name`, and only inherits an existing rationale after probing `is_rationale_aligned()` against the candidate stop. Evidence: `app/agent/swap.py:235-248`, `app/agent/critique/checks.py:298-321`, `tests/unit/test_swap.py:608-686`. | closed |
| T-05-02 | Tampering | Legacy conversation state | mitigate | A pre-Phase-5 placeholder already stored in conversation state is denied inheritance when it contains `"Walking-distance alternative for"`, forcing the deterministic fallback instead. Evidence: `app/agent/swap.py:239`, `tests/unit/test_swap.py:689-731`. | closed |
| T-05-03 | Spoofing / XSS | Backend reply rendered by frontend chat | mitigate | The backend summary is deterministic plain text, and the first-party frontend escapes reply text before inserting `<br>` and rendering via `dangerouslySetInnerHTML`. Evidence: `app/agent/revision.py:295-310`, `frontend/src/api/chat.js:72-82`, `frontend/src/components/ChatMessage.jsx:133-142`, `frontend/src/api/chat.test.js`. | closed |
| T-05-04 | Denial of Service | Closure-swap hot path | mitigate | The Phase 5 rationale fix does not add LLM, DB, or network calls to candidate rationale generation. It uses a local pure scorer helper and a deterministic fallback string, while route/search/enrichment behavior remains bounded by existing swap-node controls. Evidence: `app/agent/swap.py:235-248`, `app/agent/critique/checks.py:298-321`, `tests/unit/test_swap_node.py:266-356`. | closed |

---

## Accepted Risks Log

No accepted risks.

---

## Unregistered Flags

No `## Threat Flags` entries were present in `05-01-SUMMARY.md` or
`05-02-SUMMARY.md`.

---

## Verification Notes

Focused checks run for this audit:

| Check | Result |
|-------|--------|
| `.venv/bin/pytest tests/unit/test_swap.py tests/unit/test_swap_node.py tests/unit/test_critique_checks.py::test_rationale_stop_alignment_catches_closure_swap_placeholder_bleed -v` | 38 passed |
| `npm test -- src/api/chat.test.js` from `frontend/` | 8 passed |
| `.venv/bin/ruff check app/agent/swap.py tests/unit/test_swap.py tests/unit/test_swap_node.py` | All checks passed |
| Manual legacy-placeholder probe through `_candidates_to_matches` | Returned `Walking-distance alternative to Pizzeria Delfina, featuring Tony's Pizza Napoletana.` and `is_rationale_aligned(...) == True` |

Notes:

- `pytest` and `poetry` were not on the shell PATH. The repository-local
  `.venv/bin/pytest`, `.venv/bin/python`, and `.venv/bin/ruff` were used
  instead.
- `frontend/src/components/ChatMessage.jsx` uses `dangerouslySetInnerHTML`, but
  `frontend/src/api/chat.js` escapes backend reply text before it reaches that
  renderer.
- The literal legacy substring remains in `app/agent/swap.py` only as a
  deny-list guard, and in tests/docs as regression evidence. It is no longer
  emitted as the candidate rationale.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-27 | 4 | 4 | 0 | Codex gsd-secure-phase |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-27
