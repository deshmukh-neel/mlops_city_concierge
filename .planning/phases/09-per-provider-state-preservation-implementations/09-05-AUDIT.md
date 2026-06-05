---
phase: 09
plan: 05
audit_type: revertability
date: 2026-06-05
audit_head_at_start: 218cf5da749a11f5f32d46c600792a14eec01207
status: in-progress
---

# Phase 9 / PROV-05 Revertability Audit

**Phase:** `09-per-provider-state-preservation-implementations`
**Plan:** `09-05-revertability-audit`
**Branch:** `gsd/phase-09-per-provider-state-preservation-implementations`
**HEAD at audit start:** `218cf5da749a11f5f32d46c600792a14eec01207` (`docs(09-04): complete gemini3-experimental-adapter plan (PROV-04 SHIPPED-STRUCTURAL)`)
**Pre-audit working tree:** clean (`git status --short` = no output)
**Pre-audit baseline `make test-unit`:** `1051 passed, 7 skipped` in 489.70s (current HEAD)

## TL;DR

PROV-05 SC #5 verdict pending Part 2 (per-sub-phase revert dry-run); updated below after each revert lands.

This audit is two-part:

1. **Part 1 — Static import-isolation grep (D-09-07 enforcement).** Each of the 4 adapter files (`openai_gpt5.py`, `deepseek.py`, `anthropic.py`, `gemini.py`) is grepped for sibling-adapter imports. Convention: imports ONLY from `app.agent.adapters` base + `langchain_core` + stdlib.
2. **Part 2 — Per-sub-phase revert dry-run + `make test` (PROV-05 SC #5 enforcement).** For each PROV-NN, dry-run revert that sub-phase's commits via `git revert --no-commit <range>`, run `make test`, capture the result, then `git reset --hard HEAD` to restore.

## PROV-05 SC #5 acceptance text (verbatim from ROADMAP.md Phase 9)

> "Each provider sub-phase ships as an independently revertable commit; reverting any one sub-phase leaves the remaining adapters and the v2.0 `openai/gpt-4o-mini` anchor fully functional in prod; verified by running `make test` after each revert (PROV-05)."

---

## Per-Sub-Phase Import-Isolation Audit

**Convention under test (D-09-07, 09-CONTEXT.md lines 44):**
> "PROV-05 revert atomicity enforced by CONTEXT.md convention, not test: `app/agent/adapters/<provider>.py` imports ONLY from `app.agent.adapters` base + `langchain_core` + stdlib; never from a sibling adapter file."

**Exemption (D-09-07):** `app/agent/adapters/__init__.py` is the assembly point — it MUST import `from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter` (and 3 siblings) to populate the `ADAPTERS` registry at module load. Per-provider files do NOT import each other; THAT is the PROV-05 isolation guarantee.

### Grep 1 — full import lists per adapter (literal `grep -nE "^(from|import) " app/agent/adapters/<file>.py`)

**`app/agent/adapters/openai_gpt5.py`:**
```
32:from __future__ import annotations
34:from langchain_core.messages import AIMessage, BaseMessage
36:from app.agent.adapters import ProviderAdapter, StatePayload
```

**`app/agent/adapters/deepseek.py`:**
```
32:from __future__ import annotations
34:from langchain_core.messages import AIMessage, BaseMessage
36:from app.agent.adapters import ProviderAdapter, StatePayload
```

**`app/agent/adapters/anthropic.py`:**
```
43:from __future__ import annotations
45:from langchain_core.messages import AIMessage, BaseMessage
47:from app.agent.adapters import ProviderAdapter, StatePayload
```

**`app/agent/adapters/gemini.py`:**
```
90:from __future__ import annotations
92:from langchain_core.messages import AIMessage, BaseMessage
94:from app.agent.adapters import ProviderAdapter, StatePayload
```

### Grep 2 — sibling-adapter import count (must be 0 for every file)

Command: `grep -cE "^from app\.agent\.adapters\.(openai_gpt5|deepseek|anthropic|gemini) " app/agent/adapters/<file>.py`

| File              | sibling-adapter imports |
|-------------------|-------------------------|
| `openai_gpt5.py`  | **0** ✅ |
| `deepseek.py`     | **0** ✅ |
| `anthropic.py`    | **0** ✅ |
| `gemini.py`       | **0** ✅ |

**Combined assertion** (single command, exit 0 if all 4 counts sum to 0):
```bash
grep -cE "^from app\.agent\.adapters\.(openai_gpt5|deepseek|anthropic|gemini) " \
  app/agent/adapters/openai_gpt5.py app/agent/adapters/deepseek.py \
  app/agent/adapters/anthropic.py app/agent/adapters/gemini.py \
  | awk -F: '{sum+=$2} END {exit (sum==0)?0:1}'
# exit 0 ✅ (no sibling imports anywhere)
```

### Grep 3 — allowed-import category sanity check

| File              | sibling-imports (must=0) | `langchain_core` | `app.agent.adapters` base | stdlib (`__future__`/typing/etc) |
|-------------------|--------------------------|------------------|---------------------------|----------------------------------|
| `openai_gpt5.py`  | 0 ✅                     | 1                | 1                         | 1 (`__future__`)                 |
| `deepseek.py`     | 0 ✅                     | 1                | 1                         | 1 (`__future__`)                 |
| `anthropic.py`    | 0 ✅                     | 1                | 1                         | 1 (`__future__`)                 |
| `gemini.py`       | 0 ✅                     | 1                | 1                         | 1 (`__future__`)                 |

All four files exhibit the identical 3-line import header:

```
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.adapters import ProviderAdapter, StatePayload
```

This is the canonical D-09-07 shape. The fact that all 4 adapters are byte-identical at the import-header level is itself the strongest evidence of convention enforcement: there is no provider-specific accident that drifted; the rule was applied uniformly.

### Part 1 Verdict — PASS

**All 4 adapter files satisfy D-09-07 import isolation.** Zero sibling-adapter imports anywhere; only the three sanctioned import categories (`__future__`, `langchain_core`, `app.agent.adapters` base) appear. PROV-05 static gate met.

---

## Per-Sub-Phase Revert Simulation

*(Part 2 — recorded in subsequent commit `audit(09-05): per-sub-phase revert dry-run + make test (PROV-05 part 2)`.)*

---
