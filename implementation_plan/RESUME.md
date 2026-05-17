# RESUME — where to pick up (written 2026-05-17)

You are on branch **`feature/agent-w8b-live-map`**. This file is your
step-by-step. Work top to bottom. Each step says WHICH BRANCH to be on.

> Why this file exists: W8/W9/W10 planning docs are committed but **branch-local
> and not on `main`**. From `main` you'd see none of this. Always start by
> checking out the branch named in each step.

---

## The big picture (what happened)

- "Add Google Maps to the frontend" grew into 3 workstreams + a root-caused agent bug.
- **W8** = live Google Map + itinerary routing. Phased w8a → w8b → w8c.
- **W9** = weather-aware recs. Documented, deferred (depends on W8c).
- **W10** = the REAL bug: the agent never produces itineraries on Gemini 3
  (`gemini_compat.py` thought-signature bypass). Fix = LangChain 0.3→1.x.
  Root-caused, branched, baselined, **paused**.
- Shared MLflow prod alias is on **Gemini v2 (~0/6, broken)** — your call to
  keep it through W10. Until W10 lands, the deployed agent makes no itineraries.

Branches (all committed, nothing lost):
| Branch | Latest commit | Has |
|---|---|---|
| `feature/agent-w8a-coord-propagation` | `0a8721b` | backend lat/lng in PlaceCard |
| `feature/agent-w8b-live-map` (HERE) | `fec33e1` | everything in w8a + full frontend map + the App.jsx bugfix + W8/W9 docs |
| `feature/agent-w10-langchain-v1` | `27b417f` | W10 plan + baseline only (no code yet) |

`w8b` already contains all of `w8a`'s work (it was branched off w8a).

---

## DO THESE IN ORDER

### STEP 1 — Verify the live map (branch: `feature/agent-w8b-live-map`)
This is the only thing blocking W8 from being "done". Needs YOU (a browser + a
Google key); code is finished and tested (24 frontend tests green).

1. `git checkout feature/agent-w8b-live-map`
2. Get a Google Maps browser key + Map ID — follow `docs/google_maps_setup.md`
   (Maps JavaScript API + Directions API; Map ID Vector, tilt/rotation OFF;
   key referrer-restricted to `http://localhost:5173/*` only).
3. Put them in `frontend/.env.development.local` (file already exists with
   blank `VITE_GOOGLE_MAPS_API_KEY=` / `VITE_GOOGLE_MAPS_MAP_ID=`).
4. Need a backend with data: start cloud-sql-proxy (port 5433) + MLflow tunnel
   (port 5050), then run the local backend (see STEP 1b). Point the frontend
   at it: `frontend/.env.development.local` already has
   `VITE_API_URL=http://localhost:8000`.
5. `cd frontend && npm run dev` (RESTART it — Vite doesn't hot-reload env).
6. Run the **11-point gate** in `docs/w8b_verification.md`. Use the query in
   there. NOTE: the agent itself is broken on Gemini (W10) — to actually see
   pins/route you need the agent to commit an itinerary, which is unreliable
   until W10. Map *rendering* (pins when given coords, fallback panel, etc.)
   can still be checked.
7. **Rotate the Google Maps API key after verifying** — the old one was
   exposed in a chat transcript. Make a new key, swap it in, delete the old.

#### STEP 1b — how to run the local backend (prod data, no local DB)
```
# terminal A: proxy   (gcloud-auth'd)
cloud-sql-proxy --port 5433 "mlops-491820:us-central1:mlops--city-concierge"
# terminal B: mlflow tunnel
make mlflow-tunnel                       # localhost:5050
# terminal C: the app
ENC=$(printf 'pjnhek@gmail.com' | sed 's/@/%40/g'); TOK=$(gcloud sql generate-login-token)
DATABASE_URL="postgresql://$ENC:$TOK@127.0.0.1:5433/mlops-city-concierge" \
MLFLOW_TRACKING_URI=http://localhost:5050 \
poetry run uvicorn app.main:app --port 8000
# /health should show the active model.
```

### STEP 2 — Decide the shared prod alias (no branch; MLflow only)
The shared alias is Gemini v2 = ~0/6 (broken for all classmates until W10).
You chose to keep it. **Reconsider before more time passes.** If you want it
working NOW while W10 is pending:
```
make mlflow-tunnel                       # if not running
MLFLOW_TRACKING_URI=http://localhost:5050 make set-production-alias VERSION=1
```
(v1 = gpt-4o-mini, the only currently-working model. Reversible.)
Memory note: `project_mlflow_prod_alias_gemini.md`.

### STEP 3 — W10: make Gemini actually work (branch: `feature/agent-w10-langchain-v1`)
The real fix. Big, breaking, do it in a FRESH session with energy.
1. `git checkout feature/agent-w10-langchain-v1`
2. Read `implementation_plan/james/w10_langchain_v1_gemini3.md` fully — it has
   the root cause, blast radius (22 files), baseline (416 pass + 1 known
   flake), and the regression oracle (6× Mission query, must go ~0/6 → passing).
3. Execute the migration in the doc's Scope section, methodically:
   bump pins → `poetry lock` → fix `langchain_core` API churn across 21 files
   → delete `app/gemini_compat.py` + its call at `app/chain.py:43` → re-run
   full suite + the convergence harness + the OpenAI path.
4. Verify both providers work. The convergence harness passing retroactively
   confirms the diagnosis.

### STEP 4 — W8c: backend Directions tool (branch: new, off `feature/agent-w8b-live-map`)
Only after W8b is verified (STEP 1). See `implementation_plan/james/w8_live_map_routing.md`
§PR3. Task #3. New branch `feature/agent-w8c-directions-retiming` off w8b.

### STEP 5 — W9: weather (later, depends on W8c)
`implementation_plan/james/w9_weather_aware.md`. Deferred. Don't start until
W8c is in.

---

## Open tasks (also in the task list)
- #3 W8c Directions tool + re-timing node — blocked on #4
- #4 W8b real-API verification — **STEP 1, do first**
- #5 W10 LangChain 1.x migration — **STEP 3, the real fix**
- (#1 w8a, #2 w8b, #6 App.jsx bugfix — DONE)

## PR status
- **w8a → PR #86 OPEN, CI FULLY GREEN, READY TO MERGE**
  (https://github.com/deshmukh-neel/mlops_city_concierge/pull/86). All 5 checks
  pass. Was failing on a stale `test_chat_endpoint.py` key assertion (W8a added
  lat/lng but that test's expected-keys set wasn't updated); fixed by
  cherry-picking `2bd55f7` onto w8a. **Merge it yourself** (don't let the agent
  run `gh pr merge`). Good first action tomorrow.
- **w8b → branch pushed, NO PR yet** — open it only AFTER STEP 1 real-API
  verification. w8b contains all of w8a + the same test fix. If #86 merges
  first, retarget/rebase w8b so its PR shows only the frontend diff.
- w10 → its own PR later (independent).

> Lesson logged: a W8a-shaped change (adding a PlaceCard field) must update
> BOTH hardcoded key-set assertions — `test_agent_state.py` AND
> `test_chat_endpoint.py`. Grep `tests/` for the field set before claiming green.

## Memory (loads automatically every session, any branch)
- `project_gemini3_thought_signatures.md` — W10 root cause
- `project_mlflow_prod_alias_gemini.md` — shared alias situation
These will resurface even if you forget this file exists.
