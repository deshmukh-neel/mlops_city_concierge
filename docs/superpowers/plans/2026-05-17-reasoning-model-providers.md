# Reasoning-Model Provider Support (DeepSeek + Kimi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent work with reasoning models by routing all LLM construction through one factory that uses provider-specific LangChain packages (`langchain-deepseek`, `langchain-moonshot`) which round-trip `reasoning_content`, instead of `ChatOpenAI(base_url=...)` which discards it by design.

**Architecture:** Provider→LLM construction is currently duplicated across 5 sites (`app/chain.py`, `app/main.py`, `app/agent/critique/vibe.py`, `scripts/eval_agent.py`, plus `app/config.py:resolve_llm_api_key`). Introduce a single `app/llm_factory.py:build_chat_model(provider, model, temperature)` that centralizes construction and is the only place new providers are added. Route every site through it. Add `deepseek`/`kimi` there using `ChatDeepSeek`/`ChatMoonshot`.

**Tech Stack:** Poetry, langchain-core 1.x, langchain-openai 1.2.1, langchain-google-genai 4.2.2, new: langchain-deepseek 1.0.1, langchain-moonshot 0.1.0. pytest. Validation: `scripts/w10_convergence_check.py` (the reusable oracle).

---

## File Structure

- **Create** `app/llm_factory.py` — single source of truth for provider→`BaseChatModel`. Owns the provider→(package, key, model-default) mapping. One responsibility: build a chat model.
- **Create** `tests/unit/test_llm_factory.py` — unit tests for the factory (mock the provider classes; assert correct class + kwargs per provider; unknown provider raises).
- **Modify** `pyproject.toml` — add `langchain-deepseek`, `langchain-moonshot`.
- **Modify** `app/config.py` — add `deepseek_api_key`, `moonshot_api_key` settings; extend `resolve_llm_api_key`.
- **Modify** `app/chain.py` — replace inline if/elif with `build_chat_model`.
- **Modify** `app/main.py:103-109` — extend provider validation to delegate to factory list.
- **Modify** `app/agent/critique/vibe.py:99-124` — judge construction via `build_chat_model`.
- **Modify** `scripts/eval_agent.py` — `LlmProvider` type + `build_eval_llm` via factory.
- **Modify** `scripts/log_model_to_mlflow.py` — widen `Literal`/`choices` to factory's providers.
- **Modify** `.env.example` — document new keys.
- **Modify** `scripts/w10_convergence_check.py` — route deepseek/kimi through factory (drop the ad-hoc `_OPENAI_COMPAT` ChatOpenAI shim).

---

### Task 1: Add provider packages

**Files:**
- Modify: `pyproject.toml` (AI / embeddings dependency block, ~line 27)

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, after the `langchain-openai` line in `[tool.poetry.dependencies]`:

```toml
langchain-deepseek = ">=1.0.0,<2.0.0"
langchain-moonshot = ">=0.1.0,<1.0.0"
```

- [ ] **Step 2: Lock + install**

Run: `poetry lock && poetry install`
Expected: resolves cleanly (both require `langchain-core <2,>=1` and `langchain-openai <2,>=1` — already satisfied by 1.4.0 / 1.2.1).

- [ ] **Step 3: Verify importability + constructor surface**

Run:
```bash
poetry run python -c "from langchain_deepseek import ChatDeepSeek; from langchain_moonshot import ChatMoonshot; import inspect; print('deepseek', sorted(ChatDeepSeek.model_fields)[:8]); print('moonshot', sorted(ChatMoonshot.model_fields)[:8])"
```
Expected: both import; field lists include `model`/`temperature`/an api-key field. **Record the exact api-key field name for each** (e.g. `api_key` — LangChain convention is `api_key: SecretStr`; Moonshot may use `moonshot_api_key`). Task 3's code MUST use the names observed here.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "build(w10): add langchain-deepseek + langchain-moonshot"
```

---

### Task 2: Add API-key settings + resolver

**Files:**
- Modify: `app/config.py:93` (after `anthropic_api_key`), `app/config.py:144-159` (`resolve_llm_api_key`)
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
def test_resolve_llm_api_key_supports_deepseek_and_kimi(monkeypatch) -> None:
    from app.config import get_settings, resolve_llm_api_key

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key")
    get_settings.cache_clear()
    assert resolve_llm_api_key("deepseek") == "ds-key"
    assert resolve_llm_api_key("kimi") == "ms-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_config.py::test_resolve_llm_api_key_supports_deepseek_and_kimi -v`
Expected: FAIL — `ValueError: Unsupported llm_provider: deepseek`

- [ ] **Step 3: Add settings fields**

In `app/config.py`, immediately after line 93 (`anthropic_api_key: str = ""`):

```python
    deepseek_api_key: str = ""
    moonshot_api_key: str = ""
```

- [ ] **Step 4: Extend resolver**

In `app/config.py`, in `resolve_llm_api_key`, after the `elif provider == "anthropic":` block (line 154-155) and before `else:`:

```python
    elif provider == "deepseek":
        api_key = s.deepseek_api_key
    elif provider == "kimi":
        api_key = s.moonshot_api_key
```

- [ ] **Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_config.py -v`
Expected: PASS (all config tests)

- [ ] **Step 6: Commit**

```bash
git add app/config.py tests/unit/test_config.py
git commit -m "feat(w10): deepseek/kimi api-key settings + resolver"
```

---

### Task 3: Create the LLM factory

**Files:**
- Create: `app/llm_factory.py`
- Test: `tests/unit/test_llm_factory.py`

> Use the exact api-key field names recorded in Task 1 Step 3. The code below assumes the LangChain convention `api_key: SecretStr` for all four classes (`ChatOpenAI`, `ChatDeepSeek`, `ChatMoonshot`) and `google_api_key: SecretStr` for `ChatGoogleGenerativeAI` (already used in `app/chain.py:80-83`). If Task 1 observed a different field for Moonshot/DeepSeek, adjust the corresponding builder lambda only.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_llm_factory.py`:

```python
from __future__ import annotations

import pytest

from app.llm_factory import SUPPORTED_PROVIDERS, build_chat_model


@pytest.mark.parametrize(
    "provider,patch_path",
    [
        ("openai", "app.llm_factory.ChatOpenAI"),
        ("gemini", "app.llm_factory.ChatGoogleGenerativeAI"),
        ("deepseek", "app.llm_factory.ChatDeepSeek"),
        ("kimi", "app.llm_factory.ChatMoonshot"),
    ],
)
def test_build_chat_model_dispatches_per_provider(provider, patch_path, mocker, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch(patch_path, return_value=f"{provider}-llm")

    out = build_chat_model(provider, "some-model", temperature=0.3)

    assert out == f"{provider}-llm"
    cls.assert_called_once()
    _, kwargs = cls.call_args
    assert kwargs["model"] == "some-model"
    assert kwargs["temperature"] == 0.3


def test_build_chat_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        build_chat_model("anthropic", "claude", temperature=0.0)


def test_supported_providers_is_the_contract() -> None:
    assert SUPPORTED_PROVIDERS == ("openai", "gemini", "deepseek", "kimi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_llm_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.llm_factory'`

- [ ] **Step 3: Write the factory**

Create `app/llm_factory.py`:

```python
"""Single source of truth for provider -> chat model construction.

Reasoning models (DeepSeek, Kimi/Moonshot, Gemini) emit opaque reasoning
state with each tool call and require it replayed in history next turn.
`langchain_openai.ChatOpenAI` deliberately does NOT round-trip the
non-standard `reasoning_content` field (its docstring directs you to
provider-specific packages), so OpenAI-compatible reasoning models 400 on
the second tool turn. We therefore use the provider-specific LangChain
classes, which preserve reasoning state. Every provider->LLM construction
in the codebase routes through here so a provider is added in ONE place.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_moonshot import ChatMoonshot
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import resolve_llm_api_key

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "gemini", "deepseek", "kimi")


def build_chat_model(
    llm_provider: str, chat_model: str, temperature: float
) -> BaseChatModel:
    """Construct a chat model for `llm_provider`. Raises ValueError for
    unsupported providers, RuntimeError if the provider's API key is missing
    (both via resolve_llm_api_key)."""
    provider = llm_provider.lower()
    api_key = resolve_llm_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model=chat_model, api_key=SecretStr(api_key), temperature=temperature
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=SecretStr(api_key),
            temperature=temperature,
        )
    if provider == "deepseek":
        return ChatDeepSeek(
            model=chat_model, api_key=SecretStr(api_key), temperature=temperature
        )
    if provider == "kimi":
        return ChatMoonshot(
            model=chat_model, api_key=SecretStr(api_key), temperature=temperature
        )
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_llm_factory.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Typecheck + commit**

Run: `poetry run mypy app/llm_factory.py`
Expected: `Success: no issues found`

```bash
git add app/llm_factory.py tests/unit/test_llm_factory.py
git commit -m "feat(w10): central llm_factory.build_chat_model (4 providers)"
```

---

### Task 4: Route app/chain.py through the factory

**Files:**
- Modify: `app/chain.py:11-13` (imports), `app/chain.py:75-86` (construction)
- Test: existing `tests/unit/test_chain.py` (already mocks `app.chain.ChatOpenAI`/`ChatGoogleGenerativeAI`)

- [ ] **Step 1: Update the test to mock the factory**

In `tests/unit/test_chain.py`, both `test_build_rag_chain_supports_openai` and `test_build_rag_chain_supports_gemini`: replace the `mocker.patch("app.chain.ChatOpenAI", ...)` / `mocker.patch("app.chain.ChatGoogleGenerativeAI", ...)` pairs with a single factory mock. For the openai test:

```python
    retriever_cls = mocker.patch("app.chain.PgVectorRetriever", return_value=retriever)
    factory = mocker.patch("app.chain.build_chat_model", return_value="openai-llm")
    build_qa = mocker.patch("app.chain.build_retrieval_qa", return_value=chain)
```
and replace the `openai_cls.assert_called_once_with(...)` block with:
```python
    factory.assert_called_once_with("openai", "gpt-4o-mini", temperature=0.2)
```
For the gemini test, mirror it: `factory = mocker.patch("app.chain.build_chat_model", return_value="gemini-llm")` and `factory.assert_called_once_with("gemini", "gemini-2.5-flash", temperature=0.1)`. Delete the now-unused `SecretStr` assertions on the model classes.

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_chain.py -v`
Expected: FAIL — `AttributeError: <module 'app.chain'> does not have the attribute 'build_chat_model'`

- [ ] **Step 3: Refactor app/chain.py**

In `app/chain.py`, remove imports at lines 12-13 (`from langchain_google_genai import ChatGoogleGenerativeAI`, `from langchain_openai import ChatOpenAI`) and the now-unused `from pydantic import SecretStr` (verify it is unused elsewhere in the file first; `build_retrieval_qa` does not use it). Add:

```python
from .llm_factory import build_chat_model
```

Replace `app/chain.py:75-86` (the `provider = llm_provider.lower()` through the `else: raise ValueError` block) with:

```python
    llm: BaseChatModel = build_chat_model(llm_provider, chat_model, temperature)
```

(The `api_key` parameter of `build_rag_chain` is now unused for LLM construction — the factory resolves keys itself. Leave the parameter in the signature for now to avoid a wider call-site refactor; add `# noqa: ARG001`-style handling only if a linter flags it. Do NOT change `build_rag_chain`'s public signature in this task.)

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/unit/test_chain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/chain.py tests/unit/test_chain.py
git commit -m "refactor(w10): chain.py builds llm via factory"
```

---

### Task 5: Route vibe.py judge through the factory

**Files:**
- Modify: `app/agent/critique/vibe.py:99-124`
- Test: existing `tests/unit/` vibe/judge tests (find with `grep -rl make_judge tests/`)

- [ ] **Step 1: Inspect existing judge tests**

Run: `grep -rln "make_judge" tests/`
Read each. Note how they currently mock `ChatOpenAI`/`ChatGoogleGenerativeAI` inside `vibe`.

- [ ] **Step 2: Update judge tests to mock the factory**

In each test file from Step 1, replace patches of `app.agent.critique.vibe.ChatOpenAI` (and the Gemini equivalent, imported lazily inside `make_judge`) with `mocker.patch("app.agent.critique.vibe.build_chat_model", return_value=<fake>)`. Keep the existing "missing key returns None" assertions — those stay (see Step 4).

- [ ] **Step 3: Run to verify failure**

Run: `poetry run pytest <judge test files> -v`
Expected: FAIL — `vibe` has no attribute `build_chat_model`.

- [ ] **Step 4: Refactor make_judge**

In `app/agent/critique/vibe.py`, add at top-level imports: `from app.llm_factory import build_chat_model`. Replace the body of the `try:` block (lines 103-120, the two `if provider ==` branches) with:

```python
        key_attr = {"openai": "openai_api_key", "gemini": "gemini_api_key",
                    "deepseek": "deepseek_api_key", "kimi": "moonshot_api_key"}.get(provider)
        if key_attr is not None and not getattr(s, key_attr):
            _log.warning("vibe judge requested but key for %s missing; skipping", provider)
            return None
        return build_chat_model(provider, model, temperature=0.0)
```

(Preserves the existing "missing key → warn + return None → vibe skipped" behavior; the broad `except` on line 121 still guards construction failures.)

- [ ] **Step 5: Run tests**

Run: `poetry run pytest <judge test files> -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/agent/critique/vibe.py tests/
git commit -m "refactor(w10): vibe judge built via factory"
```

---

### Task 6: Route eval_agent + mlflow script + main.py + oracle through the factory

**Files:**
- Modify: `scripts/eval_agent.py:45` (`LlmProvider`), `scripts/eval_agent.py:197-210` (`build_eval_llm`)
- Modify: `scripts/log_model_to_mlflow.py:28,103` (`Literal`/`choices`)
- Modify: `app/main.py:103-109`
- Modify: `scripts/w10_convergence_check.py`

- [ ] **Step 1: eval_agent.py**

Replace `LlmProvider = Literal["openai", "gemini"]` (line 45) with:
```python
LlmProvider = Literal["openai", "gemini", "deepseek", "kimi"]
```
Replace the body of `build_eval_llm` (lines 197-210) with:
```python
    from app.llm_factory import build_chat_model

    return build_chat_model(provider, chat_model, temperature)
```
Remove now-unused imports in `eval_agent.py` (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `resolve_llm_api_key` if unused, `SecretStr` if unused) — run `poetry run ruff check scripts/eval_agent.py` and clear what it flags.

- [ ] **Step 2: log_model_to_mlflow.py**

Line 28: `llm_provider: Literal["openai", "gemini", "deepseek", "kimi"]`.
Line 103: `parser.add_argument("--llm-provider", default="openai", choices=["openai", "gemini", "deepseek", "kimi"])`.

- [ ] **Step 3: main.py**

Replace `app/main.py:103-109` (the `llm_provider == "openai" / elif "gemini" / else raise` validation) with a check against the factory contract:
```python
    from app.llm_factory import SUPPORTED_PROVIDERS

    llm_provider = (params.get("llm_provider") or "openai").lower()
    if llm_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")
```
(Keep the surrounding `chat_model` resolution lines that followed; only the provider-validation if/elif is replaced. Read lines 100-115 first and preserve any model-name defaulting logic.)

- [ ] **Step 4: oracle (w10_convergence_check.py)**

In `scripts/w10_convergence_check.py`: delete the `_OPENAI_COMPAT` dict and its branch in `make_llm`. Replace `make_llm`'s body with:
```python
    from app.llm_factory import build_chat_model

    default = {"openai": "gpt-4o-mini", "gemini": "gemini-3.1-pro-preview"}.get(provider)
    model = chat_model or default
    if model is None:
        raise SystemExit(f"--chat-model required for provider {provider}")
    return build_chat_model(provider, model, temperature), model
```
Keep `--provider` choices as `["openai", "gemini", "deepseek", "kimi"]`.

- [ ] **Step 5: Full suite + typecheck + lint**

Run: `make test`
Expected: all pass, 0 failed (count = prior green + new factory/config tests).
Run: `make typecheck` → `Success`. Run: `make lint` → `All checks passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_agent.py scripts/log_model_to_mlflow.py app/main.py scripts/w10_convergence_check.py
git commit -m "refactor(w10): route eval/mlflow/main/oracle through llm_factory"
```

---

### Task 7: .env.example + end-to-end oracle validation

**Files:**
- Modify: `.env.example`
- Validation only (no code): `scripts/w10_convergence_check.py`

- [ ] **Step 1: Document keys**

In `.env.example`, near the existing `GEMINI_API_KEY`/`OPENAI_API_KEY` lines, add:
```
# Reasoning-model providers (OpenAI-compatible but need provider-specific
# LangChain packages to round-trip reasoning_content — see app/llm_factory.py)
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
MOONSHOT_API_KEY=
MOONSHOT_MODEL=kimi-k2.6
```

- [ ] **Step 2: Commit the doc**

```bash
git add .env.example
git commit -m "docs(w10): document deepseek/moonshot env keys"
```

- [ ] **Step 3: Oracle — DeepSeek (the bug → fixed)**

Run: `poetry run python scripts/w10_convergence_check.py --provider deepseek --runs 6`
Expected: NO `reasoning_content must be passed back` 400s. Record converged/6. **Success criterion: zero reasoning_content 400 errors** (convergence rate is secondary — the library-gap bug is fixed when the 400s are gone; remaining non-convergence is the separate, already-documented thin-query/finalize behavior).

- [ ] **Step 4: Oracle — Kimi**

Run: `poetry run python scripts/w10_convergence_check.py --provider kimi --runs 6`
Expected: NO `reasoning_content is missing in assistant tool call message` 400s. Record converged/6.

- [ ] **Step 5: Oracle — OpenAI regression check**

Run: `poetry run python scripts/w10_convergence_check.py --provider openai --chat-model gpt-4o-mini --temperature 0.0 --runs 6`
Expected: still works (≥4/6, matching the pre-refactor v2 baseline) — the factory refactor must not regress the OpenAI path.

- [ ] **Step 6: Oracle — Gemini (no regression to its partial state)**

Run: `poetry run python scripts/w10_convergence_check.py --provider gemini --chat-model gemini-3.1-pro-preview --runs 6`
Expected: ≥1/6 (no worse than the documented post-prompt-fix v2 baseline; Gemini still routes through lcgg unchanged).

- [ ] **Step 7: Record results**

Append a results table (provider, model, converged/6, any 400s) to `implementation_plan/james/w10_langchain_v1_gemini3.md` under a new `## Reasoning-model provider support (DeepSeek/Kimi)` section. Commit:
```bash
git add implementation_plan/james/w10_langchain_v1_gemini3.md
git commit -m "docs(w10): record reasoning-model provider oracle results"
```

---

## Self-Review

**Spec coverage:** User goal = "do langchain-deepseek integration now, make reasoning models work, validate via oracle." Covered: Task 1 (deps), Task 2 (keys), Task 3 (factory w/ ChatDeepSeek+ChatMoonshot), Tasks 4-6 (route all 5 duplicate sites through it — DRY), Task 7 (oracle validation incl. OpenAI/Gemini no-regression). The original W10 migration + v2 + prompt fix are already committed (`fcd8083`, `ded9fd8`, `361f52d`, `099e819`, `d9601a6`) — out of scope here, not re-touched.

**Placeholder scan:** Task 1 Step 3 intentionally defers exact api-key field names to runtime observation (the only honest option without the packages installed) with an explicit fallback rule — not a placeholder, a verification gate. All code steps contain full code.

**Type consistency:** `build_chat_model(llm_provider: str, chat_model: str, temperature: float) -> BaseChatModel` and `SUPPORTED_PROVIDERS: tuple[str, ...]` used identically in Tasks 3, 4, 5, 6. `resolve_llm_api_key` extended in Task 2 is consumed by the factory in Task 3.

**Risk note:** Task 1 Step 3 is the single point where a wrong assumption (api-key kwarg name for `ChatMoonshot`/`ChatDeepSeek`) would propagate — it is explicitly gated to observe-then-adjust before Task 3 writes the factory.
