# Phase 9 / PROV-01 — gpt-5-mini AIMessage shape probe

**Probed:** 2026-06-04
**Plan:** 09-01 (openai-gpt5-adapter)
**Decision:** D-09-03 (probe-then-build)

## langchain-openai version

`1.2.2`

## gpt-5-mini chat_model used

- provider: `openai`
- model: `gpt-5-mini`
- temperature: `1.0`
- built via: `app.llm_factory.build_chat_model("openai", "gpt-5-mini", 1.0)`
- probe query: `'search for a bar in mission'`

## AIMessage additional_kwargs keys

```python
['refusal']
```

Values (redacted for sk- patterns):

```python
{'refusal': 'None'}
```

## AIMessage response_metadata

```python
{'token_usage': {'completion_tokens': 525, 'prompt_tokens': 12, 'total_tokens': 537, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 384, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': '<redacted-for-probe>', 'id': '<redacted-for-probe>', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}
```

## AIMessage content shape

str (len=541)

## AIMessage usage_metadata

```python
{'input_tokens': 12, 'output_tokens': 525, 'total_tokens': 537, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 384}}
```

## AIMessage tool_calls

```python
[]
```

## Raw dict(message) dump

```python
{'content': 'Do you mean the Mission District in San Francisco, the city of Mission (e.g., Mission, BC), or some other "Mission"? \n\nAlso tell me:\n- Any vibe or type you want (cocktails, dive bar, craft beer, wine bar, karaoke, late-night, patio, dog-friendly, etc.)\n- How far you want to go (walking distance, <2 miles, drive)\n- Whether you want ratings, hours, directions or to make a reservation\n\nI can then search and give a short list with addresses, hours, ratings and a map link. If you want, I can use your current location to find nearby options.', 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 525, 'prompt_tokens': 12, 'total_tokens': 537, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 384, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-mini-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-DnFK79F7WbXz1DQMvCWZMfhwDgDlF', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'lc_run--019e95c7-982f-7b50-aae3-766ad4ee7e98-0', 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 12, 'output_tokens': 525, 'total_tokens': 537, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 384}}}
```

## Verdict

subclass required

### Interpretation

- `reasoning_content` in additional_kwargs: **False**
- `reasoning` in response_metadata: **False**

If the verdict is `kwarg path works`, Plan 09-01 Task 2 takes **Path A** — the
adapter's `capture_reasoning_state` reads `message.additional_kwargs.get("reasoning_content")`
directly and `replay_reasoning_state` writes the same key back on the most-recent
outbound `AIMessage`.

If the verdict is `subclass required`, Plan 09-01 Task 2 takes **Path B** — we
introduce an `OpenAIReasoningChatModel(ChatOpenAI)` in `app/llm_factory.py` that
overrides `_generate` to lift the raw response's `reasoning_content` field into
`AIMessage.additional_kwargs["reasoning_content"]` BEFORE LangChain's normalizer
drops it. The adapter then reads from the subclass-enriched message.

If the verdict is `neither — escalate`, PROV-01 is library-blocked and the
Phase 9 PR cannot ship per D-09-02; we open `09-PROV-01-BLOCKER.md`.
