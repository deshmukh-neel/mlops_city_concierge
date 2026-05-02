from __future__ import annotations

import openai
import psycopg2
from google.api_core import exceptions as google_exceptions


def status_for_invoke_error(exc: BaseException) -> tuple[int, str]:
    """Map an exception raised by rag_chain.invoke to (status_code, public_message_prefix).

    Returns the most specific match. Vendor exception classes are imported here so
    routes.py doesn't need to know about openai/psycopg2/google internals.
    """
    if isinstance(exc, openai.RateLimitError | google_exceptions.ResourceExhausted):
        return 429, "LLM rate limit hit"
    if isinstance(exc, openai.OpenAIError | google_exceptions.GoogleAPIError):
        return 502, "Upstream LLM error"
    if isinstance(exc, psycopg2.OperationalError):
        return 503, "Database temporarily unavailable"
    if isinstance(exc, psycopg2.DatabaseError):
        return 500, "Database error"
    return 500, "Internal server error"
