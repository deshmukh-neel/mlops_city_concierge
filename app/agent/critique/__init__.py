"""Deterministic + cheap-LLM critique for the agent.

`checks.py` is the canonical home for the itinerary check functions. W6's
eval pipeline imports from here, not the reverse — when W6 lands it will
reuse these to compute per-query metrics.
"""
