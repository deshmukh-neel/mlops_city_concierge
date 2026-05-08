"""Deterministic + cheap-LLM critique for the agent.

`checks.py` is the canonical home for the itinerary check functions. W6's
eval pipeline imports from here, not the reverse — when W6 lands it will
reuse these to compute per-query metrics.
"""

# Single source of truth for critique-message prefixes. The LLM is told to
# recognize these via REVISION_GUIDANCE in prompts.py; the graph emits them
# from the critique node. Keep them in sync by importing both sides.
CRITIQUE_STEP = "[critique:step]"
CRITIQUE_ITINERARY = "[critique:itinerary]"
CRITIQUE_VIBE = "[critique:vibe]"
