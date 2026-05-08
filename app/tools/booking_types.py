"""Closed set of booking providers, factored out so both app.tools.booking
(producer) and app.agent.state (consumer, via Stop / PlaceCard) can share one
definition without a circular import.

Adding a new provider is a single-source-of-truth change here; both the URL
builder and the persisted Stop will pick it up."""

from __future__ import annotations

from typing import Literal

Provider = Literal["resy", "tock", "opentable", "google_maps", "unknown"]
