from __future__ import annotations

from typing import Any, Protocol


class PredictionEngine(Protocol):
    """Contract for engines that produce per-match prediction rows."""

    def predict_upcoming(
        self,
        sport_key: str,
        *,
        completed_events: list[dict[str, Any]],
        upcoming_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def predict_what_if(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ProbabilityEngine(Protocol):
    """Contract for engines that produce outcome probabilities per event id."""

    def probabilities_for_upcoming(
        self,
        sport_key: str,
        *,
        completed_events_from_feed: list[dict[str, Any]],
        upcoming_events: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]: ...
