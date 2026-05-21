from __future__ import annotations

from pathlib import Path
from typing import Any

from webapp.adaptive_model import adaptive_probabilities_for_upcoming
from webapp.prediction_engine import predict_upcoming_matches, run_what_if_scenario


class HeuristicPredictionEngine:
    """Adapter around the heuristic prediction module."""

    def predict_upcoming(
        self,
        sport_key: str,
        *,
        completed_events: list[dict[str, Any]],
        upcoming_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return predict_upcoming_matches(
            sport_key,
            completed_events=completed_events,
            upcoming_events=upcoming_events,
        )

    def predict_what_if(self, payload: dict[str, Any]) -> dict[str, Any]:
        return run_what_if_scenario(payload)


class AdaptiveProbabilityEngine:
    """Adapter around the adaptive scikit-learn training/probability module."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def probabilities_for_upcoming(
        self,
        sport_key: str,
        *,
        completed_events_from_feed: list[dict[str, Any]],
        upcoming_events: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        return adaptive_probabilities_for_upcoming(
            sport_key,
            completed_events_from_feed=completed_events_from_feed,
            upcoming_events=upcoming_events,
            db_path=self._db_path,
        )
