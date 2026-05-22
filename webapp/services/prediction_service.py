from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from webapp.config import AppConfig
from webapp.data_sources import SUPPORTED_SPORTS, fetch_sport_events_with_meta
from webapp.errors import ValidationError
from webapp.history_store import save_prediction_snapshot
from webapp.injury_store import get_injury_adjustments_map
from webapp.ml.engines import AdaptiveProbabilityEngine, HeuristicPredictionEngine
from webapp.ml.interfaces import PredictionEngine, ProbabilityEngine
from webapp.services.explainability import enrich_prediction_explainability

LOGGER = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CacheRow:
    ts: float
    payload: dict[str, Any]


class PredictionService:
    def __init__(
        self,
        config: AppConfig,
        *,
        prediction_engine: PredictionEngine | None = None,
        probability_engine: ProbabilityEngine | None = None,
    ) -> None:
        self._config = config
        self._cache: dict[str, CacheRow] = {}
        self._prediction_engine = prediction_engine or HeuristicPredictionEngine()
        self._probability_engine = probability_engine or AdaptiveProbabilityEngine(config.db_path)

    def clear_cache(self, sport: str | None = None) -> None:
        if sport:
            self._cache.pop(sport, None)
            return
        self._cache.clear()

    @staticmethod
    def validate_sport(sport_key: str) -> None:
        if sport_key not in SUPPORTED_SPORTS:
            raise ValidationError(f"Unsupported sport: {sport_key}")

    @staticmethod
    def sport_payload() -> list[dict[str, Any]]:
        return [{"key": cfg.key, "label": cfg.label} for cfg in SUPPORTED_SPORTS.values()]

    @staticmethod
    def serialize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
        start_time = row.get("start_time")
        if isinstance(start_time, datetime):
            kickoff = start_time.astimezone(timezone.utc).isoformat()
        else:
            kickoff = str(start_time)

        return {
            "event_id": row["event_id"],
            "sport": row["sport"],
            "league": row["league"],
            "kickoff_utc": kickoff,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "venue": row["venue"],
            "status": row["status"],
            "model_source": row.get("model_source", "heuristic"),
            "predicted_winner": row["predicted_winner"],
            "predicted_result": row["predicted_result"],
            "confidence": row["confidence"],
            "home_win_probability": row["home_win_probability"],
            "draw_probability": row["draw_probability"],
            "away_win_probability": row["away_win_probability"],
            "predicted_score": row["predicted_score"],
            "factors": row["factors"],
            "confidence_tier": row.get("confidence_tier", "Very Uncertain"),
            "risk_indicators": row.get("risk_indicators", []),
            "factor_contributions": row.get("factor_contributions", []),
            "top_factors": row.get("top_factors", []),
            "explanation": row.get("explanation", ""),
        }

    @staticmethod
    def apply_probabilities(prediction: dict[str, Any], probs: dict[str, float]) -> None:
        home = float(probs.get("H", 0.0))
        draw = float(probs.get("D", 0.0))
        away = float(probs.get("A", 0.0))
        total = home + draw + away
        if total <= 0:
            return

        home /= total
        draw /= total
        away /= total

        prediction["home_win_probability"] = round(home, 4)
        prediction["draw_probability"] = round(draw, 4)
        prediction["away_win_probability"] = round(away, 4)

        winner = "H"
        confidence = home
        predicted_winner = prediction["home_team"]
        if away > confidence:
            winner = "A"
            confidence = away
            predicted_winner = prediction["away_team"]
        if draw > confidence:
            winner = "D"
            confidence = draw
            predicted_winner = "Draw"

        prediction["predicted_result"] = winner
        prediction["predicted_winner"] = predicted_winner
        prediction["confidence"] = round(confidence, 4)

    @staticmethod
    def apply_injury_adjustments(prediction: dict[str, Any], injury_map: dict[str, dict[str, float]]) -> None:
        home_adj = injury_map.get(prediction["home_team"], {})
        away_adj = injury_map.get(prediction["away_team"], {})

        hr = float(home_adj.get("rating_delta", 0.0))
        hf = float(home_adj.get("form_delta", 0.0))
        ho = float(home_adj.get("offense_delta", 0.0))
        hd = float(home_adj.get("defense_delta", 0.0))

        ar = float(away_adj.get("rating_delta", 0.0))
        af = float(away_adj.get("form_delta", 0.0))
        ao = float(away_adj.get("offense_delta", 0.0))
        ad = float(away_adj.get("defense_delta", 0.0))

        signal_home = (hr / 120.0) + (hf * 0.9) + (ho * 0.25) - (hd * 0.2)
        signal_away = (ar / 120.0) + (af * 0.9) + (ao * 0.25) - (ad * 0.2)
        signal_delta = signal_home - signal_away

        home = float(prediction.get("home_win_probability", 0.0))
        draw = float(prediction.get("draw_probability", 0.0))
        away = float(prediction.get("away_win_probability", 0.0))
        if (home + draw + away) <= 0:
            return

        home *= math.exp(signal_delta)
        away *= math.exp(-signal_delta)
        draw *= math.exp(-abs(signal_delta) * 0.2)

        total = home + draw + away
        home /= total
        draw /= total
        away /= total

        prediction["home_win_probability"] = round(home, 4)
        prediction["draw_probability"] = round(draw, 4)
        prediction["away_win_probability"] = round(away, 4)

        winner = "H"
        confidence = home
        predicted_winner = prediction["home_team"]
        if away > confidence:
            winner = "A"
            confidence = away
            predicted_winner = prediction["away_team"]
        if draw > confidence:
            winner = "D"
            confidence = draw
            predicted_winner = "Draw"
        prediction["predicted_result"] = winner
        prediction["predicted_winner"] = predicted_winner
        prediction["confidence"] = round(confidence, 4)

        prediction.setdefault("factors", {})
        prediction["factors"]["injury_adjustment"] = {
            "home": {"rating_delta": hr, "form_delta": hf, "offense_delta": ho, "defense_delta": hd},
            "away": {"rating_delta": ar, "form_delta": af, "offense_delta": ao, "defense_delta": ad},
            "signal_delta": round(signal_delta, 4),
        }

    def build_upcoming_response(self, sport_key: str, *, force_refresh: bool = False) -> dict[str, Any]:
        self.validate_sport(sport_key)

        now = time.time()
        cache_hit = self._cache.get(sport_key)
        if not force_refresh and cache_hit and (now - cache_hit.ts) < self._config.upcoming_cache_ttl_seconds:
            return cache_hit.payload

        completed, upcoming, source_meta = fetch_sport_events_with_meta(
            sport_key,
            days_back=270,
            days_ahead=30,
            force_demo_mode=bool(self._config.demo_mode),
        )
        predictions = self._prediction_engine.predict_upcoming(
            sport_key,
            completed_events=completed,
            upcoming_events=upcoming,
        )
        adaptive_probs, training_info = self._probability_engine.probabilities_for_upcoming(
            sport_key,
            completed_events_from_feed=completed,
            upcoming_events=upcoming,
        )
        injury_map = get_injury_adjustments_map(sport_key, db_path=self._config.db_path)

        for prediction in predictions:
            probs = adaptive_probs.get(str(prediction.get("event_id")))
            if probs:
                self.apply_probabilities(prediction, probs)
                prediction["model_source"] = "adaptive_ml"
            else:
                prediction["model_source"] = "heuristic"
            self.apply_injury_adjustments(prediction, injury_map)
            enrich_prediction_explainability(prediction)

        serialized = [self.serialize_prediction_row(item) for item in predictions[:120]]
        snapshot_id = save_prediction_snapshot(
            sport=sport_key, matches=serialized, db_path=self._config.db_path
        )
        payload = {
            "sport": sport_key,
            "sport_label": SUPPORTED_SPORTS[sport_key].label,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_upcoming_matches": len(predictions),
            "snapshot_id": snapshot_id,
            "training": training_info,
            "injury_adjustments_count": len(injury_map),
            "demo_mode": bool(source_meta.get("mode") == "demo"),
            "data_mode": source_meta.get("mode", "live"),
            "data_source_reason": source_meta.get("reason"),
            "data_provider": source_meta.get("provider"),
            "matches": serialized,
        }
        self._cache[sport_key] = CacheRow(ts=now, payload=payload)
        LOGGER.info(
            "upcoming_predictions_computed",
            extra={"context": {"sport": sport_key, "count": len(predictions)}},
        )
        return payload

    def run_what_if(self, payload: dict[str, Any]) -> dict[str, Any]:
        required_fields = ["sport", "home_team", "away_team"]
        missing = [field for field in required_fields if not payload.get(field)]
        if missing:
            raise ValidationError(f"Missing required field(s): {', '.join(missing)}")

        sport_key = str(payload["sport"])
        self.validate_sport(sport_key)

        prediction = self._prediction_engine.predict_what_if(payload)
        if not _as_bool(payload.get("ignore_injuries", False)):
            injury_map = get_injury_adjustments_map(sport_key, db_path=self._config.db_path)
            self.apply_injury_adjustments(prediction, injury_map)
        enrich_prediction_explainability(prediction)
        return self.serialize_prediction_row(prediction)
