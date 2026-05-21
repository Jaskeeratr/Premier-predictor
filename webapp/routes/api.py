from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from webapp.adaptive_model import get_latest_model_summary
from webapp.errors import AppError, ValidationError
from webapp.services import HistoryService, InjuryService, PredictionService

LOGGER = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _as_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValidationError("limit must be an integer") from exc


def _prediction_service() -> PredictionService:
    service = current_app.extensions["prediction_service"]
    assert isinstance(service, PredictionService)
    return service


def _history_service() -> HistoryService:
    service = current_app.extensions["history_service"]
    assert isinstance(service, HistoryService)
    return service


def _injury_service() -> InjuryService:
    service = current_app.extensions["injury_service"]
    assert isinstance(service, InjuryService)
    return service


@api_bp.errorhandler(AppError)
def handle_app_error(exc: AppError):
    return jsonify({"error": str(exc)}), exc.status_code


@api_bp.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    LOGGER.exception("api_unhandled_exception")
    return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@api_bp.get("/sports")
def sports() -> Any:
    return jsonify({"sports": _prediction_service().sport_payload()})


@api_bp.get("/model-summary")
def model_summary() -> Any:
    sport_key = request.args.get("sport", "football")
    _prediction_service().validate_sport(sport_key)
    db_path_value = current_app.config.get("DB_PATH")
    summary = get_latest_model_summary(sport_key, db_path=Path(db_path_value) if db_path_value else None)
    if not summary:
        return jsonify({"status": "not_available", "sport": sport_key})
    return jsonify({"status": "ok", "sport": sport_key, "summary": summary})


@api_bp.get("/upcoming")
def upcoming() -> Any:
    sport_key = request.args.get("sport", "football")
    force_refresh = str(request.args.get("force", "0")).strip().lower() in {"1", "true", "yes"}
    payload = _prediction_service().build_upcoming_response(sport_key, force_refresh=force_refresh)
    return jsonify(payload)


@api_bp.post("/what-if")
def what_if() -> Any:
    payload = request.get_json(silent=True) or {}
    return jsonify(_prediction_service().run_what_if(payload))


@api_bp.get("/injuries")
def injuries() -> Any:
    sport_key = request.args.get("sport")
    team_term = request.args.get("team")
    limit = _as_int(request.args.get("limit"), default=500)

    if sport_key:
        _prediction_service().validate_sport(sport_key)
    rows = _injury_service().list_rows(sport=sport_key, team_term=team_term, limit=limit)
    return jsonify({"count": len(rows), "rows": rows})


@api_bp.put("/injuries")
def upsert_injury() -> Any:
    payload = request.get_json(silent=True) or {}
    sport_key = payload.get("sport")
    if sport_key:
        _prediction_service().validate_sport(str(sport_key))
    row = _injury_service().upsert(payload)
    _prediction_service().clear_cache(str(sport_key))
    return jsonify({"saved": row})


@api_bp.delete("/injuries")
def delete_injury() -> Any:
    payload = request.get_json(silent=True) or {}
    sport_key = payload.get("sport")
    team = payload.get("team")
    if sport_key:
        _prediction_service().validate_sport(str(sport_key))
    deleted = _injury_service().delete(sport=sport_key, team=team)
    if sport_key:
        _prediction_service().clear_cache(str(sport_key))
    else:
        _prediction_service().clear_cache()
    return jsonify({"deleted_rows": deleted})


@api_bp.get("/history")
def history() -> Any:
    sport_key = request.args.get("sport")
    team = request.args.get("team")
    league = request.args.get("league")
    limit = _as_int(request.args.get("limit"), default=200)

    if sport_key:
        _prediction_service().validate_sport(sport_key)
    rows = _history_service().list_rows(sport=sport_key, team=team, league=league, limit=limit)
    return jsonify({"count": len(rows), "rows": rows})


@api_bp.delete("/history")
def clear_history() -> Any:
    payload = request.get_json(silent=True) or {}
    sport_key = payload.get("sport")
    if sport_key:
        _prediction_service().validate_sport(str(sport_key))
    deleted = _history_service().clear(sport=sport_key)
    if sport_key:
        _prediction_service().clear_cache(str(sport_key))
    else:
        _prediction_service().clear_cache()
    return jsonify({"deleted_rows": deleted, "sport": sport_key or "all"})
