from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

from flask import Flask, jsonify, render_template, request

from webapp.adaptive_model import adaptive_probabilities_for_upcoming, init_model_store
from webapp.data_sources import SUPPORTED_SPORTS, fetch_sport_events
from webapp.history_store import clear_prediction_history, init_history_db, list_prediction_history, save_prediction_snapshot
from webapp.prediction_engine import predict_upcoming_matches, run_what_if_scenario

app = Flask(__name__, template_folder="templates", static_folder="static")

_CACHE_TTL_SECONDS = 300
_UPCOMING_CACHE: dict[str, dict[str, Any]] = {}
init_history_db()
init_model_store()


def _sport_payload() -> list[dict[str, Any]]:
    return [{"key": cfg.key, "label": cfg.label} for cfg in SUPPORTED_SPORTS.values()]


def _serialize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
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
    }


def _validate_sport(sport_key: str) -> None:
    if sport_key not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport: {sport_key}")


def _apply_probabilities(prediction: dict[str, Any], probs: dict[str, float]) -> None:
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

    # Keep rough score aligned with predicted result.
    score_text = str(prediction.get("predicted_score", ""))
    if "-" in score_text:
        try:
            left_raw, right_raw = score_text.split("-", maxsplit=1)
            left = int(round(float(left_raw.strip())))
            right = int(round(float(right_raw.strip())))
            if winner == "D":
                mid = int(round((left + right) / 2))
                prediction["predicted_score"] = f"{mid}-{mid}"
            elif winner == "H" and left <= right:
                prediction["predicted_score"] = f"{right + 1}-{right}"
            elif winner == "A" and right <= left:
                prediction["predicted_score"] = f"{left}-{left + 1}"
        except (ValueError, TypeError):
            pass


def _build_upcoming_response(sport_key: str, *, force_refresh: bool = False) -> dict[str, Any]:
    now = time.time()
    cache_hit = _UPCOMING_CACHE.get(sport_key)
    if (not force_refresh) and cache_hit and (now - cache_hit["ts"]) < _CACHE_TTL_SECONDS:
        return cache_hit["payload"]

    completed, upcoming = fetch_sport_events(sport_key, days_back=270, days_ahead=30)
    predictions = predict_upcoming_matches(sport_key, completed_events=completed, upcoming_events=upcoming)
    adaptive_probs, training_info = adaptive_probabilities_for_upcoming(
        sport_key,
        completed_events_from_feed=completed,
        upcoming_events=upcoming,
    )
    for prediction in predictions:
        probs = adaptive_probs.get(str(prediction.get("event_id")))
        if probs:
            _apply_probabilities(prediction, probs)
            prediction["model_source"] = "adaptive_ml"
        else:
            prediction["model_source"] = "heuristic"
    serialized_matches = [_serialize_prediction_row(item) for item in predictions[:120]]
    snapshot_id = save_prediction_snapshot(sport=sport_key, matches=serialized_matches)
    payload = {
        "sport": sport_key,
        "sport_label": SUPPORTED_SPORTS[sport_key].label,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_upcoming_matches": len(predictions),
        "snapshot_id": snapshot_id,
        "training": training_info,
        "matches": serialized_matches,
    }
    _UPCOMING_CACHE[sport_key] = {"ts": now, "payload": payload}
    return payload


@app.get("/")
def index() -> str:
    return render_template("index.html", sports=_sport_payload())


@app.get("/api/sports")
def sports() -> Any:
    return jsonify({"sports": _sport_payload()})


@app.get("/api/upcoming")
def upcoming() -> Any:
    sport_key = request.args.get("sport", "football")
    force_refresh = str(request.args.get("force", "0")).strip().lower() in {"1", "true", "yes"}
    try:
        _validate_sport(sport_key)
        payload = _build_upcoming_response(sport_key, force_refresh=force_refresh)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Failed to load matches: {exc}"}), 500


@app.post("/api/what-if")
def what_if() -> Any:
    payload = request.get_json(silent=True) or {}
    required_fields = ["sport", "home_team", "away_team"]
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    sport_key = payload["sport"]
    try:
        _validate_sport(sport_key)
        prediction = run_what_if_scenario(payload)
        return jsonify(_serialize_prediction_row(prediction))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"What-if simulation failed: {exc}"}), 500


@app.get("/api/history")
def history() -> Any:
    sport_key = request.args.get("sport")
    team = request.args.get("team")
    league = request.args.get("league")
    limit_raw = request.args.get("limit", "200")
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    try:
        if sport_key:
            _validate_sport(sport_key)
        rows = list_prediction_history(sport=sport_key, team=team, league=league, limit=limit)
        return jsonify({"count": len(rows), "rows": rows})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Failed to load history: {exc}"}), 500


@app.delete("/api/history")
def clear_history() -> Any:
    payload = request.get_json(silent=True) or {}
    sport_key = payload.get("sport")
    try:
        if sport_key:
            _validate_sport(sport_key)
        deleted = clear_prediction_history(sport=sport_key)
        if sport_key:
            _UPCOMING_CACHE.pop(sport_key, None)
        else:
            _UPCOMING_CACHE.clear()
        return jsonify({"deleted_rows": deleted, "sport": sport_key or "all"})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Failed to clear history: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
