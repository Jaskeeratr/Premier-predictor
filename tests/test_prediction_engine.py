from __future__ import annotations

from datetime import datetime, timezone

from webapp.prediction_engine import TeamStats, predict_match, run_what_if_scenario


def test_predict_match_probability_outputs_sum_to_one():
    event = {
        "event_id": "test-1",
        "league": "Test League",
        "sport": "football",
        "start_time": datetime(2026, 5, 20, tzinfo=timezone.utc),
        "home_team": "Home FC",
        "away_team": "Away FC",
        "venue": "Test Ground",
        "status": "scheduled",
    }
    teams = {
        "Home FC": TeamStats(
            rating=1540, games=8, points_for=16, points_against=8, form_results=[1, 1, 0.5, 1, 0.5]
        ),
        "Away FC": TeamStats(
            rating=1495, games=8, points_for=11, points_against=12, form_results=[0, 0.5, 1, 0, 0.5]
        ),
    }
    context = {"avg_home": 1.5, "avg_away": 1.1, "draw_rate": 0.24}

    prediction = predict_match("football", event, teams, context)

    total = (
        prediction["home_win_probability"]
        + prediction["draw_probability"]
        + prediction["away_win_probability"]
    )
    assert round(total, 3) == 1.0
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["predicted_score"]


def test_what_if_neutral_site_removes_home_edge():
    base = {
        "sport": "basketball",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "home_rating": 1500,
        "away_rating": 1500,
        "home_form": 0.5,
        "away_form": 0.5,
        "home_avg_scored": 110,
        "away_avg_scored": 110,
        "home_avg_allowed": 110,
        "away_avg_allowed": 110,
    }

    home_edge = run_what_if_scenario({**base, "neutral_site": False})
    neutral = run_what_if_scenario({**base, "neutral_site": True})

    assert home_edge["home_win_probability"] > neutral["home_win_probability"]
