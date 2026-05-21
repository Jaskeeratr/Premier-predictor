from __future__ import annotations

from webapp.config import AppConfig
from webapp.services.prediction_service import PredictionService


def test_injury_adjustment_reweights_probabilities(tmp_path):
    service = PredictionService(
        AppConfig(
            app_env="test",
            debug=False,
            log_level="WARNING",
            upcoming_cache_ttl_seconds=1,
            demo_mode=True,
            db_path=tmp_path / "db.sqlite",
            model_dir=tmp_path / "models",
        )
    )
    prediction = {
        "home_team": "Home",
        "away_team": "Away",
        "predicted_score": "2-1",
        "home_win_probability": 0.45,
        "draw_probability": 0.20,
        "away_win_probability": 0.35,
    }
    injury_map = {
        "Home": {"rating_delta": -120, "form_delta": -0.2, "offense_delta": -0.5, "defense_delta": 0.4},
        "Away": {"rating_delta": 80, "form_delta": 0.15, "offense_delta": 0.2, "defense_delta": -0.2},
    }

    service.apply_injury_adjustments(prediction, injury_map)
    total = (
        prediction["home_win_probability"]
        + prediction["draw_probability"]
        + prediction["away_win_probability"]
    )
    assert abs(total - 1.0) < 0.001
    assert prediction["predicted_winner"] in {"Home", "Away", "Draw"}
    assert "injury_adjustment" in prediction["factors"]


def test_probability_application_normalizes():
    prediction = {
        "home_team": "Home",
        "away_team": "Away",
        "home_win_probability": 0.0,
        "draw_probability": 0.0,
        "away_win_probability": 0.0,
    }
    PredictionService.apply_probabilities(prediction, {"H": 2.0, "D": 1.0, "A": 1.0})
    assert (
        abs(
            (
                prediction["home_win_probability"]
                + prediction["draw_probability"]
                + prediction["away_win_probability"]
            )
            - 1.0
        )
        < 0.001
    )
    assert prediction["confidence"] >= 0.0
