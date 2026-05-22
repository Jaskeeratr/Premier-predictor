from __future__ import annotations

from webapp.services.explainability import confidence_tier, enrich_prediction_explainability


def test_confidence_tier_thresholds():
    assert confidence_tier(0.80) == "Strong Pick"
    assert confidence_tier(0.67) == "Lean"
    assert confidence_tier(0.55) == "Toss-up"
    assert confidence_tier(0.49) == "Very Uncertain"


def test_enrich_prediction_adds_explainability_fields():
    row = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "predicted_winner": "Arsenal",
        "predicted_result": "H",
        "confidence": 0.64,
        "predicted_score": "2-1",
        "factors": {
            "home_rating": 1560.0,
            "away_rating": 1510.0,
            "home_form": 0.63,
            "away_form": 0.45,
            "component_breakdown": {
                "rating": 0.22,
                "form": 0.16,
                "attack": 0.12,
                "defense": -0.06,
            },
        },
    }
    enrich_prediction_explainability(row)
    assert row["confidence_tier"] == "Lean"
    assert isinstance(row["risk_indicators"], list)
    assert isinstance(row["factor_contributions"], list)
    assert len(row["factor_contributions"]) > 0
    assert isinstance(row["top_factors"], list)
    assert len(row["top_factors"]) > 0
    assert "projected to win" in row["explanation"].lower()
