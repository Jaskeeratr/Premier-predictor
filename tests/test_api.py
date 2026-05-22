from __future__ import annotations


def test_health_and_sports_smoke(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    sports = client.get("/api/sports")
    assert sports.status_code == 200
    payload = sports.get_json()
    assert "sports" in payload
    assert len(payload["sports"]) >= 5


def test_unsupported_sport_rejected(client):
    response = client.get("/api/upcoming?sport=quidditch")
    assert response.status_code == 400
    assert "Unsupported sport" in response.get_json()["error"]


def test_what_if_api_returns_probabilities(client):
    response = client.post(
        "/api/what-if",
        json={
            "sport": "football",
            "home_team": "Alpha FC",
            "away_team": "Beta FC",
            "home_rating": 1530,
            "away_rating": 1480,
            "home_form": 0.64,
            "away_form": 0.44,
            "home_avg_scored": 1.8,
            "away_avg_scored": 1.2,
            "home_avg_allowed": 1.0,
            "away_avg_allowed": 1.4,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["predicted_winner"] in {"Alpha FC", "Beta FC", "Draw"}
    total = payload["home_win_probability"] + payload["draw_probability"] + payload["away_win_probability"]
    assert abs(total - 1.0) < 0.01
    assert payload["confidence_tier"] in {"Strong Pick", "Lean", "Toss-up", "Very Uncertain"}
    assert isinstance(payload["risk_indicators"], list)
    assert isinstance(payload["explanation"], str)


def test_model_health_endpoint(client):
    response = client.get("/api/model-health?sport=football")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["scope"] == "single"
    assert payload["sport"] == "football"
    assert "summary" in payload
    assert payload["summary"]["status"] in {
        "Healthy",
        "Needs More Data",
        "Undertrained",
        "Low Confidence",
    }
