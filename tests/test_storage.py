from __future__ import annotations

from webapp.history_store import (
    clear_prediction_history,
    init_history_db,
    list_prediction_history,
    save_prediction_snapshot,
)
from webapp.injury_store import (
    delete_injury_adjustments,
    get_injury_adjustments_map,
    init_injury_store,
    list_injury_adjustments,
    upsert_injury_adjustment,
)


def test_history_snapshot_round_trip(tmp_path):
    db_path = tmp_path / "history.db"
    init_history_db(db_path)
    match = {
        "event_id": "event-1",
        "kickoff_utc": "2026-05-20T18:00:00+00:00",
        "league": "Demo League",
        "home_team": "Home",
        "away_team": "Away",
        "venue": "Arena",
        "predicted_winner": "Home",
        "predicted_result": "H",
        "confidence": 0.61,
        "predicted_score": "2-1",
        "home_win_probability": 0.61,
        "draw_probability": 0.2,
        "away_win_probability": 0.19,
        "factors": {"rating": 12},
    }

    snapshot_id = save_prediction_snapshot("football", [match], db_path=db_path)
    rows = list_prediction_history(sport="football", db_path=db_path)

    assert snapshot_id is not None
    assert len(rows) == 1
    assert rows[0]["predicted_winner"] == "Home"
    assert clear_prediction_history(sport="football", db_path=db_path) == 1


def test_injury_adjustments_round_trip(tmp_path):
    db_path = tmp_path / "injuries.db"
    init_injury_store(db_path)

    saved = upsert_injury_adjustment(
        "football",
        "Arsenal",
        rating_delta=-25,
        form_delta=-0.08,
        offense_delta=-0.2,
        notes="starting forward out",
        db_path=db_path,
    )
    rows = list_injury_adjustments(sport="football", db_path=db_path)
    mapping = get_injury_adjustments_map("football", db_path=db_path)

    assert saved["team"] == "Arsenal"
    assert len(rows) == 1
    assert mapping["Arsenal"]["rating_delta"] == -25
    assert delete_injury_adjustments(sport="football", team="Arsenal", db_path=db_path) == 1
