from __future__ import annotations

from datetime import date

from webapp.data_sources import fetch_sport_events_with_meta


def test_force_demo_mode_generates_completed_and_upcoming_events():
    completed, upcoming, meta = fetch_sport_events_with_meta(
        "basketball",
        today=date(2026, 5, 20),
        days_back=45,
        days_ahead=14,
        force_demo_mode=True,
    )

    assert meta["mode"] == "demo"
    assert len(completed) > 0
    assert len(upcoming) > 0
    assert all(event["sport"] == "basketball" for event in completed + upcoming)
