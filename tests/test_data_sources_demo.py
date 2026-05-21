from __future__ import annotations

import requests

from webapp.data_sources import fetch_sport_events_with_meta


def test_force_demo_mode_returns_demo_data():
    completed, upcoming, meta = fetch_sport_events_with_meta(
        "football",
        days_back=30,
        days_ahead=10,
        force_demo_mode=True,
    )
    assert meta["mode"] == "demo"
    assert len(completed) > 0
    assert len(upcoming) > 0


def test_auto_demo_fallback_when_requests_fail(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("webapp.data_sources.requests.get", fail)
    completed, upcoming, meta = fetch_sport_events_with_meta("basketball", days_back=20, days_ahead=7)
    assert meta["mode"] == "demo"
    assert meta["reason"] == "live_feed_unavailable_or_sparse"
    assert len(completed) > 0
    assert len(upcoming) > 0
