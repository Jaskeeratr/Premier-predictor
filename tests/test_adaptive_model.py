from __future__ import annotations

import sqlite3

from webapp.adaptive_model import adaptive_probabilities_for_upcoming, init_model_store
from webapp.data_sources import generate_demo_events


def test_adaptive_returns_not_enough_data(tmp_path):
    from datetime import datetime, timezone

    db_path = tmp_path / "adaptive.sqlite"
    model_dir = tmp_path / "models"
    init_model_store(db_path, model_dir)
    anchor = datetime.now(timezone.utc).date()
    completed, upcoming = generate_demo_events("volleyball", anchor_day=anchor, days_back=5, days_ahead=2)
    probs, info = adaptive_probabilities_for_upcoming(
        "volleyball",
        completed_events_from_feed=completed[:8],
        upcoming_events=upcoming[:4],
        db_path=db_path,
    )
    assert probs == {}
    assert info["status"] in {"not_enough_data", "not_enough_samples"}


def test_adaptive_handles_corrupt_latest_artifact(tmp_path):
    db_path = tmp_path / "adaptive.sqlite"
    model_dir = tmp_path / "models"
    init_model_store(db_path, model_dir)
    from datetime import datetime, timezone

    anchor = datetime.now(timezone.utc).date()
    completed, upcoming = generate_demo_events("football", anchor_day=anchor, days_back=120, days_ahead=20)

    # First run trains and stores a valid artifact.
    probs1, info1 = adaptive_probabilities_for_upcoming(
        "football",
        completed_events_from_feed=completed,
        upcoming_events=upcoming[:8],
        db_path=db_path,
    )
    assert info1["status"] == "ok"
    assert len(probs1) > 0

    # Insert a corrupt latest run path; loader should fallback to prior valid artifact.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_runs (
                sport, trained_at_utc, sample_count, completed_events_count,
                best_model, best_params_json, holdout_accuracy, model_path,
                version, cv_accuracy, cv_log_loss, cv_brier, cv_ece, summary_json, is_active, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "football",
                "2026-01-01T00:00:00+00:00",
                100,
                120,
                "logistic_regression",
                "{}",
                0.55,
                str(tmp_path / "bad.joblib"),
                "corrupt-latest",
                0.5,
                1.0,
                0.4,
                0.2,
                "{}",
                1,
                str(tmp_path / "bad.joblib"),
            ),
        )
        conn.commit()

    probs2, info2 = adaptive_probabilities_for_upcoming(
        "football",
        completed_events_from_feed=completed,
        upcoming_events=upcoming[:8],
        db_path=db_path,
    )
    assert info2["status"] == "ok"
    assert len(probs2) > 0
