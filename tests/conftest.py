from __future__ import annotations

from pathlib import Path

import pytest

from webapp import create_app
from webapp.config import AppConfig


@pytest.fixture()
def app(tmp_path: Path):
    cfg = AppConfig(
        app_env="test",
        debug=False,
        log_level="WARNING",
        upcoming_cache_ttl_seconds=1,
        demo_mode=True,
        db_path=tmp_path / "test_prediction_history.db",
        model_dir=tmp_path / "models",
    )
    flask_app = create_app(cfg)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
