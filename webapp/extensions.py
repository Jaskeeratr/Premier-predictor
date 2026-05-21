from __future__ import annotations

from flask import Flask

from webapp.adaptive_model import init_model_store
from webapp.config import AppConfig
from webapp.history_store import init_history_db
from webapp.injury_store import init_injury_store
from webapp.services import HistoryService, InjuryService, PredictionService


def init_platform(app: Flask, config: AppConfig) -> None:
    init_history_db(config.db_path)
    init_model_store(config.db_path, config.model_dir)
    init_injury_store(config.db_path)

    app.extensions["prediction_service"] = PredictionService(config)
    app.extensions["history_service"] = HistoryService(config.db_path)
    app.extensions["injury_service"] = InjuryService(config.db_path)
