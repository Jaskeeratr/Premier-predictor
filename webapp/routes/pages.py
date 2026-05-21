from __future__ import annotations

from flask import Blueprint, current_app, render_template

from webapp.services.prediction_service import PredictionService

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def index() -> str:
    service = current_app.extensions["prediction_service"]
    assert isinstance(service, PredictionService)
    return render_template("index.html", sports=service.sport_payload())
