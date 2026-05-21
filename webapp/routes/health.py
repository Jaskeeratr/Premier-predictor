from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "sports-predictor",
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


@health_bp.get("/ready")
def ready():
    db_path = current_app.config.get("DB_PATH")
    model_dir = current_app.config.get("MODEL_DIR")
    db_ok = bool(db_path and Path(str(db_path)).exists())
    models_ok = bool(model_dir and Path(str(model_dir)).exists())
    ready_status = db_ok and models_ok
    return (
        jsonify(
            {
                "status": "ready" if ready_status else "degraded",
                "db_ok": db_ok,
                "models_ok": models_ok,
                "demo_mode": bool(current_app.config.get("DEMO_MODE", False)),
            }
        ),
        200 if ready_status else 503,
    )
