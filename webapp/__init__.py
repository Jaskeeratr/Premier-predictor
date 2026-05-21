from __future__ import annotations

from flask import Flask, jsonify

from webapp.config import AppConfig
from webapp.errors import AppError
from webapp.extensions import init_platform
from webapp.logging_utils import configure_logging
from webapp.routes import api_bp, health_bp, pages_bp


def create_app(config: AppConfig | None = None) -> Flask:
    app_config = config or AppConfig.from_env()
    configure_logging(app_config.log_level)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["APP_ENV"] = app_config.app_env
    app.config["DEBUG"] = app_config.debug
    app.config["DEMO_MODE"] = app_config.demo_mode
    app.config["DB_PATH"] = str(app_config.db_path) if app_config.db_path else None
    app.config["MODEL_DIR"] = str(app_config.model_dir) if app_config.model_dir else None
    app.config["UPCOMING_CACHE_TTL_SECONDS"] = app_config.upcoming_cache_ttl_seconds

    init_platform(app, app_config)
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(health_bp)

    @app.errorhandler(AppError)
    def handle_app_error(exc: AppError):
        return jsonify({"error": str(exc)}), exc.status_code

    return app
