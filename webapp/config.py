from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    upcoming_cache_ttl_seconds: int = 300
    demo_mode: bool = False
    db_path: Path | None = None
    model_dir: Path | None = None

    @staticmethod
    def from_env() -> AppConfig:
        root = Path(__file__).resolve().parent
        db_path_env = os.getenv("PREDICTOR_DB_PATH")
        model_dir_env = os.getenv("PREDICTOR_MODEL_DIR")

        return AppConfig(
            app_env=os.getenv("APP_ENV", "development"),
            debug=_as_bool(os.getenv("APP_DEBUG"), default=True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            upcoming_cache_ttl_seconds=int(os.getenv("UPCOMING_CACHE_TTL_SECONDS", "300")),
            demo_mode=_as_bool(os.getenv("PREDICTOR_DEMO_MODE"), default=False),
            db_path=Path(db_path_env) if db_path_env else root / "prediction_history.db",
            model_dir=Path(model_dir_env) if model_dir_env else root / "models",
        )
