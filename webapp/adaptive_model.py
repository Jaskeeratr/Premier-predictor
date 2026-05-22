from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from webapp.data_sources import SUPPORTED_SPORTS
from webapp.history_store import DEFAULT_DB_PATH
from webapp.prediction_engine import SPORT_PROFILES

LOGGER = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "home_rating",
    "away_rating",
    "rating_diff",
    "home_form",
    "away_form",
    "form_diff",
    "home_avg_for",
    "away_avg_for",
    "attack_edge",
    "home_avg_against",
    "away_avg_against",
    "defense_edge",
    "margin_edge",
    "home_games",
    "away_games",
    "games_delta",
    "home_adv",
    "elo_expected_home",
    "league_avg_for",
    "league_avg_against",
]


@dataclass(frozen=True)
class SportModelConfig:
    min_samples: int
    min_classes: int
    retrain_after_new_events: int
    retrain_interval_hours: float
    logistic_cs: tuple[float, ...]
    gb_depths: tuple[int, ...]
    gb_estimators: tuple[int, ...]
    rf_depths: tuple[int, ...]
    allow_rf: bool


SPORT_MODEL_CONFIGS: dict[str, SportModelConfig] = {
    "football": SportModelConfig(
        min_samples=50,
        min_classes=3,
        retrain_after_new_events=8,
        retrain_interval_hours=18.0,
        logistic_cs=(0.2, 0.6, 1.0, 2.0, 4.0),
        gb_depths=(2, 3),
        gb_estimators=(120, 180),
        rf_depths=(6, 8),
        allow_rf=False,
    ),
    "cricket": SportModelConfig(
        min_samples=45,
        min_classes=2,
        retrain_after_new_events=8,
        retrain_interval_hours=18.0,
        logistic_cs=(0.2, 0.6, 1.0, 3.0),
        gb_depths=(2, 3),
        gb_estimators=(100, 160),
        rf_depths=(6, 8),
        allow_rf=False,
    ),
    "basketball": SportModelConfig(
        min_samples=60,
        min_classes=2,
        retrain_after_new_events=10,
        retrain_interval_hours=12.0,
        logistic_cs=(0.1, 0.4, 1.0, 3.0),
        gb_depths=(2, 3),
        gb_estimators=(80, 140),
        rf_depths=(6, 8),
        allow_rf=True,
    ),
    "american_football": SportModelConfig(
        min_samples=55,
        min_classes=2,
        retrain_after_new_events=8,
        retrain_interval_hours=18.0,
        logistic_cs=(0.1, 0.5, 1.0, 2.0),
        gb_depths=(2, 3),
        gb_estimators=(100, 150),
        rf_depths=(6, 8),
        allow_rf=True,
    ),
    "volleyball": SportModelConfig(
        min_samples=35,
        min_classes=2,
        retrain_after_new_events=6,
        retrain_interval_hours=24.0,
        logistic_cs=(0.2, 0.8, 1.5, 3.0),
        gb_depths=(2, 3),
        gb_estimators=(80, 120),
        rf_depths=(6,),
        allow_rf=False,
    ),
}


@dataclass
class TeamState:
    rating: float = 1500.0
    games: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    form: list[float] = field(default_factory=list)

    @property
    def avg_for(self) -> float:
        return self.points_for / self.games if self.games else 0.0

    @property
    def avg_against(self) -> float:
        return self.points_against / self.games if self.games else 0.0

    @property
    def form_index(self) -> float:
        if not self.form:
            return 0.5
        recent = self.form[-5:]
        weighted = 0.0
        total = 0.0
        for idx, val in enumerate(recent, start=1):
            weighted += idx * val
            total += idx
        return weighted / total if total else 0.5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row[1]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_model_store(db_path: Path | None = None, model_dir: Path | None = None) -> None:
    global MODEL_DIR
    if model_dir is not None:
        MODEL_DIR = model_dir
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_events (
                sport TEXT NOT NULL,
                event_id TEXT NOT NULL,
                start_time_utc TEXT NOT NULL,
                league TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score REAL,
                away_score REAL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (sport, event_id)
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                trained_at_utc TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                completed_events_count INTEGER NOT NULL,
                best_model TEXT NOT NULL,
                best_params_json TEXT,
                holdout_accuracy REAL,
                model_path TEXT NOT NULL
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                sport TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                split_name TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES model_runs(id)
            )
            """)
        _ensure_column(conn, "model_runs", "version", "TEXT")
        _ensure_column(conn, "model_runs", "cv_accuracy", "REAL")
        _ensure_column(conn, "model_runs", "cv_log_loss", "REAL")
        _ensure_column(conn, "model_runs", "cv_brier", "REAL")
        _ensure_column(conn, "model_runs", "cv_ece", "REAL")
        _ensure_column(conn, "model_runs", "summary_json", "TEXT")
        _ensure_column(conn, "model_runs", "is_active", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(conn, "model_runs", "artifact_path", "TEXT")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_events_sport_time ON training_events(sport, start_time_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_runs_sport_time ON model_runs(sport, trained_at_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_metrics_run_metric ON model_metrics(run_id, metric_name)"
        )
        conn.commit()


def sync_completed_events(
    sport: str, completed_events: list[dict[str, Any]], db_path: Path | None = None
) -> int:
    if not completed_events:
        return 0
    rows = []
    now = _utc_now_iso()
    for event in completed_events:
        if event.get("home_score") is None or event.get("away_score") is None:
            continue
        rows.append(
            (
                sport,
                str(event.get("event_id")),
                str(event.get("start_time")),
                event.get("league"),
                event.get("home_team"),
                event.get("away_team"),
                float(event.get("home_score")),
                float(event.get("away_score")),
                now,
            )
        )
    if not rows:
        return 0

    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO training_events (
                sport, event_id, start_time_utc, league, home_team, away_team, home_score, away_score, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, event_id) DO UPDATE SET
                start_time_utc=excluded.start_time_utc,
                league=excluded.league,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                updated_at_utc=excluded.updated_at_utc
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def _load_training_events(sport: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT event_id, start_time_utc, league, home_team, away_team, home_score, away_score
            FROM training_events
            WHERE sport = ?
            ORDER BY start_time_utc
            """,
            (sport,),
        ).fetchall()
    return [dict(row) for row in rows]


def _load_model_runs(sport: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM model_runs
            WHERE sport = ?
            ORDER BY id DESC
            """,
            (sport,),
        ).fetchall()
    return [dict(row) for row in rows]


def _load_latest_model_run(sport: str, db_path: Path | None = None) -> dict[str, Any] | None:
    runs = _load_model_runs(sport, db_path=db_path)
    return runs[0] if runs else None


def _record_model_run(
    sport: str,
    *,
    version: str,
    sample_count: int,
    completed_events_count: int,
    best_model: str,
    best_params: dict[str, Any],
    cv_metrics: dict[str, float],
    holdout_accuracy: float,
    artifact_path: Path,
    summary: dict[str, Any],
    db_path: Path | None = None,
) -> dict[str, Any]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO model_runs (
                sport, trained_at_utc, sample_count, completed_events_count,
                best_model, best_params_json, holdout_accuracy, model_path,
                version, cv_accuracy, cv_log_loss, cv_brier, cv_ece,
                summary_json, is_active, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                _utc_now_iso(),
                sample_count,
                completed_events_count,
                best_model,
                json.dumps(best_params),
                holdout_accuracy,
                str(artifact_path),
                version,
                float(cv_metrics.get("accuracy", 0.0)),
                float(cv_metrics.get("log_loss", 0.0)),
                float(cv_metrics.get("brier", 0.0)),
                float(cv_metrics.get("ece", 0.0)),
                json.dumps(summary),
                1,
                str(artifact_path),
            ),
        )
        run_id = int(cur.lastrowid)
        metric_rows = [
            (run_id, sport, "cv_accuracy", float(cv_metrics.get("accuracy", 0.0)), "cv", _utc_now_iso()),
            (run_id, sport, "cv_log_loss", float(cv_metrics.get("log_loss", 0.0)), "cv", _utc_now_iso()),
            (run_id, sport, "cv_brier", float(cv_metrics.get("brier", 0.0)), "cv", _utc_now_iso()),
            (run_id, sport, "cv_ece", float(cv_metrics.get("ece", 0.0)), "cv", _utc_now_iso()),
            (run_id, sport, "holdout_accuracy", float(holdout_accuracy), "holdout", _utc_now_iso()),
        ]
        conn.executemany(
            """
            INSERT INTO model_metrics (run_id, sport, metric_name, metric_value, split_name, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            metric_rows,
        )
        conn.commit()
    return _load_latest_model_run(sport, db_path=db_path) or {}


def _expected_home(home_rating: float, away_rating: float, home_adv_elo: float) -> float:
    return 1 / (1 + 10 ** (((away_rating) - (home_rating + home_adv_elo)) / 400))


def _outcome_label(home_score: float, away_score: float, supports_draw: bool) -> str | None:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    if supports_draw:
        return "D"
    return None


def _update_team_states(
    sport: str,
    home: TeamState,
    away: TeamState,
    *,
    home_score: float,
    away_score: float,
) -> None:
    profile = SPORT_PROFILES[sport]
    exp_home = _expected_home(home.rating, away.rating, profile.home_adv_elo)
    if home_score > away_score:
        actual_home = 1.0
        home.form.append(1.0)
        away.form.append(0.0)
    elif home_score < away_score:
        actual_home = 0.0
        home.form.append(0.0)
        away.form.append(1.0)
    else:
        actual_home = 0.5
        home.form.append(0.5)
        away.form.append(0.5)

    shift = profile.k_factor * (actual_home - exp_home)
    home.rating += shift
    away.rating -= shift

    home.games += 1
    away.games += 1
    home.points_for += home_score
    home.points_against += away_score
    away.points_for += away_score
    away.points_against += home_score

    if len(home.form) > 8:
        home.form.pop(0)
    if len(away.form) > 8:
        away.form.pop(0)


def _feature_row(
    sport: str,
    home: TeamState,
    away: TeamState,
    *,
    home_adv: float,
    league_avg_for: float,
    league_avg_against: float,
) -> list[float]:
    rating_diff = home.rating - away.rating
    form_diff = home.form_index - away.form_index
    attack_edge = home.avg_for - away.avg_for
    defense_edge = away.avg_against - home.avg_against
    margin_edge = (home.avg_for - home.avg_against) - (away.avg_for - away.avg_against)
    exp_home = _expected_home(home.rating, away.rating, SPORT_PROFILES[sport].home_adv_elo * home_adv)
    return [
        home.rating,
        away.rating,
        rating_diff,
        home.form_index,
        away.form_index,
        form_diff,
        home.avg_for,
        away.avg_for,
        attack_edge,
        home.avg_against,
        away.avg_against,
        defense_edge,
        margin_edge,
        float(home.games),
        float(away.games),
        float(home.games - away.games),
        float(home_adv),
        exp_home,
        league_avg_for,
        league_avg_against,
    ]


def _build_training_matrix(
    sport: str, completed_events: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, TeamState], dict[str, float]]:
    supports_draw = SUPPORTED_SPORTS[sport].supports_draw
    teams: dict[str, TeamState] = {}
    x_rows: list[list[float]] = []
    y_rows: list[str] = []
    total_home = 0.0
    total_away = 0.0
    total_games = 0

    for event in completed_events:
        home_team = event["home_team"]
        away_team = event["away_team"]
        home_score = float(event["home_score"])
        away_score = float(event["away_score"])

        home = teams.setdefault(home_team, TeamState())
        away = teams.setdefault(away_team, TeamState())

        league_avg_for = (total_home / total_games) if total_games else SPORT_PROFILES[sport].base_home_score
        league_avg_against = (
            (total_away / total_games) if total_games else SPORT_PROFILES[sport].base_away_score
        )
        x_rows.append(
            _feature_row(
                sport,
                home,
                away,
                home_adv=1.0,
                league_avg_for=league_avg_for,
                league_avg_against=league_avg_against,
            )
        )
        label = _outcome_label(home_score, away_score, supports_draw=supports_draw)
        if label is None:
            x_rows.pop()
        else:
            y_rows.append(label)

        _update_team_states(sport, home, away, home_score=home_score, away_score=away_score)
        total_home += home_score
        total_away += away_score
        total_games += 1

    context = {
        "avg_home": (total_home / total_games) if total_games else SPORT_PROFILES[sport].base_home_score,
        "avg_away": (total_away / total_games) if total_games else SPORT_PROFILES[sport].base_away_score,
    }
    return np.array(x_rows, dtype=float), np.array(y_rows), teams, context


def _sport_config(sport: str) -> SportModelConfig:
    return SPORT_MODEL_CONFIGS.get(
        sport,
        SportModelConfig(
            min_samples=45,
            min_classes=2,
            retrain_after_new_events=8,
            retrain_interval_hours=18.0,
            logistic_cs=(0.3, 1.0, 2.0),
            gb_depths=(2, 3),
            gb_estimators=(120,),
            rf_depths=(6,),
            allow_rf=False,
        ),
    )


def _candidate_models(sport: str) -> list[tuple[str, dict[str, Any], Any]]:
    cfg = _sport_config(sport)
    supports_draw = SUPPORTED_SPORTS[sport].supports_draw
    class_weight = "balanced" if supports_draw else None
    candidates: list[tuple[str, dict[str, Any], Any]] = []

    for c_value in cfg.logistic_cs:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        C=float(c_value),
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        candidates.append(("logistic_regression", {"C": float(c_value)}, model))

    for depth in cfg.gb_depths:
        for estimators in cfg.gb_estimators:
            gb = GradientBoostingClassifier(
                n_estimators=int(estimators),
                learning_rate=0.05,
                max_depth=int(depth),
                random_state=42,
            )
            candidates.append(
                (
                    "gradient_boosting",
                    {"n_estimators": int(estimators), "max_depth": int(depth), "learning_rate": 0.05},
                    gb,
                )
            )

    if cfg.allow_rf and not supports_draw:
        for depth in cfg.rf_depths:
            rf = RandomForestClassifier(
                n_estimators=320,
                max_depth=int(depth),
                min_samples_leaf=2,
                random_state=42,
            )
            candidates.append(("random_forest", {"n_estimators": 320, "max_depth": int(depth)}, rf))

    return candidates


def _multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> float:
    idx = {label: pos for pos, label in enumerate(labels)}
    y_one_hot = np.zeros((len(y_true), len(labels)), dtype=float)
    for row_idx, label in enumerate(y_true):
        y_one_hot[row_idx, idx[str(label)]] = 1.0
    return float(np.mean(np.sum((y_one_hot - y_prob) ** 2, axis=1)))


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], bins: int = 10
) -> float:
    if len(y_true) == 0:
        return 0.0
    class_idx = {label: i for i, label in enumerate(labels)}
    true_idx = np.array([class_idx[str(v)] for v in y_true], dtype=int)
    pred_idx = np.argmax(y_prob, axis=1)
    conf = np.max(y_prob, axis=1)
    corr = (pred_idx == true_idx).astype(float)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if not np.any(mask):
            continue
        bucket_acc = float(np.mean(corr[mask]))
        bucket_conf = float(np.mean(conf[mask]))
        weight = float(np.mean(mask))
        ece += abs(bucket_acc - bucket_conf) * weight
    return float(ece)


def _evaluate_candidate(
    estimator: Any,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    n_splits: int,
) -> dict[str, Any]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(x):
        model = clone(estimator)
        model.fit(x[train_idx], y[train_idx])
        preds = model.predict(x[val_idx])
        probs = model.predict_proba(x[val_idx])

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y[val_idx], preds)),
                "log_loss": float(log_loss(y[val_idx], probs, labels=labels)),
                "brier": _multiclass_brier(y[val_idx], probs, labels),
                "ece": _expected_calibration_error(y[val_idx], probs, labels),
            }
        )

    summary = {
        metric: float(np.mean([row[metric] for row in fold_metrics]))
        for metric in ("accuracy", "log_loss", "brier", "ece")
    }
    summary["fold_count"] = len(fold_metrics)
    return {"metrics": summary, "fold_metrics": fold_metrics}


def _fit_best_model(
    sport: str,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Any, dict[str, Any], dict[str, float], float]:
    labels = sorted({str(v) for v in y})
    n_splits = max(3, min(5, len(x) // 25))
    split_idx = int(len(x) * 0.85)
    split_idx = max(25, min(split_idx, len(x) - 8))
    x_train, x_hold = x[:split_idx], x[split_idx:]
    y_train, y_hold = y[:split_idx], y[split_idx:]

    best_model: Any = None
    best_desc: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}
    best_score = -10_000.0
    comparisons: list[dict[str, Any]] = []

    for name, params, estimator in _candidate_models(sport):
        eval_result = _evaluate_candidate(estimator, x_train, y_train, labels, n_splits=n_splits)
        metrics = eval_result["metrics"]
        score = metrics["accuracy"] - (metrics["log_loss"] * 0.15) - (metrics["ece"] * 0.10)
        comparisons.append({"model": name, "params": params, "score": round(score, 6), **metrics})
        if score > best_score:
            best_score = score
            best_model = clone(estimator)
            best_desc = {"name": name, "params": params, "comparisons": comparisons}
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError(f"No model candidate succeeded for sport={sport}")

    best_model.fit(x_train, y_train)
    hold_pred = best_model.predict(x_hold)
    holdout_accuracy = float(accuracy_score(y_hold, hold_pred))

    best_model.fit(x, y)
    best_desc["comparisons"] = comparisons
    return best_model, best_desc, best_metrics, holdout_accuracy


def _extract_feature_importance(model: Any) -> list[dict[str, float | str]]:
    model_obj = model
    if isinstance(model, Pipeline):
        model_obj = model.named_steps.get("model", model)

    values: np.ndarray | None = None
    if hasattr(model_obj, "coef_"):
        coef = np.asarray(model_obj.coef_, dtype=float)
        if coef.ndim == 2:
            values = np.mean(np.abs(coef), axis=0)
    elif hasattr(model_obj, "feature_importances_"):
        values = np.asarray(model_obj.feature_importances_, dtype=float)

    if values is None:
        return []

    pairs = []
    for idx, feature_name in enumerate(FEATURE_NAMES):
        if idx >= len(values):
            break
        pairs.append({"feature": feature_name, "importance": float(values[idx])})
    pairs.sort(key=lambda row: float(row["importance"]), reverse=True)
    return pairs[:12]


def _artifact_path(sport: str, version: str) -> Path:
    sport_dir = MODEL_DIR / sport
    sport_dir.mkdir(parents=True, exist_ok=True)
    return sport_dir / f"{version}.joblib"


def _save_model_artifact(
    sport: str,
    model: Any,
    *,
    version: str,
    context: dict[str, Any],
) -> Path:
    path = _artifact_path(sport, version)
    artifact = {
        "sport": sport,
        "version": version,
        "saved_at_utc": _utc_now_iso(),
        "feature_names": FEATURE_NAMES,
        "model": model,
        "context": context,
    }
    joblib.dump(artifact, path)
    return path


def _load_model_artifact(path: Path) -> dict[str, Any] | None:
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
    except Exception:  # pragma: no cover - handled by caller
        return None
    return None


def _load_latest_valid_artifact(
    sport: str, db_path: Path | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for run in _load_model_runs(sport, db_path=db_path):
        path_str = run.get("artifact_path") or run.get("model_path")
        if not path_str:
            continue
        artifact = _load_model_artifact(Path(path_str))
        if artifact is not None:
            return artifact, run
    return None, None


def _needs_retrain(sport: str, completed_events_count: int, db_path: Path | None = None) -> bool:
    cfg = _sport_config(sport)
    latest = _load_latest_model_run(sport, db_path=db_path)
    if not latest:
        return True

    seen = int(latest.get("completed_events_count", 0) or 0)
    if completed_events_count >= seen + cfg.retrain_after_new_events:
        return True

    try:
        trained_at = datetime.fromisoformat(str(latest["trained_at_utc"]))
        age_hours = (
            datetime.now(timezone.utc) - trained_at.astimezone(timezone.utc)
        ).total_seconds() / 3600.0
        return age_hours >= cfg.retrain_interval_hours
    except Exception:
        return True


def _probabilities_from_model(model: Any, row: np.ndarray, supports_draw: bool) -> dict[str, float]:
    classes = [str(v) for v in model.classes_]
    probs = model.predict_proba(row.reshape(1, -1))[0]
    mapping = {cls: float(prob) for cls, prob in zip(classes, probs, strict=False)}
    home = mapping.get("H", 0.0)
    draw = mapping.get("D", 0.0) if supports_draw else 0.0
    away = mapping.get("A", 0.0)

    total = home + draw + away
    if total <= 0:
        if supports_draw:
            return {"H": 0.42, "D": 0.18, "A": 0.40}
        return {"H": 0.5, "D": 0.0, "A": 0.5}
    return {"H": home / total, "D": draw / total, "A": away / total}


def get_latest_model_summary(sport: str, db_path: Path | None = None) -> dict[str, Any] | None:
    latest = _load_latest_model_run(sport, db_path=db_path)
    if not latest:
        return None
    try:
        summary_json = latest.get("summary_json")
        if summary_json:
            summary = json.loads(str(summary_json))
            summary["run_id"] = int(latest["id"])
            summary["trained_at_utc"] = latest.get("trained_at_utc")
            summary["best_model"] = latest.get("best_model")
            return summary
    except Exception:
        return None
    return None


def get_model_health_summary(sport: str, db_path: Path | None = None) -> dict[str, Any]:
    cfg = _sport_config(sport)
    latest = _load_latest_model_run(sport, db_path=db_path)
    event_count = len(_load_training_events(sport, db_path=db_path))

    if not latest:
        return {
            "sport": sport,
            "status": "Needs More Data",
            "reason": "No trained model run found.",
            "sample_count": 0,
            "required_min_samples": cfg.min_samples,
            "completed_events_count": event_count,
        }

    sample_count = int(latest.get("sample_count", 0) or 0)
    cv_accuracy = float(latest.get("cv_accuracy", 0.0) or 0.0)
    cv_log_loss = float(latest.get("cv_log_loss", 0.0) or 0.0)
    cv_brier = float(latest.get("cv_brier", 0.0) or 0.0)
    cv_ece = float(latest.get("cv_ece", 0.0) or 0.0)
    holdout_accuracy = float(latest.get("holdout_accuracy", 0.0) or 0.0)
    trained_at = str(latest.get("trained_at_utc") or "")

    if sample_count < cfg.min_samples:
        status = "Needs More Data"
        reason = f"Sample count {sample_count} is below required minimum {cfg.min_samples}."
    elif sample_count < int(cfg.min_samples * 1.25):
        status = "Undertrained"
        reason = "Model is trained, but sample volume is still thin for stable calibration."
    elif cv_accuracy < 0.52 or holdout_accuracy < 0.50 or cv_ece > 0.18:
        status = "Low Confidence"
        reason = "Performance and calibration metrics indicate unstable prediction quality."
    else:
        status = "Healthy"
        reason = "Model has sufficient data and acceptable accuracy/calibration metrics."

    return {
        "sport": sport,
        "status": status,
        "reason": reason,
        "sample_count": sample_count,
        "required_min_samples": cfg.min_samples,
        "completed_events_count": int(latest.get("completed_events_count", 0) or 0),
        "last_trained_at_utc": trained_at,
        "active_model": str(latest.get("best_model") or "unknown"),
        "version": str(latest.get("version") or ""),
        "metrics": {
            "cv_accuracy": cv_accuracy,
            "cv_log_loss": cv_log_loss,
            "cv_brier": cv_brier,
            "cv_ece": cv_ece,
            "holdout_accuracy": holdout_accuracy,
        },
    }


def adaptive_probabilities_for_upcoming(
    sport: str,
    completed_events_from_feed: list[dict[str, Any]],
    upcoming_events: list[dict[str, Any]],
    db_path: Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    init_model_store(db_path)
    sync_completed_events(sport, completed_events_from_feed, db_path=db_path)
    completed_events = _load_training_events(sport, db_path=db_path)
    cfg = _sport_config(sport)
    supports_draw = SUPPORTED_SPORTS[sport].supports_draw

    if len(completed_events) < cfg.min_samples:
        return {}, {"status": "not_enough_data", "sample_count": len(completed_events)}

    x, y, team_states, league_context = _build_training_matrix(sport, completed_events)
    if len(x) < cfg.min_samples:
        return {}, {"status": "not_enough_samples", "sample_count": int(len(x))}
    if len(set(y.tolist())) < cfg.min_classes:
        return {}, {"status": "not_enough_classes", "sample_count": int(len(x))}

    retrain = _needs_retrain(sport, len(completed_events), db_path=db_path)
    artifact, model_run = _load_latest_valid_artifact(sport, db_path=db_path)

    if retrain or artifact is None:
        model, best_desc, cv_metrics, holdout_acc = _fit_best_model(sport, x, y)
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        feature_importance = _extract_feature_importance(model)
        summary = {
            "feature_importance": feature_importance,
            "cv_metrics": {k: round(float(v), 6) for k, v in cv_metrics.items()},
            "model_comparisons": best_desc["comparisons"],
            "class_labels": sorted({str(v) for v in y}),
        }
        artifact_path = _save_model_artifact(
            sport,
            model,
            version=version,
            context={"sample_count": int(len(x)), "class_labels": summary["class_labels"]},
        )
        model_run = _record_model_run(
            sport,
            version=version,
            sample_count=int(len(x)),
            completed_events_count=len(completed_events),
            best_model=str(best_desc["name"]),
            best_params=dict(best_desc["params"]),
            cv_metrics=cv_metrics,
            holdout_accuracy=holdout_acc,
            artifact_path=artifact_path,
            summary=summary,
            db_path=db_path,
        )
        artifact, _ = _load_latest_valid_artifact(sport, db_path=db_path)

    if artifact is None:
        return {}, {"status": "model_unavailable", "sample_count": int(len(x))}

    model = artifact["model"]
    probs_by_event: dict[str, dict[str, float]] = {}
    for event in upcoming_events:
        home = team_states.get(event["home_team"], TeamState())
        away = team_states.get(event["away_team"], TeamState())
        row = np.array(
            _feature_row(
                sport,
                home,
                away,
                home_adv=1.0,
                league_avg_for=float(league_context["avg_home"]),
                league_avg_against=float(league_context["avg_away"]),
            ),
            dtype=float,
        )
        probs_by_event[str(event["event_id"])] = _probabilities_from_model(
            model, row, supports_draw=supports_draw
        )

    info = {
        "status": "ok",
        "sample_count": int(len(x)),
        "completed_events_count": len(completed_events),
        "best_model": (model_run or {}).get("best_model", "unknown"),
        "holdout_accuracy": float((model_run or {}).get("holdout_accuracy", 0.0) or 0.0),
        "cv_accuracy": float((model_run or {}).get("cv_accuracy", 0.0) or 0.0),
        "cv_log_loss": float((model_run or {}).get("cv_log_loss", 0.0) or 0.0),
        "cv_brier": float((model_run or {}).get("cv_brier", 0.0) or 0.0),
        "cv_ece": float((model_run or {}).get("cv_ece", 0.0) or 0.0),
        "trained_at_utc": (model_run or {}).get("trained_at_utc"),
        "version": (model_run or {}).get("version"),
    }
    summary = get_latest_model_summary(sport, db_path=db_path)
    if summary:
        info["top_features"] = summary.get("feature_importance", [])[:8]
    LOGGER.info(
        "adaptive_probabilities_ready",
        extra={"context": {"sport": sport, "events": len(upcoming_events), "model": info.get("best_model")}},
    )
    return probs_by_event, info
