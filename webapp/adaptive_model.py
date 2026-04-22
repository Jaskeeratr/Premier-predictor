from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from webapp.data_sources import SUPPORTED_SPORTS
from webapp.history_store import DEFAULT_DB_PATH
from webapp.prediction_engine import SPORT_PROFILES

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


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


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_model_store(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
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
            """
        )
        conn.execute(
            """
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
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_events_sport_time ON training_events(sport, start_time_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_sport_time ON model_runs(sport, trained_at_utc)")
        conn.commit()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_completed_events(sport: str, completed_events: list[dict[str, Any]], db_path: Path | None = None) -> int:
    if not completed_events:
        return 0
    now = _utc_now_iso()
    rows = []
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


def _load_latest_model_run(sport: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM model_runs
            WHERE sport = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (sport,),
        ).fetchone()
        return dict(row) if row else None


def _record_model_run(
    sport: str,
    *,
    sample_count: int,
    completed_events_count: int,
    best_model: str,
    best_params: dict[str, Any],
    holdout_accuracy: float,
    model_path: Path,
    db_path: Path | None = None,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_runs (
                sport, trained_at_utc, sample_count, completed_events_count,
                best_model, best_params_json, holdout_accuracy, model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sport,
                _utc_now_iso(),
                sample_count,
                completed_events_count,
                best_model,
                json.dumps(best_params),
                float(holdout_accuracy),
                str(model_path),
            ),
        )
        conn.commit()


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

    rating_shift = profile.k_factor * (actual_home - exp_home)
    home.rating += rating_shift
    away.rating -= rating_shift

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
    attack_edge = (home.avg_for - away.avg_for)
    defense_edge = (away.avg_against - home.avg_against)
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
    sport: str,
    completed_events: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, TeamState], dict[str, float]]:
    supports_draw = SUPPORTED_SPORTS[sport].supports_draw
    teams: dict[str, TeamState] = {}
    X: list[list[float]] = []
    y: list[str] = []
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
        league_avg_against = (total_away / total_games) if total_games else SPORT_PROFILES[sport].base_away_score

        row = _feature_row(
            sport,
            home,
            away,
            home_adv=1.0,
            league_avg_for=league_avg_for,
            league_avg_against=league_avg_against,
        )
        label = _outcome_label(home_score, away_score, supports_draw)
        if label is not None:
            X.append(row)
            y.append(label)

        _update_team_states(sport, home, away, home_score=home_score, away_score=away_score)
        total_home += home_score
        total_away += away_score
        total_games += 1

    context = {
        "avg_home": (total_home / total_games) if total_games else SPORT_PROFILES[sport].base_home_score,
        "avg_away": (total_away / total_games) if total_games else SPORT_PROFILES[sport].base_away_score,
    }
    return np.array(X, dtype=float), np.array(y), teams, context


def _candidate_models(sport: str, supports_draw: bool) -> list[tuple[str, dict[str, Any], Any]]:
    models: list[tuple[str, dict[str, Any], Any]] = []
    for c in [0.1, 0.3, 0.7, 1.0, 2.0, 4.0]:
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3500,
                        C=c,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        models.append(("logreg", {"C": c}, lr))

    if not supports_draw:
        for depth in [6, 8, 10]:
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=2,
                random_state=42,
            )
            models.append(("random_forest", {"max_depth": depth, "n_estimators": 300}, rf))
    return models


def _fit_best_model(sport: str, X: np.ndarray, y: np.ndarray) -> tuple[Any, str, dict[str, Any], float]:
    supports_draw = SUPPORTED_SPORTS[sport].supports_draw
    split_idx = int(len(X) * 0.82)
    split_idx = max(20, min(split_idx, len(X) - 10))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    best_model = None
    best_name = ""
    best_params: dict[str, Any] = {}
    best_acc = -1.0

    for name, params, model in _candidate_models(sport, supports_draw):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = float(acc)
            best_model = model
            best_name = name
            best_params = params

    if best_model is None:
        raise RuntimeError(f"Model training failed for sport={sport}")

    best_model.fit(X, y)
    return best_model, best_name, best_params, best_acc


def _model_path(sport: str) -> Path:
    return MODEL_DIR / f"{sport}_latest.joblib"


def _save_model_artifact(
    sport: str,
    model: Any,
    *,
    context: dict[str, Any],
) -> Path:
    path = _model_path(sport)
    artifact = {
        "sport": sport,
        "model": model,
        "context": context,
        "saved_at_utc": _utc_now_iso(),
    }
    joblib.dump(artifact, path)
    return path


def _load_model_artifact(sport: str) -> dict[str, Any] | None:
    path = _model_path(sport)
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
        return artifact if isinstance(artifact, dict) else None
    except Exception:
        return None


def _needs_retrain(sport: str, completed_events_count: int, db_path: Path | None = None) -> bool:
    latest = _load_latest_model_run(sport, db_path)
    if not latest:
        return True

    seen = int(latest.get("completed_events_count", 0))
    if completed_events_count >= seen + 4:
        return True

    try:
        trained_at = datetime.fromisoformat(str(latest["trained_at_utc"]))
        age_hours = (datetime.now(timezone.utc) - trained_at.astimezone(timezone.utc)).total_seconds() / 3600.0
        if age_hours >= 24:
            return True
    except Exception:
        return True

    return False


def _probabilities_from_model(model: Any, row: np.ndarray, supports_draw: bool) -> dict[str, float]:
    classes = list(model.classes_)
    probs = model.predict_proba(row.reshape(1, -1))[0]
    mapping = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

    home = mapping.get("H", 0.0)
    away = mapping.get("A", 0.0)
    draw = mapping.get("D", 0.0) if supports_draw else 0.0

    total = home + away + draw
    if total <= 0:
        return {"H": 0.5, "D": 0.0 if not supports_draw else 0.2, "A": 0.5}
    return {"H": home / total, "D": draw / total, "A": away / total}


def adaptive_probabilities_for_upcoming(
    sport: str,
    completed_events_from_feed: list[dict[str, Any]],
    upcoming_events: list[dict[str, Any]],
    db_path: Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    init_model_store(db_path)
    sync_completed_events(sport, completed_events_from_feed, db_path)
    completed_events = _load_training_events(sport, db_path)

    supports_draw = SUPPORTED_SPORTS[sport].supports_draw
    if len(completed_events) < 30:
        return {}, {"status": "not_enough_data", "sample_count": len(completed_events)}

    X, y, team_states, league_context = _build_training_matrix(sport, completed_events)
    if len(X) < 25 or len(set(y)) < 2:
        return {}, {"status": "not_enough_classes", "sample_count": len(X)}

    model_run = _load_latest_model_run(sport, db_path)
    retrain = _needs_retrain(sport, len(completed_events), db_path=db_path)
    artifact = _load_model_artifact(sport)

    if retrain or artifact is None:
        model, best_name, best_params, holdout_acc = _fit_best_model(sport, X, y)
        model_path = _save_model_artifact(sport, model, context={"sample_count": int(len(X))})
        _record_model_run(
            sport,
            sample_count=int(len(X)),
            completed_events_count=len(completed_events),
            best_model=best_name,
            best_params=best_params,
            holdout_accuracy=holdout_acc,
            model_path=model_path,
            db_path=db_path,
        )
        model_run = _load_latest_model_run(sport, db_path)
        artifact = _load_model_artifact(sport)

    if artifact is None:
        return {}, {"status": "model_unavailable", "sample_count": len(X)}

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
        probs_by_event[str(event["event_id"])] = _probabilities_from_model(model, row, supports_draw)

    info = {
        "status": "ok",
        "sample_count": int(len(X)),
        "completed_events_count": len(completed_events),
        "best_model": model_run.get("best_model") if model_run else "unknown",
        "holdout_accuracy": float(model_run.get("holdout_accuracy", 0.0)) if model_run else 0.0,
        "trained_at_utc": model_run.get("trained_at_utc") if model_run else None,
    }
    return probs_by_event, info
