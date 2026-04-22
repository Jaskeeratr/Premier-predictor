from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

from webapp.history_store import DEFAULT_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_injury_store(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injury_adjustments (
                sport TEXT NOT NULL,
                team TEXT NOT NULL,
                rating_delta REAL NOT NULL DEFAULT 0,
                form_delta REAL NOT NULL DEFAULT 0,
                offense_delta REAL NOT NULL DEFAULT 0,
                defense_delta REAL NOT NULL DEFAULT 0,
                notes TEXT,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (sport, team)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_injury_sport ON injury_adjustments(sport)")
        conn.commit()


def upsert_injury_adjustment(
    sport: str,
    team: str,
    *,
    rating_delta: float = 0.0,
    form_delta: float = 0.0,
    offense_delta: float = 0.0,
    defense_delta: float = 0.0,
    notes: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    team = team.strip()
    if not team:
        raise ValueError("team is required")

    payload = (
        sport,
        team,
        float(rating_delta),
        float(form_delta),
        float(offense_delta),
        float(defense_delta),
        (notes or "").strip(),
        _utc_now(),
    )

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO injury_adjustments (
                sport, team, rating_delta, form_delta, offense_delta, defense_delta, notes, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sport, team) DO UPDATE SET
                rating_delta=excluded.rating_delta,
                form_delta=excluded.form_delta,
                offense_delta=excluded.offense_delta,
                defense_delta=excluded.defense_delta,
                notes=excluded.notes,
                updated_at_utc=excluded.updated_at_utc
            """,
            payload,
        )
        conn.commit()
    return {
        "sport": sport,
        "team": team,
        "rating_delta": float(rating_delta),
        "form_delta": float(form_delta),
        "offense_delta": float(offense_delta),
        "defense_delta": float(defense_delta),
        "notes": (notes or "").strip(),
    }


def list_injury_adjustments(
    *,
    sport: str | None = None,
    team_term: str | None = None,
    limit: int = 500,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 2000))
    where = []
    params: list[Any] = []
    if sport:
        where.append("sport = ?")
        params.append(sport)
    if team_term:
        where.append("LOWER(team) LIKE ?")
        params.append(f"%{team_term.lower()}%")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    query = f"""
        SELECT sport, team, rating_delta, form_delta, offense_delta, defense_delta, notes, updated_at_utc
        FROM injury_adjustments
        {where_sql}
        ORDER BY updated_at_utc DESC
        LIMIT ?
    """
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_injury_adjustments_map(sport: str, db_path: Path | None = None) -> dict[str, dict[str, float]]:
    rows = list_injury_adjustments(sport=sport, limit=5000, db_path=db_path)
    mapping: dict[str, dict[str, float]] = {}
    for row in rows:
        mapping[row["team"]] = {
            "rating_delta": float(row.get("rating_delta", 0.0)),
            "form_delta": float(row.get("form_delta", 0.0)),
            "offense_delta": float(row.get("offense_delta", 0.0)),
            "defense_delta": float(row.get("defense_delta", 0.0)),
        }
    return mapping


def delete_injury_adjustments(
    *,
    sport: str | None = None,
    team: str | None = None,
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        if sport and team:
            deleted = conn.execute(
                "DELETE FROM injury_adjustments WHERE sport = ? AND team = ?",
                (sport, team.strip()),
            ).rowcount
        elif sport:
            deleted = conn.execute(
                "DELETE FROM injury_adjustments WHERE sport = ?",
                (sport,),
            ).rowcount
        else:
            deleted = conn.execute("DELETE FROM injury_adjustments").rowcount
        conn.commit()
        return int(deleted)
