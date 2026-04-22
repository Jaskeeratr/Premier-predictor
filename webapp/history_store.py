from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "prediction_history.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                generated_at_utc TEXT NOT NULL,
                signature TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                sport TEXT NOT NULL,
                event_id TEXT,
                kickoff_utc TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                venue TEXT,
                predicted_winner TEXT,
                predicted_result TEXT,
                confidence REAL,
                predicted_score TEXT,
                home_win_probability REAL,
                draw_probability REAL,
                away_win_probability REAL,
                factors_json TEXT,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(snapshot_id) REFERENCES prediction_snapshots(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_sport_created ON prediction_history(sport, created_at_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_teams ON prediction_history(home_team, away_team)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_league ON prediction_history(league)")
        conn.commit()


def _snapshot_signature(matches: list[dict[str, Any]]) -> str:
    reduced = [
        {
            "event_id": item.get("event_id"),
            "kickoff_utc": item.get("kickoff_utc"),
            "predicted_winner": item.get("predicted_winner"),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
            "predicted_score": item.get("predicted_score"),
        }
        for item in matches
    ]
    reduced_sorted = sorted(reduced, key=lambda row: (str(row["kickoff_utc"]), str(row["event_id"])))
    raw = json.dumps(reduced_sorted, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_prediction_snapshot(
    sport: str,
    matches: list[dict[str, Any]],
    *,
    generated_at_utc: str | None = None,
    db_path: Path | None = None,
) -> int | None:
    if not matches:
        return None

    created_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    signature = _snapshot_signature(matches)
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()

    with _connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id FROM prediction_snapshots
            WHERE sport = ? AND signature = ? AND generated_at_utc >= ?
            ORDER BY id DESC LIMIT 1
            """,
            (sport, signature, cutoff),
        ).fetchone()
        if existing:
            return int(existing["id"])

        cur = conn.execute(
            "INSERT INTO prediction_snapshots (sport, generated_at_utc, signature) VALUES (?, ?, ?)",
            (sport, created_at, signature),
        )
        snapshot_id = int(cur.lastrowid)

        rows = []
        for item in matches:
            rows.append(
                (
                    snapshot_id,
                    sport,
                    item.get("event_id"),
                    item.get("kickoff_utc"),
                    item.get("league"),
                    item.get("home_team"),
                    item.get("away_team"),
                    item.get("venue"),
                    item.get("predicted_winner"),
                    item.get("predicted_result"),
                    float(item.get("confidence", 0.0)),
                    item.get("predicted_score"),
                    float(item.get("home_win_probability", 0.0)),
                    float(item.get("draw_probability", 0.0)),
                    float(item.get("away_win_probability", 0.0)),
                    json.dumps(item.get("factors", {})),
                    created_at,
                )
            )
        conn.executemany(
            """
            INSERT INTO prediction_history (
                snapshot_id, sport, event_id, kickoff_utc, league, home_team, away_team, venue,
                predicted_winner, predicted_result, confidence, predicted_score,
                home_win_probability, draw_probability, away_win_probability,
                factors_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return snapshot_id


def list_prediction_history(
    *,
    sport: str | None = None,
    team: str | None = None,
    league: str | None = None,
    limit: int = 200,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    where = []
    params: list[Any] = []

    if sport:
        where.append("sport = ?")
        params.append(sport)
    if league:
        where.append("LOWER(league) LIKE ?")
        params.append(f"%{league.lower()}%")
    if team:
        where.append("(LOWER(home_team) LIKE ? OR LOWER(away_team) LIKE ?)")
        term = f"%{team.lower()}%"
        params.extend([term, term])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    query = f"""
        SELECT
            id, snapshot_id, sport, event_id, kickoff_utc, league, home_team, away_team, venue,
            predicted_winner, predicted_result, confidence, predicted_score,
            home_win_probability, draw_probability, away_win_probability,
            created_at_utc
        FROM prediction_history
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def clear_prediction_history(*, sport: str | None = None, db_path: Path | None = None) -> int:
    with _connect(db_path) as conn:
        if sport:
            target_snapshots = conn.execute(
                "SELECT id FROM prediction_snapshots WHERE sport = ?",
                (sport,),
            ).fetchall()
            snapshot_ids = [int(row["id"]) for row in target_snapshots]
            if not snapshot_ids:
                return 0
            placeholder = ",".join("?" for _ in snapshot_ids)
            deleted = conn.execute(
                f"DELETE FROM prediction_history WHERE snapshot_id IN ({placeholder})",
                snapshot_ids,
            ).rowcount
            conn.execute(
                f"DELETE FROM prediction_snapshots WHERE id IN ({placeholder})",
                snapshot_ids,
            )
            conn.commit()
            return int(deleted)

        deleted = conn.execute("DELETE FROM prediction_history").rowcount
        conn.execute("DELETE FROM prediction_snapshots")
        conn.commit()
        return int(deleted)
