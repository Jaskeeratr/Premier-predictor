from __future__ import annotations

from pathlib import Path
from typing import Any

from webapp.errors import ValidationError
from webapp.injury_store import delete_injury_adjustments, list_injury_adjustments, upsert_injury_adjustment


class InjuryService:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def list_rows(self, *, sport: str | None, team_term: str | None, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValidationError("limit must be a positive integer")
        return list_injury_adjustments(sport=sport, team_term=team_term, limit=limit, db_path=self._db_path)

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        sport_key = payload.get("sport")
        team = str(payload.get("team", "")).strip()
        if not sport_key or not team:
            raise ValidationError("sport and team are required")

        return upsert_injury_adjustment(
            sport_key,
            team,
            rating_delta=float(payload.get("rating_delta", 0.0)),
            form_delta=float(payload.get("form_delta", 0.0)),
            offense_delta=float(payload.get("offense_delta", 0.0)),
            defense_delta=float(payload.get("defense_delta", 0.0)),
            notes=str(payload.get("notes", "")),
            db_path=self._db_path,
        )

    def delete(self, *, sport: str | None, team: str | None) -> int:
        return delete_injury_adjustments(sport=sport, team=team, db_path=self._db_path)
