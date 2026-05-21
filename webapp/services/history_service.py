from __future__ import annotations

from pathlib import Path
from typing import Any

from webapp.errors import ValidationError
from webapp.history_store import clear_prediction_history, list_prediction_history


class HistoryService:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def list_rows(
        self, *, sport: str | None, team: str | None, league: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValidationError("limit must be a positive integer")
        return list_prediction_history(
            sport=sport, team=team, league=league, limit=limit, db_path=self._db_path
        )

    def clear(self, *, sport: str | None) -> int:
        return clear_prediction_history(sport=sport, db_path=self._db_path)
