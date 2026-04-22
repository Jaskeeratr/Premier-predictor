from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re
from typing import Any

from bs4 import BeautifulSoup
import requests

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
SPORTSDB_EVENTS_DAY_URL = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
SPORTSDB_BROWSE_CALENDAR_URL = "https://www.thesportsdb.com/browse_calendar/"


@dataclass(frozen=True)
class SportConfig:
    key: str
    label: str
    provider: str
    espn_sport_slug: str | None = None
    espn_leagues: tuple[str, ...] = ()
    supports_draw: bool = False
    score_units: str = "points"


SUPPORTED_SPORTS: dict[str, SportConfig] = {
    "football": SportConfig(
        key="football",
        label="Football",
        provider="espn",
        espn_sport_slug="soccer",
        espn_leagues=("eng.1", "esp.1", "ger.1", "uefa.champions", "usa.1"),
        supports_draw=True,
        score_units="goals",
    ),
    "american_football": SportConfig(
        key="american_football",
        label="American Football",
        provider="espn",
        espn_sport_slug="football",
        espn_leagues=("ufl", "nfl", "college-football"),
        supports_draw=False,
        score_units="points",
    ),
    "volleyball": SportConfig(
        key="volleyball",
        label="Volleyball",
        provider="espn",
        espn_sport_slug="volleyball",
        espn_leagues=("mens-college-volleyball",),
        supports_draw=False,
        score_units="sets",
    ),
    "basketball": SportConfig(
        key="basketball",
        label="Basketball",
        provider="espn",
        espn_sport_slug="basketball",
        espn_leagues=("nba", "wnba"),
        supports_draw=False,
        score_units="points",
    ),
    "cricket": SportConfig(
        key="cricket",
        label="Cricket",
        provider="sportsdb",
        supports_draw=True,
        score_units="runs",
    ),
}


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_espn_event(event: dict[str, Any], sport_key: str, league_slug: str) -> dict[str, Any] | None:
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    if len(competitors) < 2:
        return None

    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
    status_state = ((event.get("status") or {}).get("type") or {}).get("state", "pre")
    status_lookup = {"pre": "scheduled", "in": "in_progress", "post": "completed"}

    home_score = _to_int(home.get("score"))
    away_score = _to_int(away.get("score"))

    return {
        "event_id": str(event.get("id", "")),
        "sport": sport_key,
        "source_league_slug": league_slug,
        "league": (competition.get("league") or {}).get("name") or league_slug,
        "start_time": _parse_datetime(event.get("date")),
        "home_team": (home.get("team") or {}).get("displayName", "Home Team"),
        "away_team": (away.get("team") or {}).get("displayName", "Away Team"),
        "venue": (competition.get("venue") or {}).get("fullName", "Unknown Venue"),
        "neutral_site": bool(competition.get("neutralSite", False)),
        "status": status_lookup.get(status_state, "scheduled"),
        "home_score": home_score,
        "away_score": away_score,
        "score_units": SUPPORTED_SPORTS[sport_key].score_units,
    }


def _fetch_espn_events(
    config: SportConfig,
    start: date,
    end: date,
    leagues: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    dates_range = f"{start:%Y%m%d}-{end:%Y%m%d}"

    target_leagues = leagues or config.espn_leagues
    for league_slug in target_leagues:
        url = ESPN_SCOREBOARD_URL.format(sport=config.espn_sport_slug, league=league_slug)
        params = {"dates": dates_range, "limit": 500}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            continue

        payload = response.json()
        for event in payload.get("events") or []:
            normalized = _normalize_espn_event(event, sport_key=config.key, league_slug=league_slug)
            if not normalized:
                continue
            events[normalized["event_id"]] = normalized

    return sorted(events.values(), key=lambda item: item["start_time"])


def _normalize_sportsdb_event(event: dict[str, Any], sport_key: str) -> dict[str, Any] | None:
    event_date = event.get("dateEvent")
    event_time = event.get("strTime") or "00:00:00"
    if not event_date:
        return None

    start_dt = _parse_datetime(f"{event_date}T{event_time}")
    home_score = _to_int(event.get("intHomeScore"))
    away_score = _to_int(event.get("intAwayScore"))
    status = "completed" if home_score is not None and away_score is not None else "scheduled"

    return {
        "event_id": str(event.get("idEvent", f"{event_date}-{event.get('strEvent', '')}")),
        "sport": sport_key,
        "league": event.get("strLeague", "Cricket"),
        "start_time": start_dt,
        "home_team": event.get("strHomeTeam", "Home Team"),
        "away_team": event.get("strAwayTeam", "Away Team"),
        "venue": event.get("strVenue", "Unknown Venue"),
        "neutral_site": False,
        "status": status,
        "home_score": home_score,
        "away_score": away_score,
        "score_units": SUPPORTED_SPORTS[sport_key].score_units,
    }


def _fetch_sportsdb_events(config: SportConfig, start: date, end: date) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    current_day = start
    sport_name = config.label

    while current_day <= end:
        params = {"d": current_day.isoformat(), "s": sport_name}
        response = requests.get(SPORTSDB_EVENTS_DAY_URL, params=params, timeout=30)
        if response.status_code == 200:
            payload = response.json()
            for event in payload.get("events") or []:
                normalized = _normalize_sportsdb_event(event, sport_key=config.key)
                if not normalized:
                    continue
                events[normalized["event_id"]] = normalized
        current_day += timedelta(days=1)

    return sorted(events.values(), key=lambda item: item["start_time"])


def _fetch_sportsdb_calendar_events(config: SportConfig, start: date, end: date) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    current_day = start

    while current_day <= end:
        params = {"d": current_day.isoformat(), "s": config.label.lower()}
        response = requests.get(SPORTSDB_BROWSE_CALENDAR_URL, params=params, timeout=30)
        if response.status_code != 200:
            current_day += timedelta(days=1)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            kickoff_text = cells[0].get_text(" ", strip=True).replace("UTC", "").strip()
            league_text = cells[2].get_text(" ", strip=True)
            match_text = cells[4].get_text(" ", strip=True)

            if " vs " not in match_text:
                continue
            home_team, away_team = [part.strip() for part in match_text.split(" vs ", maxsplit=1)]

            event_link = row.find("a", href=re.compile(r"/event/\d+-"))
            event_id = None
            if event_link and event_link.get("href"):
                match = re.search(r"/event/(\d+)-", event_link.get("href", ""))
                if match:
                    event_id = match.group(1)
            event_id = event_id or f"{current_day.isoformat()}-{home_team}-{away_team}"

            if kickoff_text:
                kickoff_dt = _parse_datetime(f"{current_day.isoformat()}T{kickoff_text}:00")
            else:
                kickoff_dt = _parse_datetime(f"{current_day.isoformat()}T00:00:00")

            events[event_id] = {
                "event_id": event_id,
                "sport": config.key,
                "league": league_text or config.label,
                "start_time": kickoff_dt,
                "home_team": home_team,
                "away_team": away_team,
                "venue": "Unknown Venue",
                "neutral_site": False,
                "status": "scheduled",
                "home_score": None,
                "away_score": None,
                "score_units": config.score_units,
            }
        current_day += timedelta(days=1)

    return sorted(events.values(), key=lambda item: item["start_time"])


def fetch_sport_events(
    sport_key: str,
    *,
    today: date | None = None,
    days_back: int = 120,
    days_ahead: int = 14,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sport_key not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport: {sport_key}")

    config = SUPPORTED_SPORTS[sport_key]
    anchor_day = today or datetime.now(timezone.utc).date()
    start_day = anchor_day - timedelta(days=days_back)
    end_day = anchor_day + timedelta(days=days_ahead)
    anchor_dt = datetime.combine(anchor_day, datetime.min.time(), tzinfo=timezone.utc)

    if config.provider == "espn":
        events = _fetch_espn_events(config, start=start_day, end=end_day)
        if sport_key == "basketball":
            nba_present = any(event.get("source_league_slug") == "nba" and event["start_time"] >= anchor_dt for event in events)
            if not nba_present:
                extended_nba_events = _fetch_espn_events(
                    config,
                    start=anchor_day,
                    end=anchor_day + timedelta(days=240),
                    leagues=("nba",),
                )
                merged_events = {event["event_id"]: event for event in events}
                for event in extended_nba_events:
                    merged_events[event["event_id"]] = event
                events = sorted(merged_events.values(), key=lambda item: item["start_time"])
    else:
        # The free JSON feed can be sparse for future cricket fixtures, so we combine:
        # 1) scored event feed for recent form
        # 2) calendar page scrape for upcoming fixtures
        recent_start = anchor_day - timedelta(days=min(days_back, 45))
        scored_events = _fetch_sportsdb_events(config, start=recent_start, end=anchor_day)
        calendar_events = _fetch_sportsdb_calendar_events(config, start=anchor_day, end=end_day)
        merged: dict[str, dict[str, Any]] = {e["event_id"]: e for e in scored_events}
        for event in calendar_events:
            merged[event["event_id"]] = event
        events = sorted(merged.values(), key=lambda item: item["start_time"])

    completed = [e for e in events if e["home_score"] is not None and e["away_score"] is not None and e["start_time"] < anchor_dt]
    upcoming = [e for e in events if e["start_time"] >= anchor_dt]
    return completed, upcoming
