from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

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

DEMO_TEAMS: dict[str, list[str]] = {
    "football": [
        "Arsenal",
        "Liverpool",
        "Manchester City",
        "Chelsea",
        "Tottenham",
        "Aston Villa",
        "Newcastle",
        "Brighton",
    ],
    "american_football": ["Chiefs", "Bills", "Ravens", "49ers", "Cowboys", "Packers", "Lions", "Eagles"],
    "basketball": ["Lakers", "Celtics", "Nuggets", "Bucks", "Knicks", "Mavericks", "Suns", "Warriors"],
    "volleyball": ["Falcons VC", "Orcas VC", "Titans VC", "Storm VC", "Atlas VC", "River VC"],
    "cricket": [
        "Mumbai Indians",
        "Chennai Super Kings",
        "RCB",
        "Kolkata Knight Riders",
        "Karachi Kings",
        "Lahore Qalandars",
    ],
}

DEMO_LEAGUES: dict[str, str] = {
    "football": "Demo Premier League",
    "american_football": "Demo Pro Football",
    "basketball": "Demo Basketball League",
    "volleyball": "Demo Volleyball Series",
    "cricket": "Demo Cricket Championship",
}

DEMO_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "football": (0, 4),
    "american_football": (10, 41),
    "basketball": (85, 132),
    "volleyball": (0, 3),
    "cricket": (120, 210),
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
        "home_score": _to_int(home.get("score")),
        "away_score": _to_int(away.get("score")),
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
    consecutive_failures = 0
    for league_slug in target_leagues:
        url = ESPN_SCOREBOARD_URL.format(sport=config.espn_sport_slug, league=league_slug)
        try:
            response = requests.get(url, params={"dates": dates_range, "limit": 500}, timeout=5)
            if response.status_code != 200:
                continue
            payload = response.json()
            consecutive_failures = 0
        except requests.RequestException as exc:
            LOGGER.warning("espn_fetch_failed", extra={"context": {"league": league_slug, "error": str(exc)}})
            consecutive_failures += 1
            if consecutive_failures >= 2 and not events:
                break
            continue
        for event in payload.get("events") or []:
            normalized = _normalize_espn_event(event, sport_key=config.key, league_slug=league_slug)
            if normalized:
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
    consecutive_failures = 0
    while current_day <= end:
        try:
            response = requests.get(
                SPORTSDB_EVENTS_DAY_URL,
                params={"d": current_day.isoformat(), "s": config.label},
                timeout=5,
            )
            if response.status_code == 200:
                payload = response.json()
                for event in payload.get("events") or []:
                    normalized = _normalize_sportsdb_event(event, sport_key=config.key)
                    if normalized:
                        events[normalized["event_id"]] = normalized
                consecutive_failures = 0
        except requests.RequestException as exc:
            LOGGER.warning(
                "sportsdb_fetch_failed",
                extra={"context": {"day": current_day.isoformat(), "error": str(exc)}},
            )
            consecutive_failures += 1
            if consecutive_failures >= 3 and not events:
                break
        current_day += timedelta(days=1)
    return sorted(events.values(), key=lambda item: item["start_time"])


def _fetch_sportsdb_calendar_events(config: SportConfig, start: date, end: date) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    current_day = start
    consecutive_failures = 0
    while current_day <= end:
        try:
            response = requests.get(
                SPORTSDB_BROWSE_CALENDAR_URL,
                params={"d": current_day.isoformat(), "s": config.label.lower()},
                timeout=5,
            )
            if response.status_code != 200:
                current_day += timedelta(days=1)
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            consecutive_failures = 0
        except requests.RequestException as exc:
            LOGGER.warning(
                "sportsdb_calendar_failed",
                extra={"context": {"day": current_day.isoformat(), "error": str(exc)}},
            )
            consecutive_failures += 1
            if consecutive_failures >= 3 and not events:
                break
            current_day += timedelta(days=1)
            continue

        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                continue
            kickoff_text = cells[0].get_text(" ", strip=True).replace("UTC", "").strip()
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
            kickoff_dt = _parse_datetime(f"{current_day.isoformat()}T{kickoff_text or '00:00'}:00")
            events[event_id] = {
                "event_id": event_id,
                "sport": config.key,
                "league": cells[2].get_text(" ", strip=True) or config.label,
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


def _rand01(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / ((1 << 64) - 1)


def _rand_int(key: str, low: int, high: int) -> int:
    if high <= low:
        return low
    return low + int(_rand01(key) * ((high - low) + 1))


def generate_demo_events(
    sport_key: str,
    *,
    anchor_day: date,
    days_back: int,
    days_ahead: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teams = DEMO_TEAMS[sport_key]
    score_low, score_high = DEMO_SCORE_RANGES[sport_key]
    league = DEMO_LEAGUES[sport_key]
    units = SUPPORTED_SPORTS[sport_key].score_units
    supports_draw = SUPPORTED_SPORTS[sport_key].supports_draw
    anchor_dt = datetime.combine(anchor_day, datetime.min.time(), tzinfo=timezone.utc)

    events: list[dict[str, Any]] = []
    event_counter = 0
    for day_offset in range(-days_back, days_ahead + 1):
        day = anchor_day + timedelta(days=day_offset)
        if day.weekday() not in {1, 3, 5, 6}:
            continue
        slot_count = 2 if len(teams) >= 8 else 1
        for slot in range(slot_count):
            home_idx = (day.toordinal() + slot * 3) % len(teams)
            away_idx = (home_idx + 1 + slot * 2) % len(teams)
            if away_idx == home_idx:
                away_idx = (away_idx + 1) % len(teams)
            home_team = teams[home_idx]
            away_team = teams[away_idx]
            kickoff = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
                hours=18 + (slot * 2)
            )
            event_id = f"demo-{sport_key}-{day.isoformat()}-{slot}-{event_counter}"
            event_counter += 1

            is_completed = kickoff < anchor_dt
            home_score: int | None = None
            away_score: int | None = None
            if is_completed:
                home_score = _rand_int(f"{event_id}:h", score_low, score_high)
                away_score = _rand_int(f"{event_id}:a", score_low, score_high)
                if supports_draw:
                    if _rand01(f"{event_id}:draw") < 0.18:
                        away_score = home_score
                elif home_score == away_score:
                    away_score = max(score_low, away_score - 1)

            events.append(
                {
                    "event_id": event_id,
                    "sport": sport_key,
                    "league": league,
                    "start_time": kickoff,
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": f"{home_team} Arena",
                    "neutral_site": False,
                    "status": "completed" if is_completed else "scheduled",
                    "home_score": home_score,
                    "away_score": away_score,
                    "score_units": units,
                }
            )

    completed = [e for e in events if e["home_score"] is not None and e["away_score"] is not None]
    upcoming = [e for e in events if e["start_time"] >= anchor_dt]
    return sorted(completed, key=lambda x: x["start_time"]), sorted(upcoming, key=lambda x: x["start_time"])


def fetch_sport_events_with_meta(
    sport_key: str,
    *,
    today: date | None = None,
    days_back: int = 120,
    days_ahead: int = 14,
    force_demo_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sport_key not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport: {sport_key}")

    config = SUPPORTED_SPORTS[sport_key]
    anchor_day = today or datetime.now(timezone.utc).date()
    start_day = anchor_day - timedelta(days=days_back)
    end_day = anchor_day + timedelta(days=days_ahead)
    anchor_dt = datetime.combine(anchor_day, datetime.min.time(), tzinfo=timezone.utc)

    if force_demo_mode:
        completed, upcoming = generate_demo_events(
            sport_key,
            anchor_day=anchor_day,
            days_back=days_back,
            days_ahead=days_ahead,
        )
        return (
            completed,
            upcoming,
            {"mode": "demo", "reason": "forced_by_config", "provider": "demo_generator"},
        )

    events: list[dict[str, Any]] = []
    try:
        if config.provider == "espn":
            events = _fetch_espn_events(config, start=start_day, end=end_day)
            if sport_key == "basketball":
                nba_present = any(
                    event.get("source_league_slug") == "nba" and event["start_time"] >= anchor_dt
                    for event in events
                )
                if not nba_present:
                    extended = _fetch_espn_events(
                        config,
                        start=anchor_day,
                        end=anchor_day + timedelta(days=240),
                        leagues=("nba",),
                    )
                    merged = {event["event_id"]: event for event in events}
                    for event in extended:
                        merged[event["event_id"]] = event
                    events = sorted(merged.values(), key=lambda item: item["start_time"])
        else:
            recent_start = anchor_day - timedelta(days=min(days_back, 45))
            scored = _fetch_sportsdb_events(config, start=recent_start, end=anchor_day)
            calendar = _fetch_sportsdb_calendar_events(config, start=anchor_day, end=end_day)
            merged = {e["event_id"]: e for e in scored}
            for event in calendar:
                merged[event["event_id"]] = event
            events = sorted(merged.values(), key=lambda item: item["start_time"])
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.exception(
            "live_fetch_unexpected_failure", extra={"context": {"sport": sport_key, "error": str(exc)}}
        )
        events = []

    completed = [
        e
        for e in events
        if e["home_score"] is not None and e["away_score"] is not None and e["start_time"] < anchor_dt
    ]
    upcoming = [e for e in events if e["start_time"] >= anchor_dt]

    # If live feeds are unavailable or sparse, fallback to demo mode automatically.
    if len(events) == 0 or len(upcoming) == 0 or len(completed) < 12:
        demo_completed, demo_upcoming = generate_demo_events(
            sport_key,
            anchor_day=anchor_day,
            days_back=days_back,
            days_ahead=days_ahead,
        )
        reason = "live_feed_unavailable_or_sparse"
        return demo_completed, demo_upcoming, {"mode": "demo", "reason": reason, "provider": "demo_generator"}

    return completed, upcoming, {"mode": "live", "reason": "live_feed_ok", "provider": config.provider}


def fetch_sport_events(
    sport_key: str,
    *,
    today: date | None = None,
    days_back: int = 120,
    days_ahead: int = 14,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed, upcoming, _ = fetch_sport_events_with_meta(
        sport_key,
        today=today,
        days_back=days_back,
        days_ahead=days_ahead,
        force_demo_mode=False,
    )
    return completed, upcoming
