from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import math

from webapp.data_sources import SUPPORTED_SPORTS


@dataclass
class SportProfile:
    base_home_score: float
    base_away_score: float
    home_adv_elo: float
    k_factor: float
    default_draw_rate: float
    score_floor: float
    score_ceiling: float
    score_diff_weight: float


SPORT_PROFILES: dict[str, SportProfile] = {
    "football": SportProfile(
        base_home_score=1.5,
        base_away_score=1.1,
        home_adv_elo=55,
        k_factor=22,
        default_draw_rate=0.24,
        score_floor=0.0,
        score_ceiling=5.0,
        score_diff_weight=0.35,
    ),
    "american_football": SportProfile(
        base_home_score=24,
        base_away_score=21,
        home_adv_elo=38,
        k_factor=18,
        default_draw_rate=0.01,
        score_floor=7.0,
        score_ceiling=45.0,
        score_diff_weight=0.08,
    ),
    "basketball": SportProfile(
        base_home_score=112,
        base_away_score=108,
        home_adv_elo=30,
        k_factor=16,
        default_draw_rate=0.0,
        score_floor=70.0,
        score_ceiling=140.0,
        score_diff_weight=0.035,
    ),
    "volleyball": SportProfile(
        base_home_score=2.1,
        base_away_score=1.6,
        home_adv_elo=20,
        k_factor=18,
        default_draw_rate=0.0,
        score_floor=0.0,
        score_ceiling=3.0,
        score_diff_weight=0.45,
    ),
    "cricket": SportProfile(
        base_home_score=168,
        base_away_score=162,
        home_adv_elo=25,
        k_factor=20,
        default_draw_rate=0.08,
        score_floor=90.0,
        score_ceiling=210.0,
        score_diff_weight=0.015,
    ),
}


@dataclass
class TeamStats:
    rating: float = 1500.0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    form_results: list[float] = field(default_factory=list)

    @property
    def avg_for(self) -> float:
        return self.points_for / self.games if self.games else 0.0

    @property
    def avg_against(self) -> float:
        return self.points_against / self.games if self.games else 0.0

    @property
    def form_index(self) -> float:
        if not self.form_results:
            return 0.5
        weighted = 0.0
        total_weight = 0.0
        for idx, result in enumerate(self.form_results[-5:], start=1):
            weight = idx
            weighted += result * weight
            total_weight += weight
        return weighted / total_weight if total_weight else 0.5


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _append_form(team: TeamStats, outcome: float) -> None:
    team.form_results.append(outcome)
    if len(team.form_results) > 8:
        team.form_results.pop(0)


def _build_team_stats(completed_events: list[dict[str, Any]], sport_key: str) -> tuple[dict[str, TeamStats], dict[str, float]]:
    profile = SPORT_PROFILES[sport_key]
    teams: dict[str, TeamStats] = {}
    total_home = 0.0
    total_away = 0.0
    total_games = 0
    draw_games = 0

    sorted_events = sorted(completed_events, key=lambda event: event["start_time"])
    for event in sorted_events:
        if event["home_score"] is None or event["away_score"] is None:
            continue

        home_name = event["home_team"]
        away_name = event["away_team"]
        home_score = float(event["home_score"])
        away_score = float(event["away_score"])

        home = teams.setdefault(home_name, TeamStats())
        away = teams.setdefault(away_name, TeamStats())

        expected_home = 1 / (1 + 10 ** (((away.rating) - (home.rating + profile.home_adv_elo)) / 400))
        if home_score > away_score:
            actual_home = 1.0
            home.wins += 1
            away.losses += 1
            _append_form(home, 1.0)
            _append_form(away, 0.0)
        elif home_score < away_score:
            actual_home = 0.0
            home.losses += 1
            away.wins += 1
            _append_form(home, 0.0)
            _append_form(away, 1.0)
        else:
            actual_home = 0.5
            draw_games += 1
            home.draws += 1
            away.draws += 1
            _append_form(home, 0.5)
            _append_form(away, 0.5)

        rating_shift = profile.k_factor * (actual_home - expected_home)
        home.rating += rating_shift
        away.rating -= rating_shift

        home.games += 1
        away.games += 1
        home.points_for += home_score
        home.points_against += away_score
        away.points_for += away_score
        away.points_against += home_score

        total_home += home_score
        total_away += away_score
        total_games += 1

    empirical_avg_home = (total_home / total_games) if total_games else profile.base_home_score
    empirical_avg_away = (total_away / total_games) if total_games else profile.base_away_score
    empirical_draw = (draw_games / total_games) if total_games else profile.default_draw_rate

    # Low-data stabilization: blend strongly toward sport baselines until enough completed games exist.
    reliability = _clamp(total_games / 30.0, 0.0, 1.0)
    league_context = {
        "avg_home": (profile.base_home_score * (1.0 - reliability)) + (empirical_avg_home * reliability),
        "avg_away": (profile.base_away_score * (1.0 - reliability)) + (empirical_avg_away * reliability),
        "draw_rate": (profile.default_draw_rate * (1.0 - reliability)) + (empirical_draw * reliability),
    }
    return teams, league_context


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _team_bias(team_name: str) -> float:
    digest = hashlib.sha256(team_name.lower().encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (raw / ((1 << 64) - 1)) - 0.5


def _with_team_priors(sport_key: str, team_name: str, team: TeamStats, league_context: dict[str, float]) -> TeamStats:
    if team.games > 0:
        return team

    bias = _team_bias(f"{sport_key}:{team_name}")
    if sport_key == "cricket":
        rating_span = 80.0
        form_span = 0.30
        attack_span = 0.10
        defend_span = 0.08
    elif sport_key == "basketball":
        rating_span = 110.0
        form_span = 0.45
        attack_span = 0.18
        defend_span = 0.12
    else:
        rating_span = 120.0
        form_span = 0.55
        attack_span = 0.22
        defend_span = 0.12

    base_rating = 1500 + (bias * rating_span)
    form = _clamp(0.5 + (bias * form_span), 0.2, 0.8)

    avg_home = float(league_context.get("avg_home", SPORT_PROFILES[sport_key].base_home_score))
    avg_away = float(league_context.get("avg_away", SPORT_PROFILES[sport_key].base_away_score))
    blend_avg = (avg_home + avg_away) / 2.0
    attack_adj = 1.0 + (bias * attack_span)
    defend_adj = 1.0 - (bias * defend_span)

    return TeamStats(
        rating=base_rating,
        games=8,
        wins=4,
        losses=4,
        draws=0,
        points_for=blend_avg * 8 * _clamp(attack_adj, 0.8, 1.2),
        points_against=blend_avg * 8 * _clamp(defend_adj, 0.85, 1.18),
        form_results=[form, form, form, form, form],
    )


def _safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def _build_win_components(
    sport_key: str,
    *,
    home_stats: TeamStats,
    away_stats: TeamStats,
    league_context: dict[str, float],
    rating_diff: float,
    form_diff: float,
    home_adv_multiplier: float,
) -> tuple[float, dict[str, float]]:
    avg_home = float(league_context["avg_home"])
    avg_away = float(league_context["avg_away"])

    home_attack = _safe_ratio(home_stats.avg_for, avg_home)
    away_attack = _safe_ratio(away_stats.avg_for, avg_away)
    home_defense = _safe_ratio(home_stats.avg_against, avg_away)
    away_defense = _safe_ratio(away_stats.avg_against, avg_home)

    attack_edge = home_attack - away_attack
    defense_edge = away_defense - home_defense
    margin_edge = _safe_ratio(
        (home_stats.avg_for - home_stats.avg_against) - (away_stats.avg_for - away_stats.avg_against),
        (avg_home + avg_away) / 2.0,
        0.0,
    )
    home_term = home_adv_multiplier

    if sport_key == "football":
        components = {
            "rating": rating_diff / 185.0,
            "form": form_diff * 1.10,
            "attack": attack_edge * 0.75,
            "defense": defense_edge * 0.60,
            "home_adv": home_term * 0.22,
            "matchup": margin_edge * 0.30,
        }
    elif sport_key == "american_football":
        components = {
            "rating": rating_diff / 215.0,
            "form": form_diff * 0.75,
            "offense": attack_edge * 0.80,
            "defense": defense_edge * 0.72,
            "home_adv": home_term * 0.18,
            "game_script": margin_edge * 0.55,
        }
    elif sport_key == "basketball":
        pace_edge = _safe_ratio((home_stats.avg_for + away_stats.avg_for), (avg_home + avg_away), 1.0) - 1.0
        components = {
            "rating": rating_diff / 245.0,
            "form": form_diff * 0.70,
            "offense": attack_edge * 0.88,
            "defense": defense_edge * 0.72,
            "home_adv": home_term * 0.14,
            "pace": pace_edge * 0.25,
            "efficiency": margin_edge * 0.60,
        }
    elif sport_key == "volleyball":
        components = {
            "rating": rating_diff / 175.0,
            "form": form_diff * 1.20,
            "set_attack": attack_edge * 0.65,
            "set_defense": defense_edge * 0.60,
            "home_adv": home_term * 0.16,
            "matchup": margin_edge * 0.70,
        }
    elif sport_key == "cricket":
        components = {
            "rating": rating_diff / 205.0,
            "form": form_diff * 0.95,
            "batting": attack_edge * 0.85,
            "bowling": defense_edge * 0.78,
            "home_adv": home_term * 0.12,
            "run_rate": margin_edge * 0.58,
        }
    else:
        components = {
            "rating": rating_diff / 200.0,
            "form": form_diff,
            "attack": attack_edge * 0.70,
            "defense": defense_edge * 0.60,
            "home_adv": home_term * 0.18,
        }

    win_logit = sum(components.values())
    rounded_components = {name: round(value, 4) for name, value in components.items()}
    return win_logit, rounded_components


def _estimate_score(
    sport_key: str,
    home_stats: TeamStats,
    away_stats: TeamStats,
    league_context: dict[str, float],
    rating_diff: float,
    form_diff: float,
) -> tuple[float, float]:
    profile = SPORT_PROFILES[sport_key]
    avg_home = league_context["avg_home"]
    avg_away = league_context["avg_away"]

    home_attack = (home_stats.avg_for / avg_home) if avg_home else 1.0
    away_attack = (away_stats.avg_for / avg_away) if avg_away else 1.0
    home_def_weakness = (home_stats.avg_against / avg_away) if avg_away else 1.0
    away_def_weakness = (away_stats.avg_against / avg_home) if avg_home else 1.0

    if sport_key == "cricket":
        low, high = 0.92, 1.08
    elif sport_key == "basketball":
        low, high = 0.78, 1.24
    else:
        low, high = 0.70, 1.35

    home_attack = _clamp(home_attack if home_attack > 0 else 1.0, low, high)
    away_attack = _clamp(away_attack if away_attack > 0 else 1.0, low, high)
    home_def_weakness = _clamp(home_def_weakness if home_def_weakness > 0 else 1.0, low, high)
    away_def_weakness = _clamp(away_def_weakness if away_def_weakness > 0 else 1.0, low, high)

    home_expected = avg_home * home_attack * away_def_weakness
    away_expected = avg_away * away_attack * home_def_weakness

    if sport_key == "cricket":
        momentum_scale_home = 0.06
        momentum_scale_away = 0.05
    else:
        momentum_scale_home = 0.16
        momentum_scale_away = 0.14

    momentum = (rating_diff / 400.0) + (form_diff * 0.8)
    home_expected *= 1 + (momentum_scale_home * momentum)
    away_expected *= 1 - (momentum_scale_away * momentum)

    home_expected = _clamp(home_expected, profile.score_floor, profile.score_ceiling)
    away_expected = _clamp(away_expected, profile.score_floor, profile.score_ceiling)
    return home_expected, away_expected


def _format_score(sport_key: str, winner: str, home_score: float, away_score: float) -> str:
    if sport_key == "volleyball":
        if winner == "Home":
            if home_score - away_score >= 1.5:
                return "3-0"
            if home_score - away_score >= 0.8:
                return "3-1"
            return "3-2"
        if winner == "Away":
            if away_score - home_score >= 1.5:
                return "0-3"
            if away_score - home_score >= 0.8:
                return "1-3"
            return "2-3"
        return "2-2"

    h = int(round(home_score))
    a = int(round(away_score))
    if winner == "Home" and h <= a:
        h = a + 1
    elif winner == "Away" and a <= h:
        a = h + 1
    elif winner == "Draw":
        mid = int(round((h + a) / 2))
        h = mid
        a = mid
    return f"{h}-{a}"


def predict_match(
    sport_key: str,
    event: dict[str, Any],
    teams: dict[str, TeamStats],
    league_context: dict[str, float],
    *,
    home_adv_multiplier: float = 1.0,
) -> dict[str, Any]:
    profile = SPORT_PROFILES[sport_key]
    supports_draw = SUPPORTED_SPORTS[sport_key].supports_draw

    home_team = event["home_team"]
    away_team = event["away_team"]
    home = _with_team_priors(sport_key, home_team, teams.get(home_team, TeamStats()), league_context)
    away = _with_team_priors(sport_key, away_team, teams.get(away_team, TeamStats()), league_context)

    rating_diff = (home.rating - away.rating) + (profile.home_adv_elo * home_adv_multiplier)
    form_diff = home.form_index - away.form_index

    home_expected, away_expected = _estimate_score(
        sport_key=sport_key,
        home_stats=home,
        away_stats=away,
        league_context=league_context,
        rating_diff=rating_diff,
        form_diff=form_diff,
    )

    win_logit, components = _build_win_components(
        sport_key,
        home_stats=home,
        away_stats=away,
        league_context=league_context,
        rating_diff=rating_diff,
        form_diff=form_diff,
        home_adv_multiplier=home_adv_multiplier,
    )
    win_logit += (home_expected - away_expected) * profile.score_diff_weight
    home_raw = _sigmoid(win_logit)

    draw_probability = 0.0
    if supports_draw:
        base_draw = _clamp(league_context.get("draw_rate", profile.default_draw_rate), 0.03, 0.35)
        draw_probability = base_draw - abs(home_raw - 0.5) * 0.18 - abs(home_expected - away_expected) * 0.03
        draw_probability = _clamp(draw_probability, 0.02, 0.28)

    remaining = 1.0 - draw_probability
    home_probability = home_raw * remaining
    away_probability = (1.0 - home_raw) * remaining

    total = home_probability + away_probability + draw_probability
    home_probability /= total
    away_probability /= total
    draw_probability /= total

    winner_side = "Home"
    winner_name = home_team
    winner_probability = home_probability
    predicted_result = "H"
    if away_probability > winner_probability:
        winner_side = "Away"
        winner_name = away_team
        winner_probability = away_probability
        predicted_result = "A"
    if draw_probability > winner_probability:
        winner_side = "Draw"
        winner_name = "Draw"
        winner_probability = draw_probability
        predicted_result = "D"

    rough_score = _format_score(sport_key, winner_side, home_expected, away_expected)

    return {
        "event_id": event["event_id"],
        "league": event["league"],
        "sport": sport_key,
        "start_time": event["start_time"],
        "home_team": home_team,
        "away_team": away_team,
        "venue": event["venue"],
        "status": event["status"],
        "predicted_result": predicted_result,
        "predicted_winner": winner_name,
        "confidence": round(winner_probability, 4),
        "home_win_probability": round(home_probability, 4),
        "draw_probability": round(draw_probability, 4),
        "away_win_probability": round(away_probability, 4),
        "predicted_score": rough_score,
        "factors": {
            "home_rating": round(home.rating, 1),
            "away_rating": round(away.rating, 1),
            "home_form": round(home.form_index, 3),
            "away_form": round(away.form_index, 3),
            "component_breakdown": components,
        },
    }


def predict_upcoming_matches(
    sport_key: str,
    completed_events: list[dict[str, Any]],
    upcoming_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    teams, context = _build_team_stats(completed_events, sport_key)
    predictions = [predict_match(sport_key, event, teams, context) for event in upcoming_events]
    predictions.sort(key=lambda item: item["start_time"])
    return predictions


def run_what_if_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    sport_key = payload["sport"]
    home_team = payload["home_team"]
    away_team = payload["away_team"]
    neutral_site = bool(payload.get("neutral_site", False))

    baseline_games = 10
    home_avg_scored = float(payload.get("home_avg_scored", 20))
    away_avg_scored = float(payload.get("away_avg_scored", 20))
    home_avg_allowed = float(payload.get("home_avg_allowed", 20))
    away_avg_allowed = float(payload.get("away_avg_allowed", 20))

    home_stats = TeamStats(
        rating=float(payload.get("home_rating", 1500)),
        games=baseline_games,
        wins=5,
        losses=5,
        points_for=home_avg_scored * baseline_games,
        points_against=home_avg_allowed * baseline_games,
        form_results=[float(payload.get("home_form", 0.5))] * 5,
    )
    away_stats = TeamStats(
        rating=float(payload.get("away_rating", 1500)),
        games=baseline_games,
        wins=5,
        losses=5,
        points_for=away_avg_scored * baseline_games,
        points_against=away_avg_allowed * baseline_games,
        form_results=[float(payload.get("away_form", 0.5))] * 5,
    )

    profile = SPORT_PROFILES[sport_key]
    context = {
        "avg_home": profile.base_home_score,
        "avg_away": profile.base_away_score,
        "draw_rate": profile.default_draw_rate,
    }
    event = {
        "event_id": "what-if",
        "league": "What-If Scenario",
        "sport": sport_key,
        "start_time": "",
        "home_team": home_team,
        "away_team": away_team,
        "venue": "Custom Venue",
        "status": "custom",
    }
    return predict_match(
        sport_key=sport_key,
        event=event,
        teams={home_team: home_stats, away_team: away_stats},
        league_context=context,
        home_adv_multiplier=0.0 if neutral_site else 1.0,
    )
