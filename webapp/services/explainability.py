from __future__ import annotations

from typing import Any

CONFIDENCE_TIERS = (
    (0.75, "Strong Pick"),
    (0.60, "Lean"),
    (0.52, "Toss-up"),
    (0.00, "Very Uncertain"),
)

COMPONENT_LABELS = {
    "rating": "Rating edge",
    "form": "Recent form",
    "attack": "Attack strength",
    "defense": "Defensive edge",
    "home_adv": "Home advantage",
    "matchup": "Matchup trend",
    "offense": "Offensive form",
    "game_script": "Game script fit",
    "pace": "Pace fit",
    "efficiency": "Efficiency edge",
    "set_attack": "Set attack",
    "set_defense": "Set defense",
    "batting": "Batting strength",
    "bowling": "Bowling edge",
    "run_rate": "Run-rate trend",
}


def confidence_tier(confidence: float) -> str:
    value = max(0.0, min(1.0, float(confidence)))
    for threshold, label in CONFIDENCE_TIERS:
        if value >= threshold:
            return label
    return "Very Uncertain"


def _winner_side(row: dict[str, Any]) -> str:
    result = str(row.get("predicted_result", "")).upper()
    if result == "A":
        return "away"
    if result == "D":
        return "draw"
    return "home"


def _build_factor_contributions(row: dict[str, Any]) -> list[dict[str, Any]]:
    factors = row.get("factors") or {}
    components = (factors.get("component_breakdown") or {}).copy()
    if not isinstance(components, dict):
        return []

    items = []
    for raw_key, raw_value in components.items():
        value = float(raw_value)
        items.append(
            {
                "key": str(raw_key),
                "label": COMPONENT_LABELS.get(str(raw_key), str(raw_key).replace("_", " ").title()),
                "value": round(value, 4),
            }
        )
    if not items:
        return []

    max_abs = max(abs(item["value"]) for item in items) or 1.0
    winner_side = _winner_side(row)
    for item in items:
        positive_for_home = item["value"] >= 0
        if winner_side == "home":
            supports_winner = positive_for_home
        elif winner_side == "away":
            supports_winner = not positive_for_home
        else:
            supports_winner = abs(item["value"]) < 0.03
        item["supports_winner"] = supports_winner
        item["impact_pct"] = round((abs(item["value"]) / max_abs) * 100.0, 1)

    items.sort(key=lambda x: abs(float(x["value"])), reverse=True)
    return items[:6]


def _risk_tags(row: dict[str, Any], tier: str) -> list[str]:
    tags: list[str] = []
    confidence = float(row.get("confidence", 0.0))
    factors = row.get("factors") or {}

    if tier == "Strong Pick":
        tags.append("Strong Pick")
    if confidence < 0.52:
        tags.append("High Risk")

    home_rating = float(factors.get("home_rating", 1500))
    away_rating = float(factors.get("away_rating", 1500))
    form_delta = abs(float(factors.get("home_form", 0.5)) - float(factors.get("away_form", 0.5)))
    winner = str(row.get("predicted_winner", ""))
    home_team = str(row.get("home_team", ""))
    away_team = str(row.get("away_team", ""))

    if winner == away_team and away_rating < home_rating - 45:
        tags.append("Upset Alert")
    if winner == home_team and home_rating < away_rating - 45:
        tags.append("Upset Alert")
    if form_delta >= 0.18:
        tags.append("Momentum Shift")

    # Preserve deterministic order and uniqueness.
    seen: set[str] = set()
    deduped = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _top_factor_notes(row: dict[str, Any], contributions: list[dict[str, Any]]) -> list[str]:
    if not contributions:
        return []
    winner_side = _winner_side(row)
    notes: list[str] = []
    for factor in contributions[:4]:
        supports_winner = bool(factor.get("supports_winner"))
        prefix = "+" if supports_winner else "-"
        label = str(factor["label"])
        if winner_side == "draw":
            note = f"{prefix} {label} is balanced"
        else:
            edge = "supports winner" if supports_winner else "leans against winner"
            note = f"{prefix} {label} {edge}"
        notes.append(note)

    injury = (row.get("factors") or {}).get("injury_adjustment")
    if isinstance(injury, dict):
        signal = float(injury.get("signal_delta", 0.0))
        if abs(signal) >= 0.08:
            if winner_side == "home":
                injury_note = (
                    "+ Injury adjustment favors home side"
                    if signal > 0
                    else "- Injury adjustment hurts home side"
                )
            elif winner_side == "away":
                injury_note = (
                    "+ Injury adjustment favors away side"
                    if signal < 0
                    else "- Injury adjustment hurts away side"
                )
            else:
                injury_note = "+ Injury adjustment increases match volatility"
            notes.append(injury_note)
    return notes[:5]


def _summary_text(row: dict[str, Any], tier: str, tags: list[str], notes: list[str]) -> str:
    winner = row.get("predicted_winner", "Unknown")
    confidence_pct = round(float(row.get("confidence", 0.0)) * 100.0, 1)
    score = row.get("predicted_score", "-")
    signal = notes[0] if notes else "mixed factors"
    tag_text = ", ".join(tags) if tags else "No special tag"
    return (
        f"{winner} is projected to win at {confidence_pct}% ({tier}). "
        f"Projected score {score}. Primary signal: {signal}. Tags: {tag_text}."
    )


def enrich_prediction_explainability(row: dict[str, Any]) -> None:
    confidence = float(row.get("confidence", 0.0))
    tier = confidence_tier(confidence)
    contributions = _build_factor_contributions(row)
    tags = _risk_tags(row, tier)
    notes = _top_factor_notes(row, contributions)

    row["confidence_tier"] = tier
    row["risk_indicators"] = tags
    row["factor_contributions"] = contributions
    row["top_factors"] = notes
    row["explanation"] = _summary_text(row, tier, tags, notes)
