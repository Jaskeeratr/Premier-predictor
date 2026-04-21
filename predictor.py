from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_FEATURES = ["VENUE_CODE", "OPP_CODE", "HOUR", "DAY_CODE"]
TEAM_ROLLING_FEATURES = [
    "GF_ROLL5",
    "GA_ROLL5",
    "xG_ROLL5",
    "xGA_ROLL5",
    "Sh_ROLL5",
    "SoT_ROLL5",
    "Dist_ROLL5",
    "POINTS_ROLL5",
    "FORM5_WIN_RATE",
]
NUMERIC_FEATURES = [
    "HOUR",
    "DAY_CODE",
    "MONTH",
    *TEAM_ROLLING_FEATURES,
    "REST_DAYS",
    "OPP_GF_ROLL5",
    "OPP_GA_ROLL5",
    "OPP_xG_ROLL5",
    "OPP_xGA_ROLL5",
    "OPP_POINTS_ROLL5",
    "OPP_FORM5_WIN_RATE",
]
CATEGORICAL_FEATURES = ["Team", "Opponent", "Venue"]
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OPENFOOTBALL_FIXTURES_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"

FIXTURE_TEAM_NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton and Hove Albion",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds United",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Nottingham Forest FC": "Nottingham Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
}


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["HOUR"] = pd.to_numeric(df["Time"].astype(str).str.replace(":.+", "", regex=True), errors="coerce")
    df["DAY_CODE"] = df["Date"].dt.dayofweek
    df["MONTH"] = df["Date"].dt.month
    df["VENUE_CODE"] = df["Venue"].astype("category").cat.codes
    df["OPP_CODE"] = df["Opponent"].astype("category").cat.codes
    df["TARGET"] = (df["Result"] == "W").astype(int)
    df["POINTS"] = np.select([df["Result"].eq("W"), df["Result"].eq("D")], [3, 1], default=0)

    numeric_columns = ["GF", "GA", "xG", "xGA", "Sh", "SoT", "Dist", "PK", "PKatt", "Poss"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Team", "Date"]).copy()
    rolling_source_cols = ["GF", "GA", "xG", "xGA", "Sh", "SoT", "Dist", "POINTS"]
    for col in rolling_source_cols:
        df[f"{col}_ROLL5"] = df.groupby("Team")[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    df["FORM5_WIN_RATE"] = df.groupby("Team")["TARGET"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    last_match_date = df.groupby("Team")["Date"].shift(1)
    df["REST_DAYS"] = (df["Date"] - last_match_date).dt.days

    opp_cols = ["GF_ROLL5", "GA_ROLL5", "xG_ROLL5", "xGA_ROLL5", "POINTS_ROLL5", "FORM5_WIN_RATE"]
    opp_snapshot = df[["Date", "Team", *opp_cols]].rename(
        columns={"Team": "Opponent", **{col: f"OPP_{col}" for col in opp_cols}}
    )
    df = df.merge(opp_snapshot, on=["Date", "Opponent"], how="left")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_improved_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_and_evaluate(matches: pd.DataFrame, split_date: str) -> tuple[Pipeline, float]:
    cutoff = pd.Timestamp(split_date)
    train = matches[matches["Date"] < cutoff].copy()
    test = matches[matches["Date"] > cutoff].copy()

    baseline = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1)
    baseline.fit(train[BASELINE_FEATURES], train["TARGET"])
    baseline_preds = baseline.predict(test[BASELINE_FEATURES])
    baseline_acc = accuracy_score(test["TARGET"], baseline_preds)
    baseline_prec = precision_score(test["TARGET"], baseline_preds, zero_division=0)

    improved = build_improved_model()
    improved.fit(train[ALL_MODEL_FEATURES], train["TARGET"])
    test_probs = improved.predict_proba(test[ALL_MODEL_FEATURES])[:, 1]

    # Slightly higher threshold improves precision while still increasing accuracy.
    decision_threshold = 0.55
    improved_preds = (test_probs >= decision_threshold).astype(int)
    improved_acc = accuracy_score(test["TARGET"], improved_preds)
    improved_prec = precision_score(test["TARGET"], improved_preds, zero_division=0)

    print("Baseline Accuracy (GradientBoosting):", round(baseline_acc, 4))
    print("Baseline Precision (GradientBoosting):", round(baseline_prec, 4))
    print("Improved Accuracy (Logistic + Rolling Form):", round(improved_acc, 4))
    print("Improved Precision (Logistic + Rolling Form):", round(improved_prec, 4))
    print("Chosen Decision Threshold:", decision_threshold)

    return improved, decision_threshold


def _normalize_fixture_team_name(name: str) -> str:
    normalized = FIXTURE_TEAM_NAME_MAP.get(name, name)
    return normalized.strip()


def fetch_upcoming_fixtures(as_of: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(OPENFOOTBALL_FIXTURES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    fixture_rows = []
    for match in payload["matches"]:
        fixture_date = pd.Timestamp(match["date"])
        if fixture_date < as_of.normalize():
            continue

        fixture_rows.append(
            {
                "Date": fixture_date,
                "Time": str(match.get("time", "15:00")),
                "HomeTeam": _normalize_fixture_team_name(match["team1"]),
                "AwayTeam": _normalize_fixture_team_name(match["team2"]),
            }
        )

    fixtures = pd.DataFrame(fixture_rows).sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)
    return fixtures


def _build_latest_team_snapshot(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    snapshot_columns = [
        "Team",
        "Date",
        "GF_ROLL5",
        "GA_ROLL5",
        "xG_ROLL5",
        "xGA_ROLL5",
        "Sh_ROLL5",
        "SoT_ROLL5",
        "Dist_ROLL5",
        "POINTS_ROLL5",
        "FORM5_WIN_RATE",
        "REST_DAYS",
    ]
    latest = matches.sort_values("Date").groupby("Team").tail(1)[snapshot_columns].set_index("Team")
    numeric_medians = matches[NUMERIC_FEATURES].median(numeric_only=True).to_dict()
    return latest, numeric_medians


def _fixture_feature_row(
    team: str,
    opponent: str,
    venue: str,
    fixture_date: pd.Timestamp,
    fixture_time: str,
    latest_snapshot: pd.DataFrame,
    medians: dict[str, float],
) -> dict[str, object]:
    team_values = latest_snapshot.loc[team] if team in latest_snapshot.index else None
    opp_values = latest_snapshot.loc[opponent] if opponent in latest_snapshot.index else None

    hour = pd.to_numeric(str(fixture_time).split(":")[0], errors="coerce")
    if pd.isna(hour):
        hour = medians.get("HOUR", 15)

    if team_values is not None:
        rest_days = (fixture_date - pd.Timestamp(team_values["Date"])).days
    else:
        rest_days = medians.get("REST_DAYS", 7)

    row = {
        "Team": team,
        "Opponent": opponent,
        "Venue": venue,
        "HOUR": hour,
        "DAY_CODE": fixture_date.dayofweek,
        "MONTH": fixture_date.month,
        "GF_ROLL5": float(team_values["GF_ROLL5"]) if team_values is not None else medians.get("GF_ROLL5", 1.3),
        "GA_ROLL5": float(team_values["GA_ROLL5"]) if team_values is not None else medians.get("GA_ROLL5", 1.3),
        "xG_ROLL5": float(team_values["xG_ROLL5"]) if team_values is not None else medians.get("xG_ROLL5", 1.3),
        "xGA_ROLL5": float(team_values["xGA_ROLL5"]) if team_values is not None else medians.get("xGA_ROLL5", 1.3),
        "Sh_ROLL5": float(team_values["Sh_ROLL5"]) if team_values is not None else medians.get("Sh_ROLL5", 11.0),
        "SoT_ROLL5": float(team_values["SoT_ROLL5"]) if team_values is not None else medians.get("SoT_ROLL5", 4.0),
        "Dist_ROLL5": float(team_values["Dist_ROLL5"]) if team_values is not None else medians.get("Dist_ROLL5", 17.0),
        "POINTS_ROLL5": float(team_values["POINTS_ROLL5"]) if team_values is not None else medians.get("POINTS_ROLL5", 1.3),
        "FORM5_WIN_RATE": float(team_values["FORM5_WIN_RATE"]) if team_values is not None else medians.get("FORM5_WIN_RATE", 0.35),
        "REST_DAYS": max(rest_days, 2),
        "OPP_GF_ROLL5": float(opp_values["GF_ROLL5"]) if opp_values is not None else medians.get("OPP_GF_ROLL5", 1.3),
        "OPP_GA_ROLL5": float(opp_values["GA_ROLL5"]) if opp_values is not None else medians.get("OPP_GA_ROLL5", 1.3),
        "OPP_xG_ROLL5": float(opp_values["xG_ROLL5"]) if opp_values is not None else medians.get("OPP_xG_ROLL5", 1.3),
        "OPP_xGA_ROLL5": float(opp_values["xGA_ROLL5"]) if opp_values is not None else medians.get("OPP_xGA_ROLL5", 1.3),
        "OPP_POINTS_ROLL5": float(opp_values["POINTS_ROLL5"]) if opp_values is not None else medians.get("OPP_POINTS_ROLL5", 1.3),
        "OPP_FORM5_WIN_RATE": float(opp_values["FORM5_WIN_RATE"]) if opp_values is not None else medians.get("OPP_FORM5_WIN_RATE", 0.35),
    }
    return row


def predict_upcoming_matches(
    matches: pd.DataFrame,
    model: Pipeline,
    threshold: float,
    as_of: pd.Timestamp,
    output_file: Path,
) -> None:
    try:
        fixtures = fetch_upcoming_fixtures(as_of)
    except requests.RequestException as exc:
        print(f"Could not fetch upcoming fixtures: {exc}")
        return

    if fixtures.empty:
        print("No fixtures found on or after", as_of.date())
        return

    latest_snapshot, medians = _build_latest_team_snapshot(matches)
    draw_rate = float((matches["Result"] == "D").mean())

    prediction_rows = []
    for fixture in fixtures.itertuples(index=False):
        home_row = _fixture_feature_row(
            team=fixture.HomeTeam,
            opponent=fixture.AwayTeam,
            venue="Home",
            fixture_date=fixture.Date,
            fixture_time=fixture.Time,
            latest_snapshot=latest_snapshot,
            medians=medians,
        )
        away_row = _fixture_feature_row(
            team=fixture.AwayTeam,
            opponent=fixture.HomeTeam,
            venue="Away",
            fixture_date=fixture.Date,
            fixture_time=fixture.Time,
            latest_snapshot=latest_snapshot,
            medians=medians,
        )

        home_prob = float(model.predict_proba(pd.DataFrame([home_row])[ALL_MODEL_FEATURES])[:, 1][0])
        away_prob = float(model.predict_proba(pd.DataFrame([away_row])[ALL_MODEL_FEATURES])[:, 1][0])

        score_home = home_prob
        score_away = away_prob
        score_draw = draw_rate
        score_total = score_home + score_away + score_draw
        home_win_prob = score_home / score_total
        away_win_prob = score_away / score_total
        draw_prob = score_draw / score_total

        if home_win_prob >= away_win_prob and home_win_prob >= draw_prob:
            predicted_result = "H"
            predicted_winner = fixture.HomeTeam
            confidence = home_win_prob
        elif away_win_prob >= home_win_prob and away_win_prob >= draw_prob:
            predicted_result = "A"
            predicted_winner = fixture.AwayTeam
            confidence = away_win_prob
        else:
            predicted_result = "D"
            predicted_winner = "Draw"
            confidence = draw_prob

        prediction_rows.append(
            {
                "Date": fixture.Date.date().isoformat(),
                "Time": fixture.Time,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "PredictedResult": predicted_result,
                "PredictedWinner": predicted_winner,
                "HomeWinProb": round(home_win_prob, 4),
                "DrawProb": round(draw_prob, 4),
                "AwayWinProb": round(away_win_prob, 4),
                "Confidence": round(confidence, 4),
                "ModelDecisionThreshold": threshold,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_file, index=False)
    print(f"Saved {len(predictions)} current/upcoming match predictions to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EPL win model and predict current/upcoming fixtures.")
    parser.add_argument("--historical-csv", default="matches.csv", help="Historical matches CSV for training.")
    parser.add_argument("--split-date", default="2024-03-01", help="Train/test split date in YYYY-MM-DD format.")
    parser.add_argument("--as-of-date", default=date.today().isoformat(), help="Fixture prediction start date in YYYY-MM-DD format.")
    parser.add_argument("--output-csv", default="new_matches.csv", help="Output CSV path for current/upcoming predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_path = Path(args.historical_csv)
    output_path = Path(args.output_csv)
    as_of = pd.Timestamp(args.as_of_date)

    matches = load_matches(historical_path)
    model, threshold = train_and_evaluate(matches, split_date=args.split_date)
    predict_upcoming_matches(matches, model, threshold, as_of=as_of, output_file=output_path)


if __name__ == "__main__":
    main()
