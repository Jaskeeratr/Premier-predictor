from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OPENFOOTBALL_FIXTURES_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"
)

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

CATEGORICAL_FEATURES = ["Team", "Opponent", "Venue"]
ROLLING_SOURCE_COLS = ["GF", "GA", "xG", "xGA", "Sh", "SoT", "Dist", "POINTS"]


@dataclass(frozen=True)
class FeatureConfig:
    rolling_window: int = 5
    weighted_span: int = 5
    min_periods: int = 1


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, labels: list[int]) -> float:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_one_hot = np.zeros((len(y_true), len(labels)), dtype=float)
    for row_idx, label in enumerate(y_true):
        y_one_hot[row_idx, label_to_idx[int(label)]] = 1.0
    return float(np.mean(np.sum((y_one_hot - y_prob) ** 2, axis=1)))


def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    conf = np.asarray(y_prob, dtype=float)
    pred = (conf >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if not np.any(mask):
            continue
        bucket_acc = float(np.mean(correct[mask]))
        bucket_conf = float(np.mean(conf[mask]))
        ece += abs(bucket_acc - bucket_conf) * float(np.mean(mask))
    return float(ece)


def _normalize_fixture_team_name(name: str) -> str:
    return FIXTURE_TEAM_NAME_MAP.get(name, name).strip()


def _numeric_features(window: int) -> list[str]:
    rolling = [f"{col}_ROLL{window}" for col in ROLLING_SOURCE_COLS]
    weighted = [f"{col}_EWM{window}" for col in ["GF", "GA", "xG", "xGA", "POINTS", "TARGET"]]
    opp_roll = [f"OPP_{col}" for col in rolling]
    opp_weighted = [f"OPP_{col}" for col in weighted]
    matchup = [
        f"xG_DIFF_ROLL{window}",
        f"GA_DIFF_ROLL{window}",
        f"POINTS_DIFF_ROLL{window}",
        "REST_DIFF",
        "HOUR",
        "DAY_CODE",
        "MONTH",
        "REST_DAYS",
    ]
    return [*rolling, *weighted, *opp_roll, *opp_weighted, *matchup]


def _all_model_features(window: int) -> list[str]:
    return _numeric_features(window) + CATEGORICAL_FEATURES


def load_matches(path: Path, feature_cfg: FeatureConfig) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["HOUR"] = pd.to_numeric(df["Time"].astype(str).str.replace(":.+", "", regex=True), errors="coerce")
    df["DAY_CODE"] = df["Date"].dt.dayofweek
    df["MONTH"] = df["Date"].dt.month
    df["TARGET"] = (df["Result"] == "W").astype(int)
    df["POINTS"] = np.select([df["Result"].eq("W"), df["Result"].eq("D")], [3, 1], default=0)

    numeric_columns = ["GF", "GA", "xG", "xGA", "Sh", "SoT", "Dist", "PK", "PKatt", "Poss"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Team", "Date"]).copy()
    win = feature_cfg.rolling_window
    span = feature_cfg.weighted_span

    for col in ROLLING_SOURCE_COLS:
        df[f"{col}_ROLL{win}"] = df.groupby("Team")[col].transform(
            lambda s: s.shift(1).rolling(win, min_periods=feature_cfg.min_periods).mean()
        )
    for col in ["GF", "GA", "xG", "xGA", "POINTS", "TARGET"]:
        df[f"{col}_EWM{win}"] = df.groupby("Team")[col].transform(
            lambda s: s.shift(1).ewm(span=span, min_periods=feature_cfg.min_periods, adjust=False).mean()
        )

    last_match_date = df.groupby("Team")["Date"].shift(1)
    df["REST_DAYS"] = (df["Date"] - last_match_date).dt.days

    own_cols = [c for c in df.columns if c.endswith(f"_ROLL{win}") or c.endswith(f"_EWM{win}")]
    opp_snapshot = df[["Date", "Team", *own_cols]].rename(
        columns={"Team": "Opponent", **{col: f"OPP_{col}" for col in own_cols}}
    )
    df = df.merge(opp_snapshot, on=["Date", "Opponent"], how="left")

    df[f"xG_DIFF_ROLL{win}"] = df[f"xG_ROLL{win}"] - df[f"OPP_xG_ROLL{win}"]
    df[f"GA_DIFF_ROLL{win}"] = df[f"GA_ROLL{win}"] - df[f"OPP_GA_ROLL{win}"]
    df[f"POINTS_DIFF_ROLL{win}"] = df[f"POINTS_ROLL{win}"] - df[f"OPP_POINTS_ROLL{win}"]
    df["REST_DIFF"] = df["REST_DAYS"] - df.groupby("Team")["REST_DAYS"].transform("median")

    return df.sort_values("Date").reset_index(drop=True)


def build_model_pipeline(*, model_name: str, numeric_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
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
    if model_name == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.06, max_depth=3, random_state=42)
    else:
        model = LogisticRegression(max_iter=3500, class_weight="balanced", C=1.0)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def evaluate_with_time_cv(
    data: pd.DataFrame,
    *,
    features: list[str],
    model_name: str,
    folds: int = 5,
) -> dict[str, float]:
    splitter = TimeSeriesSplit(n_splits=folds)
    metrics: list[dict[str, float]] = []
    x = data[features]
    y = data["TARGET"].astype(int).to_numpy()

    for train_idx, val_idx in splitter.split(x):
        model = build_model_pipeline(
            model_name=model_name, numeric_features=[f for f in features if f not in CATEGORICAL_FEATURES]
        )
        x_train = x.iloc[train_idx]
        y_train = y[train_idx]
        x_val = x.iloc[val_idx]
        y_val = y[val_idx]

        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.55).astype(int)
        val_prob_matrix = np.column_stack([1.0 - val_prob, val_prob])
        metrics.append(
            {
                "accuracy": float(accuracy_score(y_val, val_pred)),
                "precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "recall": float(recall_score(y_val, val_pred, zero_division=0)),
                "log_loss": float(log_loss(y_val, val_prob_matrix, labels=[0, 1])),
                "brier": _multiclass_brier(y_val, val_prob_matrix, [0, 1]),
                "ece": _ece_binary(y_val, val_prob),
            }
        )

    return {name: float(np.mean([m[name] for m in metrics])) for name in metrics[0].keys()}


def select_best_model_by_cv(train_data: pd.DataFrame, features: list[str]) -> tuple[str, dict[str, float]]:
    candidates = ["logistic_regression", "gradient_boosting"]
    scored: list[tuple[str, dict[str, float], float]] = []
    for name in candidates:
        summary = evaluate_with_time_cv(train_data, features=features, model_name=name)
        rank_score = summary["accuracy"] + (0.25 * summary["precision"]) - (0.2 * summary["log_loss"])
        scored.append((name, summary, rank_score))

    scored.sort(key=lambda row: row[2], reverse=True)
    best_name, best_summary, _ = scored[0]
    return best_name, best_summary


def train_and_evaluate(
    matches: pd.DataFrame,
    *,
    split_date: str,
    feature_cfg: FeatureConfig,
) -> tuple[Pipeline, float, pd.DataFrame, dict[str, Any]]:
    cutoff = pd.Timestamp(split_date)
    train = matches[matches["Date"] < cutoff].copy()
    test = matches[matches["Date"] > cutoff].copy()
    features = _all_model_features(feature_cfg.rolling_window)

    best_model_name, cv_summary = select_best_model_by_cv(train, features)
    model = build_model_pipeline(
        model_name=best_model_name,
        numeric_features=[f for f in features if f not in CATEGORICAL_FEATURES],
    )
    model.fit(train[features], train["TARGET"])

    test_probs = model.predict_proba(test[features])[:, 1]
    threshold = 0.55
    test_preds = (test_probs >= threshold).astype(int)
    test_prob_matrix = np.column_stack([1.0 - test_probs, test_probs])

    eval_summary = {
        "model": best_model_name,
        "threshold": threshold,
        "cv_accuracy": round(cv_summary["accuracy"], 4),
        "cv_precision": round(cv_summary["precision"], 4),
        "cv_recall": round(cv_summary["recall"], 4),
        "cv_log_loss": round(cv_summary["log_loss"], 4),
        "cv_brier": round(cv_summary["brier"], 4),
        "cv_ece": round(cv_summary["ece"], 4),
        "test_accuracy": round(float(accuracy_score(test["TARGET"], test_preds)), 4),
        "test_precision": round(float(precision_score(test["TARGET"], test_preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(test["TARGET"], test_preds, zero_division=0)), 4),
        "test_log_loss": round(float(log_loss(test["TARGET"], test_prob_matrix, labels=[0, 1])), 4),
    }

    historical_predictions = test[["Date", "Team", "Opponent", "Result"]].copy()
    historical_predictions["ActualWin"] = (
        historical_predictions["Result"].eq("W").map({True: "Yes", False: "No"})
    )
    historical_predictions["PredWinProb"] = np.round(test_probs, 4)
    historical_predictions["PredWin"] = (test_probs >= threshold).astype(int).map({1: "Yes", 0: "No"})
    historical_predictions["Date"] = historical_predictions["Date"].dt.strftime("%Y-%m-%d")
    historical_predictions["Match"] = (
        historical_predictions["Team"] + " vs " + historical_predictions["Opponent"]
    )
    historical_predictions["PredWinProb"] = historical_predictions["PredWinProb"].map(_format_percent)
    historical_predictions = historical_predictions[
        ["Date", "Match", "Result", "ActualWin", "PredWin", "PredWinProb"]
    ].sort_values("Date", ascending=False)

    return model, threshold, historical_predictions, eval_summary


def fetch_upcoming_fixtures(as_of: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(OPENFOOTBALL_FIXTURES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for match in payload["matches"]:
        fixture_date = pd.Timestamp(match["date"])
        if fixture_date < as_of.normalize():
            continue
        rows.append(
            {
                "Date": fixture_date,
                "Time": str(match.get("time", "15:00")),
                "HomeTeam": _normalize_fixture_team_name(match["team1"]),
                "AwayTeam": _normalize_fixture_team_name(match["team2"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)


def _build_latest_team_snapshot(
    matches: pd.DataFrame, feature_cfg: FeatureConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    win = feature_cfg.rolling_window
    snapshot_cols = [
        "Team",
        "Date",
        f"GF_ROLL{win}",
        f"GA_ROLL{win}",
        f"xG_ROLL{win}",
        f"xGA_ROLL{win}",
        f"Sh_ROLL{win}",
        f"SoT_ROLL{win}",
        f"Dist_ROLL{win}",
        f"POINTS_ROLL{win}",
        f"TARGET_EWM{win}",
        "REST_DAYS",
    ]
    latest = matches.sort_values("Date").groupby("Team").tail(1)[snapshot_cols].set_index("Team")
    medians = matches[_numeric_features(feature_cfg.rolling_window)].median(numeric_only=True).to_dict()
    return latest, medians


def _fixture_feature_row(
    team: str,
    opponent: str,
    venue: str,
    fixture_date: pd.Timestamp,
    fixture_time: str,
    latest_snapshot: pd.DataFrame,
    medians: dict[str, float],
    feature_cfg: FeatureConfig,
) -> dict[str, object]:
    win = feature_cfg.rolling_window
    team_values = latest_snapshot.loc[team] if team in latest_snapshot.index else None
    opp_values = latest_snapshot.loc[opponent] if opponent in latest_snapshot.index else None
    hour = pd.to_numeric(str(fixture_time).split(":")[0], errors="coerce")
    if pd.isna(hour):
        hour = medians.get("HOUR", 15)

    rest_days = medians.get("REST_DAYS", 7)
    if team_values is not None:
        rest_days = (fixture_date - pd.Timestamp(team_values["Date"])).days
    opp_rest = medians.get("REST_DAYS", 7)
    if opp_values is not None:
        opp_rest = (fixture_date - pd.Timestamp(opp_values["Date"])).days

    row: dict[str, object] = {
        "Team": team,
        "Opponent": opponent,
        "Venue": venue,
        "HOUR": hour,
        "DAY_CODE": fixture_date.dayofweek,
        "MONTH": fixture_date.month,
        "REST_DAYS": max(float(rest_days), 2.0),
    }
    for col in ROLLING_SOURCE_COLS:
        key = f"{col}_ROLL{win}"
        row[key] = float(team_values[key]) if team_values is not None else medians.get(key, 1.3)
        row[f"OPP_{key}"] = (
            float(opp_values[key]) if opp_values is not None else medians.get(f"OPP_{key}", 1.3)
        )
    for col in ["GF", "GA", "xG", "xGA", "POINTS", "TARGET"]:
        key = f"{col}_EWM{win}"
        row[key] = float(team_values[key]) if team_values is not None else medians.get(key, 0.4)
        row[f"OPP_{key}"] = (
            float(opp_values[key]) if opp_values is not None else medians.get(f"OPP_{key}", 0.4)
        )

    row[f"xG_DIFF_ROLL{win}"] = float(row[f"xG_ROLL{win}"]) - float(row[f"OPP_xG_ROLL{win}"])
    row[f"GA_DIFF_ROLL{win}"] = float(row[f"GA_ROLL{win}"]) - float(row[f"OPP_GA_ROLL{win}"])
    row[f"POINTS_DIFF_ROLL{win}"] = float(row[f"POINTS_ROLL{win}"]) - float(row[f"OPP_POINTS_ROLL{win}"])
    row["REST_DIFF"] = float(rest_days) - float(opp_rest)
    return row


def predict_upcoming_matches(
    matches: pd.DataFrame,
    model: Pipeline,
    threshold: float,
    *,
    as_of: pd.Timestamp,
    output_file: Path,
    feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    try:
        fixtures = fetch_upcoming_fixtures(as_of)
    except requests.RequestException as exc:
        print(f"Could not fetch upcoming fixtures: {exc}")
        return pd.DataFrame()
    if fixtures.empty:
        print(f"No fixtures found on or after {as_of.date()}")
        return pd.DataFrame()

    latest_snapshot, medians = _build_latest_team_snapshot(matches, feature_cfg)
    draw_rate = float((matches["Result"] == "D").mean())
    features = _all_model_features(feature_cfg.rolling_window)
    rows = []

    for fixture in fixtures.itertuples(index=False):
        home_row = _fixture_feature_row(
            team=fixture.HomeTeam,
            opponent=fixture.AwayTeam,
            venue="Home",
            fixture_date=fixture.Date,
            fixture_time=fixture.Time,
            latest_snapshot=latest_snapshot,
            medians=medians,
            feature_cfg=feature_cfg,
        )
        away_row = _fixture_feature_row(
            team=fixture.AwayTeam,
            opponent=fixture.HomeTeam,
            venue="Away",
            fixture_date=fixture.Date,
            fixture_time=fixture.Time,
            latest_snapshot=latest_snapshot,
            medians=medians,
            feature_cfg=feature_cfg,
        )
        home_prob = float(model.predict_proba(pd.DataFrame([home_row])[features])[:, 1][0])
        away_prob = float(model.predict_proba(pd.DataFrame([away_row])[features])[:, 1][0])

        score_home = home_prob
        score_away = away_prob
        score_draw = draw_rate
        score_total = score_home + score_away + score_draw
        home_win_prob = score_home / score_total
        away_win_prob = score_away / score_total
        draw_prob = score_draw / score_total

        if home_win_prob >= away_win_prob and home_win_prob >= draw_prob:
            pred_result = "H"
            pred_winner = fixture.HomeTeam
            conf = home_win_prob
        elif away_win_prob >= home_win_prob and away_win_prob >= draw_prob:
            pred_result = "A"
            pred_winner = fixture.AwayTeam
            conf = away_win_prob
        else:
            pred_result = "D"
            pred_winner = "Draw"
            conf = draw_prob

        rows.append(
            {
                "Date": fixture.Date.date().isoformat(),
                "Time": fixture.Time,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "PredictedResult": pred_result,
                "PredictedWinner": pred_winner,
                "HomeWinProb": round(home_win_prob, 4),
                "DrawProb": round(draw_prob, 4),
                "AwayWinProb": round(away_win_prob, 4),
                "Confidence": round(conf, 4),
                "ModelDecisionThreshold": threshold,
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_file, index=False)
    print(f"Saved {len(pred_df)} current/upcoming match predictions to {output_file}")
    return pred_df


def save_simple_prediction_report(predictions: pd.DataFrame, html_path: Path, csv_path: Path) -> None:
    if predictions.empty:
        return
    report = predictions.copy()
    report["Game"] = report["HomeTeam"] + " vs " + report["AwayTeam"]
    report["Confidence"] = report["Confidence"].map(_format_percent)
    report = report[["Date", "Time", "Game", "PredictedWinner", "Confidence"]].rename(
        columns={"PredictedWinner": "Predicted Winner"}
    )
    report.to_csv(csv_path, index=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Match Predictions</title>
  <style>
    body {{ font-family: "Segoe UI", Tahoma, Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin: 0 0 6px 0; }}
    p {{ margin: 0 0 16px 0; color: #334155; }}
    table {{ border-collapse: collapse; width: 100%; background: #ffffff; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08); border-radius: 10px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #0f172a; color: #ffffff; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Premier League Predictions</h1>
  <p>Simple view: game, predicted winner, and confidence.</p>
  {report.to_html(index=False, escape=True)}
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EPL win model and predict current/upcoming fixtures.")
    parser.add_argument(
        "--historical-csv", default="matches.csv", help="Historical matches CSV for training."
    )
    parser.add_argument("--split-date", default="2024-03-01", help="Train/test split date (YYYY-MM-DD).")
    parser.add_argument(
        "--as-of-date", default=date.today().isoformat(), help="Fixture prediction start date (YYYY-MM-DD)."
    )
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling feature window size.")
    parser.add_argument("--weighted-span", type=int, default=5, help="EWM span for recent-form weighting.")
    parser.add_argument(
        "--output-csv", default="new_matches.csv", help="Output CSV for current/upcoming predictions."
    )
    parser.add_argument("--report-html", default="prediction_report.html", help="Simple HTML report path.")
    parser.add_argument("--report-csv", default="prediction_table.csv", help="Simple CSV report path.")
    parser.add_argument(
        "--metrics-json", default="reports/premier_metrics.json", help="Evaluation summary JSON path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cfg = FeatureConfig(
        rolling_window=max(2, args.rolling_window), weighted_span=max(2, args.weighted_span)
    )
    historical_path = Path(args.historical_csv)
    output_path = Path(args.output_csv)
    html_report_path = Path(args.report_html)
    csv_report_path = Path(args.report_csv)
    metrics_json_path = Path(args.metrics_json)
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(args.as_of_date)

    matches = load_matches(historical_path, feature_cfg=feature_cfg)
    model, threshold, _, eval_summary = train_and_evaluate(
        matches,
        split_date=args.split_date,
        feature_cfg=feature_cfg,
    )
    print("Evaluation Summary:", json.dumps(eval_summary, indent=2))
    metrics_json_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    new_predictions = predict_upcoming_matches(
        matches,
        model,
        threshold,
        as_of=as_of,
        output_file=output_path,
        feature_cfg=feature_cfg,
    )
    if not new_predictions.empty:
        save_simple_prediction_report(new_predictions, html_path=html_report_path, csv_path=csv_report_path)
        print(f"Saved simple table CSV to {csv_report_path}")
        print(f"Saved simple HTML report to {html_report_path}")
        print(f"Saved evaluation metrics to {metrics_json_path}")


if __name__ == "__main__":
    main()
