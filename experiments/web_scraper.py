from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

SEASON_URLS = {
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "2025-26": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
}


def load_season_csv(season: str, url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df["Season"] = season
    return df


def main() -> None:
    collected: list[pd.DataFrame] = []

    for season, url in SEASON_URLS.items():
        try:
            season_df = load_season_csv(season, url)
            collected.append(season_df)
            print(f"Loaded {season}: {len(season_df)} rows")
        except requests.RequestException as exc:
            print(f"Failed to download {season}: {exc}")

    if not collected:
        print("No data downloaded.")
        return

    combined = pd.concat(collected, ignore_index=True)
    combined.to_csv("premier_league_latest_results.csv", index=False)
    print("Saved latest EPL data to premier_league_latest_results.csv")


if __name__ == "__main__":
    main()
