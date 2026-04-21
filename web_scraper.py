import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

years = list(range(2025, 2020, -1))  # This will loop over 2025 and 2024
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features="html.parser")
    standings_table = soup.select('table.stats_table')[0]
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("Stats", "").replace("-", "")
        
        data = requests.get(team_url)
        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")
        
        soup_team = BeautifulSoup(data.text, features="lxml")  # Using lxml explicitly
        team_links = [l.get("href") for l in soup_team.find_all('a')]
        team_links = [l for l in team_links if l and 'all_comps/shooting' in l]
        
        try:
            shooting_data = requests.get(f"https://fbref.com{team_links[0]}")
            shooting = pd.read_html(StringIO(shooting_data.text), match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()
        except (ValueError, IndexError):
            print(f"No 'Shooting' table found for {team_name} in {year}. Skipping...")
            continue  # Skip to the next team
        
        try:
            team_data = matches[0].merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        
        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        
        time.sleep(5)

# Combine all the DataFrames and save to CSV
if all_matches:
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    all_matches_df.to_csv("premier_league_matches.csv", index=False)
    print("Data saved to premier_league_matches.csv")
else:
    print("No match data scraped.")




