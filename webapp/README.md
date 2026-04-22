# Multi-Sport Web Predictor

## Run
1. Open PowerShell in the project root.
2. Activate the environment:
   - `.\.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `python -m pip install -r requirements.txt`
4. Start the web app:
   - `python run_webapp.py`
5. Open:
   - `http://127.0.0.1:5000`

## Features
- Sport selector for:
  - Football
  - American Football
  - Volleyball
  - Basketball (includes NBA and WNBA feeds when available)
  - Cricket
- Upcoming match feed with predicted winner, confidence, and rough score.
- League + team filter controls for upcoming matches.
- Auto-refresh interval selector (Off, 1, 3, 5, 10 min).
- SQLite prediction history snapshots with filtering and clear controls.
- What-if simulator where you can input your own ratings, form, and scoring assumptions.
- `Use` button now auto-fills a what-if scenario from that match and runs it immediately.
- Each sport uses a sport-specific prediction component model (different weighting logic by sport).
- Adaptive ML training per sport:
  - Stores completed matches in SQLite.
  - Auto-retrains when new results arrive (or model gets stale).
  - Tunes candidate model parameters and keeps the best by holdout accuracy.
  - Reuses trained model for upcoming winner probabilities.

## Notes
- Data providers:
  - ESPN public scoreboards (primary for most sports)
  - TheSportsDB calendar/results fallback (cricket)
- Predictions are heuristic and designed for exploration, not betting guarantees.

## History APIs
- `GET /api/history?sport=<key>&team=<term>&league=<term>&limit=<n>`
- `DELETE /api/history` with JSON body:
  - `{"sport": "football"}` to clear one sport
  - `{"sport": null}` (or omit) to clear all

## Adaptive Model Data
- Stored in SQLite tables in `webapp/prediction_history.db`:
  - `training_events`
  - `model_runs`
- Trained model artifacts are written to `webapp/models/`.
