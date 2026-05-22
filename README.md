# Multi-Sport Analytics And Prediction Platform

Production-oriented Flask + scikit-learn project for multi-sport match outcome forecasting, scenario simulation, and model tracking.

This repository is designed as an interview-ready engineering project:
- clear backend architecture
- explainable ML pipeline
- offline demo support
- testable services and APIs
- practical DevEx/CI setup

Predictions are analytics outputs for exploration, not betting advice.

## 1) Project Overview

The platform predicts match winners and confidence across:
- football
- basketball
- American football
- volleyball
- cricket

It combines:
- live feed ingestion (with offline fallback)
- sport-specific heuristic modeling
- adaptive retraining with scikit-learn
- injury adjustment controls
- what-if simulations
- SQLite history and model-run persistence
- dashboard visualization

## 2) Key Features

- Multi-sport dashboard with league/team filtering.
- Winner + confidence + rough score predictions.
- Deterministic explainability: confidence tiers, risk indicators, and top-factor narratives.
- Adaptive per-sport model retraining and artifact versioning.
- Cross-validation, log loss, Brier score, and calibration (ECE) tracking.
- Model summary endpoint with top feature signals.
- Model health endpoint with production-style status (`Healthy`, `Needs More Data`, `Undertrained`, `Low Confidence`).
- Injury adjustment workflow (rating/form/offense/defense deltas).
- What-if simulation with optional injury override.
- Prediction history snapshots and charting.
- Health/readiness endpoints.
- Automatic offline demo fallback with deterministic sample events.

## 3) Supported Sports

- Football
- Basketball (NBA/WNBA source coverage in live mode)
- American Football
- Volleyball
- Cricket

## 4) Architecture Overview

```text
run_webapp.py
webapp/
  __init__.py                 app factory
  config.py                   env-driven configuration
  logging_utils.py            structured JSON logging
  errors.py                   typed app exceptions
  extensions.py               storage/model/service initialization
  routes/
    pages.py                  dashboard route
    api.py                    REST endpoints
    health.py                 health/readiness endpoints
  services/
    prediction_service.py     prediction orchestration, cache, serialization
    history_service.py        history use-cases
    injury_service.py         injury use-cases
  data_sources.py             live ingestion + demo fallback generator
  prediction_engine.py        explainable heuristic model
  adaptive_model.py           ML training, evaluation tracking, artifacts
  history_store.py            SQLite history persistence
  injury_store.py             SQLite injury persistence
  static/                     frontend logic/styles
  templates/                  dashboard HTML
predictor.py                  Premier League training/prediction CLI pipeline
tests/                        pytest suite (offline-capable)
```

## 5) ML Pipeline (Applied, Explainable)

### Adaptive multi-sport model (`webapp/adaptive_model.py`)
- Feature set includes rating, form, scoring, matchup, and home-advantage signals.
- Per-sport training configuration controls:
  - minimum sample thresholds
  - retrain cadence
  - candidate model families/hyperparameters
- Candidate models:
  - Logistic Regression
  - Gradient Boosting
  - Random Forest (selected sports)
- Time-series cross-validation (chronological splits).
- Metrics captured per run:
  - CV accuracy
  - CV log loss
  - CV Brier score
  - CV expected calibration error (ECE)
  - holdout accuracy
- Versioned artifacts stored per sport in `webapp/models/<sport>/`.
- Rollback-safe loading: if latest artifact is corrupt/missing, fallback to last valid run.

### Premier League CLI pipeline (`predictor.py`)
- Configurable rolling windows and exponentially weighted recent form.
- Matchup features (xG differential, points differential, rest differential).
- Time-series CV model selection between logistic and boosting baselines.
- Evaluation summary export to JSON.

## 6) Tech Stack

- Backend: Flask
- ML: scikit-learn, numpy, pandas
- Persistence: SQLite
- Frontend: vanilla JS + HTML/CSS
- Tooling: pytest, ruff, black
- DevOps: Docker, GitHub Actions

## 7) Screenshots (Placeholders)

Add screenshots under `docs/screenshots/` and update these links:
- Dashboard home: `docs/screenshots/dashboard-home.png`
- Demo mode banner: `docs/screenshots/demo-mode.png`
- Model summary panel: `docs/screenshots/model-summary.png`
- Model health card: `docs/screenshots/model-health.png`
- What-if simulator: `docs/screenshots/what-if.png`

## 8) Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python run_webapp.py
```

Open: `http://127.0.0.1:5000`

## 9) Docker

```powershell
docker compose up --build
```

Container persists DB/model artifacts via named volume (`predictor_data`).

## 10) Demo Mode

Forced demo mode:

```powershell
$env:PREDICTOR_DEMO_MODE="true"
python run_webapp.py
```

Automatic demo mode:
- If live feeds fail or are sparse, API auto-falls back to deterministic sample data.
- UI shows a demo-mode banner.

## 11) Evaluation Metrics

### Adaptive multi-sport runs
- `cv_accuracy`
- `cv_log_loss`
- `cv_brier`
- `cv_ece`
- `holdout_accuracy`

### Premier League CLI output
- Cross-validation: accuracy/precision/recall/log-loss/Brier/ECE
- Holdout test: accuracy/precision/recall/log-loss
- Saved to `reports/premier_metrics.json` by default.

## 12) API Endpoints

- `GET /api/health`
- `GET /api/ready`
- `GET /api/sports`
- `GET /api/upcoming?sport=football&force=1`
- `GET /api/model-summary?sport=football`
- `GET /api/model-health?sport=football`
- `GET /api/model-health?sport=all`
- `POST /api/what-if`
- `GET /api/history`
- `DELETE /api/history`
- `GET /api/injuries`
- `PUT /api/injuries`
- `DELETE /api/injuries`

## 13) Repository Structure Notes

- `sample_data/`: non-production sample CSVs.
- `reports/`: generated outputs (CSV/HTML/metrics).
- `docs/screenshots/`: recruiter-facing dashboard screenshots.
- `docs/sample_outputs/`: demo CSV/JSON outputs for portfolio walkthroughs.
- `experiments/`: non-core analysis utilities.
- `webapp/models/`: persisted model artifacts.

## 14) Limitations And Disclaimers

- Public sports feeds can be delayed or incomplete.
- Injury adjustments are manual inputs, not auto-scraped injury intelligence.
- Confidence values are probabilistic rankings, not certainty guarantees.
- Model behavior depends on historical data coverage and data quality.

## 15) Future Roadmap

- Provider-layer caching and retry backoff.
- Drift checks and alerting for stale model performance.
- Richer charting and model diagnostics in dashboard.
- Optional authentication and role-based write actions for injury/history admin flows.
- Scheduled retraining jobs and artifact retention policies.
