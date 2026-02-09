# EnergyCast — Household Energy Forecasting (End-to-End MLE Demo)

A production-style forecasting system for **household electricity consumption / generation / net load** using Latvian smart-meter data(sadalestikls.lv/en/innovation) with external signals (weather(open-meteo.com), calendar).  
Focus: Modular and scalable design with reproducible pipeline, clean evaluation, experiment tracking, automatic testing and following good engineering practices

---

## What’s implemented (current state)

### Data & pipeline
- Layered data layout: `data/original → data/raw → data/processed`
- Processed dataset split into train/dev/test (time-based) for 24h horizon

### Models & evaluation
- Baselines:
  - Seasonal Naïve (per-customer variant)
- ML model:
  - LightGBM per-customer training/testing pipeline
- Metrics:` MAE, RMSE, sMAPE`
- Runs tracked via MLflow (`mlflow.db` + `mlruns`)

### Engineering
- Modular `src/energycast` python package with mirrored `tests/energycast` with coverage about 50%
- GitHub Actions CI for unit tests
- YAML configs for splits/models (`configs/`)

Results are tracked in MLflow and saved under `mlruns/` with `artifacts/` per model run.

---

## Repository map (high-level)

- `src/energycast/` — library code (ingestion, features, normalization, models, evaluation)
- `scripts/` — runnable entrypoints (data splitting and preprocessing, baseline, LGBM training)
- `configs/` — reproducible configs (splits + model params)
- `data/` — original/raw/processed layers
- `artifacts/` — model run artifacts (baseline + model)
- `mlruns/`, `mlflow.db` — MLflow tracking store 
- `infra/postgres/`, `docker-compose.yml` — DB infra for ingestion experiments

---

## Configuration

- Splits: `configs/splits/default_24.yaml`
- Models:
  - `configs/models/naive_24.yaml`
  - `configs/models/lgbm_per_customer_24.yaml`

---

## Project next steps 

- Complete an end-to-end pipeline: one reproducible command that process raw data -> ingest validated data to the DB -> runs data processing → splits → baselines → model training → evaluation → MLflow logging → saves artifacts -> updates monitoring.

 - Experiment with global deep learning models: train and compare global forecasters (single model across all households) such as N-HiTS.

- Experiment with pretrained / foundation time-series models: evaluate on the current dataset pretrained model like  Chronos2 and TimesFM-style

- Move from offline backtesting to an online-like replay system: implement scheduled batch prediction that simulates real-time data ingestion and model retraining.