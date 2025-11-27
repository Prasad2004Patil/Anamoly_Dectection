# Cloud-Based Anomaly Detection — Full Implementation

This repository implements an end-to-end anomaly detection project (ML models, REST API, Streamlit dashboard).

## Contents
- `data_simulation.py` — generate synthetic e-commerce data
- `model_training.py` — train Isolation Forest, One-Class SVM, Autoencoder (saves artifacts/)
- `api/app.py` — Flask API to serve predictions and retraining
- `dashboard/dashboard.py` — Streamlit dashboard connecting to API
- `utils.py` — helper functions
- `artifacts/` — models and scaler saved here after training
- `Dockerfile.api`, `Dockerfile.streamlit`, `docker-compose.yml` — container configuration
- `run_all.sh` — convenient runner

## Quick local setup
1. Create virtual env and install:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

