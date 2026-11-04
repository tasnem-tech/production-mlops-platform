# Production MLOps Platform (Starter)

**An educational engineering starter, not a production MLOps platform.** Trains a reproducible logistic regression model on *synthetic* customer-churn labels, exposes predictions with FastAPI, and provides a Streamlit UI. Includes validation, tests, Docker and GitHub Actions.

> **Important:** Simulated labels and test metrics do not establish real-world predictive accuracy. Do not use predictions for actual customer decisions.

## Requirements
- Python 3.10+ (Python 3.11 recommended)
- PyCharm or a terminal

## Run on macOS / Linux (PyCharm Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m uvicorn src.api:app --reload
```

API documentation: http://localhost:8000/docs

**In a second PyCharm Terminal** (activate `.venv` there too):

```bash
source .venv/bin/activate
python -m streamlit run src/dashboard.py
```

Dashboard: http://localhost:8501

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m uvicorn src.api:app --reload
```

Open a second PowerShell terminal, activate `.venv`, then run `python -m streamlit run src/dashboard.py`.

## API request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tenure_months":12,"monthly_spend":90,"support_tickets":3}'
```

API validates ranges and rejects extra fields. `GET /health` provides a basic readiness check.

## Architecture

`Synthetic generator → stratified train/test split → StandardScaler + LogisticRegression → local joblib artifact → FastAPI → Streamlit UI`

The scaler is fit only on the training split. Metrics are saved to `artifacts/metrics.json` and include accuracy, ROC-AUC and average precision on a held-out synthetic test set. This is a single split, not robust production evaluation.

## Project files

```text
src/
  __init__.py
  features.py
  train.py
  api.py
  dashboard.py
tests/
  test_train.py
  test_api.py
.github/workflows/tests.yml
Dockerfile
.dockerignore
.gitignore
requirements.txt
README.md
```

## Docker (API only)

```bash
docker build -t synthetic-churn-api .
docker run --rm -p 8000:8000 synthetic-churn-api
```

The Docker image trains a model at build time. The dashboard is not included as a running service in the container.

## Security and deployment notes

- Do not load untrusted `.joblib` model artifacts: deserialization can execute code.
- The API has **no authentication, rate limiting or TLS termination**. Run locally; do not expose directly to the public internet.
- Streamlit calls the API via `API_URL` (default `http://localhost:8000`). Deploying Streamlit alone will not deploy the API.
- Do not upload real customer data, API secrets or model artifacts from untrusted sources.
- CI runs tests only; it does not deploy the application.

## Roadmap

- [ ] Cross-validation, calibration and uncertainty assessment
- [ ] MLflow tracking and model registry
- [ ] Evidently drift and data-quality reporting
- [ ] Artifact provenance, signed builds and versioning
- [ ] Structured logging, observability, authentication and rate limiting
- [ ] Managed cloud deployment with rollback
- [ ] Real-world validation with consented, representative data

## Ethics

All data are simulated. This project demonstrates software engineering patterns rather than evidence of real-world churn prediction.
