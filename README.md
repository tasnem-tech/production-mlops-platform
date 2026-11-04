# Production MLOps Platform (Starter)

An **engineering starter** for a synthetic customer-churn classification demo. Includes a reproducible model training function, FastAPI prediction service, Streamlit dashboard, input validation, tests, GitHub Actions and Docker.

> **Scope:** This is not yet a production-ready MLOps system. MLflow tracking, drift monitoring, robust evaluation, model registry, authentication, observability and production deployment remain future work. Synthetic labels are generated from a simulated process and are not evidence of real-world accuracy.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn src.api:app --reload
```

Visit http://localhost:8000/docs for API documentation.

In a separate terminal:
```bash
streamlit run src/dashboard.py
```

## API example

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"tenure_months":12,"monthly_spend":90,"support_tickets":3}'
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Architecture

Synthetic customer features → training pipeline → local serialized model → FastAPI / Streamlit

## Roadmap

- [ ] Train/validation/test split, cross-validation, class-imbalance analysis, calibration and PR-AUC
- [ ] MLflow experiment tracking and model registry
- [ ] Evidently data-quality and drift reports
- [ ] Persisted model versioning and safe artifact provenance
- [ ] Structured logs, metrics, authentication and rate limiting
- [ ] Cloud deployment with documented CI/CD and rollback

## Ethics

No personal data is used. This project is for learning and portfolio demonstration only.
