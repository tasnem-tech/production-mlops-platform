import pytest
from fastapi.testclient import TestClient
from src import api
from src.train import train

@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    train(model_path=path, metrics_path=tmp_path / "metrics.json", n=300)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    with TestClient(api.app) as client:
        yield client

def test_health(client):
    assert client.get("/health").json()["status"] == "ok"

def test_predict(client):
    r = client.post("/predict", json={"tenure_months": 12, "monthly_spend": 90, "support_tickets": 3})
    assert r.status_code == 200
    assert 0 <= r.json()["churn_probability"] <= 1

def test_validation(client):
    r = client.post("/predict", json={"tenure_months": -1, "monthly_spend": 90, "support_tickets": 3})
    assert r.status_code == 422
    r = client.post("/predict", json={"tenure_months": 12, "monthly_spend": 90, "support_tickets": 3, "secret": "x"})
    assert r.status_code == 422
