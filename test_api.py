from fastapi.testclient import TestClient
from src.api import app
client = TestClient(app)

def test_health():
    assert client.get('/health').json()['status'] == 'ok'

def test_predict():
    r = client.post('/predict', json={'tenure_months': 12, 'monthly_spend': 90, 'support_tickets': 3})
    assert r.status_code == 200
    assert 0 <= r.json()['churn_probability'] <= 1

def test_validation():
    r = client.post('/predict', json={'tenure_months': -1, 'monthly_spend': 90, 'support_tickets': 3})
    assert r.status_code == 422
