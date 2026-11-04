import numpy as np
from src.train import synthetic_data, train

def test_reproducible_data():
    a, y1 = synthetic_data(200, 42)
    b, y2 = synthetic_data(200, 42)
    assert a.equals(b)
    assert np.array_equal(y1, y2)

def test_training_writes_artifacts(tmp_path):
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    result = train(model, metrics, n=300)
    assert model.is_file() and metrics.is_file()
    assert 0 <= result["roc_auc"] <= 1
