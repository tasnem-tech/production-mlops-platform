from src.model import generate_data, predict, train_and_evaluate

def test_reproducible():
    a, b = generate_data(100)
    c, d = generate_data(100)
    assert (a == c).all() and (b == d).all()

def test_probability():
    assert 0 <= predict(24, 75, 2) <= 1

def test_evaluation():
    _, metrics = train_and_evaluate()
    assert 0 <= metrics['roc_auc'] <= 1
    assert metrics['test_samples'] == 400
