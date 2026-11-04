"""Deterministic synthetic churn model. Educational use only."""
from functools import lru_cache
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    spend = rng.uniform(10, 200, n)
    tickets = rng.integers(0, 9, n)
    X = np.column_stack((tenure, spend, tickets))
    logits = -1.2 - .025 * tenure + .009 * spend + .32 * tickets
    y = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return X, y

def train_and_evaluate():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, {'roc_auc': float(roc_auc_score(y_test, probabilities)), 'average_precision': float(average_precision_score(y_test, probabilities)), 'test_samples': len(y_test), 'data_source': 'synthetic'}

@lru_cache(maxsize=1)
def get_model():
    return train_and_evaluate()[0]

def predict(tenure_months, monthly_spend, support_tickets):
    return float(get_model().predict_proba([[tenure_months, monthly_spend, support_tickets]])[0, 1])
