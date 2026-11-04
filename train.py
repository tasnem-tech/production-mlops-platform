"""Train a deterministic classifier on explicitly synthetic data."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "churn_model.joblib"
DEFAULT_METRICS = ROOT / "artifacts" / "metrics.json"

def synthetic_data(n=3000, seed=42):
    if n < 100:
        raise ValueError("n must be at least 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    spend = rng.uniform(15, 200, n).round(2)
    tickets = np.clip(rng.poisson(2, n), 0, 15)
    # Simulated probabilities are not observed customer outcomes.
    logit = -0.7 - 0.035 * tenure + 0.009 * (spend - 80) + 0.42 * tickets
    prob = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob)
    X = pd.DataFrame({"tenure_months": tenure, "monthly_spend": spend, "support_tickets": tickets})
    return X, target

def train(model_path=DEFAULT_MODEL, metrics_path=DEFAULT_METRICS, seed=42, n=3000):
    X, y = synthetic_data(n, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(X_train[FEATURES], y_train)
    p = model.predict_proba(X_test[FEATURES])[:, 1]
    metrics = {
        "dataset": "synthetic", "seed": seed, "samples": n,
        "train_samples": len(X_train), "test_samples": len(X_test),
        "positive_rate_test": round(float(np.mean(y_test)), 4),
        "accuracy": round(float(accuracy_score(y_test, p >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
        "average_precision": round(float(average_precision_score(y_test, p)), 4),
        "decision_threshold": 0.5,
        "warning": "Synthetic evaluation only; not real-world model validation."
    }
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(train(seed=args.seed, n=args.samples), indent=2))
