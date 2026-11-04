"""Local demonstration API. Load only trusted locally trained model artifacts."""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, ConfigDict, Field
from src.features import FEATURES

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[1] / "artifacts" / "churn_model.joblib")))
model_holder = {"model": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError("Model missing. Run: python -m src.train")
    # joblib/pickle can execute code when loaded. Never load untrusted files.
    model_holder["model"] = load(MODEL_PATH)
    yield
    model_holder["model"] = None

app = FastAPI(title="Synthetic Churn Prediction API", version="0.1.0", lifespan=lifespan)

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=10000, allow_inf_nan=False)
    support_tickets: int = Field(ge=0, le=100)

class Prediction(BaseModel):
    churn_probability: float
    predicted_churn: bool
    threshold: float
    note: str

@app.get("/health")
def health():
    return {"status": "ok" if model_holder["model"] is not None else "model_not_loaded"}

@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    model = model_holder["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    frame = pd.DataFrame([customer.model_dump()], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0, 1])
    return Prediction(churn_probability=round(probability, 6), predicted_churn=probability >= 0.5,
                      threshold=0.5, note="Synthetic-data demonstration; not a real-world churn forecast.")
