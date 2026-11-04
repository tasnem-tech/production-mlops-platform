from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model import predict
app = FastAPI(title='Synthetic Churn Prediction API')
class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=10000)
    support_tickets: int = Field(ge=0, le=100)
@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'synthetic-demo'}
@app.post('/predict')
def prediction(features: CustomerFeatures):
    return {'churn_probability': predict(features.tenure_months, features.monthly_spend, features.support_tickets), 'model': 'synthetic-demo'}
