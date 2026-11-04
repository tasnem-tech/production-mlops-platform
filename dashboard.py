"""Streamlit UI calling the separate FastAPI service."""
import os
import requests
import streamlit as st

st.set_page_config(page_title="Churn MLOps Starter", page_icon="📊", layout="centered")
st.title("📊 Synthetic Customer Churn Demo")
st.warning("Educational simulation only. Predictions are not validated on real customers.")
api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.caption(f"Prediction API: {api_url}")
with st.form("prediction"):
    tenure = st.slider("Tenure (months)", 0, 120, 12)
    spend = st.number_input("Monthly spend", min_value=0.0, max_value=10000.0, value=90.0, step=5.0)
    tickets = st.number_input("Support tickets", min_value=0, max_value=100, value=3, step=1)
    submitted = st.form_submit_button("Predict simulated churn", type="primary")
if submitted:
    try:
        response = requests.post(f"{api_url}/predict", json={"tenure_months": tenure, "monthly_spend": spend,
            "support_tickets": tickets}, timeout=8)
        response.raise_for_status()
        result = response.json()
        st.metric("Simulated churn probability", f"{result['churn_probability']:.1%}")
        st.progress(min(max(float(result["churn_probability"]), 0), 1))
        st.write("Classification:", "Churn" if result["predicted_churn"] else "No churn")
        st.caption(result["note"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.error(f"API request failed. Check that FastAPI is running. Details: {exc}")
