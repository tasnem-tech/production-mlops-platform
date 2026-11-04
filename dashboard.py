import streamlit as st
from src.model import predict
st.set_page_config(page_title='Churn Risk Demo')
st.title('Customer Churn Risk Demo')
st.warning('Synthetic data only. Not validated for real-world decisions.')
tenure = st.slider('Tenure (months)', 0, 120, 24)
spend = st.number_input('Monthly spend', min_value=0., max_value=10000., value=75.)
tickets = st.slider('Support tickets', 0, 100, 2)
if st.button('Estimate churn probability'):
    st.metric('Estimated probability', f'{predict(tenure, spend, tickets):.1%}')
