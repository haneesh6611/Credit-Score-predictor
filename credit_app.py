import streamlit as st
import pickle
import numpy as np

# Load model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔍 Credit Score Predictor")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income (₹)", value=50000)
loan = st.number_input("Loan Amount", value=10000)
history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Good" if x == 1 else "Bad")
debt = st.number_input("Outstanding Debt", value=2000)
cards = st.number_input("Number of Credit Cards", min_value=0, value=2)

if st.button("Predict"):
    input_data = np.array([[age, income, loan, history, debt, cards]])
    pred = model.predict(input_data)[0]
    conf = model.predict_proba(input_data).max() * 100

    if pred == 1:
        st.success(f"✅ Good Credit Score (Confidence: {conf:.2f}%)")
    else:
        st.error(f"⚠️ Bad Credit Score (Confidence: {conf:.2f}%)")
