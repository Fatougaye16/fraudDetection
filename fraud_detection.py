import streamlit as st
import pandas as pd
import joblib, pickle
import numpy
import sys

try:
    # First try normal loading
        model = joblib.load("stacking_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    
st.title("Fraud Detection prediction")

st.markdown("Please enter the transaction details and click predict")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrg = st.number_input("New balance (Sender)", min_value=0.0, value=1000.0)
oldbalanceDest = st.number_input("Old balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([
        {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrg,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }
    ])
    
    prediction = model.predict(input_data)[0]
    
    st.subheader(f"Prediction : '{int(prediction)}'")
    
    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction is not fraud")