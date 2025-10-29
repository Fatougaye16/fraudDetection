import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import numpy as np

# Optional SHAP explainability
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

# Page setup
st.set_page_config(page_title="Fraud Detection Real-Time Dashboard", layout="wide")

# Load model
try:
    model = joblib.load("stacking_model_new.pkl")
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Simulate transactions
def generate_transaction():
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
    return {
        "Timestamp": datetime.now(),
        "type": random.choice(types),
        "amount": round(random.uniform(10, 10000), 2),
        "oldbalanceOrg": round(random.uniform(0, 20000), 2),
        "newbalanceOrig": round(random.uniform(0, 20000), 2),
        "oldbalanceDest": round(random.uniform(0, 20000), 2),
        "newbalanceDest": round(random.uniform(0, 20000), 2),
    }

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "fraud_log" not in st.session_state:
    st.session_state.fraud_log = []

# Auto refresh
st_autorefresh(interval=1000, key="fraud_dashboard")

# Generate new transaction each refresh
new_tx = generate_transaction()
input_df = pd.DataFrame([new_tx])

# Get predictions
try:
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
except Exception:
    try:
        pred = model.predict(input_df)[0]
        prob = 1.0 if pred == 1 else 0.0
    except Exception:
        pred = 0
        prob = 0.0

new_tx["Prediction"] = "Fraud" if pred == 1 else "Not Fraud"
new_tx["Fraud Probability"] = round(prob, 3)
st.session_state.transactions.insert(0, new_tx)
st.session_state.transactions = st.session_state.transactions[:50]

# Add to fraud log if flagged
if pred == 1:
    st.session_state.fraud_log.insert(0, new_tx)

# Convert to DataFrame
df = pd.DataFrame(st.session_state.transactions)

# Sidebar Filters
st.sidebar.header("🔍 Filters")
if not df.empty:
    selected_type = st.sidebar.multiselect(
        "Transaction Type", df["type"].unique(), default=df["type"].unique()
    )
    amount_min, amount_max = st.sidebar.slider(
        "Amount Range", float(df["amount"].min() -1), float(df["amount"].max()),
        (float(df["amount"].min()), float(df["amount"].max()))
    )

    filtered_df = df[
        (df["type"].isin(selected_type)) &
        (df["amount"].between(amount_min, amount_max))
    ]
else:
    filtered_df = df

# Metrics row
fraud_count = df["Prediction"].value_counts().get("Fraud", 0)
not_fraud_count = df["Prediction"].value_counts().get("Not Fraud", 0)

col1, col2, col3 = st.columns(3)
col1.metric("🚨 Fraudulent Transactions", fraud_count)
col2.metric("✅ Legitimate Transactions", not_fraud_count)
col3.metric("📊 Total Transactions", len(df))

# Trend chart
if not df.empty:
    chart_data = (
        df.groupby(["Timestamp", "Prediction"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    st.subheader("📈 Transaction Trend Over Time")
    st.line_chart(chart_data, use_container_width=True)

# Highlight function
def highlight_fraud(row):
    color = '#ff4b4b' if row["Prediction"] == "Fraud" else '#4caf50'
    return [
        f'background-color: {color}; color: white' if col == "Prediction" else ''
        for col in row.index
    ]

# Display filtered table
st.subheader("📌 Filtered Transactions")
if not filtered_df.empty:
    st.dataframe(
        filtered_df.style.apply(highlight_fraud, axis=1),
        use_container_width=True,
        height=400
    )
else:
    st.info("No transactions match the current filter.")

# Fraud log table
st.subheader("🚨 Fraud Log (Flagged Transactions)")
if st.session_state.fraud_log:
    fraud_df = pd.DataFrame(st.session_state.fraud_log)
    st.dataframe(fraud_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
else:
    st.info("No fraudulent transactions detected yet.")

# SHAP explainability (optional)


if shap_available and not input_df.empty:
    try:
        st.subheader("🧠 Fraud Explanation for Latest Transaction")

        # ✅ Wrap your model in a callable
        f = lambda x: model.predict_proba(x)
        background = input_df.copy() # background data for SHAP

        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(input_df)

        st.write("**Latest transaction:**")
        st.json(new_tx)

        # SHAP plot for class 1 (fraud)
        shap.initjs()
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][0,:],
            input_df.iloc[0,:],
            matplotlib=True
        )
        st.pyplot(bbox_inches="tight")

    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")
else:
    st.caption("ℹ️ Install `shap` for feature attribution (e.g., `pip install shap`).")


# Download fraud log
if st.session_state.fraud_log:
    fraud_csv = pd.DataFrame(st.session_state.fraud_log).to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download Fraud Log",
        data=fraud_csv,
        file_name="fraud_log.csv",
        mime="text/csv",
    )

# Optional alert sound or visual cue
if pred == 1:
    st.warning("🚨 Fraudulent transaction detected!")
 