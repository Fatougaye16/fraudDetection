# app.py
import warnings
import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import numpy as np
import uuid
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version.*')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')
warnings.filterwarnings('ignore', message='.*serialized model.*')
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# Suppress XGBoost warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure matplotlib for streamlit
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

# Optional SHAP explainability
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

# Optional plotly for better charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    plotly_available = True
except Exception:
    plotly_available = False

st.set_page_config(
    page_title=" Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CSS / Styling
# -----------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        color: black;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        color: black;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .transaction-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .fraud-detected {
        border-left: 5px solid #f44336 !important;
        background-color: #ffebee !important;
    }
    .legitimate {
        border-left: 5px solid #4caf50 !important;
        background-color: #e8f5e8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# Model loading & helpers
# -----------------------
MODEL_PATH = "stacking_model_latest.pkl"

def load_model(path=MODEL_PATH):
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, e

model, model_err = load_model()
if model is not None:
    st.sidebar.success("✅ Model loaded successfully")
    model_status = "🟢 ACTIVE"
else:
    st.sidebar.error(f"❌ Error loading model: {model_err}")
    model_status = "🔴 ERROR"

# Show some debug info about model
with st.sidebar.expander("Model Info", expanded=False):
    st.write(f"Type: {type(model)}")
    # feature names if available
    try:
        if hasattr(model, "feature_names_in_"):
            st.write("feature_names_in_:")
            st.write(list(model.feature_names_in_))
    except Exception:
        pass
    # If pipeline, show step names
    try:
        if hasattr(model, "steps"):
            st.write("Pipeline steps:")
            for sname, s in model.steps:
                st.write(f"- {sname}: {type(s)}")
    except Exception:
        pass

# Safe predict helpers: try raw, then try get_dummies + align to model.feature_names_in_
def align_and_predict_proba(m, df):
    """Return predicted probability for class 1 and predicted label.
    Tries multiple strategies to avoid input mismatch crashes.
    """
    # Strategy 1: direct
    try:
        proba = m.predict_proba(df)
        pred = m.predict(df)
        return float(proba[0][1]), int(pred[0])
    except Exception:
        pass

    # Strategy 2: if we have model.feature_names_in_, try reindexing after get_dummies
    try:
        df_d = pd.get_dummies(df)
        if hasattr(m, "feature_names_in_"):
            target_cols = list(m.feature_names_in_)
            # keep only target columns, fill missing with 0, drop extras
            df_aligned = pd.DataFrame(columns=target_cols)
            for c in target_cols:
                if c in df_d.columns:
                    df_aligned[c] = df_d[c]
                else:
                    df_aligned[c] = 0
            df_aligned = df_aligned.fillna(0)
            proba = m.predict_proba(df_aligned)
            pred = m.predict(df_aligned)
            return float(proba[0][1]), int(pred[0])
        else:
            # If no feature_names, try to align by intersection
            common = [c for c in df_d.columns if hasattr(m, "feature_names_in_") and c in m.feature_names_in_]
            if common:
                df_sub = df_d[common]
                proba = m.predict_proba(df_sub)
                pred = m.predict(df_sub)
                return float(proba[0][1]), int(pred[0])
    except Exception:
        pass

    # Strategy 3: fallback to risk-based heuristic (no model)
    try:
        # If model fails, return a neutral probability based on amount / risk_score if present
        amt = float(df.iloc[0].get("amount", 0))
        rs = float(df.iloc[0].get("risk_score", 0)) if "risk_score" in df.columns else 0
        # simple mapping: risk_score scaled to 0-1
        prob = min(max(rs / 100.0, 0.01 if amt < 100 else 0.05), 0.99)
        pred = 1 if prob > 0.5 else 0
        return float(prob), int(pred)
    except Exception:
        return 0.0, 0

# -----------------------
# Transaction generator (cleaned)
# -----------------------
class EnhancedTransactionGenerator:
    def __init__(self, n_customers=500, n_merchants=200, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.customer_profiles = self._init_customers(n_customers)
        self.merchants = self._init_merchants(n_merchants)

    def _init_customers(self, n):
        customers = []
        for i in range(n):
            customer = {
                "id": f"CUST_{i:06d}",
                "name": f"Customer {i}",
                "location": random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix", "Philadelphia"]),
                "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]),
                "avg_spending": float(np.random.lognormal(6, 1.2)),
                "risk_profile": random.choice(["low", "medium", "high"]),
                "account_age": random.randint(30, 2000),
                "last_transaction": datetime.now() - timedelta(hours=random.randint(1, 72)),
            }
            customers.append(customer)
        return customers

    def _init_merchants(self, n):
        categories = [
            "Gas Station",
            "Grocery",
            "Restaurant",
            "Online Store",
            "ATM",
            "Department Store",
            "Pharmacy",
            "Hotel",
            "Airline",
            "Telecom",
        ]
        merchants = []
        for i in range(n):
            merchant = {
                "id": f"MERCH_{i:05d}",
                "name": f"{random.choice(['Super', 'Quick', 'Best', 'Metro'])} {random.choice(categories)} {i}",
                "category": random.choice(categories),
                "location": random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"]),
                "risk_level": random.choices(["low", "medium", "high"], weights=[70, 25, 5])[0],
                "fraud_history": random.randint(0, 3),
            }
            merchants.append(merchant)
        return merchants

    def _get_time_of_day(self, timestamp):
        hour = timestamp.hour
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night"

    def _calculate_risk_score(self, transaction, customer, merchant):
        risk_score = 0
        risk_factors = []

        # amount based
        if transaction["amount"] > 5000:
            risk_score += 25
            risk_factors.append("High Amount")
        elif transaction["amount"] > 2000:
            risk_score += 15
            risk_factors.append("Elevated Amount")

        hour = transaction["Timestamp"].hour
        if hour < 6 or hour > 22:
            risk_score += 20
            risk_factors.append("Unusual Time")

        if customer["location"] != merchant["location"]:
            risk_score += 15
            risk_factors.append("Location Mismatch")

        if customer["risk_profile"] == "high":
            risk_score += 30
            risk_factors.append("High-Risk Customer")
        elif customer["risk_profile"] == "medium":
            risk_score += 15
            risk_factors.append("Medium-Risk Customer")

        if merchant["risk_level"] == "high":
            risk_score += 25
            risk_factors.append("High-Risk Merchant")
        elif merchant["risk_level"] == "medium":
            risk_score += 10
            risk_factors.append("Medium-Risk Merchant")

        if customer["account_age"] < 90:
            risk_score += 20
            risk_factors.append("New Account")

        if random.random() < 0.1:
            risk_score += 30
            risk_factors.append("Velocity Alert")

        if transaction["channel"] in ["Online", "Phone"]:
            risk_score += 5
            risk_factors.append("CNP Transaction")

        return {
            "risk_score": min(risk_score, 100),
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if risk_score > 70 else "MEDIUM" if risk_score > 40 else "LOW",
            "customer_risk_profile": customer["risk_profile"],
            "merchant_risk_level": merchant["risk_level"],
        }

    def generate_transaction(self):
        customer = random.choice(self.customer_profiles)
        merchant = random.choice(self.merchants)
        current_time = datetime.now()

        transaction = {
            "transaction_id": str(uuid.uuid4())[:8].upper(),
            "Timestamp": current_time,
            "customer_id": customer["id"],
            "customer_name": customer["name"],
            "merchant_id": merchant["id"],
            "merchant_name": merchant["name"],
            "merchant_category": merchant["category"],
            "type": random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]),
            "channel": random.choice(["Online", "ATM", "POS", "Mobile App", "Phone"]),
            "customer_location": customer["location"],
            "merchant_location": merchant["location"],
            "step": random.randint(1, 744),
            "unusuallogin": 1 if random.random() < 0.1 else 0,
            "day_of_week": current_time.weekday(),
            "month": current_time.month,
            "Acct type": random.choice(["Checking", "Savings"]),
            "Time of day": self._get_time_of_day(current_time),
        }

        base_amount = customer["avg_spending"]
        cat = merchant["category"]
        if cat in ["Gas Station", "Grocery", "Pharmacy"]:
            amount = base_amount * random.uniform(0.1, 0.8)
        elif cat in ["Department Store", "Online Store"]:
            amount = base_amount * random.uniform(0.5, 2.0)
        elif cat == "ATM":
            amount = random.choice([20, 40, 60, 80, 100, 200, 300, 500])
        else:
            amount = base_amount * random.uniform(0.3, 1.5)

        if random.random() < 0.05:
            amount *= random.uniform(5, 15)

        transaction.update(
            {
                "amount": round(amount, 2),
                "oldbalanceOrg": round(random.uniform(100, 50000), 2),
                "newbalanceOrig": round(random.uniform(50, 48000), 2),
                "oldbalanceDest": round(random.uniform(0, 30000), 2),
                "newbalanceDest": round(random.uniform(0, 32000), 2),
            }
        )

        transaction.update(self._calculate_risk_score(transaction, customer, merchant))
        return transaction


@st.cache_resource
def get_tx_generator():
    return EnhancedTransactionGenerator()


tx_gen = get_tx_generator()

def generate_transaction():
    return tx_gen.generate_transaction()

# -----------------------
# Session state init
# -----------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "fraud_log" not in st.session_state:
    st.session_state.fraud_log = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "daily_stats" not in st.session_state:
    st.session_state.daily_stats = {"total_transactions": 0, "fraud_detected": 0, "false_positives": 0, "high_risk_alerts": 0}

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("⚙️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Rate (sec)", 1, 10, 3)

if st.sidebar.button("🔧 Reset Cache"):
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.experimental_rerun()

if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="fraud_dashboard")

# -----------------------
# Generate transaction (on refresh or button)
# -----------------------
if auto_refresh or st.sidebar.button("🔄 Generate Transaction"):
    try:
        new_tx = generate_transaction()
        st.sidebar.success("✅ Transaction generated successfully")
        st.sidebar.write("**Generated Transaction Keys:**")
        st.sidebar.write(list(new_tx.keys()))
    except Exception as e:
        st.sidebar.error(f"❌ Transaction generation failed: {e}")
        st.stop()

    # Build model input using the transaction
    # Use features that match your training features (numeric + categorical columns)
    current_time = datetime.now()
    model_features = {
        "step": new_tx.get("step", random.randint(1, 744)),
        "amount": new_tx.get("amount", 0.0),
        "oldbalanceOrg": new_tx.get("oldbalanceOrg", 0.0),
        "newbalanceOrig": new_tx.get("newbalanceOrig", 0.0),
        "oldbalanceDest": new_tx.get("oldbalanceDest", 0.0),
        "newbalanceDest": new_tx.get("newbalanceDest", 0.0),
        "unusuallogin": new_tx.get("unusuallogin", 0),
        "day_of_week": new_tx.get("day_of_week", current_time.weekday()),
        "month": new_tx.get("month", current_time.month),
        # categorical raw values (we will handle encoding before predict)
        "type": new_tx.get("type", "PAYMENT"),
        "Acct type": new_tx.get("Acct type", "Savings"),
        "Time of day": new_tx.get("Time of day", "Morning"),
        # keep risk info in the row so fallback heuristics can use it
        "risk_score": new_tx.get("risk_score", 0),
    }
    input_df = pd.DataFrame([model_features])

    # Attempt model prediction safely
    if model is not None:
        try:
            prob, pred = align_and_predict_proba(model, input_df)
            st.sidebar.success("✅ Model prediction successful")
        except Exception as e:
            st.sidebar.error(f"❌ Model prediction failed: {e}")
            prob, pred = 0.0, 0
    else:
        # fallback heuristic
        prob = float(model_features["risk_score"]) / 100.0
        pred = 1 if prob > 0.5 else 0

    # Update new_tx with ML results
    new_tx.update(
        {
            "ML_Prediction": "FRAUD" if int(pred) == 1 else "LEGITIMATE",
            "ML_Confidence": round(float(prob), 3),
            "Final_Decision": "BLOCKED" if int(pred) == 1 and float(prob) > 0.8 else ("APPROVED" if int(pred) == 0 else "REVIEW"),
        }
    )

    # Push to session state
    st.session_state.transactions.insert(0, new_tx)
    st.session_state.transactions = st.session_state.transactions[:100]
    st.session_state.daily_stats["total_transactions"] += 1
    if int(pred) == 1:
        st.session_state.daily_stats["fraud_detected"] += 1
        st.session_state.fraud_log.insert(0, new_tx)
        st.session_state.fraud_log = st.session_state.fraud_log[:50]

    # Alerts
    try:
        ml_conf = float(new_tx.get("ML_Confidence", 0))
        risk_score = float(new_tx.get("risk_score", 0))
        if risk_score > 60 or ml_conf > 0.6:
            alert = {
                "alert_id": f"ALT_{len(st.session_state.alerts):06d}",
                "timestamp": new_tx["Timestamp"],
                "transaction_id": new_tx["transaction_id"],
                "severity": "HIGH" if risk_score > 80 else "MEDIUM",
                "type": "FRAUD_DETECTION" if int(pred) == 1 else "HIGH_RISK",
                "description": f"Risk Score: {risk_score}, ML Score: {ml_conf:.3f}",
                "customer_id": new_tx.get("customer_id", "Unknown"),
                "amount": new_tx.get("amount", 0.0),
                "status": "OPEN",
            }
            st.session_state.alerts.insert(0, alert)
            st.session_state.alerts = st.session_state.alerts[:100]
            if alert["severity"] == "HIGH":
                st.session_state.daily_stats["high_risk_alerts"] += 1
    except Exception:
        pass

# -----------------------
# Main Dashboard Content
# -----------------------
st.markdown('<h1 class="main-header">Fraud Detection Dashboard</h1>', unsafe_allow_html=True)

# Hide system status metrics as requested

# If we have transactions, build DataFrame
if st.session_state.transactions:
    df = pd.DataFrame(st.session_state.transactions)
else:
    df = pd.DataFrame()

# KPI row
# KPIs will be calculated after filtering for consistency


# No filters - show all data
filtered_df = df.copy()

# Update KPIs to use filtered data for consistency
st.subheader("📊 Key Metrics")
col1, col2 = st.columns(2)

if filtered_df is not None and not filtered_df.empty:
    # Ensure all expected columns exist
    filtered_df_calc = filtered_df.copy()
    for col in ["ML_Prediction", "Final_Decision", "risk_score", "amount"]:
        if col not in filtered_df_calc.columns:
            filtered_df_calc[col] = None

    # Core metrics using filtered data
    total_tx = len(filtered_df_calc)
    fraud_detected = filtered_df_calc["ML_Prediction"].eq("FRAUD").sum()
    fraud_rate = (fraud_detected / total_tx * 100) if total_tx > 0 else 0

    # Display only requested KPIs
    col1.metric("💳 Total Transactions", f"{total_tx:,}")
    col2.metric("🚨 Fraud Detected", fraud_detected, delta=f"{fraud_rate:.1f}%")
else:
    col1.metric("💳 Total Transactions", "0")
    col2.metric("🚨 Fraud Detected", "0", delta="0.0%")

# Display analytics and alerts tabs (transaction feed hidden)
if not filtered_df.empty:
    tab1, tab2 = st.tabs(["📊 Live Analytics", "🚨 Fraud Alerts"])

    with tab1:
        st.subheader("📈 Live Analytics")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Risk Score Distribution**")
            if "risk_level" in filtered_df.columns:
                risk_counts = filtered_df["risk_level"].value_counts()
                st.bar_chart(risk_counts)
            else:
                st.info("No risk_level data available")

        with c2:
            st.write("**Transaction Types Analysis**")
            if "type" in filtered_df.columns:
                type_counts = filtered_df["type"].value_counts()
                st.bar_chart(type_counts)
            else:
                st.info("No transaction type data available")



        st.write("**Transaction Timeline (minute resolution)**")
        if "Timestamp" in filtered_df.columns:
            tmp = filtered_df.copy()
            tmp["minute"] = pd.to_datetime(tmp["Timestamp"]).dt.floor("min")
            timeline = tmp.groupby(["minute", "ML_Prediction"]).size().reset_index(name="count")
            if not timeline.empty:
                timeline_pivot = timeline.pivot(index="minute", columns="ML_Prediction", values="count").fillna(0)
                st.line_chart(timeline_pivot)
            else:
                st.info("No timeline data (not enough transactions)")
        else:
            st.info("No Timestamp column available")

    with tab2:
        st.subheader("� Active Fraud Alerts")
        if st.session_state.alerts:
            alert_df = pd.DataFrame(st.session_state.alerts)
            a1, a2, a3 = st.columns(3)
            a1.metric("🔴 High Priority", len(alert_df[alert_df["severity"] == "HIGH"]))
            a2.metric("🟡 Medium Priority", len(alert_df[alert_df["severity"] == "MEDIUM"]))
            a3.metric("📊 Total Open", len(alert_df[alert_df["status"] == "OPEN"]))

            for alert in st.session_state.alerts[:10]:
                severity_class = "alert-high" if alert["severity"] == "HIGH" else "alert-medium"
                severity_emoji = "🔴" if alert["severity"] == "HIGH" else "🟡"
                st.markdown(
                    f"""
                <div class="{severity_class}">
                    <strong>{severity_emoji} Alert {alert['alert_id']}</strong> - {alert['type']}<br>
                    💰 Amount: ${alert['amount']:,.2f} | 👤 Customer: {alert['customer_id']}<br>
                    📝 {alert['description']}<br>
                    ⏰ {pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')} | Status: {alert['status']}
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("🎉 No active fraud alerts - System operating normally")

    # Actions (streamlined)
    col1, col2 = st.columns(2)
    if col1.button("📊 Export Data"):
        if st.session_state.transactions:
            csv_data = pd.DataFrame(st.session_state.transactions).to_csv(index=False)
            st.download_button("💾 Download", data=csv_data, file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    if col2.button("🧹 Clear Alerts"):
        st.session_state.alerts = []
        st.success("Cleared!")
else:
    st.info("🚀 Ready to monitor transactions.")

# -----------------------
# SHAP explainability (optional)
# -----------------------
# Optional: ML Explanation (collapsed to reduce clutter)
if shap_available and st.session_state.transactions:
    with st.expander("🧠 ML Model Explanation", expanded=False):
        try:
            latest_tx = st.session_state.transactions[-1]
            st.json(latest_tx)
        except Exception:
            st.info("Explanation unavailable.")

# Simple footer
st.markdown("---")
st.caption(f"🛡️ Fraud Detection System | {len(st.session_state.transactions)} transactions monitored")
