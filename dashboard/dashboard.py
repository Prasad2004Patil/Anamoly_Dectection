# dashboard/dashboard.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import json
import time

API_URL = os.getenv("API_URL", "http://localhost:5000")
STORAGE_LOCAL = "logs_local.json"

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("E-commerce Anomaly Detection Dashboard")

# --- Initialize session state ---
if "df" not in st.session_state:
    st.session_state.df = None

col1, col2 = st.columns([2, 1])

# --- LEFT PANEL ---
with col1:
    st.header("Upload / Live Predict")

    uploaded = st.file_uploader("Upload transactions CSV (or use simulated data below)", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.success("✅ Uploaded file loaded successfully!")
    else:
        if st.button("Generate sample data"):
            from data_simulation import simulate_transactions
            st.session_state.df = simulate_transactions(1000)
            st.success("✅ Generated sample data successfully!")

    # Display current data
    if st.session_state.df is not None:
        st.subheader("Preview of Data")
        st.dataframe(st.session_state.df.head(10))
    else:
        st.info("No data loaded yet. Upload a file or generate sample data.")

    # --- Send to API for predictions ---
    if st.button("Send to API for predictions"):
        if st.session_state.df is None:
            st.warning("⚠️ Please generate or upload data first!")
        else:
            df_to_send = st.session_state.df.copy()
            for col in df_to_send.select_dtypes(include=["datetime", "datetimetz"]).columns:
                df_to_send[col] = df_to_send[col].astype(str)
            payload = {"transactions": df_to_send.to_dict(orient='records')}
            with st.spinner("Sending data to API..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
                    if resp.status_code == 200:
                        res = resp.json()
                        if isinstance(res, dict):
                            res = [res]
                        records = []
                        for r in res:
                            t = r.get("transaction", {})
                            scores = r.get("scores", {})
                            records.append({
                                "transaction_id": t.get("transaction_id"),
                                "user_id": t.get("user_id"),
                                "amount": t.get("amount"),
                                "aggregated_score": r.get("aggregated_score"),
                                "decision": r.get("decision"),
                                "isolation": scores.get("isolation_forest"),
                                "oneclass_svm": scores.get("oneclass_svm"),
                                "autoencoder": scores.get("autoencoder")
                            })
                        recdf = pd.DataFrame(records)
                        st.success("✅ Predictions received from API!")
                        st.dataframe(recdf.sort_values("aggregated_score", ascending=False).head(50))
                        st.download_button("Download results CSV", recdf.to_csv(index=False), file_name="predictions.csv")

                        # Save locally
                        try:
                            if not os.path.exists(STORAGE_LOCAL):
                                with open(STORAGE_LOCAL, "w") as f:
                                    json.dump([], f)
                            with open(STORAGE_LOCAL, "r+") as f:
                                arr = json.load(f)
                                for r in records:
                                    arr.append({"ts": datetime.utcnow().isoformat(), **r})
                                f.seek(0)
                                json.dump(arr, f, default=str)
                        except Exception as e:
                            st.warning("Could not save local logs: " + str(e))
                    else:
                        st.error(f"API error: {resp.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # Optional Reset button
    if st.button("Reset session"):
        st.session_state.df = None
        st.info("Session cleared. Generate or upload again.")
