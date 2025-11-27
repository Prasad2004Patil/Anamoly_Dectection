#!/usr/bin/env bash
# create_project.sh
# Creates full project structure and all files for the anomaly detection project.
# Run: chmod +x create_project.sh && ./create_project.sh

set -e

echo "Creating project files and folders..."

# Make directories
mkdir -p api dashboard artifacts

# 1) requirements.txt
cat > requirements.txt <<'EOF'
fastapi==0.95.2
uvicorn==0.22.0
flask==2.3.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.3.2
joblib==1.3.2
tensorflow==2.13.0
keras==2.13.1
pymongo==4.4.0
python-dotenv==1.0.0
streamlit==1.27.1
matplotlib==3.8.1
plotly==5.22.0
scipy==1.11.3
requests==2.31.0
EOF

# 2) data_simulation.py
cat > data_simulation.py <<'EOF'
# data_simulation.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def simulate_transactions(n=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    base_time = datetime.utcnow()
    rows = []
    for i in range(n):
        user_id = np.random.randint(1000, 2000)
        amount = max(1.0, np.random.exponential(scale=80.0))
        hour = np.random.randint(0,24)
        device_type = np.random.choice(['mobile','desktop','tablet'], p=[0.6,0.3,0.1])
        country = np.random.choice(['IN','US','GB','DE','FR','CN','BR'], p=[0.4,0.2,0.1,0.08,0.07,0.1,0.05])
        ip_entropy = np.random.rand()
        session_length = np.random.exponential(scale=300)
        items_in_cart = np.random.poisson(2)
        is_guest = np.random.choice([0,1], p=[0.7,0.3])
        speed_score = np.clip(np.random.normal(50 - 0.01*amount, 10), 5, 100)
        ts = base_time - timedelta(seconds=np.random.randint(0, 86400*30))
        rows.append({
            'transaction_id': f"txn_{i}",
            'user_id': user_id,
            'amount': round(amount,2),
            'hour': hour,
            'device_mobile': 1 if device_type=='mobile' else 0,
            'device_desktop': 1 if device_type=='desktop' else 0,
            'device_tablet': 1 if device_type=='tablet' else 0,
            'country': country,
            'ip_entropy': ip_entropy,
            'session_length': session_length,
            'items_in_cart': items_in_cart,
            'is_guest': is_guest,
            'speed_score': speed_score,
            'timestamp': ts
        })
    df = pd.DataFrame(rows)
    n_anom = max(10, n//200)
    anom_indices = np.random.choice(df.index, n_anom, replace=False)
    for idx in anom_indices:
        df.at[idx,'amount'] *= np.random.uniform(5,20)
        df.at[idx,'ip_entropy'] = np.random.uniform(0,0.01)
        df.at[idx,'session_length'] = np.random.uniform(1,5)
        df.at[idx,'items_in_cart'] = np.random.randint(10,50)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    df = simulate_transactions(5000)
    df.to_csv("sample_transactions.csv", index=False)
    print("Saved sample_transactions.csv with shape", df.shape)
EOF

# 3) utils.py
cat > utils.py <<'EOF'
# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tensorflow.keras.models import load_model

SCALER_PATH = "artifacts/scaler.joblib"
MODEL_DIR = "artifacts"

def ensure_artifacts_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def features_from_df(df):
    df2 = df.copy()
    country_freq = df2['country'].value_counts().to_dict()
    df2['country_freq'] = df2['country'].map(lambda x: country_freq.get(x,0))
    features = df2[['amount','hour','device_mobile','device_desktop','device_tablet',
                    'ip_entropy','session_length','items_in_cart','is_guest','speed_score','country_freq']].fillna(0)
    return features

def fit_scaler(X):
    ensure_artifacts_dir()
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, SCALER_PATH)
    return scaler

def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found, run training first.")
    return joblib.load(SCALER_PATH)

def scale_features(X, scaler=None):
    if scaler is None:
        scaler = load_scaler()
    Xs = scaler.transform(X)
    return Xs

def save_sklearn_model(model, name):
    ensure_artifacts_dir()
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def load_sklearn_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def save_keras_model(keras_model, name):
    ensure_artifacts_dir()
    path = os.path.join(MODEL_DIR, name)
    keras_model.save(path)
    return path

def load_keras_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_model(path)
EOF

# 4) model_training.py
cat > model_training.py <<'EOF'
# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import layers, models, callbacks
from data_simulation import simulate_transactions
from utils import features_from_df, fit_scaler, scale_features, save_sklearn_model, save_keras_model

def train_models(df, test_size=0.2, random_state=42):
    X = features_from_df(df)
    y = ((df['amount'] > df['amount'].mean() + 3*df['amount'].std()) | (df['items_in_cart'] > 10) | (df['ip_entropy'] < 0.02)).astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = fit_scaler(X_train)
    X_train_s = scale_features(X_train, scaler)
    X_test_s = scale_features(X_test, scaler)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=random_state)
    iso.fit(X_train_s)
    save_sklearn_model(iso, "isolation_forest")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    ocsvm.fit(X_train_s)
    save_sklearn_model(ocsvm, "oneclass_svm")
    input_dim = X_train_s.shape[1]
    latent_dim = max(4, input_dim//2)
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(latent)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    auto = models.Model(inputs=inp, outputs=out)
    auto.compile(optimizer='adam', loss='mse')
    early = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    auto.fit(X_train_s, X_train_s, validation_data=(X_test_s, X_test_s),
             epochs=100, batch_size=128, callbacks=[early], verbose=1)
    save_keras_model(auto, "autoencoder")
    eval_dict = {}
    iso_scores = -iso.decision_function(X_test_s)
    ocsvm_scores = -ocsvm.decision_function(X_test_s)
    recon = auto.predict(X_test_s)
    mse = np.mean(np.square(recon - X_test_s), axis=1)
    for name, scores in [('isolation_forest', iso_scores), ('oneclass_svm', ocsvm_scores), ('autoencoder', mse)]:
        smin, smax = scores.min(), scores.max()
        if smax - smin == 0:
            sn = np.zeros_like(scores)
        else:
            sn = (scores - smin) / (smax - smin)
        try:
            roc = roc_auc_score(y_test, sn)
        except Exception:
            roc = float('nan')
        thr = np.percentile(sn, 99)
        preds = (sn >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
        eval_dict[name] = {'roc_auc': float(roc), 'precision': float(p), 'recall': float(r), 'f1': float(f)}
    return eval_dict

if __name__ == "__main__":
    print("Generating data...")
    df = simulate_transactions(8000)
    print("Training models...")
    results = train_models(df)
    print("Evaluation results:", results)
    print("Artifacts saved in artifacts/ directory.")
EOF

# 5) api/app.py
cat > api/app.py <<'EOF'
# api/app.py
import os
import json
from flask import Flask, request, jsonify
from utils import features_from_df, load_scaler, scale_features, load_sklearn_model, load_keras_model
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import traceback

load_dotenv()

app = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI", "")
db_client = None
db = None
if MONGO_URI:
    db_client = MongoClient(MONGO_URI)
    db = db_client.get_database(os.getenv("MONGO_DBNAME", "anomaly_db"))
    logs_collection = db.get_collection("anomaly_logs")
else:
    logs_collection = None
    LOCAL_LOG = "logs_local.json"
    if not os.path.exists(LOCAL_LOG):
        with open(LOCAL_LOG, "w") as f:
            json.dump([], f)

def load_models():
    global iso_model, ocsvm_model, ae_model, scaler
    iso_model = load_sklearn_model("isolation_forest")
    ocsvm_model = load_sklearn_model("oneclass_svm")
    ae_model = load_keras_model("autoencoder")
    scaler = load_scaler()

try:
    load_models()
except Exception as e:
    print("Warning: failed to load models at startup:", e)

def log_anomaly(record):
    try:
        if logs_collection:
            logs_collection.insert_one(record)
        else:
            with open("logs_local.json","r+") as f:
                arr = json.load(f)
                arr.append(record)
                f.seek(0)
                json.dump(arr, f, default=str)
    except Exception as e:
        print("Failed to log anomaly:", e)

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error":"Expecting JSON payload"}), 400
        txs = payload.get("transactions")
        if txs is None:
            return jsonify({"error":"Missing 'transactions' field"}), 400
        single = False
        if isinstance(txs, dict):
            txs = [txs]
            single = True
        import pandas as pd
        df = pd.DataFrame(txs)
        X = features_from_df(df)
        Xs = scale_features(X, scaler)
        iso_scores = -iso_model.decision_function(Xs)
        ocsvm_scores = -ocsvm_model.decision_function(Xs)
        recon = ae_model.predict(Xs)
        mse = np.mean(np.square(recon - Xs), axis=1)
        def norm(arr):
            mi, ma = float(arr.min()), float(arr.max())
            if ma - mi == 0:
                return [0.0]*len(arr)
            return ((arr - mi)/(ma-mi)).tolist()
        iso_n = norm(iso_scores)
        ocsvm_n = norm(ocsvm_scores)
        ae_n = norm(mse)
        results = []
        for i in range(len(txs)):
            agg = float(np.mean([iso_n[i], ocsvm_n[i], ae_n[i]]))
            decision = "anomaly" if agg > 0.7 else "normal"
            rec = {
                "transaction": txs[i],
                "scores": {"isolation_forest": iso_n[i], "oneclass_svm": ocsvm_n[i], "autoencoder": ae_n[i]},
                "aggregated_score": agg,
                "decision": decision
            }
            if agg > 0.6:
                log_rec = {"timestamp": __import__("datetime").datetime.utcnow().isoformat(), "record": rec}
                log_anomaly(log_rec)
            results.append(rec)
        if single:
            return jsonify(results[0])
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"exception", "detail": str(e)}), 500

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error":"Expecting JSON payload"}), 400
        data = payload.get("retrain_data")
        if data is None:
            return jsonify({"error":"Missing 'retrain_data' field"}), 400
        import pandas as pd
        df = pd.DataFrame(data)
        from model_training import train_models
        results = train_models(df, test_size=0.2)
        load_models()
        return jsonify({"status":"retrained", "results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"exception", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
EOF

# 6) dashboard/dashboard.py
cat > dashboard/dashboard.py <<'EOF'
# dashboard/dashboard.py
import streamlit as st
import pandas as pd
import requests
import time
import os
from datetime import datetime
import json

API_URL = os.getenv("API_URL", "http://localhost:5000")
STORAGE_LOCAL = "logs_local.json"

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("E-commerce Anomaly Detection Dashboard")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Upload / Live Predict")
    uploaded = st.file_uploader("Upload transactions CSV (or use simulated data below)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if st.button("Generate sample data"):
            from data_simulation import simulate_transactions
            df = simulate_transactions(1000)
            st.write("Generated sample data")
        else:
            df = None

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head())

        if st.button("Send to API for predictions"):
            payload = {"transactions": df.to_dict(orient='records')}
            with st.spinner("Sending..."):
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
                st.success("Predictions received")
                st.dataframe(recdf.sort_values("aggregated_score", ascending=False).head(50))
                st.download_button("Download results CSV", recdf.to_csv(index=False), file_name="predictions.csv")
                try:
                    if not os.path.exists(STORAGE_LOCAL):
                        with open(STORAGE_LOCAL,"w") as f:
                            json.dump([], f)
                    with open(STORAGE_LOCAL, "r+") as f:
                        arr = json.load(f)
                        for r in records:
                            arr.append({"ts": datetime.utcnow().isoformat(), **r})
                        f.seek(0)
                        json.dump(arr, f, default=str)
                except Exception as e:
                    st.warning("Could not save local logs: " + str(e))

with col2:
    st.header("Saved Anomalies")
    try:
        with open(STORAGE_LOCAL, "r") as f:
            logs = json.load(f)
            logs_df = pd.DataFrame(logs)
    except Exception:
        logs_df = pd.DataFrame(columns=["ts","transaction_id","user_id","amount","aggregated_score","decision"])
    if not logs_df.empty:
        st.dataframe(logs_df.sort_values("aggregated_score", ascending=False).head(200))
        min_score = float(st.slider("Min aggregated score", 0.0, 1.0, 0.6, step=0.01))
        filtered = logs_df[logs_df['aggregated_score']>=min_score]
        st.write(f"Showing {len(filtered)} records with score >= {min_score}")
    else:
        st.info("No local logs. Use the left panel to generate predictions.")

st.markdown("---")
st.subheader("Real-time metrics")
try:
    li = logs_df
    total = len(li)
    anomalies = len(li[li['aggregated_score']>0.7])
    st.metric("Total predictions logged", total)
    st.metric("High anomalies (score>0.7)", anomalies)
except Exception:
    pass

st.markdown("### Top anomalies (recent)")
try:
    if not logs_df.empty:
        st.table(logs_df.sort_values("ts", ascending=False).head(10))
    else:
        st.write("No entries yet.")
except Exception:
    pass

st.markdown("---")
st.markdown("**Manual tagging**")
tag_txn = st.text_input("Transaction ID to tag")
tag_choice = st.selectbox("Tag as", ["true_anomaly","false_positive","investigate","ignore"])
if st.button("Tag transaction"):
    if tag_txn:
        try:
            with open(STORAGE_LOCAL,"r+") as f:
                arr = json.load(f)
                for item in arr:
                    if item.get("transaction_id") == tag_txn:
                        item["tag"] = tag_choice
                f.seek(0)
                json.dump(arr, f, default=str)
            st.success("Tagged (local). For production you'd send tag to DB for retraining.")
        except Exception as e:
            st.error("Failed to tag: " + str(e))
    else:
        st.warning("Enter transaction id")

st.markdown("---")
st.caption("Dashboard built with Streamlit — include artifacts/ (models + scaler) when submitting.")
EOF

# 7) Dockerfile.api
cat > Dockerfile.api <<'EOF'
# Dockerfile.api
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5000

ENV FLASK_APP=api/app.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "api/app.py"]
EOF

# 8) Dockerfile.streamlit
cat > Dockerfile.streamlit <<'EOF'
# Dockerfile.streamlit
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
EOF

# 9) docker-compose.yml
cat > docker-compose.yml <<'EOF'
version: "3.8"
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: anomaly_api:latest
    container_name: anomaly_api
    ports:
      - "5000:5000"
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs_local.json:/app/logs_local.json
    environment:
      - MONGO_URI=${MONGO_URI:-}
      - MONGO_DBNAME=${MONGO_DBNAME:-anomaly_db}

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    image: anomaly_dashboard:latest
    container_name: anomaly_dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:5000
    volumes:
      - ./logs_local.json:/app/logs_local.json
      - ./artifacts:/app/artifacts
EOF

# 10) run_all.sh (same as provided earlier)
cat > run_all.sh <<'EOF'
#!/bin/bash
# ==========================================
# Cloud-Based Anomaly Detection Project
# Auto Runner Script (Linux/macOS)
# ==========================================

# Fail on any error
set -e

echo "🚀 Starting full pipeline..."

# Step 1: Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 not found. Please install Python 3.9+."
  exit 1
fi

# Step 2: Create virtual environment if not exists
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate environment
source venv/bin/activate

# Step 3: Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Generate synthetic data if missing
if [ ! -f "sample_transactions.csv" ]; then
  echo "🧩 Generating sample data..."
  python3 data_simulation.py
else
  echo "✅ sample_transactions.csv already exists."
fi

# Step 5: Train models (if artifacts missing)
if [ ! -d "artifacts" ] || [ -z "$(ls -A artifacts 2>/dev/null)" ]; then
  echo "🤖 Training ML models..."
  python3 model_training.py
else
  echo "✅ Found existing model artifacts, skipping training."
fi

# Step 6: Start Flask API in background
echo "🌐 Starting Flask API on port 5000..."
nohup python3 api/app.py > api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 5
echo "✅ Flask API started (PID: $API_PID)."

# Step 7: Start Streamlit Dashboard
echo "📊 Launching Streamlit dashboard (port 8501)..."
nohup streamlit run dashboard/dashboard.py --server.port 8501 --server.address 0.0.0.0 > dashboard.log 2>&1 &
DASH_PID=$!

sleep 5
echo "✅ Streamlit dashboard running (PID: $DASH_PID)."

# Step 8: Auto-open in browser
if command -v open &>/dev/null; then
  open "http://localhost:8501"
elif command -v xdg-open &>/dev/null; then
  xdg-open "http://localhost:8501"
else
  echo "👉 Open your browser manually at: http://localhost:8501"
fi

echo "----------------------------------------------"
echo "💡 Project is running!"
echo "📈 Dashboard: http://localhost:8501"
echo "🌍 API: http://localhost:5000/health"
echo "----------------------------------------------"
echo "🧹 To stop all processes, run: "
echo "    kill $API_PID $DASH_PID"
echo "----------------------------------------------"

# Keep script alive until manually terminated
wait
EOF

chmod +x run_all.sh

# 11) README.md
cat > README.md <<'EOF'
# Cloud-Based Anomaly Detection — Full Implementation

This repository implements an end-to-end anomaly detection project (ML models, REST API, Streamlit dashboard).

## Contents
- `data_simulation.py` — generate synthetic e-commerce data
- `model_training.py` — train Isolation Forest, One-Class SVM, Autoencoder (saves artifacts/)
- `api/app.py` — Flask API to serve predictions and retraining
- `dashboard/dashboard.py` — Streamlit dashboard connecting to API
- `utils.py` — helper functions
- `artifacts/` — models and scaler saved here after training
- `Dockerfile.api`, `Dockerfile.streamlit`, `docker-compose.yml` — container configuration
- `run_all.sh` — convenient runner

## Quick local setup
1. Create virtual env and install:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

