# api/app.py
import os
import joblib
import sys
import json
import traceback
from pathlib import Path

# ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as k_load_model

# utils (project helpers)
from utils import features_from_df, load_scaler, load_sklearn_model, load_keras_model

# DB / env
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# ---------- Logging / files ----------
ARTIFACTS_DIR = Path("artifacts")
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
LOCAL_LOG = "logs_local.json"

# ensure local log exists
if not os.path.exists(LOCAL_LOG):
    with open(LOCAL_LOG, "w") as f:
        json.dump([], f)

# ---------- Globals for models ----------
iso_model = None
ocsvm_model = None
ae_model = None
scaler = None

def safe_load_scaler(path: Path):
    if not path.exists():
        print(f"[WARN] scaler file not found at {path}")
        return None
    try:
        s = joblib.load(path)
        print(f"[INFO] Loaded scaler from {path} (n_features_in_={getattr(s,'n_features_in_', None)})")
        return s
    except Exception as e:
        print(f"[WARN] Failed to load scaler {path}: {e}")
        return None

def safe_load_sklearn(name):
    try:
        m = load_sklearn_model(name)
        n = getattr(m, "n_features_in_", None)
        print(f"[INFO] Loaded sklearn model '{name}' (n_features_in_={n})")
        return m
    except Exception as e:
        print(f"[WARN] Failed to load sklearn model '{name}': {e}")
        return None

def safe_load_keras(path_or_key):
    try:
        m = load_keras_model(path_or_key)
        # try to infer input dim
        inp = getattr(m, "input_shape", None)
        print(f"[INFO] Loaded keras model '{path_or_key}' (input_shape={inp})")
        return m
    except Exception as e:
        print(f"[WARN] Failed to load keras model '{path_or_key}': {e}")
        return None

def load_models():
    global iso_model, ocsvm_model, ae_model, scaler
    scaler = safe_load_scaler(SCALER_PATH) or load_scaler()
    iso_model = safe_load_sklearn("isolation_forest")
    ocsvm_model = safe_load_sklearn("oneclass_svm")

    # try typical artifact names for autoencoder
    ae_model = safe_load_keras("autoencoder") or safe_load_keras("autoencoder_model.keras") or safe_load_keras(str(ARTIFACTS_DIR / "autoencoder" / "model.keras"))
    # If utils loader failed, try loading the concrete .keras file directly (avoid double-artifacts paths)
    if ae_model is None:
        try:
            direct_path = str(ARTIFACTS_DIR / 'autoencoder' / 'model.keras')
            if Path(direct_path).exists():
                ae_model = k_load_model(direct_path)
                print(f"[INFO] Directly loaded keras model from {direct_path}")
        except Exception as e:
            print(f"[WARN] direct keras load failed: {e}")


# initial load attempt
try:
    load_models()
except Exception as e:
    print("Warning: load_models() raised:", e)

# ---------- DB / logging ----------
MONGO_URI = os.getenv("MONGO_URI", "")
db_client = None
logs_collection = None
if MONGO_URI:
    try:
        db_client = MongoClient(MONGO_URI)
        db = db_client.get_database(os.getenv("MONGO_DBNAME", "anomaly_db"))
        logs_collection = db.get_collection("anomaly_logs")
    except Exception as e:
        print("Warning: failed to connect to MongoDB:", e)

def log_anomaly(record):
    try:
        if logs_collection:
            logs_collection.insert_one(record)
        else:
            with open(LOCAL_LOG, "r+") as f:
                arr = json.load(f)
                arr.append(record)
                f.seek(0)
                json.dump(arr, f, default=str)
    except Exception as e:
        print("Failed to log anomaly:", e)

# ---------- Helpers ----------
def try_cast_numeric_df(df: pd.DataFrame):
    # ensure numeric dtypes where possible to avoid sklearn complaints
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

# ---------- Endpoints ----------
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

        df = pd.DataFrame(txs)

        # feature engineering -> this must produce the expected columns used by training
        X = features_from_df(df)
        if not isinstance(X, pd.DataFrame):
            return jsonify({"error":"internal","detail":"features_from_df must return a pandas DataFrame"}), 500

        # cast numeric dtypes where possible
        X = try_cast_numeric_df(X)

        # Validate scaler presence and shape
        if scaler is None:
            return jsonify({"error":"scaler_missing","detail":"scaler not loaded at startup"}), 500

        expected_scaler = getattr(scaler, "n_features_in_", None)
        X_arr = X.values
        if expected_scaler is not None and X_arr.shape[1] != expected_scaler:
            return jsonify({
                "error":"shape_mismatch",
                "detail": f"X has {X_arr.shape[1]} features, but StandardScaler expects {expected_scaler} features."
            }), 500

        # scale inputs for models
        Xs = scaler.transform(X_arr).astype(np.float32)

        # validate sklearn models expect same number of features
        for name, model in (("isolation_forest", iso_model), ("oneclass_svm", ocsvm_model)):
            if model is None:
                return jsonify({"error":"model_missing","detail": f"{name} not loaded"}), 500
            m_expected = getattr(model, "n_features_in_", None)
            if m_expected is not None and Xs.shape[1] != m_expected:
                return jsonify({
                    "error":"shape_mismatch",
                    "detail": f"Xs has {Xs.shape[1]} features, but {name} expects {m_expected} features."
                }), 500

        # validate autoencoder input
        if ae_model is None:
            return jsonify({"error":"model_missing","detail":"autoencoder model not loaded"}), 500
        ae_input_shape = None
        try:
            ae_input_shape = ae_model.input_shape
            ae_expected = ae_input_shape[-1] if ae_input_shape is not None else None
            if ae_expected is not None and Xs.shape[1] != ae_expected:
                return jsonify({
                    "error":"shape_mismatch",
                    "detail": f"Xs has {Xs.shape[1]} features, but autoencoder expects {ae_expected} features."
                }), 500
        except Exception:
            # if model doesn't expose input_shape, proceed but guard predict
            pass

        # Run model predictions
        try:
            iso_scores = -iso_model.decision_function(Xs)
            ocsvm_scores = -ocsvm_model.decision_function(Xs)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error":"model_predict_failed","detail": str(e)}), 500

        try:
            recon = ae_model.predict(Xs)
            mse = np.mean(np.square(recon - Xs), axis=1)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error":"autoencoder_failed","detail": str(e)}), 500

        # normalize helper
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
        df = pd.DataFrame(data)
        from model_training import train_models
        results = train_models(df, test_size=0.2)
        load_models()
        return jsonify({"status":"retrained", "results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"exception", "detail": str(e)}), 500

if __name__ == "__main__":
    # Bind to loopback for local dev and show errors in dev mode
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)
