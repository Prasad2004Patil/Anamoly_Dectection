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
    # Always save with proper .keras extension for TF >= 2.17
    path = os.path.join(MODEL_DIR, f"{name}_model.keras")
    keras_model.save(path)
    return path

def load_keras_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_model(path)
