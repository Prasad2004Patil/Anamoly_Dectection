import streamlit as st
import os
import pandas as pd
import numpy as np
import bcrypt
import joblib
import traceback
from pymongo import MongoClient
from tensorflow.keras.models import load_model

# --- CRITICAL: BRIDGE SECRETS TO ENVIRONMENT VARIABLES ---
# This is the part your previous code was missing.
# It takes the password from Streamlit Cloud and gives it to the app.
if "MONGO_URI" in st.secrets:
    os.environ["MONGO_URI"] = st.secrets["MONGO_URI"]
if "MONGO_DBNAME" in st.secrets:
    os.environ["MONGO_DBNAME"] = st.secrets["MONGO_DBNAME"]
# -------------------------------------------------------

# Safe import for utils to prevent crashing if file is missing
try:
    from utils import features_from_df
except ImportError:
    st.error("⚠️ Critical Error: 'utils.py' is missing! Please upload utils.py to your GitHub.")
    st.stop()

# --- 1. CONFIGURATION & CACHING ---
st.set_page_config(page_title="Anomaly Detection", layout="wide")

# Cache heavy model loading
@st.cache_resource
def load_system_artifacts():
    def load_artifact(path):
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    # Load artifacts
    scaler = load_artifact("artifacts/scaler.joblib")
    iso = load_artifact("artifacts/isolation_forest.joblib")
    svm = load_artifact("artifacts/oneclass_svm.joblib")
    
    # Load Keras Model (Handle multiple path versions)
    ae = None
    try:
        if os.path.exists("artifacts/autoencoder/model.keras"):
            ae = load_model("artifacts/autoencoder/model.keras")
        elif os.path.exists("artifacts/autoencoder_model.keras"):
            ae = load_model("artifacts/autoencoder_model.keras")
    except Exception as e:
        print(f"Error loading Autoencoder: {e}")
        
    return scaler, iso, svm, ae

# Cache Database Connection
@st.cache_resource
def init_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    try:
        client = MongoClient(uri)
        db = client.get_database(os.getenv("MONGO_DBNAME", "anomaly_db"))
        # Send a ping to confirm connection is alive
        client.admin.command('ping')
        return db
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None

# Initialize Resources
scaler, iso_model, ocsvm_model, ae_model = load_system_artifacts()
db = init_db()
users_collection = db["users"] if db is not None else None

# --- 2. AUTHENTICATION LOGIC ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.markdown("## 🔐 Cloud Anomaly Login")
    
    if not users_collection:
        st.warning("⚠️ Database not connected. Please check 'Advanced Settings' > 'Secrets' on Streamlit Cloud.")

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                if not users_collection:
                    st.error("Database unavailable.")
                else:
                    u_data = users_collection.find_one({"username": user})
                    if u_data and bcrypt.checkpw(pwd.encode('utf-8'), u_data['password']):
                        st.session_state.logged_in = True
                        st.session_state.username = user
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")

    with tab2:
        with st.form("register"):
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            if st.form_submit_button("Sign Up"):
                if not users_collection:
                    st.error("Database unavailable.")
                elif users_collection.find_one({"username": new_u}):
                    st.error("User exists!")
                else:
                    hashed = bcrypt.hashpw(new_p.encode('utf-8'), bcrypt.gensalt())
                    users_collection.insert_one({"username": new_u, "password": hashed})
                    st.success("Created! Please log in.")

# --- 3. MAIN DASHBOARD LOGIC ---
def main_dashboard():
    with st.sidebar:
        st.write(f"User: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.title("🛡️ E-commerce Anomaly Detection")
    
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    
    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Data Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Analyze Data") and df is not None:
        if not scaler:
            st.error("Models missing. Make sure 'artifacts' folder is uploaded to GitHub.")
            return

        with st.spinner("Running AI Models..."):
            try:
                # Feature Engineering
                X = features_from_df(df)
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Scaling
                X_scaled = scaler.transform(X.values).astype(np.float32)

                # Predictions
                iso_scores = -iso_model.decision_function(X_scaled)
                svm_scores = -ocsvm_model.decision_function(X_scaled)
                
                # Autoencoder (Handle missing model)
                if ae_model:
                    recon = ae_model.predict(X_scaled)
                    mse = np.mean(np.square(recon - X_scaled), axis=1)
                else:
                    mse = np.zeros(len(X))

                # Normalization
                def norm(arr):
                    if len(arr) == 0: return arr
                    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

                iso_n = norm(iso_scores)
                svm_n = norm(svm_scores)
                ae_n = norm(mse)
                
                # Aggregation
                agg_scores = (iso_n + svm_n + ae_n) / 3
                
                # Results
                df['Anomaly_Score'] = agg_scores
                df['Decision'] = df['Anomaly_Score'].apply(lambda x: "⚠️ ANOMALY" if x > 0.7 else "✅ Normal")
                
                st.success("Analysis Complete!")
                st.dataframe(df.sort_values("Anomaly_Score", ascending=False).style.applymap(
                    lambda x: 'background-color: #ffcdd2' if x == "⚠️ ANOMALY" else '', subset=['Decision']
                ))
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.write(traceback.format_exc())

# --- 4. APP ROUTER ---
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()