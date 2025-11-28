import streamlit as st
import os
import pandas as pd
import numpy as np
import bcrypt
import joblib
import traceback
import warnings

# --- 1. SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# --- 2. IMPORTS ---
try:
    from pymongo import MongoClient
    from tensorflow.keras.models import load_model
    from utils import features_from_df
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}. Please check requirements.txt.")
    st.stop()

# --- 3. CONFIGURATION ---
st.set_page_config(page_title="Anomaly Detection", layout="wide")

# --- 4. SECRETS BRIDGE ---
if "MONGO_URI" in st.secrets:
    os.environ["MONGO_URI"] = st.secrets["MONGO_URI"]
if "MONGO_DBNAME" in st.secrets:
    os.environ["MONGO_DBNAME"] = st.secrets["MONGO_DBNAME"]

# --- 5. LOAD ARTIFACTS ---
@st.cache_resource
def load_system_artifacts():
    def load_artifact(path):
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
            return None

    # Load artifacts
    scaler = load_artifact("artifacts/scaler.joblib")
    iso = load_artifact("artifacts/isolation_forest.joblib")
    svm = load_artifact("artifacts/oneclass_svm.joblib")
    
    # Load Keras Model
    ae = None
    try:
        if os.path.exists("artifacts/autoencoder/model.keras"):
            ae = load_model("artifacts/autoencoder/model.keras")
        elif os.path.exists("artifacts/autoencoder_model.keras"):
            ae = load_model("artifacts/autoencoder_model.keras")
    except Exception as e:
        print(f"Error loading Autoencoder: {e}")
        
    return scaler, iso, svm, ae

# --- 6. DATABASE CONNECTION ---
@st.cache_resource
def init_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    try:
        client = MongoClient(uri)
        # Verify connection
        client.admin.command('ping')
        db = client.get_database(os.getenv("MONGO_DBNAME", "anomaly_db"))
        return db
    except Exception as e:
        print(f"❌ DB Connection Failed: {e}")
        return None

# Initialize Global Resources
scaler, iso_model, ocsvm_model, ae_model = load_system_artifacts()
db = init_db()

# Safe access to users collection
users_collection = None
if db is not None:
    try:
        users_collection = db["users"]
    except Exception:
        users_collection = None

# --- 7. AUTHENTICATION LOGIC ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "df" not in st.session_state:
    st.session_state.df = None

def login_page():
    st.markdown("## 🔐 Cloud Anomaly Login")
    
    # FIXED: Explicit check for None
    if users_collection is None:
        st.warning("⚠️ Database not connected. Check 'Advanced Settings' > 'Secrets'.")

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                if users_collection is None:
                    st.error("Database unavailable.")
                else:
                    try:
                        u_data = users_collection.find_one({"username": user})
                        if u_data and bcrypt.checkpw(pwd.encode('utf-8'), u_data['password']):
                            st.session_state.logged_in = True
                            st.session_state.username = user
                            st.rerun()
                        else:
                            st.error("Invalid Username or Password")
                    except Exception as e:
                        st.error(f"Login Error: {e}")

    with tab2:
        with st.form("register"):
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            if st.form_submit_button("Sign Up"):
                if users_collection is None:
                    st.error("Database unavailable.")
                else:
                    try:
                        # Check if user exists
                        existing_user = users_collection.find_one({"username": new_u})
                        if existing_user is not None:
                            st.error("User exists!")
                        else:
                            hashed = bcrypt.hashpw(new_p.encode('utf-8'), bcrypt.gensalt())
                            users_collection.insert_one({"username": new_u, "password": hashed})
                            st.success("Created! Please log in.")
                    except Exception as e:
                        st.error(f"Registration Error: {e}")

# --- 8. MAIN DASHBOARD LOGIC ---
def main_dashboard():
    with st.sidebar:
        st.write(f"User: **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.df = None
            st.rerun()

    st.title("🛡️ E-commerce Anomaly Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            try:
                st.session_state.df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
            
    with col2:
        st.write("Or generate data:")
        if st.button("Generate Sample Data"):
            try:
                # Local import
                from data_simulation import simulate_transactions
                with st.spinner("Simulating..."):
                    st.session_state.df = simulate_transactions(500)
                st.success("Generated 500 transactions!")
            except ImportError:
                st.error("data_simulation.py not found.")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # Show Data Preview & Analyze Button
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())

        if st.button("Analyze Data"):
            if not scaler:
                st.error("Models missing. Make sure 'artifacts' folder is uploaded to GitHub.")
                return

            with st.spinner("Running AI Models..."):
                try:
                    df_to_process = st.session_state.df.copy()
                    
                    # Feature Engineering
                    X = features_from_df(df_to_process)
                    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Scaling
                    X_scaled = scaler.transform(X.values).astype(np.float32)

                    # Predictions
                    iso_scores = -iso_model.decision_function(X_scaled)
                    svm_scores = -ocsvm_model.decision_function(X_scaled)
                    
                    # Autoencoder
                    if ae_model:
                        recon = ae_model.predict(X_scaled)
                        mse = np.mean(np.square(recon - X_scaled), axis=1)
                    else:
                        mse = np.zeros(len(X))

                    def norm(arr):
                        if len(arr) == 0: return arr
                        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

                    iso_n = norm(iso_scores)
                    svm_n = norm(svm_scores)
                    ae_n = norm(mse)
                    
                    agg_scores = (iso_n + svm_n + ae_n) / 3
                    
                    df_to_process['Anomaly_Score'] = agg_scores
                    df_to_process['Decision'] = df_to_process['Anomaly_Score'].apply(lambda x: "⚠️ ANOMALY" if x > 0.7 else "✅ Normal")
                    
                    st.success("Analysis Complete!")
                    st.dataframe(df_to_process.sort_values("Anomaly_Score", ascending=False).style.applymap(
                        lambda x: 'background-color: #ffcdd2' if x == "⚠️ ANOMALY" else '', subset=['Decision']
                    ))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.write(traceback.format_exc())

# --- 9. APP ROUTER ---
if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()
