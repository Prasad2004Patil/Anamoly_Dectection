import streamlit as st
import requests
import os
import pandas as pd
# ... (keep your other imports)

API_URL = os.getenv("API_URL", "http://localhost:5000")

# --- 1. Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- 2. Define the Login UI ---
def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 Access Control</h1>", unsafe_allow_html=True)
    
    # Center the form using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        # --- Login Tab ---
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Log In", use_container_width=True)
                
                if submit:
                    try:
                        resp = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
                        if resp.status_code == 200:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success("Login successful! Reloading...")
                            st.rerun()
                        else:
                            st.error(resp.json().get("error", "Login failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")

        # --- Register Tab ---
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                submit_reg = st.form_submit_button("Sign Up", use_container_width=True)
                
                if submit_reg:
                    try:
                        resp = requests.post(f"{API_URL}/auth/register", json={"username": new_user, "password": new_pass})
                        if resp.status_code == 201:
                            st.success("Account created! You can now log in.")
                        else:
                            st.error(resp.json().get("error", "Registration failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")

# --- 3. Define the Main Application ---
def main_app():
    # ... [PASTE YOUR ORIGINAL DASHBOARD CODE HERE] ...
    # From: st.set_page_config(...) 
    # To: The end of your original file
    
    # Add a sidebar logout button
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.rerun()
    
    # (Your existing code follows...)
    st.title("E-commerce Anomaly Detection Dashboard")
    # ... rest of your original logic ...

# --- 4. Execution Flow ---
if not st.session_state.logged_in:
    st.set_page_config(page_title="Login", layout="centered") # Set layout to centered for login
    login_page()
else:
    # Ensure layout is wide for the dashboard
    st.set_page_config(page_title="Anomaly Dashboard", layout="wide") 
    main_app()