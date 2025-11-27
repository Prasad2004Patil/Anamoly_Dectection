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
