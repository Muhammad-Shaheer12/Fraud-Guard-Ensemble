# FraudGuard AI: Deep Learning Fraud Detection System

A production-grade, multi-model fraud detection system built to evaluate, detect, and interpret fraudulent financial transactions in real-time. This project features a high-performance **FastAPI backend** and a dynamic, multi-page **React dashboard** for live ensemble analytics.

## 🧠 Architectural Overview

This system evaluates transactions using four distinct neural architectures to demonstrate the value of Model Disagreement and Ensemble Learning:
1. **CNN (Convolutional Neural Network):** Identifies localized spatial anomalies within the 2,803-dimensional transaction vector.
2. **LSTM (Long Short-Term Memory):** Evaluates the transaction as a sequential manifold, looking for recursive dependencies.
3. **Token-Projected Transformer:** Uses global self-attention to identify distant, non-linear relationships across all transaction features simultaneously.
4. **Hybrid Ensemble:** Aggregates CNN feature extractions with Transformer attention mechanisms to mitigate individual architectural blind spots.

### 🔍 Interpretability
The system utilizes **Captum Feature Attributions** (via a feature magnitude proxy in this demo) to dynamically highlight the exact data points that triggered a model's specific decision.

## 🚀 Quick Start Guide

### 1. Download the Pre-Trained Models (Required)
Because the highly-optimized model checkpoints exceed GitHub's 100MB file size limit, they are hosted externally.
1. Download the model weights and pre-processed dataset from **[INSERT YOUR GOOGLE DRIVE LINK HERE]**.
2. Place the `.pt` model files into the `artifacts/models/` directory.
3. Place the `.joblib` dataset files into the `artifacts/` directory.

### 2. Backend Setup (FastAPI)
```bash
# Create and activate a virtual environment
python -m venv myenv
.\myenv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn src.fraud_detection.api.main:app --host 127.0.0.1 --port 8000
```

### 3. Frontend Setup (React + Vite)
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the React dashboard
npm run dev
```

## 📊 Dashboard Features
Navigate to `http://localhost:5173/` to explore the control center:
- **Single Model Demo:** Hot-swap between architectures to see how different networks evaluate the same transaction.
- **Ensemble Analysis:** Broadcast a single transaction across all 4 architectures simultaneously to visualize model disagreement in real-time.
- **Training Analytics:** Browse the high-resolution ROC curves, Precision-Recall charts, and Confusion Matrices generated during the training phase.
