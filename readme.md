# ⚙️ Industrial Anomaly Detection System with MLOps

![Streamlit Demo](images/streamlit.png)

A production-ready **AI/ML anomaly detection system** using an **LSTM Autoencoder** on NASA turbofan engine sensor data.  
The project includes model training, real-time inference, Streamlit UI, FastAPI backend, prediction logging, monitoring, automated retraining, Docker, and GitHub Actions CI.

---

## 🚀 Project Highlights

- Built an LSTM Autoencoder for time-series anomaly detection
- Used NASA turbofan engine sensor data
- Added FastAPI backend for real-time prediction
- Built Streamlit UI for interactive testing
- Added prediction logging
- Generated monitoring reports from inference logs
- Added automated retraining trigger based on anomaly rate
- Dockerized FastAPI and Streamlit services
- Added unit tests and GitHub Actions CI pipeline

---

## 📌 Use Case

This project simulates an **industrial predictive maintenance system**.

It detects abnormal sensor behavior in machines such as:

- aircraft engines
- manufacturing systems
- industrial IoT devices
- automotive components
- production equipment

---

## 🏗️ Architecture

```text
NASA Sensor Data
      ↓
Preprocessing + Sliding Windows
      ↓
LSTM Autoencoder
      ↓
Reconstruction Error
      ↓
Anomaly Threshold
      ↓
NORMAL / ANOMALY Prediction
      ↓
FastAPI + Streamlit
      ↓
Prediction Logs
      ↓
Monitoring Report
      ↓
Automatic Retraining Trigger
```

#Model Approach

The model learns normal sensor behavior.
During inference:

Low reconstruction error → NORMAL
High reconstruction error → ANOMALY

Example:

{
  "reconstruction_error": 21.198053,
  "threshold": 0.379192,
  "is_anomaly": true,
  "status": "ANOMALY"
}

#Tech Stack

Python, PyTorch
FastAPI
Streamlit
Pandas, NumPy, Scikit-learn
Docker & Docker Compose
Pytest
GitHub Actions

#Run Locally

Install dependencies:
pip install -r requirements.txt

Run FastAPI:
uvicorn app.main:app --reload

Run Streamlit
streamlit run streamlit/streamlit_app.py


#Run with Docker
docker compose up

See  process and then run fastapi and streamlit 
