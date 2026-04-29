# ⚙️ Industrial Anomaly Detection System with MLOps



A production-ready **AI/ML anomaly detection system** using an **LSTM Autoencoder** on NASA turbofan engine sensor data.  
The project includes model training, real-time inference, Streamlit UI, FastAPI backend, prediction logging, monitoring, automated retraining, Docker, and GitHub Actions CI.

## 📸 Demo

### 🖥️ Dashboard (Input + Prediction)
![Dashboard](images/dashboard.png)

- Users can paste sensor sequence JSON or select sample data  
- Real-time prediction using trained LSTM Autoencoder  
- Displays reconstruction error, threshold, and status  

---

### 🔴 Anomaly Detection (Case 1)
![Anomaly 1](images/anomaly1.png)

- High reconstruction error compared to threshold  
- Indicates abnormal sensor behavior  
- Status: **ANOMALY**

---

### 🔴 Anomaly Detection (Case 2)
![Anomaly 2](images/anomaly2.png)

- Significant deviation from learned normal patterns  
- Simulates critical system fault scenario  
- Status: **ANOMALY**

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

## 🧠 Model Approach

The model learns normal sensor behavior. During inference:

- Low reconstruction error → **NORMAL**  
- High reconstruction error → **ANOMALY**

### Example

```json
{
  "reconstruction_error": 21.198053,
  "threshold": 0.379192,
  "is_anomaly": true,
  "status": "ANOMALY"
}
```
## Tech Stack

Python, PyTorch
FastAPI
Streamlit
Pandas, NumPy, Scikit-learn
Docker & Docker Compose
Pytest
GitHub Actions

## Run Locally

Install dependencies:
pip install -r requirements.txt

Run FastAPI:
uvicorn app.main:app --reload

##Run Streamlit
streamlit run streamlit/streamlit_app.py


## Run with Docker
docker compose up

See  process and then run fastapi and streamlit 
