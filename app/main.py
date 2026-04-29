from fastapi import FastAPI
from app.schemas import SensorSequenceRequest
from app.inference_service import run_prediction


app = FastAPI(
    title="Industrial Sensor Anomaly Detection API",
    description="LSTM Autoencoder API for detecting anomalies in engine sensor sequences.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "Industrial Anomaly Detection API is running"}


@app.post("/predict")
def predict(request: SensorSequenceRequest):
    result = run_prediction(request.sequence)
    return result