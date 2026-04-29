import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from model import LSTMAutoencoder


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        latent_size=config["model"]["latent_size"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    model.load_state_dict(
        torch.load("models/lstm_autoencoder.pth", map_location=device)
    )
    model.eval()

    return model, device


def calculate_error(model, sequence, device):
    sequence_tensor = (
        torch.tensor(sequence, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        reconstructed = model(sequence_tensor)
        error = torch.mean((sequence_tensor - reconstructed) ** 2).item()

    return error


def log_prediction(result):
    log_path = "data/logs/predictions.csv"
    os.makedirs("data/logs", exist_ok=True)

    log_row = {
        "timestamp": datetime.now().isoformat(),
        "reconstruction_error": result["reconstruction_error"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "status": result["status"],
    }

    file_exists = os.path.exists(log_path)

    pd.DataFrame([log_row]).to_csv(
        log_path,
        mode="a",
        header=not file_exists,
        index=False,
    )


def predict_anomaly(sequence):
    config = load_config()
    model, device = load_model(config)

    with open("models/threshold.json", "r") as file:
        threshold = json.load(file)["threshold"]

    sequence = np.array(sequence)
    error = calculate_error(model, sequence, device)

    result = {
        "reconstruction_error": error,
        "threshold": threshold,
        "is_anomaly": error > threshold,
        "status": "ANOMALY" if error > threshold else "NORMAL",
    }

    log_prediction(result)

    return result


if __name__ == "__main__":
    sequences = np.load("data/processed/train_sequences.npy")
    sample_sequence = sequences[0]

    result = predict_anomaly(sample_sequence)

    print(json.dumps(result, indent=4))