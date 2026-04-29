import os
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from model import LSTMAutoencoder


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def calculate_reconstruction_errors(model, dataloader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)

            batch_errors = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2))
            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def evaluate_model(config_path="config/config.yaml"):
    config = load_config(config_path)

    sequences = np.load(os.path.join(config["data"]["processed_dir"], "train_sequences.npy"))

    dataset = SensorSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        latent_size=config["model"]["latent_size"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    model.load_state_dict(torch.load("models/lstm_autoencoder.pth", map_location=device))

    errors = calculate_reconstruction_errors(model, dataloader, device)

    threshold = float(np.mean(errors) + 3 * np.std(errors))

    metrics = {
        "mean_reconstruction_error": float(np.mean(errors)),
        "std_reconstruction_error": float(np.std(errors)),
        "anomaly_threshold": threshold,
        "max_reconstruction_error": float(np.max(errors)),
        "min_reconstruction_error": float(np.min(errors)),
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/evaluation_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open("models/threshold.json", "w") as file:
        json.dump({"threshold": threshold}, file, indent=4)

    print("Evaluation completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate_model()