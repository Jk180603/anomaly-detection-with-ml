import os
import yaml
import mlflow
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import SensorSequenceDataset
from model import LSTMAutoencoder


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model(config_path="config/config.yaml"):
    config = load_config(config_path)

    processed_dir = config["data"]["processed_dir"]
    test_size = config["data"]["test_size"]

    input_size = config["model"]["input_size"]
    hidden_size = config["model"]["hidden_size"]
    latent_size = config["model"]["latent_size"]
    num_layers = config["model"]["num_layers"]

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    sequences_path = os.path.join(processed_dir, "train_sequences.npy")
    sequences = np.load(sequences_path)

    dataset = SensorSequenceDataset(sequences)

    val_size = int(len(dataset) * test_size)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment("industrial-anomaly-detection")

    with mlflow.start_run():
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("latent_size", latent_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)

                    reconstructed = model(batch_x)
                    loss = criterion(reconstructed, batch_x)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.6f} "
                f"Val Loss: {avg_val_loss:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                torch.save(
                    model.state_dict(),
                    "models/lstm_autoencoder.pth",
                )

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact("models/lstm_autoencoder.pth")

        print("Training completed.")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("Model saved at: models/lstm_autoencoder.pth")


if __name__ == "__main__":
    train_model()