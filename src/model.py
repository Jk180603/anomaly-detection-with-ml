import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, num_layers: int = 1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.latent = nn.Linear(hidden_size, latent_size)

        self.decoder_input = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        _, (hidden, _) = self.encoder(x)

        encoded = hidden[-1]
        latent_vector = self.latent(encoded)

        decoder_hidden = self.decoder_input(latent_vector)
        decoder_input = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(decoded)

        return reconstructed