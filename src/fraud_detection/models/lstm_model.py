from __future__ import annotations

import torch
from torch import nn


class LSTMFraudModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        del input_dim

        self.input_projection = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_lstm_weights()

    def _init_lstm_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.input_projection(x.unsqueeze(-1))
        output, _ = self.lstm(seq)
        pooled = output[:, -1, :]
        logits = self.classifier(pooled)
        return logits.squeeze(dim=-1)
