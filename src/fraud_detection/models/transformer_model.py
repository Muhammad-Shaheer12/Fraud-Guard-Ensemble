from __future__ import annotations

import torch
from torch import nn

class TransformerFraudModel(nn.Module):
    def __init__(self, input_dim: int, d_out: int = 1, pretrained_tokenizer_path: str | None = None):
        super().__init__()
        self.d_model = 128
        self.seq_len = 16
        hidden_dim = self.d_model * self.seq_len  # 2048
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, d_out)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x)
        x = x.view(x.size(0), self.seq_len, self.d_model)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(dim=-1)
