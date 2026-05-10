from __future__ import annotations

import torch
import torch.nn.functional as F
from tab_transformer_pytorch import TabTransformer
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .cnn_model import tabular_to_image
class HybridFraudModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        try:
            self.cnn_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained ResNet-18 weights for the Hybrid model. "
                "This project requires pretrained backbones for all models."
            ) from exc

        cnn_dim = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()

        self.tab_branch = TabTransformer(
            categories=(),
            num_continuous=feature_dim,
            dim=64,
            depth=3,
            heads=8,
            dim_out=hidden_dim,
            attn_dropout=dropout,
            ff_dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = tabular_to_image(x)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        cnn_features = self.cnn_backbone(image)
        x_categ = torch.empty((x.size(0), 0), dtype=torch.long, device=x.device)
        tab_features = self.tab_branch(x_categ=x_categ, x_cont=x)

        fused = torch.cat([cnn_features, tab_features], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(dim=-1)
