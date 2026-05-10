from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def tabular_to_image(x: torch.Tensor) -> torch.Tensor:
    batch_size, n_features = x.shape
    side = math.ceil(math.sqrt(n_features))
    total = side * side

    if total > n_features:
        pad = torch.zeros((batch_size, total - n_features), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)

    image = x.view(batch_size, 1, side, side)
    image = image.repeat(1, 3, 1, 1)
    return image


class CNNFraudModel(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        try:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained ResNet-18 weights. "
                "This project requires pretrained backbones for all models."
            ) from exc
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = tabular_to_image(x)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.backbone(image)
        return logits.squeeze(dim=-1)
