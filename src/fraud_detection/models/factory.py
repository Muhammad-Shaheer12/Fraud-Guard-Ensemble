from __future__ import annotations

from torch import nn

from .cnn_model import CNNFraudModel
from .hybrid_model import HybridFraudModel
from .lstm_model import LSTMFraudModel
from .transformer_model import TransformerFraudModel


def build_model(model_name: str, input_dim: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "cnn":
        return CNNFraudModel()
    if model_name == "lstm":
        return LSTMFraudModel(input_dim=input_dim)
    if model_name == "transformer":
        return TransformerFraudModel(input_dim=input_dim)
    if model_name == "hybrid":
        return HybridFraudModel(feature_dim=input_dim)

    raise ValueError(f"Unsupported model: {model_name}")
