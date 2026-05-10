from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TensorBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


class TabularFraudDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def to_tensor_bundle(prepared_data) -> TensorBundle:
    return TensorBundle(
        x_train=torch.as_tensor(prepared_data.x_train, dtype=torch.float32),
        y_train=torch.as_tensor(prepared_data.y_train, dtype=torch.float32),
        x_val=torch.as_tensor(prepared_data.x_val, dtype=torch.float32),
        y_val=torch.as_tensor(prepared_data.y_val, dtype=torch.float32),
        x_test=torch.as_tensor(prepared_data.x_test, dtype=torch.float32),
        y_test=torch.as_tensor(prepared_data.y_test, dtype=torch.float32),
    )
