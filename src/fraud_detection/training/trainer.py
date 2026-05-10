from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import mlflow
import numpy as np
import torch
import wandb
from mlflow.exceptions import MlflowException
from torch import nn
from torch.utils.data import DataLoader

from ..config.settings import settings
from ..data.datasets import TabularFraudDataset
from ..models.factory import build_model
from ..utils.device import build_amp_components, enforce_gpu_or_fail
from ..utils.metrics import compute_binary_metrics


@dataclass
class TrainingResult:
    model_path: Path
    best_val_f1: float
    final_metrics: dict[str, float]


def _binary_logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


class _NoOpRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_prob = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            all_prob.append(_binary_logits_to_prob(logits).detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

    probs = np.concatenate(all_prob)
    targets = np.concatenate(all_targets)
    metrics = compute_binary_metrics(targets, probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    model_name: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 7,
    accumulation_steps: int = 1,
) -> TrainingResult:
    device = enforce_gpu_or_fail()
    print(f"[INFO] Device check passed: {device} | torch={torch.__version__}", flush=True)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow_enabled = True
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("fraud-detection-dnn")
        mlflow_context = mlflow.start_run(run_name=f"{model_name}-run")
    except Exception:
        mlflow_enabled = False
        mlflow_context = _NoOpRun()

    wandb.init(
        project=settings.wandb_project,
        mode=settings.wandb_mode,
        config={"model": model_name, "epochs": epochs, "batch_size": batch_size, "lr": lr},
        reinit=True,
    )

    train_ds = TabularFraudDataset(x_train, y_train)
    val_ds = TabularFraudDataset(x_val, y_val)
    test_ds = TabularFraudDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_name=model_name, input_dim=x_train.shape[1]).to(device)
    print(
        f"[INFO] Model initialized: {model_name} | input_dim={x_train.shape[1]} | "
        f"train_batches={len(train_loader)}",
        flush=True,
    )

    pos_weight_value = (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled, scaler = build_amp_components(device)

    best_state = None
    best_val_f1 = -1.0
    wait = 0

    with mlflow_context:
        if mlflow_enabled:
            mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(device),
                "torch_version": torch.__version__,
            }
        )

        for epoch in range(1, epochs + 1):
            epoch_t0 = time.perf_counter()
            model.train()
            train_loss = 0.0
            
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * accumulation_steps * x_batch.size(0)

            epoch_train_loss = train_loss / len(train_loader.dataset)
            val_metrics = _evaluate(model, val_loader, criterion, device)
            epoch_time = time.perf_counter() - epoch_t0
            print(
                "[EPOCH {}/{}] train_loss={:.5f} val_loss={:.5f} val_f1={:.5f} val_auc={:.5f} time={:.1f}s".format(
                    epoch,
                    epochs,
                    epoch_train_loss,
                    val_metrics["loss"],
                    val_metrics["f1"],
                    val_metrics["roc_auc"],
                    epoch_time,
                ),
                flush=True,
            )

            if mlflow_enabled:
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_roc_auc": val_metrics["roc_auc"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1"],
                    },
                    step=epoch,
                )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                }
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = _evaluate(model, test_loader, criterion, device)
        print(
            "[INFO] Test metrics | loss={:.5f} auc={:.5f} precision={:.5f} recall={:.5f} f1={:.5f}".format(
                test_metrics["loss"],
                test_metrics["roc_auc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
            ),
            flush=True,
        )
        if mlflow_enabled:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

        model_path = settings.model_dir / f"{model_name}_best.pt"
        checkpoint = {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "metrics": test_metrics,
        }
        torch.save(checkpoint, model_path)
        if mlflow_enabled:
            try:
                mlflow.log_artifact(str(model_path))
            except MlflowException:
                pass

    wandb.finish()
    return TrainingResult(model_path=model_path, best_val_f1=best_val_f1, final_metrics=test_metrics)
