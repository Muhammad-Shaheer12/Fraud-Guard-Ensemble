"""Generate evaluation visualizations for the semester report.

Usage:
    python -m src.fraud_detection.training.evaluate --dataset ieee

Produces:
    artifacts/plots/confusion_matrix_<model>.png
    artifacts/plots/roc_curve_<model>.png
    artifacts/plots/precision_recall_<model>.png
    artifacts/plots/roc_overlay.png
    artifacts/plots/model_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..config.settings import settings
from ..data.datasets import TabularFraudDataset
from ..models.factory import build_model

# ── Style ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#161b22",
        "text.color": "#c9d1d9",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "grid.color": "#21262d",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "font.family": "sans-serif",
    }
)

MODEL_NAMES = ["cnn", "lstm", "transformer", "hybrid"]
PALETTE = {"cnn": "#58a6ff", "lstm": "#f97583", "transformer": "#d2a8ff", "hybrid": "#56d364"}


def _load_model_and_predict(model_name: str, x_test: np.ndarray, y_test: np.ndarray, device: torch.device):
    """Load checkpoint, run inference, return (y_true, y_prob)."""
    ckpt_path = settings.model_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"  ⚠ Checkpoint not found: {ckpt_path}, skipping {model_name}")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    try:
        model = build_model(model_name=model_name, input_dim=input_dim).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    except Exception as exc:
        print(f"  [ERROR] Failed to load {model_name} weights: {exc}")
        return None, None

    ds = TabularFraudDataset(x_test, y_test)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    all_prob, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_prob.append(prob)
            all_true.append(y_batch.numpy())

    return np.concatenate(all_true), np.concatenate(all_prob)


def plot_confusion_matrix(y_true, y_prob, model_name, out_dir):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}", fontweight="bold", fontsize=14)
    fig.savefig(out_dir / f"confusion_matrix_{model_name}.png")
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved: confusion_matrix_{model_name}.png")


def plot_roc_curve(y_true, y_prob, model_name, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE.get(model_name, "#58a6ff"), linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name.upper()}", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.savefig(out_dir / f"roc_curve_{model_name}.png")
    plt.close(fig)
    print(f"  [OK] ROC curve saved: roc_curve_{model_name}.png")


def plot_precision_recall(y_true, y_prob, model_name, out_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE.get(model_name, "#58a6ff"), linewidth=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name.upper()}", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.savefig(out_dir / f"precision_recall_{model_name}.png")
    plt.close(fig)
    print(f"  [OK] PR curve saved: precision_recall_{model_name}.png")


def plot_roc_overlay(results: dict, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE.get(name, "#ccc"), linewidth=2.5, label=f"{name.upper()} (AUC={roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontweight="bold", fontsize=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.savefig(out_dir / "roc_overlay.png")
    plt.close(fig)
    print(f"  [OK] ROC overlay saved: roc_overlay.png")


def plot_model_comparison(results: dict, out_dir):
    names = []
    metrics_data = {"AUC": [], "F1": [], "Precision": [], "Recall": []}
    for name, (y_true, y_prob) in results.items():
        y_pred = (y_prob >= 0.5).astype(int)
        names.append(name.upper())
        metrics_data["AUC"].append(roc_auc_score(y_true, y_prob))
        metrics_data["F1"].append(f1_score(y_true, y_pred, zero_division=0))
        metrics_data["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics_data["Recall"].append(recall_score(y_true, y_pred, zero_division=0))

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_colors = ["#58a6ff", "#56d364", "#d2a8ff", "#f97583"]
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=metric_colors[i], alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c9d1d9",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set Metrics", fontweight="bold", fontsize=15)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim([0, 1.12])
    fig.savefig(out_dir / "model_comparison.png")
    plt.close(fig)
    print(f"  [OK] Model comparison saved: model_comparison.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation plots for the semester report")
    parser.add_argument("--dataset", choices=["ieee", "creditcard", "paysim"], default="ieee")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    cache_path = settings.artifact_dir / f"{args.dataset}_preprocessed.joblib"
    if not cache_path.exists():
        print(f"[ERROR] No cached dataset found at {cache_path}. Run training first.")
        return 1

    print(f"[INFO] Loading cached dataset from: {cache_path}")
    prepared, _ = joblib.load(cache_path)

    out_dir = settings.artifact_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in MODEL_NAMES:
        print(f"\n[INFO] Evaluating {model_name.upper()}...")
        y_true, y_prob = _load_model_and_predict(model_name, prepared.x_test, prepared.y_test, device)
        if y_true is None:
            continue
        results[model_name] = (y_true, y_prob)
        plot_confusion_matrix(y_true, y_prob, model_name, out_dir)
        plot_roc_curve(y_true, y_prob, model_name, out_dir)
        plot_precision_recall(y_true, y_prob, model_name, out_dir)

    if len(results) >= 2:
        print("\n[INFO] Generating combined comparison plots...")
        plot_roc_overlay(results, out_dir)
        plot_model_comparison(results, out_dir)

    print(f"\n[DONE] All plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
