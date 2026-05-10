from __future__ import annotations

import atexit
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np

from ..config.settings import settings
from ..data.preprocessing import FraudPreprocessor
from .trainer import train_model


def _dataset_paths(dataset: str) -> tuple[Path, Path | None]:
    if dataset == "ieee":
        trans = Path("ieee-fraud-detection") / "train_transaction.csv"
        ident = Path("ieee-fraud-detection") / "train_identity.csv"
        return trans, ident
    if dataset == "creditcard":
        return Path("Credit Card Dataset") / "creditcard.csv", None
    if dataset == "paysim":
        return (
            Path("Synthetic Financial Datasets For Fraud Detection") / "PS_20174392719_1491204439457_log.csv",
            None,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _downsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(x) <= max_samples:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def _run_lock_path(dataset: str, model: str) -> Path:
    return settings.artifact_dir / "locks" / f"{dataset}_{model}.lock.json"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        # Use tasklist for robust PID checks across permission boundaries.
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if "No tasks are running" in output:
            return False
        return str(pid) in output

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_run_lock(dataset: str, model: str, force_reset: bool = False) -> Path:
    lock_path = _run_lock_path(dataset, model)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        if force_reset:
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "Existing run lock found for "
                f"dataset={dataset} model={model}. "
                "Refusing to start a duplicate run. "
                "If you are sure this is stale, rerun with --force-lock-reset."
            )

    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            lock_data = {}

        existing_pid = int(lock_data.get("pid", -1))
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                "Another training run is already active for "
                f"dataset={dataset} model={model} (pid={existing_pid})."
            )

        # Stale lock from a previous interrupted run.
        lock_path.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "dataset": dataset,
        "model": model,
        "started_at_unix": time.time(),
    }
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lock_path


def _release_run_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return

    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        lock_path.unlink(missing_ok=True)
        return

    # Only remove our own lock file.
    if int(lock_data.get("pid", -1)) == os.getpid():
        lock_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fraud detection deep learning models on GPU")
    parser.add_argument("--dataset", choices=["ieee", "creditcard", "paysim"], required=True)
    parser.add_argument("--model", choices=["cnn", "lstm", "transformer", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulate", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If > 0, randomly downsample training data to this size for quick smoke runs.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="If > 0, randomly downsample val/test data to this size for quick smoke runs.",
    )
    parser.add_argument(
        "--force-lock-reset",
        action="store_true",
        help="Forcefully remove an existing run lock before starting training.",
    )
    args = parser.parse_args()

    try:
        lock_path = _acquire_run_lock(args.dataset, args.model, force_reset=args.force_lock_reset)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 2

    atexit.register(_release_run_lock, lock_path)

    print(
        f"[INFO] Starting training | dataset={args.dataset} model={args.model} "
        f"epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"[INFO] Run lock: {lock_path}", flush=True)

    preprocessor = FraudPreprocessor()
    primary_path, secondary_path = _dataset_paths(args.dataset)
    print(f"[INFO] Loading and preprocessing dataset from: {primary_path}", flush=True)
    t0 = time.perf_counter()

    if args.dataset == "ieee":
        prepared = preprocessor.prepare_ieee(primary_path, secondary_path)
    elif args.dataset == "creditcard":
        prepared = preprocessor.prepare_creditcard(primary_path)
    else:
        prepared = preprocessor.prepare_paysim(primary_path)

    print(
        f"[INFO] Preprocessing done in {time.perf_counter() - t0:.1f}s | "
        f"train={len(prepared.x_train)} val={len(prepared.x_val)} test={len(prepared.x_test)}",
        flush=True,
    )

    if args.max_train_samples > 0:
        prepared.x_train, prepared.y_train = _downsample(
            prepared.x_train,
            prepared.y_train,
            max_samples=args.max_train_samples,
            seed=42,
        )
        print(f"[INFO] Downsampled train to {len(prepared.x_train)} samples", flush=True)

    if args.max_eval_samples > 0:
        prepared.x_val, prepared.y_val = _downsample(
            prepared.x_val,
            prepared.y_val,
            max_samples=args.max_eval_samples,
            seed=43,
        )
        prepared.x_test, prepared.y_test = _downsample(
            prepared.x_test,
            prepared.y_test,
            max_samples=args.max_eval_samples,
            seed=44,
        )
        print(
            f"[INFO] Downsampled eval to val={len(prepared.x_val)} test={len(prepared.x_test)}",
            flush=True,
        )

    try:
        result = train_model(
            x_train=prepared.x_train,
            y_train=prepared.y_train,
            x_val=prepared.x_val,
            y_val=prepared.y_val,
            x_test=prepared.x_test,
            y_test=prepared.y_test,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            accumulation_steps=args.accumulate,
        )

        scaler_path = settings.artifact_dir / f"{args.dataset}_scaler.joblib"
        feature_path = settings.artifact_dir / f"{args.dataset}_features.npy"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor.scaler, scaler_path)
        np.save(feature_path, np.array(prepared.feature_names, dtype=object))

        print(f"Saved model: {result.model_path}")
        print(f"Best val F1: {result.best_val_f1:.4f}")
        print("Final test metrics:", result.final_metrics)
        return 0
    finally:
        _release_run_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
