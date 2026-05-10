import torch


MIN_TORCH_VERSION = (2, 7)


def _parse_version(version: str) -> tuple[int, int]:
    parts = version.split("+")[0].split(".")
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    return major, minor


def enforce_gpu_or_fail() -> torch.device:
    version = _parse_version(torch.__version__)
    if version < MIN_TORCH_VERSION:
        raise RuntimeError(
            f"PyTorch {torch.__version__} detected. Use PyTorch >= 2.7.0 for RTX 50-series compatibility."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this project. No CUDA device detected. "
            "Install compatible CUDA-enabled PyTorch and run on your RTX 5070."
        )

    return torch.device("cuda")


def build_amp_components(device: torch.device):
    enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    return enabled, scaler
