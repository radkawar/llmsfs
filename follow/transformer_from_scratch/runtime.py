import os
import sys
from typing import Literal


# These flags must be set before PyTorch initializes its MPS backend. Fast math
# permits faster approximations; prefer-metal routes matrix multiplications to
# native Metal kernels instead of MPS Graph where PyTorch supports that path.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")
    os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

import torch


Precision = Literal["auto", "float32", "float16", "bfloat16"]


def select_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None and str(preferred) != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device, precision: Precision) -> torch.dtype | None:
    if precision == "float32":
        return None
    if precision == "auto":
        if device.type == "mps":
            # BF16 is fast on recent Apple Silicon and has FP32-like exponent
            # range, so it does not require gradient scaling.
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return None

    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise RuntimeError(f"Automatic mixed precision is not available on {device.type}")
    return dtype


def configure_runtime(device: torch.device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
