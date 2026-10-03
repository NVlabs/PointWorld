"""
Device detection and compatibility helpers for running PointWorld on Intel XPU.

The release code targets CUDA and hardcodes ``torch.autocast('cuda', ...)``
    and CUDA-specific device APIs. These helpers preserve the original CUDA
    behavior while adding an XPU path for device-sensitive operations.
"""

from __future__ import annotations

import torch


def xpu_is_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def xpu_bf16_is_supported() -> bool:
    return xpu_is_available() and torch.xpu.is_bf16_supported()


def device_type(device) -> str:
    """Normalize a torch.device/str to its type string ('cuda' | 'xpu' | 'cpu')."""
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return device.type


def amp_dtype_for(device) -> torch.dtype:
    """Pick the AMP dtype the way Trainer does: bf16 when supported, fp16 otherwise."""
    dt = device_type(device)
    if dt == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dt == "xpu" and xpu_bf16_is_supported():
        return torch.bfloat16
    if dt == "cuda":
        return torch.float16
    return torch.float32


class autocast_disabled:
    """``torch.autocast(<device>, enabled=False)`` with a valid device_type.

    The original code used ``torch.autocast('cuda', enabled=False)`` to force
    fp32 for specific ops. On XPU that device string is not valid, so resolve
    the type from the actual tensor device.
    """

    def __init__(self, device=None):
        if device is None:
            self.device_type = "cuda"
        else:
            self.device_type = device_type(device)
            if self.device_type not in ("cuda", "xpu"):
                self.device_type = "cuda"

    def __enter__(self):
        self._ctx = torch.autocast(device_type=self.device_type, enabled=False)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def autocast_for_device(device, dtype):
    target = "xpu" if device_type(device) == "xpu" else "cuda"
    return torch.autocast(target, dtype=dtype)