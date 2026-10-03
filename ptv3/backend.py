from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PTv3Backend:
    device_type: str
    SparseConvTensor: Any
    SubMConv3d: Any
    is_spconv_module: Callable[[Any], bool]
    segment_csr: Callable[..., Any]
    flash_attn_varlen_qkvpacked_func: Callable[..., Any] | None


_backend: PTv3Backend | None = None


def configure(device: Any) -> PTv3Backend:
    global _backend
    device_type = device.split(":", 1)[0] if isinstance(device, str) else device.type
    if device_type not in ("cuda", "xpu"):
        device_type = "cuda"
    if _backend is not None:
        if _backend.device_type != device_type:
            raise RuntimeError(
                f"PTv3 backend already configured for {_backend.device_type}, "
                f"cannot switch to {device_type}"
            )
        return _backend

    if device_type == "xpu":
        from . import torch_scatter_shim
        from .spconv_shim import SparseConvTensor, SubMConv3d, is_spconv_module

        _backend = PTv3Backend(
            device_type="xpu",
            SparseConvTensor=SparseConvTensor,
            SubMConv3d=SubMConv3d,
            is_spconv_module=is_spconv_module,
            segment_csr=torch_scatter_shim.segment_csr,
            flash_attn_varlen_qkvpacked_func=None,
        )
    else:
        import spconv.pytorch as spconv
        import torch_scatter
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        _backend = PTv3Backend(
            device_type="cuda",
            SparseConvTensor=spconv.SparseConvTensor,
            SubMConv3d=spconv.SubMConv3d,
            is_spconv_module=spconv.modules.is_spconv_module,
            segment_csr=torch_scatter.segment_csr,
            flash_attn_varlen_qkvpacked_func=flash_attn_varlen_qkvpacked_func,
        )
    return _backend


def get_backend() -> PTv3Backend:
    if _backend is None:
        return configure("cuda")
    return _backend