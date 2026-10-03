"""
Pure-PyTorch replacement for ``torch_scatter.segment_csr`` used by PTv3
GridPooling. torch-scatter ships CUDA/C++ kernels only and cannot be built
for the Intel XPU backend.

``segment_csr(src, idx_ptr, reduce)`` reduces consecutive CSR segments of
``src`` (boundaries given by ``idx_ptr``, length S+1) with ``reduce`` in
{"sum", "mean", "min", "max"} and returns (S, *src.shape[1:]).
"""

from __future__ import annotations

import torch


def segment_csr(src: torch.Tensor, idx_ptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    idx_ptr = idx_ptr.to(dtype=torch.long)
    S = idx_ptr.numel() - 1
    N = src.shape[0]
    device = src.device
    shape = (S,) + tuple(src.shape[1:])

    if S == 0:
        return torch.empty(shape, device=device, dtype=src.dtype)

    # Segment id of each element: idx_ptr[i] <= pos < idx_ptr[i+1] -> i.
    seg_ids = torch.searchsorted(idx_ptr, torch.arange(N, device=device), right=True) - 1
    seg_ids = seg_ids.clamp(min=0, max=S - 1)

    counts = torch.bincount(seg_ids, minlength=S)

    # Only elements within [0, idx_ptr[-1]) belong to a segment (mirrors
    # torch_scatter, where idx_ptr[-1] == src.shape[0]).
    last = int(idx_ptr[-1].item())
    if last < N:
        src = src[:last]
        seg_ids = seg_ids[:last]
        counts = counts.clamp(max=0)  # placeholder, recomputed below
        counts = torch.bincount(seg_ids, minlength=S)
        N = last

    if reduce == "sum":
        out = torch.zeros(shape, device=device, dtype=src.dtype)
        idx = seg_ids
        for _ in range(src.dim() - 1):
            idx = idx.unsqueeze(-1)
        out.scatter_add_(0, idx.expand((N,) + tuple(src.shape[1:])), src)
        return out

    if reduce == "mean":
        # Integer inputs average to float (mirrors torch_scatter behavior).
        compute_dtype = src.dtype if src.dtype.is_floating_point else torch.float32
        out = torch.zeros(shape, device=device, dtype=compute_dtype)
        idx = seg_ids
        for _ in range(src.dim() - 1):
            idx = idx.unsqueeze(-1)
        out.scatter_add_(0, idx.expand((N,) + tuple(src.shape[1:])), src.to(compute_dtype))
        denom = counts.to(compute_dtype).clamp_min(1)
        for _ in range(src.dim() - 1):
            denom = denom.unsqueeze(-1)
        return out / denom.expand(shape)

    if reduce in ("max", "min"):
        if reduce == "max":
            init = -torch.finfo(src.dtype).max if src.dtype.is_floating_point else -torch.iinfo(src.dtype).max
        else:
            init = torch.finfo(src.dtype).max if src.dtype.is_floating_point else torch.iinfo(src.dtype).max
        out = torch.full(shape, init, device=device, dtype=src.dtype)
        idx = seg_ids
        for _ in range(src.dim() - 1):
            idx = idx.unsqueeze(-1)
        out.scatter_reduce_(0, idx.expand((N,) + tuple(src.shape[1:])), src, reduce=reduce, include_self=False)
        empty = (counts == 0)
        for _ in range(src.dim() - 1):
            empty = empty.unsqueeze(-1)
        return torch.where(empty.expand(shape), torch.zeros(1, device=device, dtype=src.dtype), out)

    raise ValueError(f"Unsupported reduce op '{reduce}' (expected sum/mean/min/max)")