"""
Pure-PyTorch replacement for the spconv (sparse-convolution) pieces used by PTv3.

spconv (spconv-cu124) ships CUDA/C++ kernels only and cannot be built for the
Intel XPU backend. This shim re-implements the two spconv APIs consumed by
this repo in pure PyTorch so the identical checkpoints (which store
``SubMConv3d.weight`` with spconv's (27, in, out) layout) load and run 1:1 on
XPU/CPU/CUDA:

- ``SparseConvTensor``: features + indices (batch, x, y, z) container with
  ``replace_feature``.
- ``SubMConv3d``: 3x3x3 submanifold convolution over sparse points. Neighbor
  lookup uses an int64 voxel-key + ``searchsorted`` (no Python loops over
  points), and the 27 kernel taps are applied as one batched GEMM.
- ``is_spconv_module``: predicate used by ``PointSequential``.

The weight parameter uses the standard conv3d layout
``(out_channels, 3, 3, 3, in_channels)`` — the exact layout the released
PointWorld checkpoints store — so state_dicts load 1:1. The 27-tap order
under ``weight.view(Cout, 27, Cin)`` is row-major over (kz, ky, kx)
(index = (dz+1)*9 + (dy+1)*3 + (dx+1), x fastest), which is exactly the
neighbor-gather order used below, so the result matches a dense 3x3x3
convolution at the point locations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Voxel-key layout: 20 bits per spatial axis (max ~1M voxels per axis at the
# finest grid) + remaining high bits for the batch index. Grid coordinates in
# this codebase are non-negative (coord - min, then // grid_size).
_BITS = 20
_MASK = (1 << _BITS) - 1


def _encode_keys(indices: torch.Tensor) -> torch.Tensor:
    """(N, 4) int indices (b, x, y, z) -> int64 composite key."""
    b = indices[:, 0].long() & _MASK
    x = indices[:, 1].long() & _MASK
    y = indices[:, 2].long() & _MASK
    z = indices[:, 3].long() & _MASK
    return (b << (3 * _BITS)) | (x << (2 * _BITS)) | (y << _BITS) | z


_OFFSETS = [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
# Key delta for each (dx, dy, dz) tap under the composite key layout above.
_OFFSET_DELTAS = torch.tensor(
    [dz * (1 << (2 * _BITS)) + dy * (1 << _BITS) + dx for (dx, dy, dz) in _OFFSETS],
    dtype=torch.int64,
)


class SparseConvTensor:
    """Minimal stand-in for ``spconv.SparseConvTensor``."""

    def __init__(self, features: torch.Tensor, indices: torch.Tensor,
                 spatial_shape, batch_size: int):
        self.features = features
        self.indices = indices.contiguous()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensor":
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SparseConvTensor(features={tuple(self.features.shape)}, "
            f"indices={tuple(self.indices.shape)}, spatial_shape={self.spatial_shape}, "
            f"batch_size={self.batch_size})"
        )


class SubMConv3d(nn.Module):
    """3x3x3 submanifold convolution (pure PyTorch).

    Numerically equivalent to a dense 3x3x3 convolution evaluated at the
    sparse point locations (missing neighbors contribute zero, as in spconv's
    submanifold padding). Weight layout matches the released checkpoints:
    (out_channels, 3, 3, 3, in_channels).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3,
                 bias: bool = True, indice_key=None):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("SubMConv3d shim only supports kernel_size=3")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        # PointTransformerV3._init_weights re-applies trunc_normal_/zeros_
        # after construction; initialize to something finite regardless.
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        feat = x.features
        indices = x.indices
        N = feat.shape[0]
        if N == 0:
            return SparseConvTensor(feat, indices, x.spatial_shape, x.batch_size)

        keys = _encode_keys(indices)
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        device = feat.device

        deltas = _OFFSET_DELTAS.to(device)
        # Gather the 27 neighbor features: (27, N, C); zero where the neighbor
        # voxel is absent (spconv treats missing submanifold neighbors as 0).
        gathered = torch.empty(27, N, feat.shape[1], device=device, dtype=feat.dtype)
        for d in range(27):
            shifted = keys + deltas[d]
            pos = torch.searchsorted(sorted_keys, shifted)
            pos = pos.clamp(max=N - 1)
            found = sorted_keys[pos] == shifted
            nidx = order[pos]
            gathered[d] = torch.where(found.unsqueeze(-1), feat[nidx], 0)

        # out[n, co] = sum_d gathered[d, n, ci] * W[co, d, ci] + bias[co]
        # weight.view(Cout, 27, Cin): row-major over (kz, ky, kx) == tap order d.
        w_taps = self.weight.view(self.out_channels, 27, self.in_channels).permute(1, 2, 0).to(feat.dtype)
        out = torch.bmm(gathered, w_taps)  # (27, N, Cout)
        out = out.sum(dim=0)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return SparseConvTensor(out, indices, x.spatial_shape, x.batch_size)


def is_spconv_module(module) -> bool:
    """Mirror of ``spconv.modules.is_spconv_module`` for this shim."""
    return isinstance(module, (SubMConv3d,))