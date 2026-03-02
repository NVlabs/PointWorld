# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


def _ensure_float32(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def _ensure_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr


@dataclass(slots=True)
class FlowTimeline:
    """Stores polyline segments along with visibility per frame."""

    segments: np.ndarray  # (M, 2, 3)
    colors: np.ndarray  # (M, 2, 3)
    _indices_per_frame: List[np.ndarray]

    @classmethod
    def empty(cls, num_frames: int) -> "FlowTimeline":
        per_frame = [np.empty((0,), dtype=np.int64) for _ in range(num_frames)]
        return cls(
            segments=np.empty((0, 2, 3), dtype=np.float32),
            colors=np.empty((0, 2, 3), dtype=np.uint8),
            _indices_per_frame=per_frame,
        )

    @classmethod
    def from_segments(
        cls,
        segments: np.ndarray,
        colors: np.ndarray,
        end_frames: np.ndarray,
        num_frames: int,
    ) -> "FlowTimeline":
        segs = _ensure_float32(segments)
        cols = _ensure_uint8(colors)
        ends = np.asarray(end_frames, dtype=np.int64)
        if segs.ndim != 3 or segs.shape[1:] != (2, 3):
            raise ValueError("segments must have shape (M, 2, 3)")
        if cols.shape != segs.shape:
            raise ValueError("colors shape mismatch with segments")
        if ends.shape[0] != segs.shape[0]:
            raise ValueError("end_frames length mismatch")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        per_frame: List[np.ndarray] = []
        for frame in range(num_frames):
            mask = ends <= frame
            if not np.any(mask):
                per_frame.append(np.empty((0,), dtype=np.int64))
            else:
                per_frame.append(np.nonzero(mask)[0].astype(np.int64))
        return cls(segs, cols, per_frame)

    @property
    def num_frames(self) -> int:
        return len(self._indices_per_frame)

    def slice_for_frame(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        if not (0 <= frame < self.num_frames):
            raise IndexError(f"frame {frame} out of range [0, {self.num_frames - 1}]")
        idx = self._indices_per_frame[frame]
        if idx.size == 0:
            return (
                np.empty((0, 2, 3), dtype=np.float32),
                np.empty((0, 2, 3), dtype=np.uint8),
            )
        return self.segments[idx], self.colors[idx]


@dataclass(slots=True)
class PointTimeline:
    """Holds per-frame point clouds and colors."""

    positions: List[np.ndarray]
    colors: List[np.ndarray]

    @classmethod
    def empty(cls, num_frames: int) -> "PointTimeline":
        return cls(
            positions=[np.empty((0, 3), dtype=np.float32) for _ in range(num_frames)],
            colors=[np.empty((0, 3), dtype=np.uint8) for _ in range(num_frames)],
        )

    @property
    def num_frames(self) -> int:
        return len(self.positions)

    def frame(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        if not (0 <= frame < self.num_frames):
            raise IndexError(f"frame {frame} out of range [0, {self.num_frames - 1}]")
        pts = _ensure_float32(self.positions[frame])
        cols = _ensure_uint8(self.colors[frame])
        return pts, cols


def build_point_timeline(
    positions: np.ndarray,
    colors: np.ndarray,
    exists: np.ndarray,
) -> PointTimeline:
    pts = _ensure_float32(positions)
    cols = _ensure_uint8(colors)
    exists_mask = np.asarray(exists, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if cols.shape not in {pts.shape, (pts.shape[1], 3)}:
        raise ValueError("colors must be (T, N, 3) or (N, 3)")
    if exists_mask.shape != pts.shape[:2]:
        raise ValueError("exists mask must have shape (T, N)")

    if cols.ndim == 2:
        cols = np.broadcast_to(cols[None, ...], pts.shape).copy()

    frames: List[np.ndarray] = []
    frame_colors: List[np.ndarray] = []
    for t in range(pts.shape[0]):
        mask_t = exists_mask[t]
        if not np.any(mask_t):
            frames.append(np.empty((0, 3), dtype=np.float32))
            frame_colors.append(np.empty((0, 3), dtype=np.uint8))
            continue
        frames.append(pts[t, mask_t].astype(np.float32, copy=False))
        frame_colors.append(cols[t, mask_t].astype(np.uint8, copy=False))

    return PointTimeline(frames, frame_colors)


def build_rainbow_flow_timeline(
    positions: np.ndarray,
    exists: np.ndarray,
    *,
    colormap: Callable[[np.ndarray], np.ndarray],
    min_brightness: float,
) -> FlowTimeline:
    pts = _ensure_float32(positions)
    mask = np.asarray(exists, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if mask.shape != pts.shape[:2]:
        raise ValueError("exists mask must match positions (T, N)")

    T, N = pts.shape[:2]
    if T == 0 or N == 0:
        return FlowTimeline.empty(T)

    timeline = np.linspace(0.0, 1.0, T, dtype=np.float32)
    cmap_vals = np.asarray(colormap(timeline)[..., :3], dtype=np.float32)
    brightness = np.linspace(float(min_brightness), 1.0, max(T - 1, 1), dtype=np.float32)

    segments: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    end_frames: List[int] = []

    for n in range(N):
        for t in range(T - 1):
            if not (mask[t, n] and mask[t + 1, n]):
                continue
            seg = pts[[t, t + 1], n, :]
            if not np.all(np.isfinite(seg)):
                continue
            col0 = cmap_vals[t] * brightness[t]
            col1 = cmap_vals[t + 1] * brightness[t]
            seg_colors = np.stack([col0, col1], axis=0)
            segments.append(seg.astype(np.float32, copy=False))
            colors.append((seg_colors * 255.0).clip(0, 255).astype(np.uint8))
            end_frames.append(t + 1)

    if not segments:
        return FlowTimeline.empty(T)

    segment_array = np.stack(segments, axis=0).astype(np.float32, copy=False)
    color_array = np.stack(colors, axis=0).astype(np.uint8, copy=False)
    end_array = np.asarray(end_frames, dtype=np.int64)
    return FlowTimeline.from_segments(segment_array, color_array, end_array, T)


def build_constant_flow_timeline(
    positions: np.ndarray,
    exists: np.ndarray,
    *,
    active_mask: np.ndarray | None,
    color_rgb: Sequence[int],
    min_brightness: float,
) -> FlowTimeline:
    pts = _ensure_float32(positions)
    mask = np.asarray(exists, dtype=bool)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError("positions must have shape (T, N, 3)")
    if mask.shape != pts.shape[:2]:
        raise ValueError("exists mask must match positions (T, N)")

    T, N = pts.shape[:2]
    if T == 0 or N == 0:
        return FlowTimeline.empty(T)

    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (N,):
            raise ValueError("active_mask must have shape (N,)")
    else:
        active = np.ones((N,), dtype=bool)

    base_color = np.asarray(color_rgb, dtype=np.float32) / 255.0
    brightness = np.linspace(float(min_brightness), 1.0, max(T - 1, 1), dtype=np.float32)

    segments: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    end_frames: List[int] = []

    for n in range(N):
        if not active[n]:
            continue
        for t in range(T - 1):
            if not (mask[t, n] and mask[t + 1, n]):
                continue
            seg = pts[[t, t + 1], n, :]
            if not np.all(np.isfinite(seg)):
                continue
            bright = brightness[t]
            seg_color = (base_color * bright * 255.0).clip(0, 255).astype(np.uint8)
            seg_colors = np.stack([seg_color, seg_color], axis=0)
            segments.append(seg.astype(np.float32, copy=False))
            colors.append(seg_colors)
            end_frames.append(t + 1)

    if not segments:
        return FlowTimeline.empty(T)

    segment_array = np.stack(segments, axis=0).astype(np.float32, copy=False)
    color_array = np.stack(colors, axis=0).astype(np.uint8, copy=False)
    end_array = np.asarray(end_frames, dtype=np.int64)
    return FlowTimeline.from_segments(segment_array, color_array, end_array, T)


def blend_with_green(colors: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = _ensure_uint8(colors)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != base.shape[:2]:
        raise ValueError("mask must have shape (T, N)")
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("colors must have shape (T, N, 3)")

    if not np.any(mask_bool):
        return base

    green = np.array([0, 255, 0], dtype=np.float32)
    blended = base.astype(np.float32)
    idx = np.where(mask_bool)
    blended[idx] = (1.0 - alpha) * blended[idx] + alpha * green
    return blended.clip(0.0, 255.0).astype(np.uint8)
