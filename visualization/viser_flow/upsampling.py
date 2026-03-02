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

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from .timeline import PointTimeline


_GREEN = np.array([0, 255, 0], dtype=np.float32)


@dataclasses.dataclass(slots=True)
class VoxelAssignment:
    background_points: np.ndarray
    background_colors: np.ndarray
    initial_points: np.ndarray
    flow_to_points: List[np.ndarray]
    static_indices: np.ndarray

    def build_timeline(
        self,
        flow_positions: np.ndarray,
        flow_exists: np.ndarray,
        *,
        supervised: Optional[np.ndarray] = None,
        tint_alpha: float = 0.5,
    ) -> PointTimeline:
        positions = np.asarray(flow_positions, dtype=np.float32)
        exists = np.asarray(flow_exists, dtype=bool)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("flow_positions must have shape (T, N, 3)")
        if exists.shape != positions.shape[:2]:
            raise ValueError("flow_exists must have shape (T, N)")

        if supervised is not None:
            supervised_mask = np.asarray(supervised, dtype=bool)
            if supervised_mask.shape != exists.shape:
                raise ValueError("supervised mask must match flow_exists shape")
        else:
            supervised_mask = None

        T, N = positions.shape[:2]
        static_points = self.background_points[self.static_indices]
        static_colors = self.background_colors[self.static_indices]
        frames: List[np.ndarray] = []
        frame_colors: List[np.ndarray] = []

        for t in range(T):
            pts_segments: List[np.ndarray] = []
            col_segments: List[np.ndarray] = []
            if static_points.size:
                pts_segments.append(static_points.astype(np.float32, copy=False))
                col_segments.append(static_colors.astype(np.uint8, copy=False))
            for n in range(N):
                point_indices = self.flow_to_points[n]
                if point_indices.size == 0:
                    continue
                if not exists[t, n]:
                    continue
                delta = positions[t, n] - self.initial_points[n]
                updated = self.background_points[point_indices] + delta
                cols = self.background_colors[point_indices].astype(np.float32, copy=True)
                if supervised_mask is not None and not supervised_mask[t, n]:
                    cols = (1.0 - float(tint_alpha)) * cols + float(tint_alpha) * _GREEN
                pts_segments.append(updated.astype(np.float32, copy=False))
                col_segments.append(cols.clip(0.0, 255.0).astype(np.uint8))

            if pts_segments:
                frames.append(np.concatenate(pts_segments, axis=0).astype(np.float32, copy=False))
                frame_colors.append(np.concatenate(col_segments, axis=0).astype(np.uint8, copy=False))
            else:
                frames.append(np.empty((0, 3), dtype=np.float32))
                frame_colors.append(np.empty((0, 3), dtype=np.uint8))

        return PointTimeline(frames, frame_colors)


def build_voxel_assignment(
    background_points: np.ndarray,
    background_colors: np.ndarray,
    flow_positions: np.ndarray,
    flow_exists: np.ndarray,
    *,
    grid_size: float,
) -> VoxelAssignment:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive for voxel assignment")

    bg_pts = np.asarray(background_points, dtype=np.float32)
    bg_cols = np.asarray(background_colors, dtype=np.uint8)
    flows = np.asarray(flow_positions, dtype=np.float32)
    exists = np.asarray(flow_exists, dtype=bool)

    if flows.ndim != 3 or flows.shape[-1] != 3:
        raise ValueError("flow_positions must have shape (T, N, 3)")
    if exists.shape != flows.shape[:2]:
        raise ValueError("flow_exists must match flow_positions shape")

    initial_positions = flows[0]
    initial_exists = exists[0]

    voxel_map: Dict[tuple[int, int, int], List[int]] = {}
    for idx in range(initial_positions.shape[0]):
        if not initial_exists[idx]:
            continue
        key = tuple(np.floor(initial_positions[idx] / grid_size).astype(np.int64))
        voxel_map.setdefault(key, []).append(idx)

    assigned = np.zeros((bg_pts.shape[0],), dtype=bool)
    flow_to_points: List[List[int]] = [[] for _ in range(initial_positions.shape[0])]

    for i, point in enumerate(bg_pts):
        key = tuple(np.floor(point / grid_size).astype(np.int64))
        candidates = voxel_map.get(key)
        if not candidates:
            continue
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            initial_positions_subset = initial_positions[candidates]
            distances = np.linalg.norm(initial_positions_subset - point, axis=1)
            chosen = candidates[int(np.argmin(distances))]
        assigned[i] = True
        flow_to_points[chosen].append(i)

    flow_point_arrays: List[np.ndarray] = []
    for lst in flow_to_points:
        if lst:
            flow_point_arrays.append(np.asarray(lst, dtype=np.int64))
        else:
            flow_point_arrays.append(np.empty((0,), dtype=np.int64))

    static_indices = np.nonzero(~assigned)[0]

    return VoxelAssignment(
        background_points=bg_pts,
        background_colors=bg_cols,
        initial_points=initial_positions.astype(np.float32, copy=False),
        flow_to_points=flow_point_arrays,
        static_indices=static_indices.astype(np.int64, copy=False),
    )
