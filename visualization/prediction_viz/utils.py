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

import numpy as np
import open3d as o3d


def _fps_select_indices_o3d(points: np.ndarray, k: int) -> np.ndarray:
    """Select K points via Open3D farthest-point downsampling and return indices.

    Uses KDTreeFlann to map sampled points back to original indices.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0 or k <= 0:
        return np.empty((0,), dtype=np.int64)
    N = pts.shape[0]
    if k >= N:
        return np.arange(N, dtype=np.int64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    down = pcd.farthest_point_down_sample(int(k))
    down_pts = np.asarray(down.points, dtype=np.float64)
    if down_pts.size == 0:
        return np.empty((0,), dtype=np.int64)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    sel: list[int] = []
    for q in down_pts:
        _, idx, _ = kdtree.search_knn_vector_3d(q, 1)
        if idx:
            sel.append(int(idx[0]))
    # Deduplicate preserving order
    seen = set()
    uniq: list[int] = []
    for i in sel:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return np.asarray(uniq, dtype=np.int64)


def _build_workspace_bounding_box(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    bmin = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    bmax = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    corners = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )
    segs = np.stack([corners[edges[:, 0]], corners[edges[:, 1]]], axis=1)
    return segs.astype(np.float32, copy=False)


def _ensure_float_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _filter_background_to_point_voxels(
    background_points: np.ndarray,
    background_colors: np.ndarray,
    point_positions_t0: np.ndarray,
    exists_mask_t0: np.ndarray,
    grid_size: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only background points whose voxel contains at least one scene point at t=0.

    - background_points/colors: merged RGB-D cloud (world frame)
    - point_positions_t0: (N,3) scene points at t=0
    - exists_mask_t0: (N,) mask for valid points at t=0
    - grid_size: voxel size (meters)
    """
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive for voxel filtering")
    bg_pts = np.asarray(background_points, dtype=np.float32)
    bg_cols = np.asarray(background_colors, dtype=np.uint8)
    if bg_pts.ndim != 2 or bg_pts.shape[1] != 3:
        raise ValueError("background_points must be (M,3)")
    p0 = np.asarray(point_positions_t0, dtype=np.float32)
    m0 = np.asarray(exists_mask_t0, dtype=bool)
    if p0.ndim != 2 or p0.shape[1] != 3:
        raise ValueError("point_positions_t0 must be (N,3)")
    if m0.shape[0] != p0.shape[0]:
        raise ValueError("exists_mask_t0 length must match point_positions_t0")

    if not np.any(m0):
        # No valid points: keep only out-of-bounds points (inside points are dropped)
        bm = np.asarray(bounds_min, dtype=np.float32).reshape(3)
        bM = np.asarray(bounds_max, dtype=np.float32).reshape(3)
        inside = (
            (bg_pts[:, 0] >= bm[0]) & (bg_pts[:, 0] <= bM[0]) &
            (bg_pts[:, 1] >= bm[1]) & (bg_pts[:, 1] <= bM[1]) &
            (bg_pts[:, 2] >= bm[2]) & (bg_pts[:, 2] <= bM[2])
        )
        keep_mask = ~inside
        return bg_pts[keep_mask].astype(np.float32, copy=False), bg_cols[keep_mask].astype(np.uint8, copy=False)

    def _voxels(x: np.ndarray) -> np.ndarray:
        return np.floor(x / float(grid_size)).astype(np.int64)

    point_keys = _voxels(p0[m0])
    bg_keys = _voxels(bg_pts)

    key_dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    point_view = point_keys.view(key_dtype).reshape(-1)
    bg_view = bg_keys.view(key_dtype).reshape(-1)
    point_unique = np.unique(point_view)

    # Inside/outside workspace mask
    bm = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    bM = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    inside = (
        (bg_pts[:, 0] >= bm[0]) & (bg_pts[:, 0] <= bM[0]) &
        (bg_pts[:, 1] >= bm[1]) & (bg_pts[:, 1] <= bM[1]) &
        (bg_pts[:, 2] >= bm[2]) & (bg_pts[:, 2] <= bM[2])
    )
    in_keep = np.isin(bg_view, point_unique, assume_unique=False)
    # Keep all outside points; inside points only if their voxel has at least one point
    keep_mask = (~inside) | (inside & in_keep)

    if not np.any(keep_mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return bg_pts[keep_mask].astype(np.float32, copy=False), bg_cols[keep_mask].astype(np.uint8, copy=False)


__all__ = [
    "_fps_select_indices_o3d",
    "_build_workspace_bounding_box",
    "_ensure_float_array",
    "_filter_background_to_point_voxels",
]
