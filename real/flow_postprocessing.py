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
import numpy as np
from sklearn.cluster import DBSCAN


def remove_outlier_flows(
    flows: np.ndarray,
    visibility_mask: np.ndarray | None = None,
    depth_valid_mask: np.ndarray | None = None,
    eps_values: list[float] | None = None,
    min_points: int = 5,
    min_outlier_frames_ratio: float = 0.3,
    verbose: bool = False,
):
    """
    Remove outlier 3D flow tracks based on spatial clustering and temporal consistency.

    Args:
        flows: (T, N, 3) array of 3D flow points.
        visibility_mask: (T, N) bool mask for tracker visibility.
        depth_valid_mask: (T, N) bool mask for depth validity.
        eps_values: List of epsilon values for multi-scale DBSCAN (meters).
        min_points: Minimum points to form a DBSCAN core cluster.
        min_outlier_frames_ratio: Ratio of frames where a track is an outlier to drop it.
        verbose: If True, prints progress information.

    Returns:
        filtered_flows: (T, N_kept, 3) array with outlier tracks removed.
        filtered_visibility_mask: (T, N_kept) bool mask or None.
        filtered_depth_valid_mask: (T, N_kept) bool mask or None.
        outlier_indices: (N_outliers,) indices removed from the original tracks.
    """
    if flows.ndim != 3 or flows.shape[-1] != 3:
        raise ValueError(f"flows must have shape (T, N, 3), got {flows.shape}")
    if eps_values is None:
        eps_values = [0.2, 0.5, 1.0]

    T, N, _ = flows.shape
    if visibility_mask is None:
        visibility_mask = np.ones((T, N), dtype=bool)
    if depth_valid_mask is None:
        depth_valid_mask = np.ones((T, N), dtype=bool)
    if visibility_mask.shape != (T, N):
        raise ValueError(f"visibility_mask must have shape {(T, N)}, got {visibility_mask.shape}")
    if depth_valid_mask.shape != (T, N):
        raise ValueError(f"depth_valid_mask must have shape {(T, N)}, got {depth_valid_mask.shape}")

    if verbose:
        print(f"Outlier removal: T={T} N={N} eps_values={eps_values}")

    validity_mask = np.logical_and(visibility_mask, depth_valid_mask)
    outlier_counts = np.zeros(N, dtype=int)
    valid_frame_counts = np.sum(validity_mask, axis=0)

    for t in range(T):
        valid_mask = validity_mask[t]
        if not np.any(valid_mask):
            continue
        positions = flows[t, valid_mask]
        valid_indices = np.where(valid_mask)[0]

        frame_outliers = np.zeros(len(valid_indices), dtype=bool)
        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit(positions)
            frame_outliers = np.logical_or(frame_outliers, db.labels_ == -1)
        outlier_counts[valid_indices[frame_outliers]] += 1

    valid_frame_counts_safe = np.maximum(valid_frame_counts, 1)
    outlier_ratios = outlier_counts / valid_frame_counts_safe
    outlier_mask = outlier_ratios >= min_outlier_frames_ratio
    outlier_indices = np.where(outlier_mask)[0]

    if verbose:
        num_outliers = len(outlier_indices)
        outlier_percentage = (num_outliers / N) * 100
        print(f"Outlier removal: removed {num_outliers} tracks ({outlier_percentage:.2f}%)")

    keep_mask = ~outlier_mask
    filtered_flows = flows[:, keep_mask]
    filtered_visibility = visibility_mask[:, keep_mask] if visibility_mask is not None else None
    filtered_depth = depth_valid_mask[:, keep_mask] if depth_valid_mask is not None else None

    return filtered_flows, filtered_visibility, filtered_depth, outlier_indices
