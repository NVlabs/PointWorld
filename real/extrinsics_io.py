#!/usr/bin/env python3

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
"""I/O helpers for extrinsics pipeline outputs."""

from __future__ import annotations

import json
import os
import time
from typing import Callable

import h5py
import numpy as np


def load_precomputed_depth(output_dir: str,
                           uuid: str,
                           data_dict: dict,
                           wrist_serial: str,
                           log_fn: Callable[[str], None]) -> None:
    """Load precomputed FoundationStereo depth from H5 into data_dict.

    Updates data_dict in-place with:
      - stereo_depth
      - stereo_intrinsics
    """
    start = time.time()
    h5_path = os.path.join(output_dir, "depth", f"{uuid}_depth.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Precomputed depth file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "metadata" not in f:
            raise ValueError(f"Invalid depth file: missing metadata in {h5_path}")

        metadata = f["metadata"]
        if "write_complete" not in metadata.attrs:
            raise ValueError(f"Depth file missing write_complete flag: {h5_path}")
        if not metadata.attrs["write_complete"]:
            raise ValueError(f"Depth file not complete: {h5_path}")

        camera_groups = [key for key in f.keys() if key != "metadata"]
        for camera_group_name in camera_groups:
            camera_serial, _camera_type = camera_group_name.split("+")
            if camera_serial not in data_dict:
                raise ValueError(f"Camera {camera_serial} in depth file but not in data_dict")

            camera_group = f[camera_group_name]
            depth_uint16 = camera_group["depth"][:]  # [T, H, W]
            depth_timestamps = camera_group["timestamps"][:]  # [T]

            depth_meters = depth_uint16.astype(np.float32) / 1000.0

            canonical_timestamps = data_dict[camera_serial]["timestamps"]
            aligned_depth_frames = []
            for canonical_ts in canonical_timestamps:
                time_diffs = np.abs(depth_timestamps - canonical_ts)
                closest_idx = np.argmin(time_diffs)
                if time_diffs[closest_idx] > 50:
                    log_fn(
                        f"Warning: Large timestamp difference ({time_diffs[closest_idx]:.1f}ms) "
                        f"for camera {camera_serial}"
                    )
                aligned_depth_frames.append(depth_meters[closest_idx])

            aligned_depth = np.stack(aligned_depth_frames, axis=0)
            data_dict[camera_serial]["stereo_depth"] = aligned_depth
            data_dict[camera_serial]["stereo_intrinsics"] = data_dict[camera_serial]["measured_intrinsics"]

            log_fn(
                f"Loaded depth for camera {camera_serial}: {aligned_depth.shape}, "
                f"range: {aligned_depth.min():.3f}-{aligned_depth.max():.3f}m"
            )

    num_cams = len([k for k in data_dict.keys() if k != wrist_serial])
    log_fn(f"Successfully loaded precomputed depth for {num_cams} cameras (time taken: {time.time() - start:.2f}s)")


def write_camera_results(output_dir: str,
                         uuid: str,
                         scene_path: str,
                         data_dict: dict,
                         wrist_serial: str,
                         optimization_metrics: dict | None,
                         error_info: dict | None,
                         log_fn: Callable[[str], None]) -> str:
    """Write camera JSON results to output_dir/cameras."""
    cameras_dir = os.path.join(output_dir, "cameras")
    os.makedirs(cameras_dir, exist_ok=True)

    results = {
        "uuid": uuid,
        "scene_path": scene_path,
    }

    if error_info is not None:
        results["error_info"] = error_info

    if optimization_metrics:
        results["optimization_summary"] = optimization_metrics

    for camera_serial in data_dict:
        if camera_serial == wrist_serial:
            continue
        camera_data = data_dict[camera_serial]
        camera_result = {}

        if "vggt_extrinsics" in camera_data:
            camera_result["vggt_extrinsics"] = camera_data["vggt_extrinsics"].tolist()

        if error_info is None and "optimized_extrinsics" in camera_data:
            camera_result["optimized_extrinsics"] = camera_data["optimized_extrinsics"].tolist()

        if "measured_intrinsics" in camera_data:
            camera_result["measured_intrinsics"] = camera_data["measured_intrinsics"].tolist()

        if "vggt_intrinsics" in camera_data:
            camera_result["vggt_intrinsics"] = camera_data["vggt_intrinsics"].tolist()

        results[camera_serial] = camera_result

    json_path = os.path.join(cameras_dir, f"{uuid}_cameras.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    log_fn(f"Camera results saved to: {json_path}")
    return json_path
