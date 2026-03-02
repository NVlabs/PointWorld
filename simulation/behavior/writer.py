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
"""Per-clip H5 writing and aggregation for Behavior extraction."""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import h5py
import numpy as np

from shared.data_contract import EXPECTED_CAMERA_PAYLOAD_SHAPES
from shared.h5_io import save_depth_as_uint16_mm, save_rgb_as_jpeg_in_h5


def _quantize_unit_normals_to_int8(normals: np.ndarray) -> np.ndarray:
    """Quantize unit normals from [-1, 1] float space into int8 [-127, 127]."""
    normals_f32 = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(normals_f32).all():
        raise ValueError("local_normals contains non-finite values")
    clipped = np.clip(normals_f32, -1.0, 1.0)
    return np.rint(clipped * 127.0).astype(np.int8)


def _normalize_camera_payload_to_contract(
    initial_rgb: np.ndarray,
    initial_depth: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize camera payloads to the release contract resolution (H=180, W=320).

    Fails fast if RGB/depth source grids are inconsistent, since they should be
    sampled together upstream.
    """
    target_h, target_w = EXPECTED_CAMERA_PAYLOAD_SHAPES["initial_depth"]

    rgb_u8 = np.asarray(initial_rgb)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected initial_rgb with shape (H,W,3), got {rgb_u8.shape}")
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)

    depth_f32 = np.asarray(initial_depth, dtype=np.float32)
    if depth_f32.ndim != 2:
        raise ValueError(f"Expected initial_depth with shape (H,W), got {depth_f32.shape}")

    if rgb_u8.shape[:2] != depth_f32.shape:
        raise ValueError(
            f"RGB/depth source resolution mismatch: rgb={rgb_u8.shape[:2]}, depth={depth_f32.shape}"
        )

    intr = np.asarray(intrinsic, dtype=np.float32)
    if intr.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3,3), got {intr.shape}")

    src_h, src_w = depth_f32.shape
    if (src_h, src_w) == (target_h, target_w):
        return rgb_u8, depth_f32, intr

    sx = float(target_w) / float(src_w)
    sy = float(target_h) / float(src_h)
    intr_scaled = intr.copy()
    intr_scaled[0, 0] *= sx
    intr_scaled[0, 2] *= sx
    intr_scaled[1, 1] *= sy
    intr_scaled[1, 2] *= sy

    rgb_resized = cv2.resize(rgb_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_f32, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return rgb_resized, depth_resized, intr_scaled


class ClipWriter:
    """Write per-clip H5 files and aggregate to a final episode H5."""

    def __init__(self, output_file: str, get_joint_names_fn, source_file: str):
        self._output_file = output_file
        self._get_joint_names = get_joint_names_fn
        self._source_file = source_file
        self._temp_dir = self._derive_temp_dir(output_file)

    @staticmethod
    def _derive_temp_dir(output_file: str) -> str:
        """Return per-episode temp directory for storing clip files."""
        output_dir = os.path.dirname(output_file)
        base = os.path.splitext(os.path.basename(output_file))[0]
        temp_dir = os.path.join(output_dir, f"{base}.clips")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def cleanup_if_output_complete(self) -> bool:
        """Return True if output is complete and temp dir was cleaned."""
        if not os.path.isfile(self._output_file):
            return False
        with h5py.File(self._output_file, "r") as f:
            if "episode_complete" not in f.attrs:
                raise KeyError(f"episode_complete missing from {self._output_file}")
            if not f.attrs["episode_complete"]:
                return False
        if os.path.isdir(self._temp_dir):
            for fname in os.listdir(self._temp_dir):
                os.remove(os.path.join(self._temp_dir, fname))
            os.rmdir(self._temp_dir)
        return True

    def _copy_h5_items_recursive(self, src: h5py.Group, dst: h5py.Group):
        """Recursively copy all datasets and groups from src to dst including attrs."""
        for key in src.keys():
            if isinstance(src[key], h5py.Dataset):
                dst.create_dataset(key, data=src[key][()], dtype=src[key].dtype)
                for attr_key, attr_val in src[key].attrs.items():
                    dst[key].attrs[attr_key] = attr_val
            elif isinstance(src[key], h5py.Group):
                subgrp = dst.create_group(key)
                for attr_key, attr_val in src[key].attrs.items():
                    subgrp.attrs[attr_key] = attr_val
                self._copy_h5_items_recursive(src[key], subgrp)
        for attr_key, attr_val in src.attrs.items():
            dst.attrs[attr_key] = attr_val

    def write_candidate(self, candidate) -> int:
        """Flush a valid candidate to a per-clip temporary H5 file atomically."""
        clip_key = (
            f"{candidate.start_downsample_idx}:"
            f"{candidate.start_downsample_idx + candidate.frames_per_clip}"
        )
        clip_path = os.path.join(self._temp_dir, f"{clip_key}.h5")

        # Write atomically using a .part file
        tmp_path = clip_path + ".part"
        total_scene_points = 0
        with h5py.File(tmp_path, "w") as f:
            dset = f.create_dataset("world_to_robot", data=candidate.world_to_robot, dtype=np.float32)
            dset.attrs["write_complete"] = True

            for data_key, data_array in candidate.robot_series.items():
                if (
                    isinstance(data_array, np.ndarray)
                    and (
                        data_array.dtype == np.bool_
                        or data_array.dtype == bool
                        or "gripper_open" in data_key
                        or "is_grasping" in data_key
                    )
                ):
                    dset = f.create_dataset(data_key, data=data_array.astype(np.bool_), dtype=np.bool_)
                else:
                    dset = f.create_dataset(data_key, data=data_array, dtype=np.float32)
                dset.attrs["write_complete"] = True

            joint_names_list = self._get_joint_names()
            joint_names_dtype = h5py.string_dtype(encoding="utf-8")
            dset = f.create_dataset(
                "joint_names", data=np.array(joint_names_list, dtype=object), dtype=joint_names_dtype
            )
            dset.attrs["write_complete"] = True

            for camera_name, camera_data in candidate.clip_seed["cameras"].items():
                camera_group = f.create_group(f"camera_{camera_name}")

                # BEHAVIOR stores mesh-local point samples (rigid-body scenes). We use canonical naming:
                # static = points, temporal = flows.
                local_scene_points_group = camera_group.create_group("local_scene_points")
                local_scene_normals_group = camera_group.create_group("local_scene_normals")
                local_scene_colors_group = camera_group.create_group("local_scene_colors")
                scene_mesh_trajectories_group = camera_group.create_group("scene_mesh_trajectories")

                mesh_data = camera_data["mesh_data"]
                scene_points_count = 0

                for mesh_name, data in mesh_data.items():
                    dset = local_scene_points_group.create_dataset(
                        mesh_name, data=data["local_points"].astype(np.float16), dtype=np.float16
                    )
                    dset.attrs["write_complete"] = True
                    dset.attrs["prim_path"] = data["prim_path"]
                    scene_points_count += data["local_points"].shape[0]

                    local_normals_q = _quantize_unit_normals_to_int8(data["local_normals"])
                    dset = local_scene_normals_group.create_dataset(
                        mesh_name, data=local_normals_q, dtype=np.int8
                    )
                    dset.attrs["write_complete"] = True
                    dset.attrs["prim_path"] = data["prim_path"]

                    dset = local_scene_colors_group.create_dataset(
                        mesh_name, data=data["colors"], dtype=np.uint8
                    )
                    dset.attrs["write_complete"] = True
                    dset.attrs["prim_path"] = data["prim_path"]

                    if mesh_name in camera_data["mesh_trajectories"]:
                        dset = scene_mesh_trajectories_group.create_dataset(
                            mesh_name,
                            data=camera_data["mesh_trajectories"][mesh_name],
                            dtype=np.float32,
                        )
                        dset.attrs["write_complete"] = True
                        dset.attrs["prim_path"] = data["prim_path"]

                initial_rgb, initial_depth, intrinsic = _normalize_camera_payload_to_contract(
                    camera_data["initial_rgb"],
                    camera_data["initial_depth"],
                    camera_data["intrinsic"],
                )

                dset = camera_group.create_dataset("intrinsic", data=intrinsic, dtype=np.float32)
                dset.attrs["write_complete"] = True
                dset = camera_group.create_dataset("extrinsic", data=camera_data["extrinsic"], dtype=np.float32)
                dset.attrs["write_complete"] = True
                dset = camera_group.create_dataset(
                    "extrinsic_trajectory", data=camera_data["extrinsic_trajectory"], dtype=np.float32
                )
                dset.attrs["write_complete"] = True

                save_rgb_as_jpeg_in_h5(camera_group, "initial_rgb", initial_rgb)
                save_depth_as_uint16_mm(camera_group, "initial_depth", initial_depth)

                camera_group.attrs["num_scene_points"] = scene_points_count
                total_scene_points += scene_points_count

            f.attrs["clip_key"] = clip_key
            f.attrs["domain"] = "behavior"
            f.attrs["num_frames"] = candidate.frames_per_clip
            f.attrs["num_scene_points"] = total_scene_points

            f.attrs["has_transition"] = candidate.has_transition
            f.attrs["any_object_moving"] = candidate.any_object_moving
            f.attrs["gripper_moving"] = candidate.gripper_moving
            f.attrs["has_gripper_state_change"] = candidate.has_gripper_state_change
            f.attrs["robot_nonbase_moving"] = candidate.robot_nonbase_moving
            f.attrs["has_trunk_arm_collision"] = candidate.has_trunk_arm_collision
            for arm in candidate.arm_names:
                f.attrs[f"has_{arm}_gripper_finger_collision"] = bool(
                    candidate.gripper_finger_collision[arm]
                )
            f.attrs["max_object_pos_movement"] = candidate.max_object_pos_movement
            f.attrs["max_object_rot_movement"] = candidate.max_object_rot_movement
            f.attrs["max_gripper_pos_movement"] = candidate.max_gripper_pos_movement
            f.attrs["max_gripper_rot_movement"] = candidate.max_gripper_rot_movement
            f.attrs["max_joint_movement"] = candidate.max_joint_movement
            for arm in candidate.arm_names:
                min_dist_moving = candidate.min_distance_to_moving_objects[arm]
                min_dist_all = candidate.min_distance_to_all_objects[arm]
                f.attrs[f"{arm}_min_distance_to_moving_objects"] = (
                    min_dist_moving if min_dist_moving != float("inf") else -1.0
                )
                f.attrs[f"{arm}_min_distance_to_all_objects"] = (
                    min_dist_all if min_dist_all != float("inf") else -1.0
                )

            f.attrs["clip_complete"] = True

        os.replace(tmp_path, clip_path)
        return total_scene_points

    def aggregate_episode(self):
        """Aggregate per-clip temp files into final episode H5 atomically, then clean temp dir."""
        if not os.path.isdir(self._temp_dir):
            print(f"No temp clip directory found at {self._temp_dir}; nothing to aggregate.")
            return

        clip_files = []
        for fname in os.listdir(self._temp_dir):
            if not fname.endswith(".h5"):
                continue
            fpath = os.path.join(self._temp_dir, fname)
            with h5py.File(fpath, "r") as cf:
                if "clip_complete" not in cf.attrs:
                    raise KeyError(f"clip_complete missing from {fpath}")
                if not cf.attrs["clip_complete"]:
                    raise RuntimeError(f"clip_complete is False for {fpath}")
                if "clip_key" not in cf.attrs:
                    raise KeyError(f"clip_key missing from {fpath}")
                clip_files.append((cf.attrs["clip_key"], fpath))

        if len(clip_files) == 0:
            print("No completed clip files found to aggregate.")
            return

        def clip_key_sort(key):
            try:
                return int(key.split(":")[0])
            except Exception as exc:
                raise ValueError(f"Invalid clip_key format: {key}") from exc

        clip_files.sort(key=lambda x: clip_key_sort(x[0]))

        tmp_output = self._output_file + ".part"
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_scene_points_sum = 0
        with h5py.File(tmp_output, "w") as out_f:
            out_f.attrs["created_time"] = datetime_str
            out_f.attrs["last_modified_time"] = datetime_str
            out_f.attrs["source_file"] = self._source_file
            out_f.attrs["domain"] = "behavior"
            for clip_key, fpath in clip_files:
                with h5py.File(fpath, "r") as cf:
                    grp = out_f.create_group(str(clip_key))
                    self._copy_h5_items_recursive(cf, grp)
                    for k, v in cf.attrs.items():
                        if k == "clip_complete":
                            continue
                        grp.attrs[k] = v
                    grp.attrs["clip_complete"] = True

                    if "num_scene_points" not in grp.attrs:
                        raise KeyError(f"num_scene_points missing from clip {clip_key}")
                    total_scene_points_sum += int(grp.attrs["num_scene_points"])

            out_f.attrs["num_scene_points"] = total_scene_points_sum
            out_f.attrs["episode_complete"] = True

        os.replace(tmp_output, self._output_file)

        for fname in os.listdir(self._temp_dir):
            os.remove(os.path.join(self._temp_dir, fname))
        os.rmdir(self._temp_dir)
