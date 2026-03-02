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

import io
import pickle
from typing import Any, Dict

import cv2
import numpy as np

try:
    from numba import njit
except ImportError:
    print("Warning: numba not available, using pure Python (slower)")
    njit = lambda **kwargs: lambda x: x

import transform_utils
from dataset_components.constants import B1K_GRIPPER_TRANSFORM, DROID_WRIST2TCP
from dataset_components.robot import (
    _apply_eval_single_arm_feature_mask,
    _deterministic_single_arm_choice,
    _get_robot_flows,
    _remove_filtered_gripper_data,
    determine_gripper_filter,
)

QUANTIZED_NORMALS_SCALE = 127.0


def _dequantize_quantized_normals_array(normals: np.ndarray, key_name: str) -> np.ndarray:
    assert isinstance(normals, np.ndarray), f"{key_name} must decode to np.ndarray, got {type(normals)}"
    assert normals.dtype == np.int8, (
        f"{key_name} must be int8 quantized normals for release data, got {normals.dtype}"
    )
    if normals.size > 0:
        min_val = int(normals.min())
        max_val = int(normals.max())
        assert min_val >= -127 and max_val <= 127, (
            f"{key_name} values must lie in [-127, 127], got [{min_val}, {max_val}]"
        )
    return normals.astype(np.float32) / QUANTIZED_NORMALS_SCALE


def _dequantize_quantized_normals_dict(normals_by_mesh: Dict[str, Any], key_name: str) -> Dict[str, np.ndarray]:
    assert isinstance(normals_by_mesh, dict), (
        f"{key_name} must decode to dict[mesh_name, np.ndarray], got {type(normals_by_mesh)}"
    )
    output: Dict[str, np.ndarray] = {}
    for mesh_name, mesh_normals in normals_by_mesh.items():
        mesh_key = f"{key_name}[{mesh_name}]"
        output[str(mesh_name)] = _dequantize_quantized_normals_array(np.asarray(mesh_normals), mesh_key)
    return output


# Numba-accelerated kernel for efficient point transformation
@njit(cache=True, fastmath=True)
def transform_points_kernel(local_points: np.ndarray,
                              pose_matrices: np.ndarray,
                              world_flows: np.ndarray) -> None:
    """
    Numba-accelerated kernel for transforming local points to world coordinates.

    Args:
        local_points: (N, 3) local point positions
        pose_matrices: (T, 4, 4) transformation matrices for each timestep
        world_flows: (T, N, 3) output array for world points across timesteps
    """
    T, N = world_flows.shape[0], world_flows.shape[1]

    for t in range(T):
        for n in range(N):
            # Transform point: world = pose_matrix @ [local, 1]
            local_x, local_y, local_z = local_points[n, 0], local_points[n, 1], local_points[n, 2]

            world_flows[t, n, 0] = (pose_matrices[t, 0, 0] * local_x +
                                       pose_matrices[t, 0, 1] * local_y +
                                       pose_matrices[t, 0, 2] * local_z +
                                       pose_matrices[t, 0, 3])
            world_flows[t, n, 1] = (pose_matrices[t, 1, 0] * local_x +
                                       pose_matrices[t, 1, 1] * local_y +
                                       pose_matrices[t, 1, 2] * local_z +
                                       pose_matrices[t, 1, 3])
            world_flows[t, n, 2] = (pose_matrices[t, 2, 0] * local_x +
                                       pose_matrices[t, 2, 1] * local_y +
                                       pose_matrices[t, 2, 2] * local_z +
                                       pose_matrices[t, 2, 3])


@njit(cache=True, fastmath=True)
def transform_normals_kernel(local_normals: np.ndarray,
                            rotation_matrices: np.ndarray,
                            world_normals: np.ndarray) -> None:
    """
    Numba-accelerated kernel for transforming local normals to world coordinates.

    Args:
        local_normals: (N, 3) local normal vectors
        rotation_matrices: (T, 3, 3) rotation matrices for each timestep
        world_normals: (T, N, 3) output array for world normals
    """
    T, N = world_normals.shape[0], world_normals.shape[1]

    for t in range(T):
        for n in range(N):
            # Transform normal: world = rotation_matrix @ local
            local_x, local_y, local_z = local_normals[n, 0], local_normals[n, 1], local_normals[n, 2]

            world_normals[t, n, 0] = (rotation_matrices[t, 0, 0] * local_x +
                                     rotation_matrices[t, 0, 1] * local_y +
                                     rotation_matrices[t, 0, 2] * local_z)
            world_normals[t, n, 1] = (rotation_matrices[t, 1, 0] * local_x +
                                     rotation_matrices[t, 1, 1] * local_y +
                                     rotation_matrices[t, 1, 2] * local_z)
            world_normals[t, n, 2] = (rotation_matrices[t, 2, 0] * local_x +
                                     rotation_matrices[t, 2, 1] * local_y +
                                     rotation_matrices[t, 2, 2] * local_z)


class SimDataDecoder:
    """Decoder for B1K simulation WebDataset samples only."""

    def __init__(self):
        pass

    def decode_wds(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Decode WDS simulation format (behavior domain)."""
        # Extract number of frames from joint positions
        if 'joint_positions' in sample:
            num_frames = sample['joint_positions'].shape[0]
        else:
            # Fallback: get from first trajectory
            camera_keys = set()
            for key in sample.keys():
                if key.startswith('camera_'):
                    # Extract camera key (e.g., 'camera_left' from 'camera_left_scene_flows')
                    parts = key.split('_')
                    if len(parts) >= 2:
                        cam_key = f"{parts[0]}_{parts[1]}"
                        camera_keys.add(cam_key)
            camera_keys = sorted(list(camera_keys))
            if camera_keys:
                first_camera = camera_keys[0]
                traj_key = f"{first_camera}_scene_mesh_trajectories.pyd"
                if traj_key in sample:
                    trajectories = pickle.loads(sample[traj_key])
                    first_mesh = next(iter(trajectories.keys()))
                    num_frames = trajectories[first_mesh].shape[0]
                else:
                    num_frames = 1
            else:
                num_frames = 1

        decoded_data = {}

        # Process each camera
        camera_keys = set()
        for key in sample.keys():
            if key.startswith('camera_'):
                # Extract camera key (e.g., 'camera_left' from 'camera_left_scene_flows')
                parts = key.split('_')
                if len(parts) >= 2:
                    cam_key = f"{parts[0]}_{parts[1]}"
                    camera_keys.add(cam_key)
        camera_keys = sorted(list(camera_keys))

        for camera_key in camera_keys:
            # Expect extensionless keys that were decoded in dataset.decode_data
            local_points_key = f"{camera_key}_local_scene_points"
            local_colors_key = f"{camera_key}_local_scene_colors"
            local_normals_key = f"{camera_key}_local_scene_normals"
            trajectories_key = f"{camera_key}_scene_mesh_trajectories"
            if not all(k in sample for k in [local_points_key, local_colors_key, local_normals_key, trajectories_key]):
                continue

            local_points = sample[local_points_key]
            local_colors = sample[local_colors_key]
            local_normals = sample[local_normals_key]
            trajectories = sample[trajectories_key]

            # Process mesh points
            # Convert dicts to lists for generic processing
            mesh_names = list(local_points.keys())
            local_points_list = [local_points[name].astype(np.float32) for name in mesh_names]
            local_colors_list = [local_colors[name].astype(np.uint8) for name in mesh_names]
            local_normals_list = [local_normals[name].astype(np.float32) for name in mesh_names]
            trajectories_list = [trajectories[name].astype(np.float32) for name in mesh_names]

            flow_data = self._process_mesh_points_list(
                local_points_list,
                local_colors_list,
                local_normals_list,
                trajectories_list,
                num_frames,
            )

            # Assemble camera data
            decoded_data[camera_key] = {
                'scene_flows': flow_data['flows'],
                'scene_colors': flow_data['colors'],
                'scene_normals': flow_data['normals'],
                'scene_visibility': flow_data['visibility'],
            }

            # Add camera image data if available
            for img_key in ['initial_rgb', 'initial_depth', 'intrinsic', 'extrinsic']:
                # extensionless keys only (already decoded in dataset.decode_data)
                key_new = f"{camera_key}_{img_key}"
                if key_new in sample:
                    decoded_data[camera_key][img_key.replace('initial_', '')] = sample[key_new]

        return decoded_data

    def _process_mesh_points_list(self,
                                     local_points_list: list,
                                     local_colors_list: list,
                                     local_normals_list: list,
                                     trajectories_list: list,
                                     num_frames: int) -> Dict[str, np.ndarray]:
        """
        Generic mesh processing that transforms per-mesh local data into world-space arrays.

        Args:
            local_points_list: list of (N_i, 3) float32 arrays
            local_colors_list: list of (N_i, 3) uint8 arrays
            local_normals_list: list of (N_i, 3) float32 arrays
            trajectories_list: list of (T, 7) float32 arrays
            num_frames: number of frames T
        Returns:
            dict with 'flows', 'colors', 'normals', 'visibility'
        """
        num_meshes = len(local_points_list)
        if num_meshes == 0:
            return {
                'flows': np.zeros((num_frames, 0, 3), dtype=np.float32),
                'colors': np.zeros((num_frames, 0, 3), dtype=np.uint8),
                'normals': np.zeros((num_frames, 0, 3), dtype=np.float32),
                'visibility': np.ones((num_frames, 0), dtype=bool),
            }

        total_points = sum(p.shape[0] for p in local_points_list)
        world_flows = np.zeros((num_frames, total_points, 3), dtype=np.float32)
        world_colors = np.zeros((num_frames, total_points, 3), dtype=np.uint8)
        world_normals = np.zeros((num_frames, total_points, 3), dtype=np.float32)
        visibility = np.ones((num_frames, total_points), dtype=bool)

        point_offset = 0
        for mesh_points, mesh_colors, mesh_normals, mesh_traj in zip(
            local_points_list, local_colors_list, local_normals_list, trajectories_list
        ):
            n_points = mesh_points.shape[0]

            pose_matrices = np.zeros((num_frames, 4, 4), dtype=np.float32)
            rotation_matrices = np.zeros((num_frames, 3, 3), dtype=np.float32)
            for t in range(num_frames):
                pose = mesh_traj[t]
                pose_mat = transform_utils.convert_pose_quat2mat(pose)
                pose_matrices[t] = pose_mat
                rotation_matrices[t] = pose_mat[:3, :3]

            mesh_world_flows = np.zeros((num_frames, n_points, 3), dtype=np.float32)
            mesh_world_normals = np.zeros((num_frames, n_points, 3), dtype=np.float32)

            transform_points_kernel(mesh_points, pose_matrices, mesh_world_flows)
            transform_normals_kernel(mesh_normals, rotation_matrices, mesh_world_normals)

            end_offset = point_offset + n_points
            world_flows[:, point_offset:end_offset] = mesh_world_flows
            world_normals[:, point_offset:end_offset] = mesh_world_normals
            for t in range(num_frames):
                world_colors[t, point_offset:end_offset] = mesh_colors

            point_offset = end_offset

        return {
            'flows': world_flows,
            'colors': world_colors,
            'normals': world_normals,
            'visibility': visibility,
        }


def decode_data(sample, domain):
    """
    Convert any ".npy" field in the sample from raw bytes to a NumPy array.
    Returns a new dict with the same keys, but arrays are loaded in memory.

    Args:
        sample: Raw sample from webdataset
        domain: Dataset domain (e.g., 'droid', 'behavior')
    """
    decoded_sample = {}

    # Verify image/camera data exists.
    camera_keys = []
    for key in sample.keys():
        if key.endswith('_initial_rgb.jpg'):
            camera_key = key.replace('_initial_rgb.jpg', '')
            if camera_key not in camera_keys:
                camera_keys.append(camera_key)
    assert len(camera_keys) > 0, "No camera keys found in sample"

    # Assert that we have the required keys for each camera
    for camera_key in camera_keys:
        required_keys = [f'{camera_key}_initial_rgb.jpg', f'{camera_key}_initial_depth.npy',
                         f'{camera_key}_intrinsic.npy', f'{camera_key}_extrinsic.npy']
        for req_key in required_keys:
            assert req_key in sample, f"Required key {req_key} not found in sample"

    for k, v in sample.items():
        if k.endswith(".npy"):
            new_key = k.split(".npy")[0]
            # v is raw bytes, so use np.load
            decoded_sample[new_key] = np.load(io.BytesIO(v))
            if isinstance(decoded_sample[new_key], np.ndarray) and decoded_sample[new_key].dtype == np.float16:
                decoded_sample[new_key] = decoded_sample[new_key].astype(np.float32)
            # Release dataset contract: scene normals must be int8 quantized and decoded with fixed scale.
            if new_key.endswith('_scene_normals'):
                decoded_sample[new_key] = _dequantize_quantized_normals_array(decoded_sample[new_key], new_key)
            # Handle depth data conversion from uint16 mm to float32 meters
            if 'initial_depth' in new_key and isinstance(decoded_sample[new_key], np.ndarray):
                if decoded_sample[new_key].dtype == np.uint16:
                    decoded_sample[new_key] = decoded_sample[new_key].astype(np.float32) / 1000.0
        elif k.endswith(".jpg"):
            # Handle JPEG RGB data
            new_key = k.split(".jpg")[0]
            jpeg_data = np.frombuffer(v, dtype=np.uint8)
            decoded_img = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
            if decoded_img is not None:
                # Convert BGR to RGB
                decoded_sample[new_key] = decoded_img[..., ::-1]
            else:
                raise RuntimeError(f"Failed to decode JPEG data for key {k}")
        elif k.endswith(".pyd") or k.endswith(".pkl"):
            # Handle pickled python dicts/objects (e.g., local points/normals/colors, trajectories)
            new_key = k.rsplit(".", 1)[0]
            decoded_sample[new_key] = pickle.loads(v)
            if new_key.endswith('_local_scene_normals'):
                decoded_sample[new_key] = _dequantize_quantized_normals_dict(decoded_sample[new_key], new_key)
            # Handle joint_names decoding for behavior domain
            if 'joint_names' in new_key:
                decoded_sample[new_key] = [name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else str(name)
                                         for name in decoded_sample[new_key]]
            # Handle clip_attributes for behavior domain - ensure they're accessible at the sample level
            if 'clip_attributes' in new_key and ('behavior' in domain):
                # Flatten clip attributes into the sample dict for easy access
                assert isinstance(decoded_sample[new_key], dict), f"clip_attributes must be a dict, got {type(decoded_sample[new_key])}"
                for attr_key, attr_value in decoded_sample[new_key].items():
                    decoded_sample[f'__{attr_key}__'] = attr_value
        else:
            try:
                decoded_sample[k] = int(v)
            except (TypeError, ValueError):
                decoded_sample[k] = v

    # For droid domain, convert gripper pose from wrist to TCP by applying a fixed wrist→TCP transform.
    # Leave renaming/flags/shape normalization to canonicalize_gripper_keys_and_flags.
    if 'droid' in domain:
        if 'gripper_pose' in decoded_sample:
            wrist2world = transform_utils.convert_pose_quat2mat(decoded_sample['gripper_pose'])  # (T,4,4) or (4,4)
            tcp2wrist = np.linalg.inv(DROID_WRIST2TCP)
            tcp2world = wrist2world @ tcp2wrist
            decoded_sample['gripper_pose'] = transform_utils.convert_pose_mat2quat(tcp2world)

    # For behavior domain, convert gripper pose from OpenGL to OpenCV coordinate frame
    elif 'behavior' in domain:
        for key in decoded_sample.keys():
            if key.endswith('gripper_pose'):
                gripper2world = transform_utils.convert_pose_quat2mat(decoded_sample[key])  # (T,4,4) or (4,4)
                gripper2world_opencv = gripper2world @ B1K_GRIPPER_TRANSFORM
                decoded_sample[key] = transform_utils.convert_pose_mat2quat(gripper2world_opencv)
    return decoded_sample


def _get_droid_scene_data(sample: dict) -> dict:
    """
    Process droid domain scene data directly from sample.

    Args:
        sample: WebDataset sample dictionary

    Returns:
        Dictionary with camera scene data in standardized format
    """
    scene_data = {}
    # Get camera keys from sample
    camera_keys = set()
    for key in sample.keys():
        if key.endswith('_scene_flows'):
            camera_keys.add(key.split('_scene_flows')[0])
    # Process each camera
    for camera_key in sorted(list(camera_keys)):
        # Load scene data directly (already in world coordinates)
        flows_key = f"{camera_key}_scene_flows"
        assert flows_key in sample, f"Expected key {flows_key} in sample but got {sample.keys()}"
        scene_data[camera_key] = {
            'scene_flows': sample[flows_key],
            'scene_colors': sample[f"{camera_key}_scene_colors"],
            'scene_normals': sample[f"{camera_key}_scene_normals"],
            'scene_visibility': sample[f"{camera_key}_scene_visibility"],
            'scene_depth_valid_mask': sample[f"{camera_key}_scene_depth_valid_mask"],
        }
        # Add camera image data
        for img_key in ['initial_rgb', 'initial_depth', 'intrinsic', 'extrinsic']:
            key_new = f"{camera_key}_{img_key}"
            assert key_new in sample, f"Expected key {key_new} in sample"
            scene_data[camera_key][img_key.replace('initial_', '')] = sample[key_new]
    return scene_data


def _flatten_camera_scene_data(decoded_scene_data: dict, raw_sample: dict) -> dict:
    """Turn per-camera decoded scene dict into flattened key-value pairs.

    - Adds scene_flows/colors/normals/visibility and depth_valid_mask
    - Reattaches image/intrinsic/extrinsic fields from the raw sample if present
    """
    camera_data = {}
    for camera_key, camera_info in decoded_scene_data.items():
        camera_data[f"{camera_key}_scene_flows"] = camera_info['scene_flows']
        camera_data[f"{camera_key}_scene_normals"] = camera_info['scene_normals']
        camera_data[f"{camera_key}_scene_colors"] = camera_info['scene_colors']
        camera_data[f"{camera_key}_scene_visibility"] = camera_info['scene_visibility']
        camera_data[f"{camera_key}_scene_depth_valid_mask"] = (
            camera_info['scene_depth_valid_mask'] if 'scene_depth_valid_mask' in camera_info
            else np.ones_like(camera_info['scene_visibility'])
        )
        for suffix in ['_initial_rgb', '_initial_depth', '_intrinsic', '_extrinsic']:
            full_key = f"{camera_key}{suffix}"
            if full_key in raw_sample:
                camera_data[full_key] = raw_sample[full_key]
    return camera_data


def build_flow_sample(sample, domain, robot_sampler, max_robot_points: int,
                      deterministic: bool = False, seed: int | None = None, force_single_arm: bool = False):
    """
    Build unified sample dict with scene and robot flows for both 'behavior' (sim) and 'droid' (real).

    Assumes all blob decoding of types (.jpg/.npy/.pyd) was already handled in decode_data.
    """
    # Determine gripper filter for behavior domain
    gripper_filter = 'both'
    forced_side: str | None = None
    if 'behavior' in domain:
        gripper_filter = determine_gripper_filter(sample, deterministic=deterministic, seed=seed)
        if force_single_arm and gripper_filter == 'both':
            forced_side = _deterministic_single_arm_choice(sample, seed=seed)

    # Get scene data based on domain
    if 'behavior' in domain:
        # Use SimDataDecoder for behavior (simulation) data
        sim_decoder = SimDataDecoder()
        decoded_scene_data = sim_decoder.decode_wds(sample)
    elif 'droid' in domain:
        # Handle droid (real) data directly
        decoded_scene_data = _get_droid_scene_data(sample)
    else:
        raise ValueError(f"Unsupported domain: {domain}. Only 'behavior' and 'droid' are supported.")
    sampler_filter = gripper_filter if not (force_single_arm and gripper_filter == 'both') else 'both'
    robot_data = _get_robot_flows(sample, robot_sampler, max_robot_points, domain, sampler_filter, seed=seed if deterministic else None)
    camera_data = _flatten_camera_scene_data(decoded_scene_data, sample)

    # Clean up filtered gripper data from the final sample dict for behavior domain
    combined_data = {**robot_data, **camera_data}
    if 'behavior' in domain:
        if gripper_filter != 'both':
            combined_data = _remove_filtered_gripper_data(combined_data, gripper_filter)
        elif forced_side is not None:
            combined_data = _apply_eval_single_arm_feature_mask(combined_data, forced_side)

    return combined_data
