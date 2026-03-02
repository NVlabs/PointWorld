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

import random
import numpy as np
import torch

import transform_utils
from dataset_components.constants import RELEASE_CONTEXT_HORIZON
from dataset_components.transforms import compute_flow_derivatives, compute_robot_distances
from dataset_components.utils import _stable_int_hash
from robot_sampler import (
    RobotSampler as TorchRobotSampler,
    convert_joints_to_dict,
    build_robotiq_joint_dict,
)


def determine_gripper_filter(sample, deterministic: bool = False, seed: int | None = None):
    """
    Helper function to determine which gripper(s) to use based on clip attributes.

    Skip a gripper if: no collision AND min distance > 20cm.
    Check left/right with 50% chance each. If one is skipped, the other is kept.
    """
    def should_skip(has_collision, min_distance):
        return not has_collision and min_distance > 0.2

    left_skip = should_skip(sample['__has_left_gripper_finger_collision__'],
                            sample['__left_min_distance_to_all_objects__'])
    right_skip = should_skip(sample['__has_right_gripper_finger_collision__'],
                             sample['__right_min_distance_to_all_objects__'])

    # Random 50/50 choice for which to check first
    if deterministic:
        key = str(sample.get('__key__', ''))
        base = 0 if seed is None else int(seed)
        h = _stable_int_hash(base, key)
        check_left_first = (h & 1) == 0
    else:
        check_left_first = random.random() < 0.5

    if check_left_first:
        if left_skip:
            return 'right'
        if right_skip:
            return 'left'
    else:
        if right_skip:
            return 'left'
        if left_skip:
            return 'right'

    return 'both'


def _deterministic_single_arm_choice(sample, seed: int | None = None) -> str:
    key = str(sample.get('__key__', ''))
    base = 0 if seed is None else int(seed)
    h = _stable_int_hash('single_arm_eval', base, key)
    return 'left' if (h & 1) else 'right'


def _apply_eval_single_arm_feature_mask(sample: dict, primary_side: str) -> dict:
    """Keep both-arm robot flows but collapse gripper features onto a single side."""
    if primary_side not in ('left', 'right'):
        return sample

    sample['__eval_single_arm_side__'] = primary_side

    # Ensure primary side data is exposed via the canonical RIGHT keys expected by single-arm models
    if primary_side == 'left':
        if 'left_gripper_pose' in sample:
            sample['right_gripper_pose'] = sample['left_gripper_pose']
        if 'left_gripper_open' in sample:
            sample['right_gripper_open'] = sample['left_gripper_open']

    sample['__has_right_gripper__'] = True
    sample['__has_left_gripper__'] = False

    # Drop left-specific keys so downstream feature gathering treats the sample as single-arm
    for key in ['left_gripper_pose', 'left_gripper_open']:
        if key in sample:
            sample.pop(key)
    return sample


def _allocate_bimanual_features(sample, feature_keys, fixed_dims, has_bimanual_robot):
    """
    Unified helper function for allocating fixed dimensions for bimanual robot features.
    """
    if has_bimanual_robot:
        T = sample['scene_flows'].shape[0]  # Get timesteps
        feat = np.zeros((T, fixed_dims), dtype=np.float32)

        # Fill available feature data into fixed slots
        # Map side -> fixed slot (RIGHT first)
        side_slot = {'right': 0, 'left': 1}

        # Fill slots deterministically by side
        for key in feature_keys:
            # Expect canonical keys: right_gripper_*, left_gripper_*
            if key.startswith('right_'):
                side = 'right'
            elif key.startswith('left_'):
                side = 'left'
            else:
                raise ValueError(f"Unexpected bimanual feature key: {key}")

            slot = side_slot[side]
            data = sample[key]  # (T, F)
            F = data.shape[-1]
            start, end = slot * F, (slot + 1) * F
            feat[:, start:end] = data
    else:
        # Original behavior for non-bimanual robot domains
        features = [sample[key] for key in sorted(feature_keys)]
        feat = np.concatenate(features, axis=-1)

    return feat


def gather_features(
    sample,
    robot_features=['robot_flows', 'robot_colors', 'robot_normals', 'gripper_open', 'robot_velocity', 'robot_acceleration'],
    scene_features=['scene_flows', 'scene_colors', 'scene_normals', 'gripper_open', 'dist2robot'],
    random_context_mode='fixed',
    context_horizon=RELEASE_CONTEXT_HORIZON,
    has_bimanual_robot=False,
    domain=None,
):
    T, NS = sample['scene_flows'].shape[:2]
    _, NR = sample['robot_flows'].shape[:2]

    # Precompute derivatives for robot and scene flows if needed
    if 'robot_velocity' in robot_features or 'robot_acceleration' in robot_features:
        robot_velocity, robot_acceleration = compute_flow_derivatives(sample['robot_flows'])
    # Check if dist2robot is needed
    need_robot_dist = ('dist2robot' in scene_features)

    # Precompute robot distances if needed
    all_dists = None
    if need_robot_dist:
        # Initialize arrays to store per-timestep robot distances
        all_dists = np.zeros((T, NS), dtype=np.float32)

        # Calculate for each timestep (always aggregate robot distances for all timesteps)
        assert context_horizon == RELEASE_CONTEXT_HORIZON, (
            f"the current implementation only supports context_horizon={RELEASE_CONTEXT_HORIZON}"
        )
        scene_points_0 = sample['scene_flows'][0]  # (NS, 3)
        for t in range(T):
            robot_points_t = sample['robot_flows'][t]  # (NR, 3)
            all_dists[t] = compute_robot_distances(scene_points_0, robot_points_t)

    # ------------------------------------------------
    # gather robot features
    # ------------------------------------------------
    feat_list = []
    for feat_name in robot_features:
        if feat_name == 'robot_colors':
            # use magenta for robot colors
            feat = np.ones((T, NR, 3), dtype=np.float32)  # (T, NR, 3)
            feat[:, :, 0] = 1.0  # Red
            feat[:, :, 1] = 0.0  # Green
            feat[:, :, 2] = 1.0  # Blue
        elif feat_name == 'gripper_open':
            if has_bimanual_robot:
                keys = [k for k in ['right_gripper_open', 'left_gripper_open'] if k in sample]
                feat = _allocate_bimanual_features(sample, keys, 2, has_bimanual_robot)
            else:
                feat = sample['right_gripper_open']  # (T, 1)
            feat = feat[:, np.newaxis].repeat(NR, axis=1)
        elif feat_name == 'robot_velocity':
            feat = robot_velocity  # Already computed above
        elif feat_name == 'robot_acceleration':
            feat = robot_acceleration  # Already computed above
        else:
            assert feat_name in sample, f"Expected key {feat_name} in sample"
            feat = sample[feat_name]
        feat_list.append(feat)

    # assert all features have the same shape except for the last dimension
    assert len(set(feat.shape[:2] for feat in feat_list)) == 1, (
        "All features must have the same shape except for the last dimension"
    )
    sample['robot_features'] = np.concatenate(feat_list, axis=-1)  # (T, NR, D)

    # ------------------------------------------------
    # gather scene features - only compute for first timestep (t=0) for memory efficiency
    # scene_features will be (1, NS, D) which can be expanded to (T, NS, D) if needed
    # ------------------------------------------------
    feat_list = []
    for feat_name in scene_features:
        if feat_name == 'gripper_open':
            if has_bimanual_robot:
                keys = [k for k in ['right_gripper_open', 'left_gripper_open'] if k in sample]
                gripper_open_all = _allocate_bimanual_features(sample, keys, 2, has_bimanual_robot)
                gripper_open_flattened = gripper_open_all.reshape(T, -1)  # (T, 2)
            else:
                gripper_open_flattened = sample['right_gripper_open'].reshape(T, -1)  # (T, 1)
            feat = gripper_open_flattened[None, None, :, :].repeat(NS, axis=1).reshape(1, NS, -1)
        elif feat_name == 'dist2robot':
            assert random_context_mode == 'fixed', (
                "the current implementation of dist2robot is only supported for fixed context mode"
            )
            assert all_dists is not None, "dist2robot is needed but robot relations were not precomputed"
            # Aggregate dist2robot across all timesteps
            feat = all_dists.transpose(1, 0)  # (NS, T)
            feat = feat[None, :, :]  # (1, NS, T)
            assert feat.shape == (1, NS, T)
        else:
            assert feat_name in sample, f"Expected key {feat_name} in sample"
            assert sample[feat_name].shape[0] == T, (
                f"Expected {T} timesteps for {feat_name}, got {sample[feat_name].shape[0]}"
            )
            feat = sample[feat_name][0:1]  # (1, NS, D)
        feat_list.append(feat)

    # assert all features have the same shape except for the last dimension
    assert len(set(feat.shape[:2] for feat in feat_list)) == 1, (
        "All scene features must have the same shape except for the last dimension"
    )
    sample['scene_features'] = np.concatenate(feat_list, axis=-1)  # (1, NS, D)
    return sample


def _get_robot_flows(sample, robot_sampler, max_robot_points: int, domain: str,
                         gripper_filter: str = 'both', seed: int | None = None) -> dict:
    if 'behavior' in domain:
        return _get_robot_flows_behavior(sample, robot_sampler, max_robot_points, gripper_filter, seed=seed)
    if 'droid' in domain:
        return _get_robot_flows_droid(sample, robot_sampler, max_robot_points, seed=seed)
    raise ValueError(f"Unsupported domain: {domain}. Only 'behavior' and 'droid' are supported.")


def _remove_filtered_gripper_data(sample, gripper_filter):
    """
    Remove gripper data for filtered out grippers from the sample dictionary.
    """
    if gripper_filter == 'both':
        return sample

    # Create a copy to avoid modifying the original
    cleaned_sample = sample.copy()

    # Determine which gripper prefixes to remove
    if gripper_filter == 'left':
        prefixes_to_remove = ['right_gripper']
    elif gripper_filter == 'right':
        prefixes_to_remove = ['left_gripper']
    else:
        # No filtering needed
        return cleaned_sample

    # Remove keys that start with the filtered gripper prefixes
    keys_to_remove = []
    for key in cleaned_sample.keys():
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                keys_to_remove.append(key)
                break

    for key in keys_to_remove:
        del cleaned_sample[key]

    return cleaned_sample


def canonicalize_gripper_keys_and_flags(sample: dict) -> dict:
    """Ensure the sample always has right_/left_ gripper pose/open with valid shapes,
    plus presence flags. Absent sides are filled with identity pose and open=0."""
    T = sample['scene_flows'].shape[0]

    def id_pose_T7():
        # [x,y,z,qx,qy,qz,qw] with identity quaternion
        return np.tile(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)[None, :], (T, 1))

    def zeros_T1():
        return np.zeros((T, 1), dtype=np.float32)

    # If singular DROID-style keys exist, map them into RIGHT, then remove singular keys
    if 'gripper_pose' in sample and 'right_gripper_pose' not in sample:
        sample['right_gripper_pose'] = sample.pop('gripper_pose')
    if 'gripper_open' in sample and 'right_gripper_open' not in sample:
        g = sample.pop('gripper_open')
        if isinstance(g, np.ndarray) and g.ndim == 1:
            g = g.reshape(-1, 1)
        sample['right_gripper_open'] = g

    has_R_pose = 'right_gripper_pose' in sample
    has_L_pose = 'left_gripper_pose' in sample
    has_R_open = 'right_gripper_open' in sample
    has_L_open = 'left_gripper_open' in sample

    if not has_R_pose:
        sample['right_gripper_pose'] = id_pose_T7()
    if not has_L_pose:
        sample['left_gripper_pose'] = id_pose_T7()
    if not has_R_open:
        g = sample.get('gripper_open', None)
        sample['right_gripper_open'] = (
            g.reshape(-1, 1) if (isinstance(g, np.ndarray) and g.ndim in (1, 2)) else zeros_T1()
        )
    if not has_L_open:
        sample['left_gripper_open'] = zeros_T1()

    # Presence flags (prefer preexisting flags if set earlier)
    if '__has_right_gripper__' not in sample:
        sample['__has_right_gripper__'] = has_R_pose or has_R_open
    if '__has_left_gripper__' not in sample:
        sample['__has_left_gripper__'] = has_L_pose or has_L_open

    # Normalize shapes
    sample['right_gripper_open'] = sample['right_gripper_open'].reshape(T, 1)
    sample['left_gripper_open'] = sample['left_gripper_open'].reshape(T, 1)
    sample['right_gripper_pose'] = sample['right_gripper_pose'].reshape(T, 7)
    sample['left_gripper_pose'] = sample['left_gripper_pose'].reshape(T, 7)
    return sample


def _get_robot_flows_behavior(sample, robot_sampler: TorchRobotSampler, max_robot_points: int,
                             gripper_filter: str = 'both', seed: int | None = None):
    """
    Get robot flows for behavior domain using the robot sampler.
    """
    # Presample robot points with gripper filtering
    robot_sampler.presample(max_robot_points, gripper_filter=gripper_filter, seed=seed)

    # Convert joint positions to the format expected by GPU robot sampler
    joint_tensor = torch.from_numpy(sample['joint_positions']).float()  # (T, 22)

    # Handle joint reordering if joint_names are available
    assert 'joint_names' in sample, "joint_names are required for behavior domain"
    # Build index mapping from sampler order -> saved order
    name_to_idx = {name: idx for idx, name in enumerate(sample['joint_names'])}
    effective_joint_names = [name for name in robot_sampler.joint_names if name in name_to_idx]
    assert effective_joint_names, (
        f"No matching joint names found between robot_sampler ({robot_sampler.joint_names}) and saved data ({sample['joint_names']}). "
    )
    reorder_indices = torch.tensor([name_to_idx[name] for name in effective_joint_names], dtype=torch.long)
    joint_dict, _ = convert_joints_to_dict(joint_tensor[:, reorder_indices], effective_joint_names)

    # Generate robot points in robot frame
    robot_flows, robot_colors, robot_normals = robot_sampler.compute_points(joint_dict)

    # Transform robot points to world frame using base pose if available
    assert 'base_pose' in sample, "base_pose is required for behavior domain with mobile base"
    T_b = transform_utils.convert_pose_quat2mat(sample['base_pose'])  # (T, 4, 4)
    robot_flows = robot_flows.cpu().numpy()  # (T, N, 3) in robot/base frame
    robot_normals = robot_normals.cpu().numpy()  # (T, N, 3) in robot/base frame
    robot_colors = robot_colors.cpu().numpy()
    # Apply per-frame rigid transform: for each t, apply T_b[t] to robot_flows[t]
    ones = np.ones(robot_flows.shape[:-1] + (1,), dtype=robot_flows.dtype)  # (T, N, 1)
    pts_h = np.concatenate([robot_flows, ones], axis=-1)  # (T, N, 4)
    # (T,4,4) @ (T,N,4)^T -> (T,N,4), then drop homogeneous coord
    robot_flows = np.einsum('tij,tkj->tki', T_b, pts_h)[..., :3]
    # Rotate normals with rotation part only, per-frame
    R_b = T_b[:, :3, :3]  # (T,3,3)
    robot_normals = np.einsum('tij,tkj->tki', R_b, robot_normals)

    # Convert colors to float32 [0,1]
    if robot_colors.dtype == np.uint8:
        robot_colors = robot_colors.astype(np.float32) / 255.0

    # Store robot data
    robot_data = {
        'robot_flows': robot_flows,
        'robot_normals': robot_normals,
        'robot_colors': robot_colors,
    }
    for key in sample.keys():
        if (not key.startswith('camera_') and
            not key.endswith('.pyd') and
            not key.endswith('.npy')):
            robot_data[key] = sample[key]

    return robot_data


def _get_robot_flows_droid(sample, robot_sampler: TorchRobotSampler, max_robot_points: int,
                               seed: int | None = None):
    """
    Get robot flows for droid domain using the robot sampler.
    """
    robot_sampler.presample(max_robot_points, seed=seed)

    device = robot_sampler.device
    dtype = robot_sampler.dtype
    joint_positions = torch.as_tensor(sample['joint_positions'], device=device, dtype=dtype)  # (T, 7)
    gripper_positions = torch.as_tensor(sample['gripper_positions'], device=device, dtype=dtype).reshape(-1)  # (T,)

    expected_joints = [f'panda_joint{i}' for i in range(1, 8)]
    missing = [name for name in expected_joints if name not in robot_sampler.joint_names]
    if missing:
        raise KeyError(f"Missing expected DROID joints in URDF: {missing}")
    if 'finger_joint' not in robot_sampler.joint_names:
        raise KeyError("Expected 'finger_joint' in URDF for Robotiq gripper")

    joint_dict = {}
    for idx, name in enumerate(expected_joints):
        joint_dict[name] = joint_positions[:, idx].reshape(-1)
    joint_dict.update(build_robotiq_joint_dict(gripper_positions, robot_sampler.joint_names))

    robot_flows, robot_colors, robot_normals = robot_sampler.compute_points(joint_dict)
    robot_flows = robot_flows.cpu().numpy()
    robot_normals = robot_normals.cpu().numpy()
    robot_colors = robot_colors.cpu().numpy()

    robot_data = {
        'robot_flows': robot_flows,
        'robot_normals': robot_normals,
        'robot_colors': robot_colors,
    }
    for key in sample.keys():
        if not key.startswith('camera_'):
            clean_key = key.split('.')[0]
            robot_data[clean_key] = sample[key]
    return robot_data


__all__ = [
    "determine_gripper_filter",
    "_deterministic_single_arm_choice",
    "_apply_eval_single_arm_feature_mask",
    "_allocate_bimanual_features",
    "gather_features",
    "_get_robot_flows",
    "_remove_filtered_gripper_data",
    "canonicalize_gripper_keys_and_flags",
    "_get_robot_flows_behavior",
    "_get_robot_flows_droid",
]
