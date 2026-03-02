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
import cv2
from numba import njit
from scipy.spatial import cKDTree

import transform_utils
from dataset_components.constants import (
    DEFAULT_SOFT_SELECTOR_TAU,
    DEFAULT_SOFT_SELECTOR_TEMP_SCALE,
    RELEASE_CONTEXT_HORIZON,
)
from dataset_components.utils import _rotate_xy, _stable_int_hash, fnv_hash_vec_nb
from utils import make_soft_selector_labels


@njit(cache=True, fastmath=True, nogil=True)
def _compute_bounds_mask_all_timesteps(flows, bounds_min, bounds_max):
    """Compute bounds mask for all timesteps at once."""
    T, N = flows.shape[0], flows.shape[1]
    keep_mask = np.ones(N, dtype=np.bool_)

    for t in range(T):
        for i in range(N):
            x, y, z = flows[t, i, 0], flows[t, i, 1], flows[t, i, 2]
            if not (bounds_min[0] < x < bounds_max[0] and
                    bounds_min[1] < y < bounds_max[1] and
                    bounds_min[2] < z < bounds_max[2]):
                keep_mask[i] = False
    return keep_mask


def _erase_scene_indices(sample: dict, idx_remove: np.ndarray) -> None:
    """
    In-place helper. Deletes exactly the same point indices from every
    scene-level array whose 2nd dim is N_scene.
    """
    if idx_remove.size == 0:
        return
    for k, v in sample.items():
        if "scene" not in k:
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] >= idx_remove.max() + 1:
            sample[k] = np.delete(v, idx_remove, axis=1)


def sphere_crop_transform(
    sample: dict,
    prob: float = 0.8,
    min_radius: float = 0.03,
    max_radius: float = 0.06,
    buffer: float = 0.05,
    max_num_spheres: int = 3,
    max_scene_points: int = 10000,
    num_candidates: int = 10,
) -> dict:
    """
    Multi-sphere probabilistic SphereCrop for temporal scene/robot flow data.
    """
    if np.random.rand() > prob:
        return sample

    robot_flat = sample["robot_flows"].reshape(-1, 3)
    robot_kdt = cKDTree(robot_flat)
    spheres_used = 0

    # Ensure at least one crop is made, then continue while we have too many points
    while (spheres_used < max_num_spheres and
           (spheres_used == 0 or sample["scene_flows"].shape[1] > max_scene_points)):
        pts0 = sample["scene_flows"][0]
        dist2robot, _ = robot_kdt.query(pts0, k=1)

        # pick the top-K farthest from robot
        cand_idx = np.argsort(-dist2robot)[:num_candidates]

        best_gain = 0
        best_center = None
        best_r = None

        scene_kdt = cKDTree(pts0)

        for idx in cand_idx:
            d = dist2robot[idx]
            if d <= buffer + min_radius:
                continue
            r = min(max_radius, max(min_radius, d - buffer))
            # just get the count, not the full list
            cnt = len(scene_kdt.query_ball_point(pts0[idx], r))
            if cnt > best_gain:
                best_gain = cnt
                best_center = idx
                best_r = r

        if best_center is None or best_gain == 0:
            break

        # now remove them for real
        idx_remove = scene_kdt.query_ball_point(pts0[best_center], best_r)
        _erase_scene_indices(sample, np.array(idx_remove, dtype=np.int64))
        spheres_used += 1

    assert sample["scene_flows"].shape[1] > 0, (
        f"no points left after sphere crop (scene_flows shape: {sample['scene_flows'].shape})"
    )
    return sample


def grid_sample_transform(sample, grid_size, mode):
    """
    Applies voxel-based downsampling (grid sampling) to "scene_flows" and
    "robot_flows". We collapse each unique voxel to exactly one point:
      - in train mode: pick a random point per voxel
      - in test mode: pick the first point per voxel
    Importantly, we only sample at time t=0 to preserve temporal correspondences
    across frames, and then apply the selected indices to all timesteps.

    Args:
        sample (dict):
            - sample['scene_flows']: (T, N, 3)
            - sample['robot_flows']: (T, N, 3), etc.
        grid_size (float): voxel size in world units
        mode (str): 'train' or 'test'

    Returns:
        sample (dict): updated in-place
    """

    def select_indices(points_t0):
        """
        Given points at t=0, return the set of indices that survive the grid-sample.
        We shift the coordinates by subtracting min, hash them, and pick one point
        (random or first) per voxel.
        """
        # shape (N, 3)
        if points_t0.shape[0] == 0:
            return np.array([], dtype=np.int64)

        # Scale by grid size, then floor => integer grid coords
        scaled = points_t0 / grid_size
        grid_coord = np.floor(scaled).astype(np.int64)

        # Subtract min for consistent hashing on nonnegative coords
        min_coord = grid_coord.min(axis=0)
        grid_coord -= min_coord

        # Hash the grid coordinates
        keys = fnv_hash_vec_nb(grid_coord)

        # Sort by key
        idx_sort = np.argsort(keys)
        keys_sorted = keys[idx_sort]

        # unique => find one index per voxel
        _, inv, counts = np.unique(keys_sorted, return_inverse=True, return_counts=True)
        cumsum_counts = np.cumsum(np.insert(counts, 0, 0)[:-1])

        if mode == "train":
            # Pick a random index offset within each voxel
            offsets = np.random.randint(0, counts.max(), size=counts.shape) % counts
        else:
            # "test": pick the first index in each voxel
            offsets = np.zeros_like(counts, dtype=np.int64)

        idx_picked = cumsum_counts + offsets
        # reorder to original indexing
        idx_selected = idx_sort[idx_picked]
        return idx_selected

    def apply_selection_to_times(sample_dict, prefix, idx_selected):
        """
        Apply the selected indices (idx_selected) to all timesteps
        for a given prefix: "scene" or "robot".
        Example keys: scene_flows, scene_colors, scene_normals, ...
        """
        base_key = f"{prefix}_flows"
        if base_key not in sample_dict:
            return
        T = sample_dict[base_key].shape[0]
        N_og = sample_dict[base_key].shape[1]

        # Flows
        sample_dict[base_key] = sample_dict[base_key][:, idx_selected, :]

        # Colors, normals, etc. if they exist
        possible_extras = [
            f"{prefix}_colors",
            f"{prefix}_normals",
            f"{prefix}_visibility",
            f"{prefix}_depth_valid_mask",
        ]
        for ex_key in possible_extras:
            if ex_key in sample_dict and sample_dict[ex_key].shape[0] == T and sample_dict[ex_key].shape[1] == N_og:
                sample_dict[ex_key] = sample_dict[ex_key][:, idx_selected]

    # ---------------------------
    # 1) Scene Particles
    # ---------------------------
    if "scene_flows" in sample and sample["scene_flows"].shape[1] > 0:
        idx_scene = select_indices(sample["scene_flows"][0])
        apply_selection_to_times(sample, "scene", idx_scene)

    # ---------------------------
    # 2) Robot Particles
    # ---------------------------
    if "robot_flows" in sample and sample["robot_flows"].shape[1] > 0:
        idx_robot = select_indices(sample["robot_flows"][0])
        apply_selection_to_times(sample, "robot", idx_robot)

    return sample


def filter_within_bounds(sample, bounds_min=[-3.0, -3.0, -3.0], bounds_max=[3.0, 3.0, 3.0]):
    """
    Filter points to be within the bounds.
    For scene_flows, if a point is out of bounds at ANY timestep,
    remove that point from all timesteps.
    """
    scene_flows = sample['scene_flows']
    T = scene_flows.shape[0]
    N = scene_flows.shape[1]

    # Convert bounds to numpy arrays with correct dtype
    if not isinstance(bounds_min, np.ndarray):
        bounds_min = np.array(bounds_min, dtype=scene_flows.dtype)
    if not isinstance(bounds_max, np.ndarray):
        bounds_max = np.array(bounds_max, dtype=scene_flows.dtype)

    # Use numba-optimized function to compute bounds mask for all timesteps at once
    keep_mask = _compute_bounds_mask_all_timesteps(scene_flows, bounds_min, bounds_max)

    if keep_mask.sum() < N:
        prefix = f"[{sample['__key__']}] " if '__key__' in sample else ''
        print(f"{prefix}Scene points out of bounds (keep percentage: {keep_mask.sum() / N:.6%})")
        sample['__out_of_bounds__'] = True
    else:
        sample['__out_of_bounds__'] = False

    # Apply the keep_mask to all scene_* tensors
    N_og = scene_flows.shape[1]
    for k, v in sample.items():
        if '__' not in k and 'scene_' in k and v.shape[0] == T and v.shape[1] == N_og:
            if v.ndim == 3:
                sample[k] = v[:, keep_mask, :]
            elif v.ndim == 2:
                sample[k] = v[:, keep_mask]
            else:
                raise ValueError(f'{k} must have 2 or 3 dimensions')
    assert sample['scene_flows'].shape[1] > 0, "No scene points left after filtering"
    return sample


def random_rotate_around_z_axis(sample):
    """
    Same signature/semantics, but heavy arithmetic is done by Numba-JIT helpers.
    """
    dtype = sample['scene_flows'].dtype
    angle = np.random.uniform(-np.pi, np.pi)
    cos_a, sin_a = np.cos(angle).astype(dtype), np.sin(angle).astype(dtype)

    # centre of the first frame's scene cloud: compute once
    xyz0 = sample['scene_flows'][0]
    center = (xyz0.min(0) + xyz0.max(0)) * 0.5

    # Shift, rotate, shift back - scene
    sample['scene_flows'] -= center
    sample['scene_flows'] = np.require(sample['scene_flows'], requirements=['C'])
    _rotate_xy(sample['scene_flows'], cos_a, sin_a)
    sample['scene_flows'] += center

    # robot
    sample['robot_flows'] -= center
    sample['robot_flows'] = np.require(sample['robot_flows'], requirements=['C'])
    _rotate_xy(sample['robot_flows'], cos_a, sin_a)
    sample['robot_flows'] += center

    # gripper pose translation and rotation
    rot2 = np.array([[cos_a, -sin_a, 0.],
                     [sin_a,  cos_a, 0.],
                     [0.,     0.,    1.]], dtype=dtype)

    for key in sample.keys():
        if key.endswith('gripper_pose'):
            gposes = transform_utils.convert_pose_quat2mat(sample[key])  # (T,4,4)
            gposes[..., :3, :3] = rot2 @ gposes[..., :3, :3]
            t = gposes[..., :3, 3]
            gposes[..., :3, 3] = (rot2 @ t.T).T + (center - rot2 @ center)
            sample[key] = transform_utils.convert_pose_mat2quat(gposes)

    # TRANSFORM EXTRINSICS: The world transformation is T = translate(center) * rotate(angle) * translate(-center)
    # We need to apply E_new = E * T^(-1) where T^(-1) = translate(center) * rotate(-angle) * translate(-center)

    # Create the inverse transformation matrix T^(-1)
    T_inv = np.eye(4, dtype=dtype)
    T_inv[:3, :3] = rot2.T  # Inverse rotation (transpose)
    T_inv[:3, 3] = center - rot2.T @ center  # Translation part for the combined transformation

    for key in sample.keys():
        if key.endswith('_extrinsic'):
            extrinsic = sample[key]  # Should be (4, 4) transformation matrix
            if extrinsic.ndim == 2 and extrinsic.shape == (4, 4):
                # Apply: E_new = E * T_inv
                sample[key] = extrinsic @ T_inv
            elif extrinsic.ndim == 3:  # Multiple cameras: (num_cams, 4, 4)
                # Apply to each camera
                for i in range(extrinsic.shape[0]):
                    extrinsic[i] = extrinsic[i] @ T_inv

    # normals
    if 'scene_normals' in sample:
        sample['scene_normals'] = np.require(sample['scene_normals'], requirements=['C'])
        _rotate_xy(sample['scene_normals'], cos_a, sin_a)
    if 'robot_normals' in sample:
        sample['robot_normals'] = np.require(sample['robot_normals'], requirements=['C'])
        _rotate_xy(sample['robot_normals'], cos_a, sin_a)

    return sample


def center_shift(sample):
    dtype = sample['scene_flows'].dtype
    first_scene = sample['scene_flows'][0]
    first_robot = sample['robot_flows'][0]
    # Mean centering across scene and robot points (first frame)
    combined = np.concatenate([first_scene, first_robot], axis=0)
    shift = combined.mean(axis=0).astype(dtype)
    sample['scene_flows'] -= shift
    sample['robot_flows'] -= shift
    for key in sample.keys():
        if key.endswith('gripper_pose'):
            sample[key][:, :3] -= shift

    # TRANSFORM EXTRINSICS: When we translate world coordinates by -shift (P_world_new = P_world - shift),
    # the world transformation matrix is T = [[I, -shift], [0, 1]]
    # We need to apply E_new = E * T^(-1) where T^(-1) = [[I, +shift], [0, 1]]
    # This gives: E_new = [[E_rot, E_trans + E_rot @ shift], [0, 1]]

    for key in sample.keys():
        if key.endswith('_extrinsic'):
            extrinsic = sample[key]  # Should be (4, 4) transformation matrix
            if extrinsic.ndim == 2 and extrinsic.shape == (4, 4):
                # Apply translation: E_new[:3, 3] = E[:3, 3] + E[:3, :3] @ shift
                extrinsic[:3, 3] = extrinsic[:3, 3] + extrinsic[:3, :3] @ shift
            elif extrinsic.ndim == 3:  # Multiple cameras: (num_cams, 4, 4)
                # Apply to each camera
                extrinsic[:, :3, 3] = extrinsic[:, :3, 3] + np.einsum('nij,j->ni', extrinsic[:, :3, :3], shift)

    if '__shift_amount__' in sample:
        sample['__shift_amount__'] -= shift
    else:
        sample['__shift_amount__'] = -shift
    return sample


def normalize_colors(sample):
    dtype = sample['scene_flows'].dtype
    sample['scene_colors'] = sample['scene_colors'].astype(dtype) / 255.0
    sample['robot_colors'] = sample['robot_colors'].astype(dtype) / 255.0
    return sample


def make_gt_copy(sample):
    # Make gt copy for scene flows (robot flows are only needed for visualization)
    sample['gt_scene_flows'] = sample['scene_flows'].copy()
    return sample


def sample_and_apply_scene_context_mask(
    sample,
    random_context_mode='fixed',
    context_horizon=RELEASE_CONTEXT_HORIZON,
):
    """
    sample and apply context mask
    """
    # ------------------------------------------------
    # sample context mask
    # ------------------------------------------------
    T, NS = sample['scene_flows'].shape[:2]
    if random_context_mode != 'fixed':
        raise ValueError(f"Release supports only random_context_mode='fixed' (got {random_context_mode})")
    scene_context_mask = np.ones((T, NS))
    scene_context_mask[context_horizon:, :] = 0
    scene_context_mask = scene_context_mask[..., np.newaxis]  # (T, NS, 1)
    sample['scene_context_mask'] = scene_context_mask  # (T, NS, 1)
    # ------------------------------------------------
    # apply context mask
    # ------------------------------------------------
    assert 'gt_scene_flows' in sample, "Expected key 'gt_scene_flows' in sample, so we don't overwrite gt values"
    # iterate through each timestep, if mask is 0, replace with last timestep
    # only apply context mask to scene_flows since we made gt copy of it at this point
    for t in range(1, T):
        sample['scene_flows'][t] = np.where(scene_context_mask[t], sample['gt_scene_flows'][t], sample['scene_flows'][t-1])
        sample['scene_colors'][t] = np.where(scene_context_mask[t], sample['scene_colors'][t], sample['scene_colors'][t-1])
        sample['scene_normals'][t] = np.where(scene_context_mask[t], sample['scene_normals'][t], sample['scene_normals'][t-1])
    return sample


def compute_flow_derivatives(flows):
    """
    Compute velocity and acceleration using mid-point method.

    Args:
        flows: (T, N, 3) array of point positions over time
    Returns:
        velocity: (T, N, 3) array of velocities
        acceleration: (T, N, 3) array of accelerations
    """
    T, N, D = flows.shape
    velocity = np.zeros_like(flows)
    acceleration = np.zeros_like(flows)

    # Compute velocity
    if T >= 2:
        # Forward diff for first timestep
        velocity[0] = flows[1] - flows[0]

        # Mid-point method for middle timesteps
        if T >= 3:
            for t in range(1, T-1):
                velocity[t] = (flows[t+1] - flows[t-1]) / 2.0

            # Backward diff for last timestep
            velocity[T-1] = flows[T-1] - flows[T-2]

    # Compute acceleration
    if T >= 3:
        # Forward diff for first timestep
        acceleration[0] = velocity[1] - velocity[0]

        # Mid-point method for middle timesteps
        if T >= 4:
            for t in range(1, T-1):
                acceleration[t] = (velocity[t+1] - velocity[t-1]) / 2.0

            # Backward diff for last timestep
            acceleration[T-1] = velocity[T-1] - velocity[T-2]

    return velocity, acceleration


def compute_robot_distances(scene_points, robot_points):
    """
    Compute distances from each scene point to the nearest robot point.

    Args:
        scene_points: (NS, 3) array of scene point positions
        robot_points: (NR, 3) array of robot point positions
    Returns:
        dists: (NS,) array of distances
    """
    tree = cKDTree(robot_points)
    dists, _ = tree.query(scene_points)
    return dists


def random_scale_transform(
    sample: dict,
    scale_range=(0.95, 1.05),
) -> dict:
    """
    Uniform scaling applied to every timestep and to the translation of gripper_pose.
    Orientations are unchanged.
    """
    dtype = sample['scene_flows'].dtype
    s = np.asarray(np.random.uniform(*scale_range), dtype=dtype)  # scalar

    # --- flows ---
    for key, arr in sample.items():
        if key.endswith('_flows'):
            sample[key] = arr * s

    # --- normals stay unit length, no change needed ---

    # --- gripper pose ---
    for key in sample.keys():
        if key.endswith('gripper_pose'):
            sample[key][:, :3] *= s

    # TRANSFORM EXTRINSICS: When we scale world coordinates by factor s (P_world_new = s * P_world),
    # the world transformation matrix is T = [[s*I, 0], [0, 1]]
    # We need to apply E_new = E * T^(-1) where T^(-1) = [[(1/s)*I, 0], [0, 1]]
    # This gives: E_new = [[E_rot * (1/s), E_trans], [0, 1]]
    # Note: Only the rotation part is scaled by 1/s, translation part stays unchanged

    inv_s = 1.0 / s  # Inverse scaling factor (scalar)
    for key in sample.keys():
        if key.endswith('_extrinsic'):
            extrinsic = sample[key]  # Should be (4, 4) transformation matrix
            if extrinsic.ndim == 2 and extrinsic.shape == (4, 4):
                # Scale only the rotation part by inverse scaling factor
                extrinsic[:3, :3] *= inv_s
                # Translation part remains unchanged
            elif extrinsic.ndim == 3:  # Multiple cameras: (num_cams, 4, 4)
                extrinsic[:, :3, :3] *= inv_s
                # Translation part remains unchanged

    # --- any cached shifts ---
    if '__shift_amount__' in sample:
        sample['__shift_amount__'] *= s

    return sample


def random_flip_transform(sample: dict, p: float = 0.5) -> dict:
    """
    With probability p, reflect the whole scene across a random axis
    (x, y) but not z. Handles flows, normals, and the full 4x4 gripper pose.
    """
    if np.random.rand() > p:
        return sample

    dtype = sample['scene_flows'].dtype
    axis = np.random.choice([0, 1])  # 0->x, 1->y

    # reflection on coordinates
    sign = np.array([[-1, 1, 1],
                     [1, -1, 1]], dtype=dtype)[axis]
    S3 = np.diag(sign)  # (3,3)
    S4 = np.eye(4, dtype=dtype)
    S4[:3, :3] = S3  # (4,4)

    # --- flows & normals ---
    for key in list(sample.keys()):
        if key.endswith('_flows'):
            sample[key] = sample[key] * sign
        if key.endswith('_normals'):
            sample[key] = sample[key] * sign

    # --- gripper pose (proper SE(3) after flip) ---
    for key in sample.keys():
        if key.endswith('gripper_pose'):
            Tmats = transform_utils.convert_pose_quat2mat(sample[key])  # (T,4,4)
            # t' = S t, R' = S R S
            Tmats[..., :3, 3] = (S3 @ Tmats[..., :3, 3].T).T
            Tmats[..., :3, :3] = S3 @ Tmats[..., :3, :3] @ S3
            sample[key] = transform_utils.convert_pose_mat2quat(Tmats)

    # --- camera extrinsics: E' = E * S  (S^{-1} = S) ---
    for key in sample.keys():
        if key.endswith('_extrinsic'):
            extr = sample[key]
            if extr.ndim == 2:
                sample[key] = extr @ S4
            else:
                sample[key] = np.einsum('nij,jk->nik', extr, S4)

    if '__shift_amount__' in sample:
        sample['__shift_amount__'] *= sign

    return sample


def image_resize_transform(sample, scale_factor=1.0):
    """
    Resize initial_rgb and initial_depth images along with their intrinsic matrices.
    """
    if scale_factor == 1.0:
        return sample

    # Process all camera data in the sample
    for key in list(sample.keys()):
        # Process RGB images
        if key.endswith('_initial_rgb'):
            rgb_img = sample[key]  # Shape: (H, W, 3)
            h, w = rgb_img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            # Use INTER_AREA for downsampling, INTER_LINEAR for upsampling
            interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
            sample[key] = cv2.resize(rgb_img, (new_w, new_h), interpolation=interpolation)

        # Process depth images
        elif key.endswith('_initial_depth'):
            depth_img = sample[key]  # Shape: (H, W)
            h, w = depth_img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            # Use INTER_NEAREST for depth to preserve depth values
            sample[key] = cv2.resize(depth_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Process intrinsic matrices
        elif key.endswith('_intrinsic'):
            intrinsic = sample[key].copy()  # Shape: (3, 3)
            assert intrinsic.ndim == 2, "intrinsic should be (3, 3)"
            # Apply scaling to focal lengths and principal point
            intrinsic[0, 0] *= scale_factor  # fx
            intrinsic[1, 1] *= scale_factor  # fy
            intrinsic[0, 2] = scale_factor * (sample[key][0, 2])  # cx
            intrinsic[1, 2] = scale_factor * (sample[key][1, 2])  # cy
            sample[key] = intrinsic
    return sample


def assert_camera_payload_resolution(sample: dict, expected_hw=(180, 320)) -> dict:
    """
    Fail-fast check that all camera image payloads match the release contract resolution.

    This intentionally does not perform resizing. The data branch is responsible for
    finalizing camera payload resolution (e.g., DROID 640x360 -> 320x180).
    """
    expected_h, expected_w = int(expected_hw[0]), int(expected_hw[1])
    if expected_h <= 0 or expected_w <= 0:
        raise ValueError(f"expected_hw must be positive, got {expected_hw}")

    rgb_keys = [k for k in sample.keys() if k.endswith("_initial_rgb")]
    depth_keys = [k for k in sample.keys() if k.endswith("_initial_depth")]
    if not rgb_keys and not depth_keys:
        raise RuntimeError("No camera image payload keys found (*_initial_rgb / *_initial_depth)")

    for k in rgb_keys:
        v = sample[k]
        if not isinstance(v, np.ndarray):
            raise RuntimeError(f"{k} must be a numpy array, got {type(v)}")
        if v.ndim != 3 or v.shape[2] != 3:
            raise RuntimeError(f"{k} must have shape (H,W,3), got {v.shape}")
        h, w = int(v.shape[0]), int(v.shape[1])
        if (h, w) != (expected_h, expected_w):
            raise RuntimeError(f"{k} resolution mismatch: got {(h, w)} expected {(expected_h, expected_w)}")

    for k in depth_keys:
        v = sample[k]
        if not isinstance(v, np.ndarray):
            raise RuntimeError(f"{k} must be a numpy array, got {type(v)}")
        if v.ndim != 2:
            raise RuntimeError(f"{k} must have shape (H,W), got {v.shape}")
        h, w = int(v.shape[0]), int(v.shape[1])
        if (h, w) != (expected_h, expected_w):
            raise RuntimeError(f"{k} resolution mismatch: got {(h, w)} expected {(expected_h, expected_w)}")

    return sample


def chromatic_auto_contrast_transform(sample, p=0.2, blend_factor=-1, min_contrast_range=50.0, eps=1e-8):
    """
    Apply auto contrast to scene colors.
    Same contrast parameters applied to all time frames.
    """
    if np.random.rand() >= p:
        return sample
    if 'scene_colors' not in sample:
        raise ValueError("Expected 'scene_colors' in sample for chromatic auto contrast")

    # work on first frame in float
    t0 = sample['scene_colors'][0].astype(np.float32)
    lo = np.min(t0, axis=0, keepdims=True)
    hi = np.max(t0, axis=0, keepdims=True)

    # Calculate the range for each channel (in 0-255 range)
    color_range = hi - lo

    # Check if any RGB channel has small range (near-uniform color)
    has_small_range = np.any(color_range[:, :3] < min_contrast_range)

    # Determine adjusted blend factor based on color range
    if has_small_range:
        # Compute how uniform each channel is (0 = uniform, 1 = full range)
        channel_diversity = np.clip(color_range[:, :3] / min_contrast_range, 0, 1)
        # Use the minimum diversity as a scaling factor for the blend
        diversity_factor = np.min(channel_diversity)

        if blend_factor < 0:
            raw_blend = np.random.rand()
            effective_blend = raw_blend * diversity_factor
        else:
            effective_blend = blend_factor * diversity_factor
    else:
        # Normal case - use full blend factor
        effective_blend = np.random.rand() if blend_factor < 0 else blend_factor

    # safe scale: avoid divide-by-zero
    denom = color_range
    scale = 255.0 / np.where(denom > eps, denom, 1.0)

    for t in range(len(sample['scene_colors'])):
        colors = sample['scene_colors'][t].astype(np.float32)
        # only apply to RGB channels
        contrast_feat = (colors[:, :3] - lo[:, :3]) * scale[:, :3]
        new_rgb = (1 - effective_blend) * colors[:, :3] + effective_blend * contrast_feat
        # clamp and cast back
        new_rgb = np.clip(new_rgb, 0, 255).astype(sample['scene_colors'][t].dtype)
        sample['scene_colors'][t][:, :3] = new_rgb

    return sample


def chromatic_translation_transform(sample, p=0.95, ratio=0.05):
    """
    Add random color translation to scene colors.
    Same translation applied to all time frames.
    """
    if np.random.rand() >= p:
        return sample

    if 'scene_colors' not in sample:
        raise ValueError("Expected 'scene_colors' in sample for chromatic translation")

    # Generate translation once
    tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * ratio

    # Apply to all scene color frames
    for t in range(len(sample['scene_colors'])):
        sample['scene_colors'][t][:, :3] = np.clip(
            sample['scene_colors'][t][:, :3] + tr, 0, 255
        )

    return sample


def chromatic_jitter_transform(sample, p=0.95, std=0.05):
    """
    Add random color jitter (noise) to scene colors.
    Different noise per point but consistent noise pattern across time.
    """
    if np.random.rand() >= p:
        return sample

    if 'scene_colors' not in sample:
        raise ValueError("Expected 'scene_colors' in sample for chromatic jitter")

    # Generate a noise pattern based on the first frame's shape
    noise_shape = sample['scene_colors'][0].shape[0]
    noise = np.random.randn(noise_shape, 3) * std * 255

    # Apply to all scene color frames (assuming point correspondence)
    for t in range(len(sample['scene_colors'])):
        if sample['scene_colors'][t].shape[0] != noise_shape:
            raise ValueError(f"Frame {t} has different point count than frame 0")

        sample['scene_colors'][t][:, :3] = np.clip(
            sample['scene_colors'][t][:, :3] + noise, 0, 255
        )

    return sample


def enforce_max_num_points(sample, max_scene_points=None,
                           deterministic: bool = False, seed: int | None = None):
    """
    Enforce maximum number of points by randomly subsampling if necessary.
    Adds flags to indicate if limits were applied.
    """
    # Initialize flags
    sample['__scene_exceeds_max__'] = False

    # Process scene points if needed
    if max_scene_points is not None and 'scene_flows' in sample:
        NS = sample['scene_flows'].shape[1]
        if NS > max_scene_points:
            # Set flag
            sample['__scene_exceeds_max__'] = True

            # Create indices for selection
            if deterministic:
                key = str(sample.get('__key__', ''))
                base = 0 if seed is None else int(seed)
                rs = np.random.RandomState(_stable_int_hash(base, key, 'scene'))
                idx = rs.choice(NS, max_scene_points, replace=False)
            else:
                idx = np.random.choice(NS, max_scene_points, replace=False)
            idx = np.sort(idx)  # Sort to maintain order

            # Apply selection to all scene-related fields
            for key in list(sample.keys()):
                if 'scene_' in key and isinstance(sample[key], np.ndarray):
                    # Check if this is a per-point field (has matching second dimension)
                    if sample[key].shape[1] == NS:
                        sample[key] = sample[key][:, idx]

    # Robot points are explicitly sampled elsewhere, so no enforcement needed

    return sample


def compute_helper_variables(sample, max_relative_movement, domain=None):
    """
    Precompute variables needed for the loss function to make the data pipeline more efficient.
    This function computes point deltas, delta norms, and identifies moving vs static points.
    """
    # Skip if there's no gt_scene_flows (should always be there for training/eval)
    if 'gt_scene_flows' not in sample:
        return sample

    # Get ground truth scene flows
    gt_scene_flows = sample['gt_scene_flows']  # (T, NS, 3)

    # Compute the per-point deltas (movement between frames)
    gt_scene_flows_delta = gt_scene_flows[1:] - gt_scene_flows[:-1]  # (T-1, NS, 3)
    # Prepend zeros for the first timestep
    zeros_first_frame = np.zeros_like(gt_scene_flows[:1])  # (1, NS, 3)
    gt_scene_flows_delta = np.concatenate([zeros_first_frame, gt_scene_flows_delta], axis=0)  # (T, NS, 3)

    # Compute per-point relative from the first frame
    gt_scene_flows_relative = gt_scene_flows - gt_scene_flows[0:1]  # (T, NS, 3)

    # Compute the delta norm (magnitude of movement)
    delta_norm = np.linalg.norm(gt_scene_flows_delta, axis=-1)  # (T, NS)

    # Create soft selector labels (used for selector-weighted loss)
    scene_selector_gt = make_soft_selector_labels(
        delta_norm, DEFAULT_SOFT_SELECTOR_TAU, DEFAULT_SOFT_SELECTOR_TEMP_SCALE
    )  # (T, NS)

    # Identify moving and static points
    moved_mask = scene_selector_gt > 0.5  # (T, NS)
    static_mask = ~moved_mask  # (T, NS)

    # Save computed variables to the sample
    sample['gt_scene_flows_delta'] = gt_scene_flows_delta
    sample['gt_scene_flows_delta_norm'] = delta_norm
    sample['gt_scene_flows_relative'] = gt_scene_flows_relative
    sample['scene_selector_gt'] = scene_selector_gt
    sample['scene_moved_mask'] = moved_mask
    sample['scene_static_mask'] = static_mask

    # Define domain-specific supervised mask if possible at this stage
    if domain is not None and 'droid' in domain:
        assert 'scene_visibility' in sample and 'scene_depth_valid_mask' in sample, (
            "Expected scene_visibility and scene_depth_valid_mask in sample for droid domain"
        )
        scene_visibility = sample['scene_visibility'].astype(bool)  # (T, NS)
        scene_depth_valid_mask = sample['scene_depth_valid_mask'].astype(bool)  # (T, NS)
        supervised_mask = scene_visibility & scene_depth_valid_mask  # (T, NS)
        sample['scene_supervised_mask'] = supervised_mask
    else:
        # For other domains, assume all points have valid supervision
        supervised_mask = np.ones_like(gt_scene_flows[..., 0], dtype=bool)  # (T, NS)
        sample['scene_supervised_mask'] = supervised_mask

    # Cap the relative movement to prevent outliers from corrupting training
    gt_scene_flows_relative = np.clip(gt_scene_flows_relative, -max_relative_movement, max_relative_movement)
    sample['gt_scene_flows_relative'] = gt_scene_flows_relative

    return sample


__all__ = [
    "sphere_crop_transform",
    "grid_sample_transform",
    "filter_within_bounds",
    "random_rotate_around_z_axis",
    "center_shift",
    "normalize_colors",
    "make_gt_copy",
    "sample_and_apply_scene_context_mask",
    "compute_flow_derivatives",
    "compute_robot_distances",
    "random_scale_transform",
    "random_flip_transform",
    "image_resize_transform",
    "assert_camera_payload_resolution",
    "chromatic_auto_contrast_transform",
    "chromatic_translation_transform",
    "chromatic_jitter_transform",
    "enforce_max_num_points",
    "compute_helper_variables",
]
