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

import re

import torch

from dataset_components.constants import RELEASE_WEIGHT_GAMMA


def custom_collate_fn(batch, args):
    """
    Collate function that pads the variable number of points across samples
    to the maximum found in the batch. Also creates new keys 'scene_exists' and
    'robot_exists' which are boolean masks indicating real (True) vs. padded (False) points.

    Args:
        batch: List of samples, where each sample is a dict of torch.Tensors
        args: Arguments object

    Assumes each item in `batch` is a dict of torch.Tensors with shapes:
        - scene_flows: (T, N_scene, D_scene)
        - robot_flows: (T, N_robot, D_robot)
        - scene_features: (T, N_scene, F_scene), etc.

    Returns:
        collated (dict of torch.Tensor):
            Keys will include all from the individual samples plus:
              - 'scene_exists': (B, T, max_N_scene)
              - 'robot_exists': (B, T, max_N_robot)
            For any field that has shape (T, N, ...), it becomes (B, T, max_N, ...).
            Fields that don't have that shape are just stacked along the batch dimension.
    """
    # A helper to check if a tensor is "per-point" shaped, i.e. (T, N, ...)
    def is_per_point_field(tensor, reference_N):
        """
        We decide it's 'per-point' if:
          1) tensor has at least 3 dimensions: (T, N, ...)
          2) the second dimension == reference_N
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        return (
            tensor.ndim >= 2
            and tensor.shape[1] == reference_N
        )

    # A helper to pad a tensor of shape (T, N, [D...]) to (T, N_max, [D...])
    def pad_time_n(tensor, N_max):
        """
        Pads the 2nd dimension of `tensor` from N to N_max with zeros.
        tensor shape: (T, N, [D...])
        returns:
          padded of shape (T, N_max, [D...])
        """
        T, N = tensor.shape[:2]
        # The "remaining" dims after (T, N)
        extra_dims = tensor.shape[2:]
        padded_shape = (T, N_max) + extra_dims
        padded = torch.zeros(
            padded_shape,
            dtype=tensor.dtype,
            device=tensor.device
        )
        padded[:, :N, ...] = tensor[:, :min(N, N_max), ...]
        return padded

    camera_suffixes = ("_initial_rgb", "_initial_depth", "_intrinsic", "_extrinsic")

    def _camera_prefix_and_suffix(key: str):
        for suffix in camera_suffixes:
            if key.endswith(suffix):
                prefix = key[:-len(suffix)]
                if re.fullmatch(r"cam\d+", prefix):
                    return prefix, suffix
        return None, None

    # 1) Find max number of scene/robot points across the batch
    N_scene_list = []
    N_robot_list = []
    for sample in batch:
        N_scene_list.append(sample['scene_flows'].shape[1])  # shape (T, N_scene, D)
        N_robot_list.append(sample['robot_flows'].shape[1])  # shape (T, N_robot, D)

    # No longer apply maximum point limits here - this is now done earlier in the pipeline
    max_N_scene = max(N_scene_list)
    max_N_robot = max(N_robot_list)

    # We'll collect results here:
    collated = {}

    # 2) Build non-camera key buffers over intersection across the batch.
    # Camera payload keys are handled separately to support mixed camera counts.
    non_camera_keys = None
    camera_prefixes = set()
    for sample in batch:
        sample_non_camera_keys = set()
        for key in sample.keys():
            prefix, _ = _camera_prefix_and_suffix(key)
            if prefix is None:
                sample_non_camera_keys.add(key)
            else:
                camera_prefixes.add(prefix)
        if non_camera_keys is None:
            non_camera_keys = sample_non_camera_keys
        else:
            non_camera_keys &= sample_non_camera_keys
    if non_camera_keys is None:
        non_camera_keys = set()

    # Stable ordering for deterministic collation behavior.
    camera_prefixes = sorted(camera_prefixes)

    # Cache one template tensor per camera payload key for zero-filling missing cameras.
    camera_templates = {}
    for prefix in camera_prefixes:
        for suffix in camera_suffixes:
            key = f"{prefix}{suffix}"
            for sample in batch:
                if key in sample:
                    camera_templates[key] = sample[key]
                    break

    first_keys = set(non_camera_keys)
    # Meta keys that we keep per-sample as lists (may not exist on all samples)
    meta_keys_union = {'joint_names', 'base_pose'}
    # Add any of these meta keys if present in any sample to ensure they are surfaced
    for sample in batch:
        for mk in list(meta_keys_union):
            if mk in sample:
                first_keys.add(mk)
    # We'll store each key's padded/stacked outputs in a python list before final stacking.
    buffer_dict = {k: [] for k in first_keys}
    for prefix in camera_prefixes:
        for suffix in camera_suffixes:
            buffer_dict[f"{prefix}{suffix}"] = []
        buffer_dict[f"{prefix}_exists"] = []

    # We also want 'scene_exists' and 'robot_exists' in the final dictionary:
    scene_exists_list = []
    robot_exists_list = []

    for sample in batch:
        # Grab shapes
        T_s, N_scene = sample['scene_flows'].shape[:2]
        T_r, N_robot = sample['robot_flows'].shape[:2]
        assert T_s == T_r, "All time dimensions T must match in scene & robot for a given sample."
        T = T_s

        # Create existence masks (accounting for potential truncation):
        scene_mask = torch.zeros((T, max_N_scene), dtype=torch.bool)
        scene_mask[:, :min(N_scene, max_N_scene)] = True
        scene_exists_list.append(scene_mask)

        robot_mask = torch.zeros((T, max_N_robot), dtype=torch.bool)
        robot_mask[:, :min(N_robot, max_N_robot)] = True
        robot_exists_list.append(robot_mask)

        # Now handle each non-camera key present on this sample.
        for k, v in sample.items():
            prefix, _ = _camera_prefix_and_suffix(k)
            if prefix is not None:
                continue
            if k not in buffer_dict:
                continue
            # If it's a per-scene-point field (T, N_scene, ...), pad to max_N_scene
            if is_per_point_field(v, N_scene) and 'scene' in k:
                buffer_dict[k].append(pad_time_n(v, max_N_scene))
            # If it's a per-robot-point field (T, N_robot, ...), pad to max_N_robot
            elif is_per_point_field(v, N_robot) and 'robot' in k:
                buffer_dict[k].append(pad_time_n(v, max_N_robot))
            else:
                # Otherwise just store as-is (will stack along batch dimension)
                # e.g. shape (T, D) or (T,) or scalars
                if k in buffer_dict.keys():
                    buffer_dict[k].append(v)

        # For unioned meta keys absent on this sample, append None to keep alignment
        for mk in meta_keys_union:
            if mk in buffer_dict and mk not in sample:
                buffer_dict[mk].append(None)

        # Fill camera payload keys for mixed camera-count batches.
        for prefix in camera_prefixes:
            required_keys = [f"{prefix}{suffix}" for suffix in camera_suffixes]
            exists = all(key in sample for key in required_keys)
            buffer_dict[f"{prefix}_exists"].append(torch.tensor(exists, dtype=torch.bool))
            for key in required_keys:
                if key in sample:
                    value = sample[key]
                else:
                    if key not in camera_templates:
                        raise KeyError(f"Missing camera template for key '{key}' during collate")
                    template = camera_templates[key]
                    if not isinstance(template, torch.Tensor):
                        raise TypeError(
                            f"Camera template for key '{key}' must be torch.Tensor, got {type(template)}"
                        )
                    value = torch.zeros_like(template)
                buffer_dict[key].append(value)

    # 3) Convert these lists into properly stacked tensors
    for k, list_of_vals in buffer_dict.items():
        if len(list_of_vals) == 0:
            continue
        # Keys collated as metadata lists (non-tensors)
        meta_keys = {
            '__key__', '__domain__', '__out_of_bounds__', '__scene_exceeds_max__',
            # Embodiment‑specific kinematic fields: keep per-sample (lists) to support mixed-domain batches
            'joint_names', 'base_pose'
        }
        if k in meta_keys:
            collated[k] = [v for v in list_of_vals]
            continue

        # Only tensors should be stacked; otherwise, keep as list to avoid attribute errors
        first_val = list_of_vals[0]
        if isinstance(first_val, torch.Tensor):
            # Safe assert: check for shape consistency across batch samples
            if len(list_of_vals) > 1:
                first_shape = first_val.shape
                for i, val in enumerate(list_of_vals[1:], 1):
                    assert isinstance(val, torch.Tensor), f"Key '{k}' expected tensor, got {type(val)}"
                    assert val.shape == first_shape, (
                        f"Shape mismatch in key '{k}': sample 0 has shape {first_shape}, "
                        f"sample {i} has shape {val.shape}. This likely indicates feature "
                        f"dimension mismatch between different domains or gripper configurations."
                    )
            collated[k] = torch.stack(list_of_vals, dim=0)
        else:
            # Keep as list to be handled by downstream code (eval visualizer handles lists)
            collated[k] = [v for v in list_of_vals]

    # 4) Add the new 'scene_exists' and 'robot_exists' keys
    collated['scene_exists'] = torch.stack(scene_exists_list, dim=0)  # (B, T, max_N_scene)
    collated['robot_exists'] = torch.stack(robot_exists_list, dim=0)  # (B, T, max_N_robot)

    # -------- final point weight tensor ------------------------------- #
    device = collated['scene_exists'].device
    selector_gt = collated['scene_selector_gt'].to(device)           # (B,T,N)
    moved = collated['scene_moved_mask'].bool().squeeze(-1).to(device)
    static = collated['scene_static_mask'].bool().squeeze(-1).to(device)
    context = collated['scene_context_mask'].bool().squeeze(-1).to(device)
    exists = collated['scene_exists'].bool().squeeze(-1).to(device)
    supervised = collated['scene_supervised_mask'].bool().squeeze(-1).to(device)
    pred = ~context
    pred_exists_supervised = pred & exists & supervised
    # num_valid_supervision = pred_exists_supervised.sum()

    B, T, N = selector_gt.shape
    weights = torch.zeros((B, T, N), dtype=torch.float64, device=device)

    w = selector_gt.pow(RELEASE_WEIGHT_GAMMA)
    w[~pred_exists_supervised] = 0.0
    weights = w

    # divide by number of valid supervision because we later will take the sum of weights * distance
    # weights_sum = num_valid_supervision
    weights_norm = weights.sum().clamp(min=1.0)
    weights = weights / weights_norm

    collated['point_weights'] = weights
    collated['weights_norm'] = weights_norm
    return collated
