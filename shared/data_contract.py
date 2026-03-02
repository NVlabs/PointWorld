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
"""Data contract definitions shared across data pipelines."""

from typing import List

SUPPORTED_DOMAINS = ("droid", "behavior")

IMAGE_KEYS = [
    "initial_rgb",
    "initial_depth",
    "intrinsic",
    "extrinsic",
]

EXPECTED_CAMERA_PAYLOAD_SHAPES = {
    "initial_depth": (180, 320),
    "intrinsic": (3, 3),
    "extrinsic": (4, 4),
}

BEHAVIOR_CLIP_ATTRIBUTE_KEYS = [
    "clip_key",
    "num_frames",
    "num_scene_points",
    "has_transition",
    "any_object_moving",
    "gripper_moving",
    "has_gripper_state_change",
    "robot_nonbase_moving",
    "has_trunk_arm_collision",
    "has_left_gripper_finger_collision",
    "has_right_gripper_finger_collision",
    "max_object_pos_movement",
    "max_object_rot_movement",
    "max_gripper_pos_movement",
    "max_gripper_rot_movement",
    "max_joint_movement",
    "left_min_distance_to_moving_objects",
    "left_min_distance_to_all_objects",
    "right_min_distance_to_moving_objects",
    "right_min_distance_to_all_objects",
    "clip_complete",
]


def validate_domain(domain: str) -> None:
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")


def get_wds_data_keys(domain: str) -> List[str]:
    validate_domain(domain)
    if domain == "behavior":
        keys = [
            "local_scene_points",
            "local_scene_colors",
            "local_scene_normals",
            "scene_mesh_trajectories",
            "left_gripper_open",
            "left_gripper_pose",
            "right_gripper_open",
            "right_gripper_pose",
            "joint_positions",
            "joint_names",
            "base_pose",
        ]
    elif domain == "droid":
        keys = [
            "scene_flows",
            "scene_colors",
            "scene_normals",
            "scene_visibility",
            "scene_depth_valid_mask",
            "gripper_open",
            "gripper_pose",
            "joint_positions",
            "gripper_positions",
        ]
    # Policy: droid/behavior always include camera image + matrix payloads.
    return keys + IMAGE_KEYS
