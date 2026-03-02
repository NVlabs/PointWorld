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
"""Data containers for Behavior (simulation) flow extraction."""

import numpy as np


class ClipCandidate:
    """Buffered clip candidate tracked during one-pass streaming."""

    def __init__(self, start_downsample_idx, frames_per_clip, world_to_robot, clip_seed, arm_names, num_joints):
        # IDs / indexing
        self.start_downsample_idx = start_downsample_idx
        self.frames_written = 0
        self.frames_per_clip = frames_per_clip

        # Transforms
        self.world_to_robot = world_to_robot

        # Robot arrays (F rows each) - dynamic sizing
        self.robot_series = {
            "joint_positions": np.zeros((frames_per_clip, num_joints), dtype=np.float32),
            "base_pose": np.zeros((frames_per_clip, 7), dtype=np.float32),
        }
        # Add gripper data for each arm - dynamic arm names
        self.arm_names = arm_names
        for arm in arm_names:
            self.robot_series[f"{arm}_gripper_pose"] = np.zeros((frames_per_clip, 7), dtype=np.float32)
            self.robot_series[f"{arm}_gripper_open"] = np.zeros((frames_per_clip, 1), dtype=np.bool_)
            self.robot_series[f"{arm}_is_grasping"] = np.zeros((frames_per_clip, 1), dtype=np.bool_)

        # Camera data from first frame
        self.clip_seed = clip_seed

        # Motion accumulators for validity tests
        self.has_transition = False
        self.any_object_moving = False
        self.gripper_moving = False
        self.robot_nonbase_moving = False
        self.has_trunk_arm_collision = False
        self.gripper_finger_collision = {arm: False for arm in arm_names}
        self.has_gripper_state_change = False

        # Detailed statistics for validation conditions
        self.max_object_pos_movement = -1.0
        self.max_object_rot_movement = -1.0
        self.max_gripper_pos_movement = -1.0
        self.max_gripper_rot_movement = -1.0
        self.max_joint_movement = -1.0

        # Per-arm gripper tracking - dynamic arm names
        self.gripper_open_history = {arm: [] for arm in arm_names}
        self.eef_pose_history = {arm: [] for arm in arm_names}
        self.prev_joint_positions = None

        # Object motion tracking (for mesh trajectories)
        self.prev_mesh_poses = {}

        # Per-arm distance tracking
        self.min_distance_to_moving_objects = {arm: float("inf") for arm in arm_names}
        self.min_distance_to_all_objects = {arm: float("inf") for arm in arm_names}
