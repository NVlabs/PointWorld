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
"""Motion and collision tracking for Behavior flow extraction."""

from __future__ import annotations

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T
import omnigibson.utils.transform_utils_np as T_np
from omnigibson.controllers import IsGraspingState
from omnigibson.utils.usd_utils import RigidContactAPI

from simulation.behavior.utils import EXCLUDE_MESH_NAMES


class MotionTracker:
    """Track per-candidate motion metrics and validity conditions."""

    def __init__(self, env, robot, args, get_joint_positions_fn):
        self.env = env
        self.robot = robot
        self.args = args
        self._get_joint_positions = get_joint_positions_fn

        self._collision_cache_t = None
        self._collision_cache_flags = (False, {})
        self._trunk_arm_indices = None
        self._gripper_finger_indices_by_arm = None

    def set_collision_indices(self, trunk_arm_indices, gripper_finger_indices_by_arm):
        """Configure per-episode collision indices (trunk/arm + finger)."""
        self._trunk_arm_indices = list(trunk_arm_indices)
        self._gripper_finger_indices_by_arm = {
            arm: list(indices) for arm, indices in gripper_finger_indices_by_arm.items()
        }
        self._collision_cache_t = None
        self._collision_cache_flags = (False, {arm: False for arm in gripper_finger_indices_by_arm})

    def update_motion_metrics(self, candidate, frame_idx, transitions_flags):
        """Update motion and collision metrics for a clip candidate."""
        clip_frame_idx = candidate.frames_written

        if transitions_flags[frame_idx]:
            candidate.has_transition = True

        proprio = self.robot._get_proprioception_dict()

        for arm in candidate.arm_names:
            gr_qpos = proprio[f"gripper_{arm}_qpos"]
            gr_qpos_np = gr_qpos.cpu().numpy()
            gr_open = np.mean(gr_qpos_np) > self.args.gripper_finger_threshold
            grasp_state_int = int(proprio[f"grasp_{arm}"].item())
            if grasp_state_int == int(IsGraspingState.TRUE):
                gr_open = False
            candidate.gripper_open_history[arm].append(gr_open)

            eef_pos = proprio[f"eef_{arm}_pos"]
            eef_quat = proprio[f"eef_{arm}_quat"]
            eef_pose_mat = T.pose2mat((eef_pos, eef_quat)).cpu().numpy()
            candidate.eef_pose_history[arm].append(eef_pose_mat)

        if not candidate.gripper_moving:
            candidate.gripper_moving = self._detect_gripper_motion(candidate)

        current_joints = self._get_joint_positions()
        if candidate.prev_joint_positions is not None:
            joint_diff = np.abs(current_joints - candidate.prev_joint_positions)
            max_joint_diff = np.max(joint_diff)
            candidate.max_joint_movement = max(candidate.max_joint_movement, max_joint_diff)
            if np.any(joint_diff > self.args.joint_movement_threshold):
                candidate.robot_nonbase_moving = True
        candidate.prev_joint_positions = current_joints.copy()

        trunk_arm_col, gripper_finger_col_by_arm = self.get_collision_flags_for_frame(frame_idx)
        if not candidate.has_trunk_arm_collision and trunk_arm_col:
            candidate.has_trunk_arm_collision = True
        for arm, collided in gripper_finger_col_by_arm.items():
            if not candidate.gripper_finger_collision[arm] and collided:
                candidate.gripper_finger_collision[arm] = True

        if not candidate.any_object_moving:
            candidate.any_object_moving = self._detect_object_motion(candidate, clip_frame_idx)

        self._update_gripper_distance_stats(candidate, clip_frame_idx)

    def _detect_gripper_motion(self, candidate):
        """Check if gripper is moving for this candidate."""
        max_pos_motion = 0.0
        max_rot_motion = 0.0

        for arm in candidate.arm_names:
            gripper_seq = candidate.gripper_open_history[arm]
            eef_seq = candidate.eef_pose_history[arm]

            if len(gripper_seq) >= 2:
                gripper_changes = np.diff(np.array(gripper_seq, dtype=int))
                if np.any(gripper_changes != 0):
                    candidate.has_gripper_state_change = True
                    return True

            if len(eef_seq) >= 2:
                total_pos_motion = 0.0
                total_rot_motion = 0.0

                for i in range(1, len(eef_seq)):
                    curr_pose = eef_seq[i]
                    prev_pose = eef_seq[i - 1]

                    pos_diff = curr_pose[:3, 3] - prev_pose[:3, 3]
                    total_pos_motion += np.linalg.norm(pos_diff)

                    rot_diff = curr_pose[:3, :3] @ prev_pose[:3, :3].T
                    rot_quat = T_np.mat2quat(rot_diff)
                    rot_error = T_np.quat2axisangle(rot_quat)
                    total_rot_motion += np.linalg.norm(rot_error)

                max_pos_motion = max(max_pos_motion, total_pos_motion)
                max_rot_motion = max(max_rot_motion, total_rot_motion)

                is_closed = np.mean(gripper_seq) < 0.5
                if is_closed:
                    pos_thresh = self.args.gripper_closed_ee_pos_threshold
                    rot_thresh = self.args.gripper_closed_ee_rot_threshold
                else:
                    pos_thresh = self.args.ee_pos_threshold
                    rot_thresh = self.args.ee_rot_threshold

                if total_pos_motion >= pos_thresh or total_rot_motion >= rot_thresh:
                    candidate.max_gripper_pos_movement = max(candidate.max_gripper_pos_movement, total_pos_motion)
                    candidate.max_gripper_rot_movement = max(candidate.max_gripper_rot_movement, total_rot_motion)
                    return True

        candidate.max_gripper_pos_movement = max(candidate.max_gripper_pos_movement, max_pos_motion)
        candidate.max_gripper_rot_movement = max(candidate.max_gripper_rot_movement, max_rot_motion)

        return False

    def _detect_object_motion(self, candidate, frame_idx):
        """Check if any objects are moving for this candidate."""
        max_pos_movement = 0.0
        max_rot_movement = 0.0

        for camera_data in candidate.clip_seed["cameras"].values():
            mesh_trajectories = camera_data["mesh_trajectories"]

            for mesh_key, trajectory in mesh_trajectories.items():
                if frame_idx == 0:
                    candidate.prev_mesh_poses[mesh_key] = trajectory[0].copy()
                    continue

                curr_pose = trajectory[frame_idx]
                init_pose = candidate.prev_mesh_poses[mesh_key]

                pos_diff = np.linalg.norm(curr_pose[:3] - init_pose[:3])
                max_pos_movement = max(max_pos_movement, pos_diff)

                curr_quat = curr_pose[3:7]
                init_quat = init_pose[3:7]
                quat_dist = T_np.quat_distance(curr_quat, init_quat)
                rot_diff = np.linalg.norm(T_np.quat2axisangle(quat_dist))

                max_rot_movement = max(max_rot_movement, rot_diff)

                if (
                    pos_diff > self.args.object_movement_pos_threshold
                    or rot_diff > self.args.object_movement_rot_threshold
                ):
                    candidate.max_object_pos_movement = max(candidate.max_object_pos_movement, pos_diff)
                    candidate.max_object_rot_movement = max(candidate.max_object_rot_movement, rot_diff)
                    return True

        candidate.max_object_pos_movement = max(candidate.max_object_pos_movement, max_pos_movement)
        candidate.max_object_rot_movement = max(candidate.max_object_rot_movement, max_rot_movement)

        return False

    def _update_gripper_distance_stats(self, candidate, frame_idx):
        """Compute per-arm min distances to scene points in ROBOT0 frame."""
        for arm in candidate.arm_names:
            eef_pose = candidate.robot_series[f"{arm}_gripper_pose"]
            assert frame_idx < eef_pose.shape[0], f"EEF pose not found for arm {arm} at frame {frame_idx}"
            eef_pos_robot0 = eef_pose[frame_idx, :3]

            min_dist_all = float("inf")
            min_dist_moving = float("inf")

            for camera_data in candidate.clip_seed["cameras"].values():
                mesh_data = camera_data["mesh_data"]
                mesh_trajectories = camera_data["mesh_trajectories"]

                for mesh_key, data in mesh_data.items():
                    prim_path = data["prim_path"]
                    if any(name in prim_path.lower() for name in EXCLUDE_MESH_NAMES):
                        continue

                    traj = mesh_trajectories[mesh_key]
                    assert frame_idx < len(traj), f"Mesh trajectory not found for mesh {mesh_key} at frame {frame_idx}"

                    pose = traj[frame_idx]
                    obj_pos = pose[:3]
                    obj_quat = pose[3:7]

                    local_points = data["local_points"]
                    assert local_points.shape[0] > 0, f"Local points not found for mesh {mesh_key}"

                    R = T_np.quat2mat(obj_quat)
                    points_robot0 = local_points @ R.T + obj_pos

                    distances = np.linalg.norm(points_robot0 - eef_pos_robot0.reshape(1, 3), axis=1)
                    min_obj_distance = float(np.min(distances))
                    if min_obj_distance < min_dist_all:
                        min_dist_all = min_obj_distance

                    init_pose = traj[0]
                    pos_diff = float(np.linalg.norm(pose[:3] - init_pose[:3]))
                    quat_dist = T_np.quat_distance(pose[3:7], init_pose[3:7])
                    rot_diff = float(np.linalg.norm(T_np.quat2axisangle(quat_dist)))

                    if (
                        pos_diff > self.args.object_movement_pos_threshold
                        or rot_diff > self.args.object_movement_rot_threshold
                    ):
                        if min_obj_distance < min_dist_moving:
                            min_dist_moving = min_obj_distance

            if min_dist_all != float("inf"):
                candidate.min_distance_to_all_objects[arm] = min(
                    candidate.min_distance_to_all_objects[arm], min_dist_all
                )
            if min_dist_moving != float("inf"):
                candidate.min_distance_to_moving_objects[arm] = min(
                    candidate.min_distance_to_moving_objects[arm], min_dist_moving
                )

    def get_collision_flags_for_frame(self, current_t):
        if self._collision_cache_t == current_t:
            return self._collision_cache_flags
        assert self._trunk_arm_indices is not None, "Collision indices not configured"
        impulses = RigidContactAPI.get_all_impulses(self.env.scene.idx)
        trunk_arm_col = th.any(th.norm(impulses[self._trunk_arm_indices], dim=-1) > 0).item()
        gripper_finger_col_by_arm = {
            arm: th.any(th.norm(impulses[idxs], dim=-1) > 0).item()
            for arm, idxs in self._gripper_finger_indices_by_arm.items()
        }
        self._collision_cache_t = current_t
        self._collision_cache_flags = (trunk_arm_col, gripper_finger_col_by_arm)
        return self._collision_cache_flags

    def is_candidate_valid(self, candidate):
        """Evaluate if a candidate clip is valid based on motion criteria."""
        no_transition = not candidate.has_transition
        no_trunk_arm_collision = not candidate.has_trunk_arm_collision
        case_1 = candidate.any_object_moving and candidate.robot_nonbase_moving
        case_2 = candidate.any_object_moving and any(candidate.gripper_finger_collision.values())
        case_3 = (not candidate.any_object_moving) and candidate.gripper_moving and candidate.robot_nonbase_moving
        return no_transition and no_trunk_arm_collision and (case_1 or case_2 or case_3)
