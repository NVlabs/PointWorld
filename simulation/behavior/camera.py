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
"""Camera extraction and mesh trajectory utilities for Behavior flows."""

from __future__ import annotations

import time

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import PoseAPI

from simulation.behavior.utils import EXCLUDE_MESH_NAMES, points_in_bounds_mask


class CameraDataExtractor:
    """Extract per-camera RGB/depth/mesh data for Behavior clips."""

    def __init__(self, env, args, workspace_bounds_min, workspace_bounds_max):
        self.env = env
        self.args = args
        self.workspace_bounds_min = workspace_bounds_min
        self.workspace_bounds_max = workspace_bounds_max

    def capture_clip_seed(self, obs, info, robot, clip_key):
        """Extract point clouds and mesh data for the first frame of a clip."""
        clip_seed = {
            "clip_key": clip_key,
            "cameras": {},
        }

        # Compute world->robot transform at start of clip (fixed for the entire clip).
        r_pos, r_quat = robot.get_position_orientation()
        robot_pose_mat = T.pose2mat((r_pos, r_quat))
        world_to_robot_t = T.pose_inv(robot_pose_mat)
        world_to_robot = world_to_robot_t.cpu().numpy()

        # Persist transform for trajectory conversion: world -> robot at clip start.
        clip_seed["world_to_robot"] = world_to_robot

        # Process each external camera.
        for camera_name in self.env.env.external_sensors:
            camera_obs = {
                "rgb": obs[f"external::{camera_name}::rgb"],
                "depth_linear": obs[f"external::{camera_name}::depth_linear"],
                "normal": obs[f"external::{camera_name}::normal"],
                "seg_instance_id": obs[f"external::{camera_name}::seg_instance_id"],
            }
            camera_params = self._get_camera_params(camera_name)
            seg_info = info["obs_info"]["external"][camera_name]["seg_instance_id"]
            camera_data = self._extract_camera_data(camera_obs, camera_params, camera_name, seg_info, world_to_robot)
            clip_seed["cameras"][camera_name] = camera_data

        return clip_seed

    def record_mesh_and_camera_poses(self, clip_seed, frame_idx):
        """Record per-prim poses for the current frame across all cameras."""
        w2r_t = th.tensor(clip_seed["world_to_robot"], dtype=th.float32)
        for camera_name, camera_data in clip_seed["cameras"].items():
            mesh_data = camera_data["mesh_data"]
            cam_params = self.env.env.external_sensors[camera_name].camera_parameters
            _, world_to_camera_extrinsic = self._get_camera_intrinsic_extrinsic(cam_params)

            w2r_f32 = clip_seed["world_to_robot"].astype(np.float32)
            robot0_to_camera = world_to_camera_extrinsic @ np.linalg.inv(w2r_f32)
            camera_data["extrinsic_trajectory"][frame_idx] = robot0_to_camera

            for mesh_key, data in mesh_data.items():
                prim_path = data["prim_path"]
                r_pos_w_t, r_quat_w_t = PoseAPI.get_world_pose(prim_path)
                h_world_rigid = T.pose2mat((r_pos_w_t, r_quat_w_t))
                h_robot = w2r_t @ h_world_rigid
                r_pos_t, r_quat_t = T.mat2pose(h_robot)
                current_pos_robot = r_pos_t.cpu().numpy()
                current_quat_robot = r_quat_t.cpu().numpy()

                if mesh_key not in camera_data["mesh_trajectories"]:
                    camera_data["mesh_trajectories"][mesh_key] = np.zeros(
                        (self.args.frames_per_clip, 7), dtype=np.float32
                    )

                camera_data["mesh_trajectories"][mesh_key][frame_idx] = np.concatenate(
                    [np.array(current_pos_robot), np.array(current_quat_robot)]
                )

    def _get_camera_params(self, camera_name, max_trials=100):
        """Get camera parameters for a given camera name."""
        for _ in range(max_trials):
            params = self.env.env.external_sensors[camera_name].camera_parameters
            if params["renderProductResolution"][0] > 0:
                return params
            print(f"Failed to get camera parameters for {camera_name}, retrying...")
            og.sim.render()
            time.sleep(0.5)
        raise RuntimeError(f"Failed to get camera parameters for {camera_name}")

    def _extract_camera_data(self, camera_obs, camera_params, camera_name, seg_info, world_to_robot):
        """Extract point cloud data for a single camera."""
        rgb = camera_obs["rgb"].cpu().numpy()[:, :, :3]
        depth = camera_obs["depth_linear"].cpu().numpy()
        normals = camera_obs["normal"][..., :3].cpu().numpy()
        seg_map = camera_obs["seg_instance_id"].cpu().numpy()

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 1e-4) & (depth < 3.0)
        assert valid.sum() > 0, "No valid points found in camera data"

        intrinsic, world_to_camera_extrinsic = self._get_camera_intrinsic_extrinsic(camera_params)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        world_to_robot_f32 = world_to_robot.astype(np.float32)
        robot0_to_camera = world_to_camera_extrinsic @ np.linalg.inv(world_to_robot_f32)

        vu = u[valid]
        vv = v[valid]
        vd = cam_z = depth[valid]
        cam_x = (vu - cx) * vd / fx
        cam_y = (vv - cy) * vd / fy
        cam_cv = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], axis=-1)

        world_T_cam = np.linalg.inv(world_to_camera_extrinsic)
        world = cam_cv @ world_T_cam.T
        world_points = world[:, :3].astype(np.float32)

        colors = rgb[valid].reshape(-1, 3)
        normals = normals[valid].reshape(-1, 3)
        normals = self._correct_normals_toward_camera(normals, world_points, world_T_cam[:3, 3])
        seg_ids = seg_map[valid].reshape(-1)

        mesh_data = self._extract_mesh_points(seg_ids, seg_info, world_points, colors, normals, world_to_robot, camera_name)

        # Initial pose converted to ROBOT frame (position + quaternion) using rigid transform.
        camera_data = {
            "initial_rgb": rgb,
            "initial_depth": depth,
            "intrinsic": intrinsic,
            "extrinsic": robot0_to_camera.astype(np.float32),
            "mesh_data": mesh_data,
            "mesh_trajectories": {},
            "extrinsic_trajectory": np.zeros((self.args.frames_per_clip, 4, 4), dtype=np.float32),
        }
        camera_data["extrinsic_trajectory"][0] = robot0_to_camera.astype(np.float32)
        return camera_data

    def _extract_mesh_points(self, seg_ids, seg_info, world_points, colors, normals, world_to_robot, camera_name):
        """Extract points for each visible USD prim using seg_instance_id mapping."""
        mesh_data = {}

        assert seg_info, f"No segmentation info available for camera {camera_name}"
        unique_seg_ids = np.unique(seg_ids)

        for seg_id in unique_seg_ids:
            if seg_id == 0:
                continue
            if seg_id not in seg_info:
                continue

            prim_path = seg_info[seg_id]
            if any(name in prim_path.lower() for name in EXCLUDE_MESH_NAMES):
                continue

            mask = seg_ids == seg_id
            if not np.any(mask):
                continue

            mesh_points_world = world_points[mask]
            mesh_colors = colors[mask]
            mesh_normals_world = normals[mask]

            if len(mesh_points_world) == 0:
                continue

            w_pos_t, w_quat_t = PoseAPI.get_world_pose(prim_path)
            world_pose_rigid = T.pose2mat((w_pos_t, w_quat_t)).cpu().numpy()
            inv_world_pose_rigid = np.linalg.inv(world_pose_rigid)
            homogeneous_points_world = np.hstack(
                (mesh_points_world, np.ones((mesh_points_world.shape[0], 1), dtype=np.float32))
            )
            local_points = (inv_world_pose_rigid @ homogeneous_points_world.T).T[:, :3]

            R = world_pose_rigid[:3, :3]
            local_normals = (R.T @ mesh_normals_world.T).T
            local_normals = local_normals / (np.linalg.norm(local_normals, axis=1, keepdims=True) + 1e-8)

            homogeneous_points_world = np.hstack(
                (mesh_points_world, np.ones((mesh_points_world.shape[0], 1), dtype=np.float32))
            )
            robot_points_for_filter = (world_to_robot @ homogeneous_points_world.T).T[:, :3]
            within = points_in_bounds_mask(robot_points_for_filter, self.workspace_bounds_min, self.workspace_bounds_max)
            if not np.any(within):
                continue
            local_points = local_points[within]
            local_normals = local_normals[within]
            mesh_colors = mesh_colors[within]

            r_pos_w_t, r_quat_w_t = PoseAPI.get_world_pose(prim_path)
            h_world_rigid = T.pose2mat((r_pos_w_t, r_quat_w_t))
            w2r_t = th.tensor(world_to_robot, dtype=th.float32)
            h_robot = w2r_t @ h_world_rigid
            r_pos_t, r_quat_t = T.mat2pose(h_robot)
            init_pose_robot = np.concatenate([r_pos_t.cpu().numpy(), r_quat_t.cpu().numpy()]).astype(np.float32)

            mesh_key = self._sanitize_prim_path_for_h5(prim_path)

            mesh_data[mesh_key] = {
                "local_points": local_points.astype(np.float32),
                "local_normals": local_normals.astype(np.float32),
                "colors": mesh_colors.astype(np.uint8),
                "initial_pose": init_pose_robot,
                "prim_path": prim_path,
            }

        if not mesh_data:
            print(f"Warning: No mesh data found for camera {camera_name}")
            return {}
        return mesh_data

    def _sanitize_prim_path_for_h5(self, prim_path):
        cleaned = prim_path.lstrip("/")
        return cleaned.replace("/", "__")

    def _correct_normals_toward_camera(self, normals, world_points, camera_position):
        """Correct normals to point toward the camera."""
        assert len(normals) == len(world_points), "Normals and world points must have the same length"
        assert len(camera_position) == 3, "Camera position must be a 3D vector"
        assert len(normals) > 0, "Normals must have at least one element"

        to_camera = camera_position.reshape(1, 3) - world_points
        to_camera = to_camera / (np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-8)
        dot_products = np.sum(normals * to_camera, axis=1)
        pointing_away = dot_products < 0
        corrected_normals = normals.copy()
        corrected_normals[pointing_away] = -corrected_normals[pointing_away]
        return corrected_normals

    def _get_camera_intrinsic_extrinsic(self, camera_params):
        """Return (intrinsic, world_to_camera_extrinsic) in OpenCV convention with T_mod applied."""
        proj = np.asarray(camera_params["cameraProjection"]).reshape(4, 4).T
        view = np.asarray(camera_params["cameraViewTransform"]).reshape(4, 4).T
        width_res, height_res = camera_params["renderProductResolution"]

        fx = proj[0, 0] * width_res / 2.0
        fy = proj[1, 1] * height_res / 2.0
        cx = width_res / 2.0
        cy = height_res / 2.0
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        T_mod = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        world_to_camera_extrinsic = (T_mod @ view).astype(np.float32)
        return intrinsic, world_to_camera_extrinsic
