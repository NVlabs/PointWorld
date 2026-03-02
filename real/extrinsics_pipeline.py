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
"""Extrinsics optimization helpers for DROID scenes."""

import os
import time

import cv2
import numpy as np
import torch

from compute_extrinsics_utils import (
    RobotVisibilityError,
    check_camera_results_exist,
    check_precomputed_depth_exists,
    deduplicate_coordinates,
    get_robot_transform,
    pose_6dof_to_matrix,
    sample_depth_with_grid_sample,
)
from real.droid_utils import (
    filter_by_timestamps,
    gather_data_dict,
    gather_trajectory,
    get_metadata,
    get_robot_serial,
    get_uuid,
)
from real.extrinsics_io import load_precomputed_depth, write_camera_results
from real.real_utils import get_time_str
from real.vggt_forward import stage_vggt_images
import transform_utils

def _print(*args, **kwargs):
    """Helper function that wraps print with automatic time prefixing and flush=True"""
    # Extract the first argument to check if it already has a timestamp
    if args and isinstance(args[0], str) and args[0].startswith(f"[{get_time_str()}]"):
        # Already has timestamp, just print with flush=True
        print(*args, flush=True, **kwargs)
    else:
        # Prepend timestamp
        if args:
            first_arg = f"[{get_time_str()}] {args[0]}"
            print(first_arg, *args[1:], flush=True, **kwargs)
        else:
            print(f"[{get_time_str()}]", flush=True, **kwargs)

MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 2.0
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(
    REPO_ROOT,
    "assets",
    "franka_description",
    "franka_panda_robotiq_2f85_og.urdf",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ExtrinsicsOptimizer:
    """Optimizes camera extrinsics for a single scene using robot mesh rendering and depth estimation."""
    
    def __init__(self, 
                 scene_path,
                 vggt_forward=None,
                 robot_renderer=None,
                 device="cuda",
                 num_frames=1,
                 downscale_ratio=1.0,
                 min_robot_points=2000):
        """
        Initialize ExtrinsicsOptimizer for a single scene.
        
        Args:
            scene_path: Path to the DROID scene
            vggt_forward: VGGT forward pass model for extrinsics estimation
            robot_renderer: Robot mesh renderer for point cloud generation
            device: Computing device ("cuda" or "cpu")
            num_frames: Number of frames to process
            downscale_ratio: Image downscaling ratio
            min_robot_points: Minimum number of robot points required for optimization
        """
        self.scene_path = scene_path
        self.vggt_forward = vggt_forward
        self.robot_renderer = robot_renderer
        self.device = device
        self.num_frames = num_frames
        self.downscale_ratio = downscale_ratio
        self.min_robot_points = min_robot_points
        
        # Initialize data containers following legacy single-stage pipeline pattern
        self.data_dict = {}
        self.proprio_dict = {}
        self.meta = get_metadata(self.scene_path)
        self.uuid = self.meta['uuid']

        # camera serials
        self.wrist_serial = self.meta["wrist_cam_serial"]
        self.ext_serials = []
        camera_keys = [k for k in self.meta.keys() if k.endswith("_cam_serial")]
        for ck in camera_keys:
            if ck == "wrist_cam_serial":
                continue
            self.ext_serials.append(self.meta[ck])
        
        # Initialize optimization metrics
        self.optimization_metrics = {}
        
    def _load_scene_data(self):
        """Load scene data using gather_data_dict and gather_trajectory with timestamp alignment"""
        start = time.time()
        # Load all frames first (similar to legacy single-stage pipeline approach)
        self.data_dict = gather_data_dict(
            self.scene_path,
            downscale_ratio=self.downscale_ratio,
            include_stereo=False,
            include_depth=False,
            include_wrist_cam=True,
            max_frames=-1
        )
        
        self.proprio_dict = gather_trajectory(self.scene_path, max_frames=-1)
        
        # Get the first camera serial to use as reference for canonical timestamps
        first_camera_serial = list(self.data_dict.keys())[0]
        
        # Get canonical timestamps from the first camera
        canonical_timestamps = self.data_dict[first_camera_serial]['timestamps']
        
        # Apply time skipping and frame sampling if num_frames is specified
        if self.num_frames > 0 and len(canonical_timestamps) > self.num_frames:
            # Uniformly sample frames to ensure they are spread out
            time_skip_ratio = max(1, len(canonical_timestamps) // self.num_frames)
            canonical_timestamps = canonical_timestamps[::time_skip_ratio]
            
            # If we still have too many frames, take the first num_frames
            if len(canonical_timestamps) > self.num_frames:
                canonical_timestamps = canonical_timestamps[:self.num_frames]
        
        # Filter data to align with canonical timestamps (following legacy single-stage pipeline approach)
        self.T = len(canonical_timestamps)
        try:
            self.data_dict = filter_by_timestamps(self.data_dict, canonical_timestamps, self.scene_path, is_proprio=False)
            self.proprio_dict = filter_by_timestamps(self.proprio_dict, canonical_timestamps, self.scene_path, is_proprio=True)
        except RuntimeError as e:
            _print(f"Error during timestamp filtering: {e}")
            raise

        # flip wrist images (vggt performs better with wrist images flipped vertically)
        flipped_wrist_rgb = []
        for frame in self.data_dict[self.wrist_serial]['rgb']:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            flipped_wrist_rgb.append(frame)
        self.data_dict[self.wrist_serial]['rgb'] = flipped_wrist_rgb

        self.T = len(self.proprio_dict['joint_positions'])

        # Rename keys to include source
        for camera_serial in self.data_dict:
            assert 'intrinsic' in self.data_dict[camera_serial], "intrinsic not found"
            self.data_dict[camera_serial]['measured_intrinsics'] = self.data_dict[camera_serial]['intrinsic'].copy()
            del self.data_dict[camera_serial]['intrinsic']

        # Load gripper to wrist camera transform (no fallback)
        robot_serial = get_robot_serial(self.scene_path)
        self.T_gripper_wrist = get_robot_transform(robot_serial=robot_serial)

        _print(f"Loaded data for {len(self.data_dict)} cameras, {self.T} frames (time taken: {time.time() - start:.2f}s)")
    
    def _compute_T_base_wrist(self, frame_idx):
        """Get base->wrist transform for a specific frame (includes vertical flip)."""
        # Get base->wrist transform for the specified frame using the proper function
        T_gripper_base_quat = self.proprio_dict['gripper_pose'][frame_idx]
        T_gripper_base = transform_utils.convert_pose_quat2mat(T_gripper_base_quat[None])[0]
        
        # base->gripper from euler
        T_base_gripper = np.linalg.inv(T_gripper_base)  # 4x4
        
        # T_base<-wrist = T_base<-gripper * (T_gripper<-wrist)
        T_base_wrist = self.T_gripper_wrist @ T_base_gripper
        
        # Apply 180-degree rotation around Z-axis to account for vertical flip of wrist images
        # This ensures the world frame definition matches the flipped wrist camera images
        # cv2.ROTATE_180 = 180° rotation around optical axis (Z-axis) in camera coordinates
        flip_transform = np.array([
            [-1,  0,  0,  0],  # 180° rotation around Z-axis
            [ 0, -1,  0,  0],  # (rotates X and Y axes)
            [ 0,  0,  1,  0],  # Z-axis unchanged
            [ 0,  0,  0,  1]
        ], dtype=np.float64)
        
        T_base_wrist_flipped = flip_transform @ T_base_wrist
        return T_base_wrist_flipped

    def _compute_frame_camera_loss(self, world_pts, optimized_T, depth_tensor, intrinsic, H, W, dedup_threshold):
        """
        Helper function for computing loss for a single frame-camera combination.
        All inputs must be tensors on the same device.
        """
        # Assert all tensor inputs are on the same device
        assert isinstance(world_pts, torch.Tensor), "world_pts must be a torch.Tensor"
        assert isinstance(optimized_T, torch.Tensor), "optimized_T must be a torch.Tensor"
        assert isinstance(depth_tensor, torch.Tensor), "depth_tensor must be a torch.Tensor"
        assert isinstance(intrinsic, torch.Tensor), "intrinsic must be a torch.Tensor"
        
        device = world_pts.device
        assert optimized_T.device == device, "optimized_T must be on the same device as world_pts"
        assert depth_tensor.device == device, "depth_tensor must be on the same device as world_pts"
        assert intrinsic.device == device, "intrinsic must be on the same device as world_pts"
        
        # Convert H, W to tensors if they aren't already
        if not isinstance(H, torch.Tensor):
            H = torch.tensor(H, device=device, dtype=torch.float32)
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, device=device, dtype=torch.float32)
        
        N_points = world_pts.shape[0]
        zero_loss = torch.tensor(0.0, device=device)
        zero_count = torch.tensor(0, device=device)
        
        # Transform all points to camera coordinates
        ones = torch.ones((N_points, 1), device=device, dtype=torch.float32)
        pts_h = torch.cat([world_pts, ones], dim=1)  # [N_points, 4]
        pts_cam = torch.mm(pts_h, optimized_T.T)
        z = pts_cam[:, 2]  # [N_points]
        
        # Filter points behind camera (vectorized)
        valid_depth = z > 0
        if not torch.any(valid_depth):
            return zero_loss, zero_count
        
        pts_cam_valid = pts_cam[valid_depth]  # [N_valid, 4]
        z_valid = z[valid_depth]  # [N_valid]
        
        # Project to image coordinates
        proj_h = torch.mm(pts_cam_valid[:, :3], intrinsic.T)
        xy = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-8)
        
        # Filter points within image bounds
        valid_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        
        if not torch.any(valid_bounds):
            return zero_loss, zero_count
        
        xy_valid = xy[valid_bounds]  # [N_inbounds, 2]
        z_final = z_valid[valid_bounds]  # [N_inbounds]
        
        # Apply coordinate deduplication (vectorized)
        if dedup_threshold > 0:
            unique_indices = deduplicate_coordinates(xy_valid, threshold_pixels=dedup_threshold)
            if len(unique_indices) == 0:
                return zero_loss, zero_count
            xy_final = xy_valid[unique_indices]
            z_final = z_final[unique_indices]
        else:
            xy_final = xy_valid
        
        # Sample depth using optimized grid sampling
        sampled_depth = sample_depth_with_grid_sample(depth_tensor, xy_final)
        
        # Compute loss
        d_pred = z_final
        d_gt = sampled_depth
        
        # Filter out depth values not within MIN_DEPTH_M and MAX_DEPTH_M
        valid_depth_range = (d_gt >= MIN_DEPTH_M) & (d_gt <= MAX_DEPTH_M)
        
        if not torch.any(valid_depth_range):
            return zero_loss, zero_count
        
        d_pred_filtered = d_pred[valid_depth_range]
        d_gt_filtered = d_gt[valid_depth_range]
        
        # Vectorized loss computation
        diff = torch.abs(d_gt_filtered - d_pred_filtered)
        frame_camera_loss = diff.mean()
        
        if torch.isfinite(frame_camera_loss):
            return frame_camera_loss, torch.tensor(d_pred_filtered.shape[0], device=device)
        else:
            return zero_loss, zero_count

    def _optimize_all_cameras_jointly(self,
                                  camera_data_dict,
                                  initial_extrinsics_dict,
                                  max_iterations,
                                  learning_rate,
                                  translation_scale,
                                  rotation_scale,
                                  dedup_threshold=0.1,
                                  optimizer_type="adam"):
        """
        V2: Optimized version that recognizes H,W are same across frames/cameras,
        constructs frame_groups outside closure, removes jitted functions, and 
        groups computation into a helper function.
        """
        
        # Initialize optimization parameters for all external cameras
        ext_camera_serials = [s for s in camera_data_dict.keys() if s != self.wrist_serial]
        n_ext_cameras = len(ext_camera_serials)
        
        # Create parameter tensors
        translation_params_norm = torch.zeros(n_ext_cameras, 3, device=self.device, dtype=torch.float32, requires_grad=True)
        rotation_params_norm = torch.zeros(n_ext_cameras, 3, device=self.device, dtype=torch.float32, requires_grad=True)
        
        # Setup optimizer
        if optimizer_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [translation_params_norm, rotation_params_norm], 
                lr=learning_rate,
                max_iter=3,
                tolerance_grad=1e-5,
                tolerance_change=1e-7
            )
        else:
            optimizer = torch.optim.Adam([translation_params_norm, rotation_params_norm], 
                                       lr=learning_rate, 
                                       eps=1e-6, 
                                       weight_decay=0.0)
        
        # Convert initial extrinsics to tensors
        extrinsic_init_tensor = torch.zeros((n_ext_cameras, 4, 4), dtype=torch.float32, device=self.device)
        camera_to_idx = {cam_serial: i for i, cam_serial in enumerate(ext_camera_serials)}
        
        for i, cam_serial in enumerate(ext_camera_serials):
            extrinsic_init_tensor[i] = torch.as_tensor(
                initial_extrinsics_dict[cam_serial], dtype=torch.float32, device=self.device
            )
        
        # Pre-batch all world points for all frames
        all_world_points = []
        for frame_idx in range(self.num_frames):
            joint_positions = self.proprio_dict['joint_positions'][frame_idx]
            gripper_positions = self.proprio_dict['gripper_positions'][frame_idx]
            fk_result = self.robot_renderer._get_forward_kinematics(joint_positions, gripper_positions)
            world_pts = self.robot_renderer._get_world_points(fk_result)
            assert world_pts.shape[0] > 0, "No valid robot points found"
            all_world_points.append(world_pts)
        all_world_points = torch.stack(all_world_points, dim=0)  # [num_frames, num_points, 3]

        # Recognize that H, W are the same across all frames and all cameras
        # Get H, W from the first available depth data
        first_cam_serial = ext_camera_serials[0]
        first_depth = camera_data_dict[first_cam_serial]['depth'][0]
        H, W = first_depth.shape
        # Construct a single "frame_groups" outside of the closure function that prepares all necessary data compactly
        frame_groups = {}
        for frame_idx in range(all_world_points.shape[0]):
            frame_groups[frame_idx] = []
            for cam_serial in ext_camera_serials:
                camera_frame_data = camera_data_dict[cam_serial]
                depth_clean = camera_frame_data['depth'][frame_idx].copy()
                depth_tensor = torch.as_tensor(depth_clean, dtype=torch.float32, device=self.device)
                intrinsic_tensor = torch.as_tensor(camera_frame_data['intrinsic'], dtype=torch.float32, device=self.device)
                cam_idx = camera_to_idx[cam_serial]
                frame_groups[frame_idx].append((cam_idx, depth_tensor, intrinsic_tensor))
        
        total_points = all_world_points.shape[1]
        _print(f"    Batched {all_world_points.shape[0]} frames, {len(ext_camera_serials)} cameras, {total_points} robot points per frame")
        _print(f"    Using {optimizer_type.upper()} optimizer")
        _print(f"    Minimum robot points threshold: {self.min_robot_points}")
        
        # Track points used in computation
        total_valid_points = 0
        
        # Track per-camera metrics across the entire optimization
        per_camera_losses_history = {cam_serial: [] for cam_serial in ext_camera_serials}
        # Per-camera point counts will be collected from the final iteration
        final_per_camera_point_counts = {cam_serial: [] for cam_serial in ext_camera_serials}
        
        # Vectorized loss computation function
        def compute_loss_vectorized():
            """Vectorized loss computation using batched operations."""
            nonlocal total_valid_points, per_camera_losses_history, final_per_camera_point_counts
            optimizer.zero_grad()
            
            # Convert normalized parameters to physical units
            translation_params = translation_params_norm * translation_scale  # [n_ext_cameras, 3]
            rotation_params = rotation_params_norm * rotation_scale  # [n_ext_cameras, 3]
            
            # Batch create all optimized extrinsics at once
            optimized_extrinsics = torch.zeros_like(extrinsic_init_tensor)  # [n_ext_cameras, 4, 4]
            
            for i in range(n_ext_cameras):
                pose_6dof = torch.cat([translation_params[i], rotation_params[i]])
                delta_T = pose_6dof_to_matrix(pose_6dof)
                optimized_extrinsics[i] = delta_T @ extrinsic_init_tensor[i]
            
            # Batch process all frame-camera combinations using the pre-constructed frame_groups
            all_losses = []
            total_valid_points = 0
            
            # Track points per (frame, camera) for visibility check
            frame_camera_point_counts = {}
            
            # Track per-camera losses for this iteration
            per_camera_losses_this_iter = {cam_serial: [] for cam_serial in ext_camera_serials}
            
            # Keep the double for loop, but group the internal computation as much as possible
            for frame_idx, camera_data_list in frame_groups.items():
                world_pts = all_world_points[frame_idx]  # [N_points, 3]
                
                # Process all cameras for this frame in batch
                for cam_idx, depth_tensor, intrinsic in camera_data_list:
                    optimized_T = optimized_extrinsics[cam_idx]  # [4, 4]

                    frame_camera_loss, valid_points_count_tensor = self._compute_frame_camera_loss(
                        world_pts, optimized_T, depth_tensor, intrinsic, H, W, dedup_threshold
                    )
                    
                    # Convert tensor to int for compatibility with existing logic
                    valid_points_count = valid_points_count_tensor.item()
                    
                    # Track points per (frame, camera) combination
                    cam_serial = ext_camera_serials[cam_idx]
                    frame_camera_point_counts[(frame_idx, cam_serial)] = valid_points_count
                    
                    # Track per-camera metrics for this iteration
                    if valid_points_count > 0:
                        all_losses.append(frame_camera_loss)
                        total_valid_points += valid_points_count
                        per_camera_losses_this_iter[cam_serial].append(frame_camera_loss.item())
            
            # Check robot visibility per (frame, camera) - raise error if ANY combination has insufficient points
            for (frame_idx, cam_serial), point_count in frame_camera_point_counts.items():
                if point_count < self.min_robot_points:
                    raise RobotVisibilityError(
                        f"Camera {cam_serial} at frame {frame_idx} sees insufficient robot points during optimization. "
                        f"Found {point_count} robot points, but require at least {self.min_robot_points} points.",
                        error_type="insufficient_robot_points_per_frame_camera"
                    )
            
            # Compute total loss
            if all_losses:
                data_loss = torch.stack(all_losses).mean()
            else:
                data_loss = torch.tensor(1e6, device=self.device, dtype=torch.float32)
            
            total_loss = data_loss
            
            # Store per-camera losses for this iteration
            for cam_serial in ext_camera_serials:
                if per_camera_losses_this_iter[cam_serial]:
                    avg_loss = np.mean(per_camera_losses_this_iter[cam_serial])
                    per_camera_losses_history[cam_serial].append(avg_loss)
                else:
                    per_camera_losses_history[cam_serial].append(0.0)
            
            # Store point counts from this iteration (will be overwritten each time, final iteration will be kept)
            for cam_serial in ext_camera_serials:
                final_per_camera_point_counts[cam_serial] = []
                for frame_idx in range(self.num_frames):
                    point_count = frame_camera_point_counts.get((frame_idx, cam_serial), 0)
                    final_per_camera_point_counts[cam_serial].append(point_count)
            
            # Backward pass
            if torch.isfinite(total_loss):
                total_loss.backward()
            else:
                _print("    Warning: Loss is not finite!")
            
            return total_loss
        
        # Check initial robot visibility
        initial_loss = compute_loss_vectorized()
        # Optimization loop
        loss_history = []
        
        for iteration in range(max_iterations):
            # Check for NaN/inf in parameters
            if not (torch.isfinite(translation_params_norm).all() and torch.isfinite(rotation_params_norm).all()):
                _print(f"    iter {iteration:03d} | NaN/inf detected in parameters, stopping")
                break
            
            # Optimizer step
            loss_value = optimizer.step(compute_loss_vectorized)
            
            # Track loss and print progress
            current_loss = loss_value.item() if torch.isfinite(loss_value) else float('inf')
            loss_history.append(current_loss)
            
            # Print progress (reduced frequency for speed)
            if iteration % 50 == 0 or iteration < 5 or iteration == max_iterations - 1:
                with torch.no_grad():
                    total_trans_diff_cm = 0.0
                    total_rot_diff_deg = 0.0
                    
                    for i, cam_serial in enumerate(ext_camera_serials):
                        translation_params = translation_params_norm[i].detach() * translation_scale
                        rotation_params = rotation_params_norm[i].detach() * rotation_scale
                        pose_6dof = torch.cat([translation_params, rotation_params])
                        delta_T = pose_6dof_to_matrix(pose_6dof)
                        optimized_T = delta_T @ extrinsic_init_tensor[i]
                        
                        trans_diff_cm = torch.linalg.norm(optimized_T[:3, 3] - extrinsic_init_tensor[i][:3, 3]).item() * 100
                        R_rel = optimized_T[:3, :3] @ extrinsic_init_tensor[i][:3, :3].T
                        rot_trace = torch.clamp((torch.trace(R_rel) - 1) / 2, -1.0, 1.0)
                        rot_diff_deg = torch.acos(rot_trace).item() * 180 / np.pi
                        
                        total_trans_diff_cm += trans_diff_cm
                        total_rot_diff_deg += rot_diff_deg
                    
                    avg_trans_diff_cm = total_trans_diff_cm / n_ext_cameras
                    avg_rot_diff_deg = total_rot_diff_deg / n_ext_cameras
                
                avg_points_per_frame = total_valid_points / (self.num_frames * n_ext_cameras)
                robot_status = f"ROBOT_VISIBLE ({avg_points_per_frame:.1f} avg per camera-frame)" 
                _print(f"    iter {iteration:03d} | loss {current_loss:.6f} | avg_trans {avg_trans_diff_cm:.2f}cm | avg_rot {avg_rot_diff_deg:.2f}deg | {robot_status}")
        
        # Compute final optimized extrinsics
        optimized_extrinsics_dict = {}
        for i, cam_serial in enumerate(ext_camera_serials):
            final_translation = translation_params_norm[i].detach() * translation_scale
            final_rotation = rotation_params_norm[i].detach() * rotation_scale
            final_pose_6dof = torch.cat([final_translation, final_rotation])
            optimized_extrinsic = (pose_6dof_to_matrix(final_pose_6dof) @ extrinsic_init_tensor[i]).cpu().numpy()
            optimized_extrinsics_dict[cam_serial] = optimized_extrinsic
            
            # Compute world-frame difference for final summary
            extrinsic_init_np = extrinsic_init_tensor[i].cpu().numpy()
            trans_change_cm = np.linalg.norm(optimized_extrinsic[:3, 3] - extrinsic_init_np[:3, 3]) * 100
            R_rel_np = optimized_extrinsic[:3, :3] @ extrinsic_init_np[:3, :3].T
            rot_change_deg = np.arccos(np.clip((np.trace(R_rel_np) - 1) / 2, -1, 1)) * 180 / np.pi
            _print(f"    Camera {cam_serial} final change: {trans_change_cm:.2f}cm translation, {rot_change_deg:.2f}deg rotation")
        
        # Compute optimization metrics
        optimization_metrics = {
            "initial_loss": float(initial_loss.item()),
            "final_loss": float(loss_history[-1]),
            "min_robot_points_threshold": self.min_robot_points
        }
        
        # Compute overall robot point statistics
        all_point_counts = []
        frames_with_sufficient_visibility = 0
        
        # Check each frame to see if all cameras have sufficient visibility
        for frame_idx in range(self.num_frames):
            frame_has_sufficient_visibility = True
            for cam_serial in ext_camera_serials:
                point_count = final_per_camera_point_counts[cam_serial][frame_idx] if frame_idx < len(final_per_camera_point_counts[cam_serial]) else 0
                all_point_counts.append(point_count)
                if point_count < self.min_robot_points:
                    frame_has_sufficient_visibility = False
            if frame_has_sufficient_visibility:
                frames_with_sufficient_visibility += 1
        
        assert all_point_counts, "No robot point counts found"
        optimization_metrics.update({
            "average_robot_points": float(np.mean(all_point_counts)),
            "max_robot_points": int(np.max(all_point_counts)),
            "min_robot_points": int(np.min(all_point_counts)),
            "frames_with_sufficient_visibility": frames_with_sufficient_visibility
        })
        
        # Add per-camera metrics
        for cam_serial in ext_camera_serials:
            camera_type = "wrist" if cam_serial == self.wrist_serial else "ext"
            prefix = f"{cam_serial}+{camera_type}_"
            
            camera_point_counts = final_per_camera_point_counts[cam_serial]
            camera_loss_history = per_camera_losses_history[cam_serial]
            
            # Compute per-camera stats
            assert camera_point_counts, f"No robot point counts found for camera {cam_serial}"
            assert camera_loss_history, f"No robot loss history found for camera {cam_serial}"
            optimization_metrics[f"{prefix}average_robot_points"] = float(np.mean(camera_point_counts))
            optimization_metrics[f"{prefix}max_robot_points"] = int(np.max(camera_point_counts))
            optimization_metrics[f"{prefix}min_robot_points"] = int(np.min(camera_point_counts))
            optimization_metrics[f"{prefix}frames_with_sufficient_visibility"] = sum(1 for count in camera_point_counts if count >= self.min_robot_points)
            optimization_metrics[f"{prefix}initial_loss"] = float(camera_loss_history[0])
            optimization_metrics[f"{prefix}final_loss"] = float(camera_loss_history[-1])
        
        return optimized_extrinsics_dict, optimization_metrics

    def add_depth_valid_masks(self):
        """Add depth valid masks following legacy single-stage pipeline pattern, type-dependent"""
        for camera_serial in self.data_dict:
            # Create a copy of keys to avoid RuntimeError when dict changes size during iteration
            keys_to_process = list(self.data_dict[camera_serial].keys())
            for key in keys_to_process:
                if key.endswith('_depth'):
                    depth_frames = self.data_dict[camera_serial][key]
                    depth_valid_mask = np.isfinite(depth_frames) & (depth_frames >= MIN_DEPTH_M) & (depth_frames <= MAX_DEPTH_M)
                    self.data_dict[camera_serial][f'{key}_valid_mask'] = depth_valid_mask

    def _prepare_vggt_input(self):
        """Prepare VGGT input ordering with external camera first."""
        aggregator_images = []
        aggregator_cam_ids = []
        aggregator_frame_idx = []

        # First external camera frames first (all frames)
        first_ext_serial = self.ext_serials[0]
        for i in range(len(self.data_dict[first_ext_serial]['rgb'])):
            aggregator_images.append(self.data_dict[first_ext_serial]['rgb'][i])
            aggregator_cam_ids.append(first_ext_serial)
            aggregator_frame_idx.append(i)

        # Then wrist camera frames (all frames)
        for i in range(len(self.data_dict[self.wrist_serial]['rgb'])):
            aggregator_images.append(self.data_dict[self.wrist_serial]['rgb'][i])
            aggregator_cam_ids.append(self.wrist_serial)
            aggregator_frame_idx.append(i)

        # Then remaining external cameras (all frames)
        for cam_serial in self.ext_serials[1:]:
            for i in range(len(self.data_dict[cam_serial]['rgb'])):
                aggregator_images.append(self.data_dict[cam_serial]['rgb'][i])
                aggregator_cam_ids.append(cam_serial)
                aggregator_frame_idx.append(i)

        return aggregator_images, aggregator_cam_ids, aggregator_frame_idx
    
    def _convert_vggt_to_base_frame(self, vggt_extrinsics, aggregator_cam_ids, aggregator_frame_idx):
        """
        Convert VGGT extrinsics from VGGT world frame to robot base frame.
        Uses pose averaging across multiple frames for more robust estimates.
        
        Args:
            vggt_extrinsics: [N, 4, 4] extrinsics from VGGT in VGGT world frame
            aggregator_cam_ids: List of camera serials corresponding to each extrinsic
            aggregator_frame_idx: List of frame indices within each camera
        Returns:
            dict: Camera serial -> extrinsic matrix in base frame
        """
        result = {}
        # VGGT world frame = first external camera frame at first frame
        # Step 1: Compute averaged estimate of T_base_ext0

        # Find wrist camera indices to get T_ext0 <- wrist across frames
        wrist_indices = [
            idx for idx, cser in enumerate(aggregator_cam_ids)
            if cser == self.wrist_serial
        ]

        assert len(wrist_indices) > 0, "No wrist camera found"

        # Step 1: Compute T_base_ext0 estimates from multiple frames
        T_base_ext0_estimates = []
        for wrist_idx in wrist_indices:
            frame_idx = aggregator_frame_idx[wrist_idx]
            T_base_wrist_t = self._compute_T_base_wrist(frame_idx)
            T_ext0_wrist_t = vggt_extrinsics[wrist_idx]
            T_wrist_t_ext0 = np.linalg.inv(T_ext0_wrist_t)
            T_base_ext0_estimate = T_wrist_t_ext0 @ T_base_wrist_t
            T_base_ext0_estimates.append(T_base_ext0_estimate)

        if len(T_base_ext0_estimates) > 1:
            T_base_ext0_avg = transform_utils.average_poses(T_base_ext0_estimates)
        else:
            T_base_ext0_avg = T_base_ext0_estimates[0]
        result[self.ext_serials[0]] = T_base_ext0_avg

        # Step 2: For all other external cameras, use ext0 as anchor and average poses
        other_ext_indices = {cam_serial: [] for cam_serial in self.ext_serials if cam_serial != self.ext_serials[0]}
        for idx, cam_serial in enumerate(aggregator_cam_ids):
            if cam_serial != self.wrist_serial and cam_serial != self.ext_serials[0]:
                assert cam_serial in self.ext_serials, f"Camera {cam_serial} not found in ext_serials"
                other_ext_indices[cam_serial].append(idx)

        for cam_serial, indices in other_ext_indices.items():
            T_ext0_cami_estimates = [vggt_extrinsics[idx] for idx in indices]
            if len(T_ext0_cami_estimates) > 1:
                T_ext0_cami_avg = transform_utils.average_poses(T_ext0_cami_estimates)
            else:
                T_ext0_cami_avg = T_ext0_cami_estimates[0]

            # Convert to base frame: T_base <- cam_i = T_ext0_cam_i @ T_base_ext0
            T_base_cam_i = T_ext0_cami_avg @ T_base_ext0_avg
            result[cam_serial] = T_base_cam_i

        return result

    def _filter_frames_by_robot_visibility(self, extrinsics_dict):
        """
        Filter frames to keep only those where all cameras see sufficient robot points.
        Uses initial robot visibility computation to identify problematic frames.
        
        Args:
            extrinsics_dict: Dict mapping camera_serial -> extrinsics matrix
            
        Returns:
            List of valid frame indices
        """
        start = time.time()
        valid_frame_indices = []
        
        for frame_idx in range(self.T):
            frame_valid = True
            
            # Get robot points for this frame
            joint_positions = self.proprio_dict['joint_positions'][frame_idx]
            gripper_positions = self.proprio_dict['gripper_positions'][frame_idx]
            fk_result = self.robot_renderer._get_forward_kinematics(joint_positions, gripper_positions)
            world_pts = self.robot_renderer._get_world_points(fk_result)
            
            # Check visibility for each external camera
            for camera_serial in self.data_dict:
                if camera_serial == self.wrist_serial:
                    continue
                    
                if camera_serial not in extrinsics_dict:
                    continue
                    
                # Get camera parameters
                extrinsic = extrinsics_dict[camera_serial]
                depth_frame = self.data_dict[camera_serial]['stereo_depth'][frame_idx]
                intrinsic = self.data_dict[camera_serial]['stereo_intrinsics']
                
                # Convert to tensors
                world_pts_tensor = torch.as_tensor(world_pts, dtype=torch.float32, device=self.device)
                extrinsic_tensor = torch.as_tensor(extrinsic, dtype=torch.float32, device=self.device)
                depth_tensor = torch.as_tensor(depth_frame, dtype=torch.float32, device=self.device)
                intrinsic_tensor = torch.as_tensor(intrinsic, dtype=torch.float32, device=self.device)
                
                H, W = depth_frame.shape
                
                # Compute robot point count for this (frame, camera) combination
                _, valid_points_count_tensor = self._compute_frame_camera_loss(
                    world_pts_tensor, extrinsic_tensor, depth_tensor, intrinsic_tensor, H, W, dedup_threshold=0.1
                )
                
                valid_points_count = valid_points_count_tensor.item()
                
                # If any camera sees insufficient points, mark frame as invalid
                if valid_points_count < self.min_robot_points:
                    _print(f"    Frame {frame_idx}: Camera {camera_serial} sees only {valid_points_count} robot points (< {self.min_robot_points}), removing frame")
                    frame_valid = False
                    break
            
            if frame_valid:
                valid_frame_indices.append(frame_idx)
        
        _print(f"Kept {len(valid_frame_indices)}/{self.T} frames after robot visibility filtering (time taken: {time.time() - start:.2f}s)")
        
        # Raise error if no frames are kept
        if len(valid_frame_indices) == 0:
            raise RobotVisibilityError(
                f"No frames have sufficient robot visibility. All {self.T} frames were filtered out because "
                f"at least one camera in each frame sees fewer than {self.min_robot_points} robot points.",
                error_type="no_valid_frames_after_filtering"
            )
        
        return valid_frame_indices

    def _update_data_for_valid_frames(self, valid_frame_indices):
        """Update all data structures to only include valid frames."""
        if len(valid_frame_indices) == self.T:
            return
            
        _print(f"Updating data structures for {len(valid_frame_indices)} valid frames...")
        
        # Update proprio_dict
        for key in self.proprio_dict:
            if isinstance(self.proprio_dict[key], list):
                self.proprio_dict[key] = [self.proprio_dict[key][i] for i in valid_frame_indices]
            elif isinstance(self.proprio_dict[key], np.ndarray) and len(self.proprio_dict[key]) == self.T:
                self.proprio_dict[key] = self.proprio_dict[key][valid_frame_indices]
        
        # Update data_dict for all cameras
        for camera_serial in self.data_dict:
            for key in self.data_dict[camera_serial]:
                if isinstance(self.data_dict[camera_serial][key], list) and len(self.data_dict[camera_serial][key]) == self.T:
                    self.data_dict[camera_serial][key] = [self.data_dict[camera_serial][key][i] for i in valid_frame_indices]
                elif isinstance(self.data_dict[camera_serial][key], np.ndarray) and len(self.data_dict[camera_serial][key]) == self.T:
                    self.data_dict[camera_serial][key] = self.data_dict[camera_serial][key][valid_frame_indices]
        
        # Update frame count
        self.T = len(valid_frame_indices)
        self.num_frames = min(self.num_frames, self.T)

    def vggt_estimation(self):
        """Estimate initial extrinsics and intrinsics using VGGT."""
        assert self.vggt_forward is not None, "No VGGT model provided"

        start = time.time()
        aggregator_images, aggregator_cam_ids, aggregator_frame_idx = self._prepare_vggt_input()

        if not aggregator_images:
            raise ValueError("No images prepared for VGGT input")

        with stage_vggt_images(aggregator_images) as temp_image_paths:
            vggt_result = self.vggt_forward(temp_image_paths)

        vggt_extrinsics_4x4 = vggt_result['extrinsics_4x4']  # [N, 4, 4]
        vggt_intrinsics = vggt_result['intrinsics']  # [N, 3, 3]

        base_frame_extrinsics = self._convert_vggt_to_base_frame(
            vggt_extrinsics_4x4,
            aggregator_cam_ids,
            aggregator_frame_idx,
        )

        for idx, cam_serial in enumerate(aggregator_cam_ids):
            if cam_serial == self.wrist_serial:
                continue
            assert cam_serial in self.data_dict, f"Camera {cam_serial} not found in data_dict"
            assert cam_serial in base_frame_extrinsics, f"Camera {cam_serial} not found in base_frame_extrinsics"

            self.data_dict[cam_serial]['vggt_extrinsics'] = base_frame_extrinsics[cam_serial]
            self.data_dict[cam_serial]['vggt_intrinsics'] = vggt_intrinsics[idx]

        _print(f"VGGT estimation completed for {len(base_frame_extrinsics)} cameras (time taken: {time.time() - start:.2f}s)")
        
        # Filter frames by robot visibility using VGGT extrinsics
        valid_frame_indices = self._filter_frames_by_robot_visibility(base_frame_extrinsics)
        
        # Update data structures to only include valid frames
        self._update_data_for_valid_frames(valid_frame_indices)

    def _collect_optimizer_inputs(self,
                                  depth_key: str,
                                  intrinsics_key: str,
                                  extrinsics_key: str):
        camera_data_dict = {}
        initial_extrinsics_dict = {}

        for camera_serial in self.data_dict:
            if camera_serial == self.wrist_serial:
                continue

            camera_data_dict[camera_serial] = {
                'rgb': self.data_dict[camera_serial]['rgb'],
                'depth': self.data_dict[camera_serial][depth_key],
                'depth_valid_mask': self.data_dict[camera_serial][f'{depth_key}_valid_mask'],
                'intrinsic': self.data_dict[camera_serial][intrinsics_key],
            }
            initial_extrinsics_dict[camera_serial] = self.data_dict[camera_serial][extrinsics_key]

        return camera_data_dict, initial_extrinsics_dict

    def optimize_extrinsics(self,
                         max_iterations=100,
                         learning_rate=0.001,
                         translation_scale=0.01,
                         rotation_scale=np.pi / 180.0,
                         dedup_threshold=0.5,
                         optimizer_type="adam"):
        """
        Optimize extrinsics using robot mesh rendering and depth-based optimization.
        Now performs joint optimization for all external cameras.
        
        Args:
            max_iterations: Maximum optimization iterations
            learning_rate: Optimization learning rate
            translation_scale: Scale for translation parameters
            rotation_scale: Scale for rotation parameters
            dedup_threshold: Coordinate deduplication threshold
            optimizer_type: Type of optimizer to use ("adam" or "lbfgs")
        """
        start = time.time()
        _print(f"  Using optimizer: {optimizer_type.upper()}")
        assert self.robot_renderer is not None, "No robot renderer available, skipping extrinsics optimization"

        camera_data_dict, initial_extrinsics_dict = self._collect_optimizer_inputs(
            depth_key='stereo_depth',
            intrinsics_key='stereo_intrinsics',
            extrinsics_key='vggt_extrinsics',
        )

        # Optimize extrinsics jointly for all external cameras
        optimized_extrinsics_dict, optimization_metrics = self._optimize_all_cameras_jointly(
            camera_data_dict=camera_data_dict,
            initial_extrinsics_dict=initial_extrinsics_dict,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            dedup_threshold=dedup_threshold,
            optimizer_type=optimizer_type
        )
        
        # Store optimized extrinsics
        for camera_serial, optimized_extrinsic in optimized_extrinsics_dict.items():
            self.data_dict[camera_serial]['optimized_extrinsics'] = optimized_extrinsic
        
        # Store optimization metrics
        self.optimization_metrics = optimization_metrics
        
        _print(f"Joint extrinsics optimization completed for {len(optimized_extrinsics_dict)} cameras (time taken: {time.time() - start:.2f}s)")

    def process(self,
                output_dir=None,
                max_iterations=100,
                learning_rate=0.001,
                translation_scale=0.01,
                rotation_scale=np.pi / 180.0,
                dedup_threshold=1.0,  # Increased for speed
                optimizer_type="adam",
                min_robot_points=None):
        """
        Main processing pipeline following legacy single-stage pipeline structure.
        
        Args:
            output_dir: Output directory for saving results and loading precomputed depth
            max_iterations: Maximum optimization iterations
            learning_rate: Optimization learning rate  
            translation_scale: Scale for translation parameters
            rotation_scale: Scale for rotation parameters
            dedup_threshold: Coordinate deduplication threshold
            optimizer_type: Type of optimizer to use ("adam" or "lbfgs")
            min_robot_points: Minimum number of robot points required (overrides instance default if provided)
        """
        # Update min_robot_points if provided at runtime
        if min_robot_points is not None:
            self.min_robot_points = min_robot_points
        
        # Load data
        self._load_scene_data()
        if self.T < self.num_frames:
            raise ValueError(f"too few frames in scene {self.scene_path}, T={self.T}, num_frames={self.num_frames}, skipping")

        if output_dir is None:
            raise ValueError("output_dir must be provided to load precomputed depth")
        load_precomputed_depth(
            output_dir=output_dir,
            uuid=self.uuid,
            data_dict=self.data_dict,
            wrist_serial=self.wrist_serial,
            log_fn=_print,
        )
        
        # Estimate initial extrinsics (no visualization; no VGGT depth required)
        self.vggt_estimation()
        
        # Add depth valid masks following legacy single-stage pipeline pattern
        self.add_depth_valid_masks()

        # Optimize extrinsics starting with VGGT
        self.optimize_extrinsics(
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            dedup_threshold=dedup_threshold,
            optimizer_type=optimizer_type,
        )
        
        # Save camera results as JSON
        if output_dir is not None:
            write_camera_results(
                output_dir=output_dir,
                uuid=self.uuid,
                scene_path=self.scene_path,
                data_dict=self.data_dict,
                wrist_serial=self.wrist_serial,
                optimization_metrics=self.optimization_metrics,
                error_info=None,
                log_fn=_print,
            )
        
        return None

################################################################################
# Helper functions
################################################################################

def determine_error_stage():
    """Determine which processing stage the error occurred in based on call stack"""
    import traceback
    import sys
    
    # Get the current exception's traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_traceback is not None:
        # Extract the traceback
        stack = traceback.extract_tb(exc_traceback)
        # Look for specific method names in the call stack to determine stage
        stack_str = str(stack)
    else:
        # Fallback to current call stack if no exception traceback available
        stack_str = str(traceback.extract_stack())
    
    if '_optimize_all_cameras_jointly' in stack_str or 'optimize_extrinsics' in stack_str:
        return "extrinsics_optimization"
    elif '_filter_frames_by_robot_visibility' in stack_str or 'vggt_estimation' in stack_str:
        return "vggt_estimation"
    elif '_load_scene_data' in stack_str:
        return "data_loading"
    else:
        return "unknown"

################################################################################
# Main function
################################################################################

def process_single_scene(scene_path, output_dir, vggt_forward, robot_renderer,
                        num_frames, downscale_ratio, min_robot_points,
                        max_iterations, lr, translation_scale, rotation_scale, dedup_threshold, optimizer,
                        debug):
    """Process a single scene for extrinsics optimization."""
    try:
        # Get UUID once for all checks
        uuid = get_uuid(scene_path)
        
        if output_dir is None:
            raise ValueError("output_dir must be provided to load precomputed depth")

        depth_exists, h5_path, error_msg = check_precomputed_depth_exists(scene_path, output_dir, uuid)
        if not depth_exists:
            raise FileNotFoundError(error_msg)

        _print(f"Verified precomputed depth file exists: {h5_path}")
        
        # Early check for existing camera results
        if output_dir is not None:
            results_exist, json_path, error_msg = check_camera_results_exist(scene_path, output_dir, uuid)
            if results_exist:
                _print(f"Camera results already exist and are readable: {json_path}, skipping scene")
                return True
        
        start = time.time()
        # Create ExtrinsicsOptimizer instance for the scene
        scene_optimizer = ExtrinsicsOptimizer(
            scene_path=scene_path,
            vggt_forward=vggt_forward,
            robot_renderer=robot_renderer,
            device=DEVICE,
            num_frames=num_frames,
            downscale_ratio=downscale_ratio,
            min_robot_points=min_robot_points,
        )
        
        # Process the scene
        scene_optimizer.process(
            output_dir=output_dir,
            max_iterations=max_iterations,
            learning_rate=lr,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            dedup_threshold=dedup_threshold,
            optimizer_type=optimizer
        )
        
        _print(f"Successfully processed scene: {scene_optimizer.uuid} (time taken: {time.time() - start:.2f}s)")
        return True
        
    except RobotVisibilityError as e:
        # Specific handling for robot visibility errors
        _print(f"Insufficient robot visibility for scene {scene_path}: {e}")
        if output_dir is not None:
            assert scene_optimizer is not None, "Optimizer not initialized"
            error_info = {
                "error_type": getattr(e, 'error_type', 'robot_visibility_error'),
                "error_message": str(e),
                "stage": determine_error_stage(),
                "min_robot_points_threshold": min_robot_points
            }
            write_camera_results(
                output_dir=output_dir,
                uuid=scene_optimizer.uuid,
                scene_path=scene_path,
                data_dict=scene_optimizer.data_dict,
                wrist_serial=scene_optimizer.wrist_serial,
                optimization_metrics=scene_optimizer.optimization_metrics,
                error_info=error_info,
                log_fn=_print,
            )
            _print(f"Saved partial results with robot visibility error for scene {scene_path}")
        return False
        
    except Exception as e:
        if 'Precomputed depth file not found' in str(e):
            _print(f"Precomputed depth file not found for scene {scene_path}, skipping")
            return False
        error_msg = f"Error processing scene {scene_path}: {e}"
        if debug:
            _print(f"{error_msg}")
            import traceback
            traceback.print_exc()
            raise
        else:
            _print(f"{error_msg}, skipping")
            return False
