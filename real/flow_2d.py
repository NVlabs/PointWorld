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
"""
2D flow tracking utilities for the real (DROID) pipeline.

This module provides tracker loading and clip slicing/filtering logic used by
the 2D flow cache generator. Visualization is intentionally omitted.
"""
from __future__ import annotations

import inspect
import os
import random
from typing import Dict, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

import transform_utils
from real.real_utils import get_time_str

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Checkpoints (explicit paths; update via CLI in the pipeline entrypoints)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COTRACKER_CKPT_PATH = os.path.join(
    REPO_ROOT,
    "checkpoints",
    "cotracker",
    "scaled_online.pth",
)


class Flow2DTracker:
    def __init__(
        self,
        cotracker_ckpt_path: str | None = None,
        deterministic_tracking: bool = False,
        deterministic_seed: int | None = None,
        tracker_query_mode: str = "legacy_skip_ratio",
    ) -> None:
        self.cotracker_ckpt_path = cotracker_ckpt_path or COTRACKER_CKPT_PATH
        self.deterministic_tracking = bool(deterministic_tracking)
        self.deterministic_seed = deterministic_seed
        if tracker_query_mode not in ("legacy_skip_ratio", "explicit_queries"):
            raise ValueError(
                "tracker_query_mode must be one of {'legacy_skip_ratio', 'explicit_queries'}; "
                f"got {tracker_query_mode!r}"
            )
        self.tracker_query_mode = tracker_query_mode
        self.tracker = None
        self._legacy_skip_ratio_supported: bool | None = None
        if self.deterministic_tracking:
            self._configure_determinism()

    def _configure_determinism(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if self.deterministic_seed is not None:
            seed = int(self.deterministic_seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    def _load_tracker(self):
        assert os.path.exists(self.cotracker_ckpt_path), (
            f"CoTracker checkpoint not found. Please download scaled_online.pth to {self.cotracker_ckpt_path}"
        )
        from cotracker.predictor import CoTrackerPredictor

        print(f"[{get_time_str()}] Loading CoTracker model...")
        tracker = CoTrackerPredictor(
            checkpoint=self.cotracker_ckpt_path,
            offline=True,
            window_len=16,
            v2=False,
        )
        tracker = tracker.to(DEFAULT_DEVICE).eval()
        print(f"[{get_time_str()}] Model loaded.")
        return tracker

    def slice_video_clips(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        proprio_dict: Dict[str, Any],
        frames_per_clip: int = 16,
        skip_every: int = 1,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Chunk each camera's video into overlapping clips of length frames_per_clip.
        Clips are sampled every skip_every frames; sequences shorter than frames_per_clip
        are padded with the last frame.
        """
        print(f"[{get_time_str()}] Chunking videos into overlapping clips...")

        T = len(next(iter(data_dict.values()))["rgb"])
        assert all(len(data_dict[camera_serial]["rgb"]) == T for camera_serial in data_dict), (
            "All camera sequences must have the same length"
        )

        start_indices = range(0, max(1, T - frames_per_clip + 1), skip_every)

        sliced_data_dict: Dict[str, Dict[str, Any]] = {}
        for camera_serial in tqdm(data_dict, desc="Chunking camera videos"):
            sliced_data_dict[camera_serial] = {}
            for start in start_indices:
                end = min(start + frames_per_clip, T)
                clip_key = f"{start}:{end}"
                sliced_data_dict[camera_serial][clip_key] = {}

                for modality in ["intrinsic", "extrinsic"]:
                    if modality in data_dict[camera_serial]:
                        sliced_data_dict[camera_serial][modality] = data_dict[camera_serial][modality]

                for modality in data_dict[camera_serial]:
                    if modality in ["intrinsic", "extrinsic", "baseline"]:
                        continue
                    data_slice = data_dict[camera_serial][modality][start:end]
                    if len(data_slice) < frames_per_clip:
                        orig_shape = data_slice.shape
                        pad_shape = list(orig_shape)
                        pad_shape[0] = frames_per_clip
                        padded_data = np.zeros(pad_shape, dtype=data_slice.dtype)
                        padded_data[:len(data_slice)] = data_slice
                        for i in range(len(data_slice), frames_per_clip):
                            padded_data[i] = data_slice[-1]
                        sliced_data_dict[camera_serial][clip_key][modality] = padded_data
                    else:
                        sliced_data_dict[camera_serial][clip_key][modality] = data_slice

        sliced_proprio_dict: Dict[str, Dict[str, Any]] = {}
        for start in start_indices:
            end = min(start + frames_per_clip, T)
            clip_key = f"{start}:{end}"
            sliced_proprio_dict[clip_key] = {}
            for modality in proprio_dict:
                data_slice = proprio_dict[modality][start:end]
                if len(data_slice) < frames_per_clip:
                    if isinstance(data_slice, np.ndarray):
                        orig_shape = data_slice.shape
                        pad_shape = list(orig_shape)
                        pad_shape[0] = frames_per_clip
                        padded_data = np.zeros(pad_shape, dtype=data_slice.dtype)
                        padded_data[:len(data_slice)] = data_slice
                        for i in range(len(data_slice), frames_per_clip):
                            padded_data[i] = data_slice[-1]
                    else:
                        assert isinstance(data_slice, list)
                        padded_data = data_slice + [data_slice[-1]] * (frames_per_clip - len(data_slice))
                    sliced_proprio_dict[clip_key][modality] = padded_data
                else:
                    sliced_proprio_dict[clip_key][modality] = data_slice

        return sliced_data_dict, sliced_proprio_dict

    def filter_clips_by_ee_motion(
        self,
        sliced_data_dict: Dict[str, Dict[str, Any]],
        sliced_proprio_dict: Dict[str, Dict[str, Any]],
        ee_pos_threshold: float = 0.005,
        ee_rot_threshold: float = 0.1,
        gripper_threshold: float = 0.1,
        gripper_closed_ee_pos_threshold: float = 0.002,
        gripper_closed_ee_rot_threshold: float = 0.05,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Filter clips based on end-effector motion. Clips with gripper changes are always kept.
        """
        clip_keys = [key for key in sliced_proprio_dict.keys() if ":" in key]
        initial_clip_count = len(clip_keys)

        valid_clips = []
        position_motion_stats = []
        rotation_motion_stats = []
        gripper_change_count = 0
        regular_ee_motion_count = 0
        gripper_closed_ee_motion_count = 0

        for clip_key in clip_keys:
            gripper_poses = sliced_proprio_dict[clip_key]["gripper_pose"]
            gripper_poses = transform_utils.convert_pose_quat2mat(gripper_poses)
            gripper_positions = sliced_proprio_dict[clip_key]["gripper_positions"]

            gripper_open = gripper_positions < gripper_threshold
            has_gripper_change = False
            if len(gripper_open) > 1:
                gripper_changes = np.diff(gripper_open.astype(int))
                has_gripper_change = np.any(gripper_changes != 0)

            pose_errors = []
            for i in range(len(gripper_poses) - 1):
                pose_error = transform_utils.get_pose_error(gripper_poses[i + 1], gripper_poses[i])
                pose_errors.append(pose_error)

            assert pose_errors, f"No pose errors found for clip {clip_key}"
            pose_errors = np.array(pose_errors)
            pos_errors = pose_errors[:, :3]
            rot_errors = pose_errors[:, 3:]

            total_pos_motion = np.sum(np.linalg.norm(pos_errors, axis=1))
            total_rot_motion = np.sum(np.linalg.norm(rot_errors, axis=1))
            position_motion_stats.append(total_pos_motion)
            rotation_motion_stats.append(total_rot_motion)

            if has_gripper_change:
                valid_clips.append(clip_key)
                gripper_change_count += 1
            else:
                is_gripper_predominantly_closed = np.mean(gripper_open) < 0.5
                if is_gripper_predominantly_closed:
                    pos_thresh = gripper_closed_ee_pos_threshold
                    rot_thresh = gripper_closed_ee_rot_threshold
                else:
                    pos_thresh = ee_pos_threshold
                    rot_thresh = ee_rot_threshold

                if total_pos_motion >= pos_thresh or total_rot_motion >= rot_thresh:
                    valid_clips.append(clip_key)
                    if is_gripper_predominantly_closed:
                        gripper_closed_ee_motion_count += 1
                    else:
                        regular_ee_motion_count += 1

        filtered_clip_count = len(valid_clips)
        filtered_percentage = (initial_clip_count - filtered_clip_count) / max(1, initial_clip_count) * 100

        print(f"[{get_time_str()}] End-effector motion filtering results:")
        print(f"  - Total clips: {initial_clip_count}")
        print(f"  - Kept due to gripper changes: {gripper_change_count}")
        print(f"  - Kept due to regular EE motion: {regular_ee_motion_count}")
        print(f"  - Kept due to gripper closed EE motion: {gripper_closed_ee_motion_count}")
        print(f"  - Total kept: {filtered_clip_count}")
        print(f"  - Filtered out: {initial_clip_count - filtered_clip_count} ({filtered_percentage:.1f}%)")

        if position_motion_stats:
            print(
                f"[{get_time_str()}] Position motion statistics: min={np.min(position_motion_stats):.4f}, "
                f"max={np.max(position_motion_stats):.4f}, mean={np.mean(position_motion_stats):.4f}, "
                f"median={np.median(position_motion_stats):.4f} m"
            )
        if rotation_motion_stats:
            print(
                f"[{get_time_str()}] Rotation motion statistics: min={np.min(rotation_motion_stats):.4f}, "
                f"max={np.max(rotation_motion_stats):.4f}, mean={np.mean(rotation_motion_stats):.4f}, "
                f"median={np.median(rotation_motion_stats):.4f} rad"
            )

        filtered_proprio_dict = {clip_key: sliced_proprio_dict[clip_key] for clip_key in valid_clips}

        filtered_data_dict: Dict[str, Dict[str, Any]] = {}
        for camera_serial in sliced_data_dict:
            filtered_data_dict[camera_serial] = {}
            for key in sliced_data_dict[camera_serial]:
                if ":" not in key:
                    filtered_data_dict[camera_serial][key] = sliced_data_dict[camera_serial][key]
            for clip_key in valid_clips:
                if clip_key in sliced_data_dict[camera_serial]:
                    filtered_data_dict[camera_serial][clip_key] = sliced_data_dict[camera_serial][clip_key]

        return filtered_data_dict, filtered_proprio_dict

    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def _tracker_inference_legacy(
        self,
        rgb_tensor: torch.Tensor,
        space_skip_ratio: int,
        seed_mask: np.ndarray | None,
    ):
        segm_mask = None
        if seed_mask is not None:
            H, W = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
            if seed_mask.shape != (H, W):
                raise ValueError(
                    f"seed_mask shape must be (H, W)=({H}, {W}), got {seed_mask.shape}"
                )
            segm_mask = (
                torch.from_numpy(seed_mask.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(rgb_tensor.device)
            )

        pred_tracks, pred_visibility = self.tracker(
            rgb_tensor,
            grid_size=0,
            grid_query_frame=0,
            backward_tracking=True,
            skip_ratio=space_skip_ratio,
            show_progress=False,
            segm_mask=segm_mask,
        )
        pred_tracks = pred_tracks[0].cpu().numpy()
        pred_visibility = pred_visibility[0, :, :].cpu().numpy()
        return pred_tracks, pred_visibility

    def _tracker_supports_legacy_skip_ratio(self) -> bool:
        if self.tracker is None:
            return False
        if self._legacy_skip_ratio_supported is None:
            try:
                sig = inspect.signature(self.tracker.forward)
                self._legacy_skip_ratio_supported = "skip_ratio" in sig.parameters
            except (TypeError, ValueError):
                self._legacy_skip_ratio_supported = False
        return bool(self._legacy_skip_ratio_supported)

    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def _tracker_inference_legacy_emulated(
        self,
        rgb_tensor: torch.Tensor,
        space_skip_ratio: int,
        seed_mask: np.ndarray | None,
    ):
        """
        Emulate legacy skip_ratio behavior for CoTracker builds that no longer
        expose the skip_ratio argument.

        Primary path mirrors the old dense tracker loop:
        - offset-grid dense queries (grid_size=192),
        - checkerboard skip filtering by `space_skip_ratio`,
        - segm-mask filtering at interpolated tracker resolution,
        - sparse tracking with `add_support_grid=False`.
        """
        _, T, _, H, W = rgb_tensor.shape
        segm_interp = None
        interp_h = interp_w = None
        scale_x = scale_y = None
        if seed_mask is not None:
            if seed_mask.shape != (H, W):
                raise ValueError(
                    f"seed_mask shape must be (H, W)=({H}, {W}), got {seed_mask.shape}"
                )
            seed_mask_t = (
                torch.from_numpy(seed_mask.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(rgb_tensor.device)
            )
            if hasattr(self.tracker, "interp_shape"):
                interp_h, interp_w = [int(v) for v in self.tracker.interp_shape]
            else:
                interp_h, interp_w = H, W
            segm_interp = torch.nn.functional.interpolate(
                seed_mask_t, (interp_h, interp_w), mode="nearest"
            )[0, 0] > 0.5
            scale_x = (interp_w - 1) / (W - 1) if W > 1 else 0.0
            scale_y = (interp_h - 1) / (H - 1) if H > 1 else 0.0

        sparse_fn = getattr(self.tracker, "_compute_sparse_tracks", None)
        if callable(sparse_fn):
            grid_size = 192
            grid_step = max(1, W // grid_size)
            grid_width = max(1, W // grid_step)
            grid_height = max(1, H // grid_step)

            sparse_sig = None
            try:
                sparse_sig = inspect.signature(sparse_fn)
            except (TypeError, ValueError):
                sparse_sig = None

            tracks_parts: list[torch.Tensor] = []
            vis_parts: list[torch.Tensor] = []
            for offset in range(grid_step * grid_step):
                ox = offset % grid_step
                oy = offset // grid_step
                x_coords = torch.arange(
                    grid_width, device=rgb_tensor.device, dtype=torch.float32
                ) * grid_step + ox
                y_coords = torch.arange(
                    grid_height, device=rgb_tensor.device, dtype=torch.float32
                ) * grid_step + oy
                mesh_x, mesh_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
                keep = (
                    (mesh_x.to(torch.long) % space_skip_ratio == 0)
                    & (mesh_y.to(torch.long) % space_skip_ratio == 0)
                )

                selected_x = mesh_x[keep].flatten()
                selected_y = mesh_y[keep].flatten()
                if selected_x.numel() == 0:
                    continue

                queries = torch.zeros(
                    (1, selected_x.shape[0], 3),
                    device=rgb_tensor.device,
                    dtype=torch.float32,
                )
                queries[0, :, 1] = selected_x
                queries[0, :, 2] = selected_y

                # Legacy segm-mask filtering happened after projection into interp space.
                if segm_interp is not None:
                    qx = torch.round(queries[0, :, 1] * scale_x).to(torch.long).clamp(0, interp_w - 1)
                    qy = torch.round(queries[0, :, 2] * scale_y).to(torch.long).clamp(0, interp_h - 1)
                    keep_interp = segm_interp[qy, qx]
                    queries = queries[:, keep_interp]
                    if queries.shape[1] == 0:
                        continue

                sparse_kwargs: Dict[str, Any] = {
                    "video": rgb_tensor,
                    "queries": queries,
                }
                optional_kwargs = {
                    "segm_mask": None,
                    "grid_size": 0,
                    "add_support_grid": False,
                    "grid_query_frame": 0,
                    "backward_tracking": True,
                }
                if sparse_sig is not None:
                    for k, v in optional_kwargs.items():
                        if k in sparse_sig.parameters:
                            sparse_kwargs[k] = v

                tracks_step, vis_step = sparse_fn(**sparse_kwargs)
                tracks_parts.append(tracks_step)
                vis_parts.append(vis_step)

            if tracks_parts:
                pred_tracks_t = torch.cat(tracks_parts, dim=2)
                pred_visibility_t = torch.cat(vis_parts, dim=2)
                return (
                    pred_tracks_t[0].cpu().numpy(),
                    pred_visibility_t[0, :, :].cpu().numpy(),
                )
            return (
                np.zeros((T, 0, 2), dtype=np.float32),
                np.zeros((T, 0), dtype=bool),
            )

        # Fallback: explicit queries with chunking when private sparse path is unavailable.
        ys = torch.arange(H, device=rgb_tensor.device, dtype=torch.float32)
        xs = torch.arange(W, device=rgb_tensor.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        keep = (
            (grid_x.to(torch.long) % space_skip_ratio == 0)
            & (grid_y.to(torch.long) % space_skip_ratio == 0)
        )
        x_flat = grid_x[keep]
        y_flat = grid_y[keep]
        if x_flat.numel() == 0:
            return (
                np.zeros((T, 0, 2), dtype=np.float32),
                np.zeros((T, 0), dtype=bool),
            )

        t_flat = torch.zeros_like(x_flat)
        queries = torch.stack([t_flat, x_flat, y_flat], dim=1).unsqueeze(0)
        if segm_interp is not None:
            qx = torch.round(queries[0, :, 1] * scale_x).to(torch.long).clamp(0, interp_w - 1)
            qy = torch.round(queries[0, :, 2] * scale_y).to(torch.long).clamp(0, interp_h - 1)
            keep_interp = segm_interp[qy, qx]
            queries = queries[:, keep_interp]
            if queries.shape[1] == 0:
                return (
                    np.zeros((T, 0, 2), dtype=np.float32),
                    np.zeros((T, 0), dtype=bool),
                )

        total_queries = queries.shape[1]
        chunk_size = 8192
        pred_tracks = np.zeros((T, total_queries, 2), dtype=np.float32)
        pred_visibility = np.zeros((T, total_queries), dtype=bool)
        for start in range(0, total_queries, chunk_size):
            end = min(start + chunk_size, total_queries)
            query_chunk = queries[:, start:end]
            tracks_chunk, vis_chunk = self.tracker(
                rgb_tensor,
                queries=query_chunk,
                backward_tracking=True,
            )
            pred_tracks[:, start:end] = tracks_chunk[0].cpu().numpy()
            pred_visibility[:, start:end] = vis_chunk[0, :, :].cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return pred_tracks, pred_visibility

    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def _tracker_inference_explicit_queries(
        self,
        rgb_tensor: torch.Tensor,
        space_skip_ratio: int,
        seed_mask: np.ndarray | None,
    ):
        _, _, _, H, W = rgb_tensor.shape
        ys = torch.arange(0, H, space_skip_ratio, device=rgb_tensor.device, dtype=torch.float32)
        xs = torch.arange(0, W, space_skip_ratio, device=rgb_tensor.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x_flat = grid_x.reshape(-1)
        y_flat = grid_y.reshape(-1)
        t_flat = torch.zeros_like(x_flat)
        queries = torch.stack([t_flat, x_flat, y_flat], dim=1).unsqueeze(0)

        if seed_mask is not None:
            if seed_mask.shape != (H, W):
                raise ValueError(
                    f"seed_mask shape must be (H, W)=({H}, {W}), got {seed_mask.shape}"
                )
            mask_bool = seed_mask.astype(bool)
            keep_np = mask_bool[
                y_flat.to(torch.long).cpu().numpy(),
                x_flat.to(torch.long).cpu().numpy(),
            ]
            keep = torch.from_numpy(keep_np.astype(np.bool_)).to(queries.device)
            queries = queries[:, keep]

        total_queries = queries.shape[1]
        T = rgb_tensor.shape[1]
        if total_queries == 0:
            return (
                np.zeros((T, 0, 2), dtype=np.float32),
                np.zeros((T, 0), dtype=bool),
            )

        chunk_size = 8192
        if total_queries <= chunk_size:
            pred_tracks, pred_visibility = self.tracker(
                rgb_tensor,
                queries=queries,
                backward_tracking=True,
            )
            pred_tracks = pred_tracks[0].cpu().numpy()
            pred_visibility = pred_visibility[0, :, :].cpu().numpy()
            return pred_tracks, pred_visibility

        pred_tracks = np.zeros((T, total_queries, 2), dtype=np.float32)
        pred_visibility = np.zeros((T, total_queries), dtype=bool)
        for start in range(0, total_queries, chunk_size):
            end = min(start + chunk_size, total_queries)
            query_chunk = queries[:, start:end]
            tracks_chunk, vis_chunk = self.tracker(
                rgb_tensor,
                queries=query_chunk,
                backward_tracking=True,
            )
            tracks_chunk = tracks_chunk[0].cpu().numpy()
            vis_chunk = vis_chunk[0, :, :].cpu().numpy()
            pred_tracks[:, start:end] = tracks_chunk
            pred_visibility[:, start:end] = vis_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return pred_tracks, pred_visibility

    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def _tracker_inference(
        self,
        rgb_seq: np.ndarray,
        space_skip_ratio: int = 4,
        seed_mask: np.ndarray | None = None,
    ):
        if self.tracker is None:
            self.tracker = self._load_tracker()

        rgb_tensor = torch.from_numpy(rgb_seq).float()
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(DEFAULT_DEVICE)

        if space_skip_ratio < 1:
            raise ValueError(f"space_skip_ratio must be >= 1, got {space_skip_ratio}")
        if self.tracker_query_mode == "legacy_skip_ratio":
            if self._tracker_supports_legacy_skip_ratio():
                return self._tracker_inference_legacy(
                    rgb_tensor=rgb_tensor,
                    space_skip_ratio=space_skip_ratio,
                    seed_mask=seed_mask,
                )
            return self._tracker_inference_legacy_emulated(
                rgb_tensor=rgb_tensor,
                space_skip_ratio=space_skip_ratio,
                seed_mask=seed_mask,
            )
        return self._tracker_inference_explicit_queries(
            rgb_tensor=rgb_tensor,
            space_skip_ratio=space_skip_ratio,
            seed_mask=seed_mask,
        )
