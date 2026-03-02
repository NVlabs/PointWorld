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

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from ..viser_tools.visualization_utils import (
    CameraObservation,
    apply_magenta_blend,
    build_camera_specs,
    filter_to_gripper_meshes,
    merge_camera_point_cloud,
    robot_clearance_mask,
)
from ..viser_flow.robot_sampler_lite import get_mesh_stable_id
from ..viser_flow.robot_flow import RobotFlow
from ..viser_flow.scene_builder import SceneBuilder
from ..viser_flow.timeline import (
    FlowTimeline,
    PointTimeline,
    build_point_timeline,
    build_rainbow_flow_timeline,
    build_constant_flow_timeline,
    blend_with_green,
)
from ..viser_flow.upsampling import (
    build_voxel_assignment,
)

import cv2

from .config import PredictionVisualizerConfig
from .sample import PredictionVisualizerSample, PredictionVisualizerLiveSession
from .utils import (
    _build_workspace_bounding_box,
    _ensure_float_array,
    _filter_background_to_point_voxels,
    _fps_select_indices_o3d,
)


class PredictionVisualizer:
    def __init__(
        self,
        config: PredictionVisualizerConfig,
        *,
        urdf_path: Path,
    ) -> None:
        self._config = config
        self._urdf_path = Path(urdf_path)
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF path not found: {self._urdf_path}")
        # Built-in per-domain URDF mapping to support mixed-domain batches.
        # Domain-specific URDF paths are required when the domain matches.
        self._urdf_map: dict[str, Path] = {
            "behavior": Path("assets/r1pro/urdf/r1pro.urdf"),
        }

    def viewer_endpoint(self) -> tuple[str, int]:
        return str(self._config.viewer_host), int(self._config.viewer_port)

    def _resolve_urdf_path(self, sample: "PredictionVisualizerSample") -> Path:
        dom = str(sample.domain or "").lower()
        # Prefer an exact or substring domain match.
        for key, path in self._urdf_map.items():
            if key in dom:
                p = Path(path)
                if not p.exists():
                    raise FileNotFoundError(
                        f"URDF path for domain '{dom}' not found: {p}"
                    )
                return p
        return self._urdf_path

    @staticmethod
    def _destroy_handles(handles: Iterable[object]) -> None:
        for handle in list(handles):
            try:
                handle.remove()
            except Exception:
                pass

    @staticmethod
    def _format_camera_pose_markdown(cam_handle) -> str:
        pos = np.asarray(cam_handle.position, dtype=np.float64).reshape(3)
        look = np.asarray(cam_handle.look_at, dtype=np.float64).reshape(3)
        up = np.asarray(cam_handle.up_direction, dtype=np.float64).reshape(3)
        fov_deg = float(np.degrees(cam_handle.fov))

        def _round_vec3(v: np.ndarray) -> list[float]:
            return [round(float(x), 8) for x in v.reshape(3)]

        keyframe_entry: Dict[str, object] = {
            "camera": {
                "position": _round_vec3(pos),
                "look_at": _round_vec3(look),
                "up": _round_vec3(up),
                "fov_deg": round(float(fov_deg), 6),
            }
        }

        import json as _json
        snippet = _json.dumps(keyframe_entry, indent=2)
        return "### Viewer camera pose\n```json\n" + snippet + "\n```"

    @staticmethod
    def _compute_bounds_shift_like_training(sample: PredictionVisualizerSample) -> np.ndarray:
        """Compute the center shift used by the training/eval data path.

        Training-time `center_shift` recenters using the first-frame mean of
        scene + robot points. For generated-H5 visualization we intentionally
        keep raw coordinates untouched, but shift workspace bounds by this same
        amount so bounds-gated visualization behavior (e.g., upsampling masks)
        matches the eval visualizer frame.
        """
        scene = np.asarray(sample.scene_ground_truth, dtype=np.float32)
        if scene.ndim != 3 or scene.shape[-1] != 3:
            raise ValueError("scene_ground_truth must have shape (T, N, 3)")
        scene_t0 = scene[0]

        if sample.robot_flows is None:
            raise ValueError(
                "robot_flows is required for bounds-shift alignment; "
                "this visualization path mirrors training/eval center_shift."
            )
        robot = _ensure_float_array(sample.robot_flows).astype(np.float32, copy=False)
        if robot.ndim != 3 or robot.shape[-1] != 3:
            raise ValueError("robot_flows must have shape (T, N, 3)")
        combined = np.concatenate([scene_t0, robot[0]], axis=0)

        if combined.ndim != 2 or combined.shape[-1] != 3 or combined.shape[0] == 0:
            raise ValueError(f"Invalid first-frame points for bounds shift: {combined.shape}")
        return combined.mean(axis=0).astype(np.float32)

    def _ensure_camera_pose_poller(
        self,
        live_session: PredictionVisualizerLiveSession,
    ) -> None:
        existing = live_session.camera_pose_poller_thread
        if existing is not None and existing.is_alive():
            return

        stop_event = threading.Event()
        live_session.camera_pose_poller_stop = stop_event

        def _poll_camera_pose() -> None:
            last_update_ts: Dict[int, float] = {}
            last_payload: Optional[str] = None
            while not stop_event.is_set():
                pose_md = live_session.camera_pose_markdown
                if pose_md is None:
                    stop_event.wait(0.5)
                    continue

                try:
                    clients = live_session.server.get_clients()
                except Exception:
                    clients = {}
                if not clients:
                    try:
                        pose_md.content = "Waiting for viewer camera pose updates..."
                    except Exception:
                        if live_session.camera_pose_markdown is pose_md:
                            live_session.camera_pose_markdown = None
                for client_id, client in clients.items():
                    try:
                        cam = client.camera
                        ts = float(cam.update_timestamp)  # raises if not initialized
                        payload = self._format_camera_pose_markdown(cam)
                        if last_update_ts.get(int(client_id)) == ts and payload == last_payload:
                            continue
                        try:
                            pose_md.content = payload
                            last_payload = payload
                        except Exception:
                            if live_session.camera_pose_markdown is pose_md:
                                live_session.camera_pose_markdown = None
                            break
                        last_update_ts[int(client_id)] = ts
                        break
                    except Exception:
                        continue
                stop_event.wait(0.5)

        poller = threading.Thread(target=_poll_camera_pose, daemon=True)
        live_session.camera_pose_poller_thread = poller
        poller.start()

    def _build_robot_flows(
        self,
        sample: PredictionVisualizerSample,
        overlay_indices: Sequence[int],
    ) -> Tuple[RobotFlow, RobotFlow, list[list[trimesh.Trimesh]]]:
        # Unified entry: build flows/overlays using whatever state is available.
        # Prefer Panda path if minimal fields exist; otherwise fall back to generic URDF path
        # that uses (joint_names, joint_positions_full, base_pose) and sample.robot_flows.
        from ..viser_tools.robot_overlays import build_robot_flows as _build

        urdf_sel = self._resolve_urdf_path(sample)
        return _build(
            urdf_path=urdf_sel,
            overlay_indices=overlay_indices,
            total_samples=int(self._config.robot_samples),
            min_samples_per_mesh=int(self._config.robot_min_samples),
            magenta_blend=float(self._config.robot_magenta_blend),
            gripper_overlay_opacity=1.0,
            full_overlay_opacity=0.5,
            # Optional states (handled inside)
            joint_positions=sample.joint_positions,
            gripper_positions=sample.gripper_positions,
            joint_names=sample.joint_names,
            joint_positions_full=sample.joint_positions_full,
            base_pose=sample.base_pose,
            robot_flows=sample.robot_flows,
        )

    def visualize(
        self,
        sample: PredictionVisualizerSample,
        *,
        launch_viewer: bool = False,
        live_session: Optional[PredictionVisualizerLiveSession] = None,
    ) -> Dict[str, object]:
        try:
            return self._visualize_impl(
                sample,
                launch_viewer=launch_viewer,
                live_session=live_session,
            )
        except KeyboardInterrupt:
            # Catch any stray interrupts to avoid killing eval loop.
            print("[prediction_visualizer] KeyboardInterrupt — skipping current sample.")
            return {"live_session": live_session}

    # Split implementation so we can cleanly intercept skip without messy state.
    def _visualize_impl(
        self,
        sample: PredictionVisualizerSample,
        *,
        launch_viewer: bool,
        live_session: Optional[PredictionVisualizerLiveSession],
    ) -> Dict[str, object]:
        # Existing implementation moved into this helper.
        # Optional per-domain camera upsampling (for visualization clarity)
        def _upsample_cam(cam: CameraObservation, scale: float) -> CameraObservation:
            s = float(scale)
            if s <= 1.0:
                return cam
            new_rgb = cam.rgb
            new_depth = cam.depth
            if cam.rgb is not None and cam.rgb.size:
                h, w = cam.rgb.shape[:2]
                new_w = max(1, int(round(w * s)))
                new_h = max(1, int(round(h * s)))
                # Use bicubic for RGB for quality
                new_rgb = cv2.resize(cam.rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if cam.depth is not None and cam.depth.size:
                h, w = cam.depth.shape[:2]
                new_w = max(1, int(round(w * s)))
                new_h = max(1, int(round(h * s)))
                # Use nearest for depth to preserve metric values
                new_depth = cv2.resize(cam.depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            intr = np.asarray(cam.intrinsic, dtype=np.float32).copy()
            if intr.shape == (3, 3):
                intr[0, 0] *= s
                intr[1, 1] *= s
                intr[0, 2] *= s
                intr[1, 2] *= s
            return CameraObservation(
                name=cam.name,
                intrinsic=intr,
                extrinsic_world_to_cam=np.asarray(cam.extrinsic_world_to_cam, dtype=np.float32),
                rgb=new_rgb,
                depth=new_depth,
            )

        dom = (sample.domain or "").lower()
        if "behavior" in dom:
            workspace_bounds_min = np.asarray(self._config.behavior_bounds_min, dtype=np.float32).copy()
            workspace_bounds_max = np.asarray(self._config.behavior_bounds_max, dtype=np.float32).copy()
            scene_point_size = float(self._config.behavior_scene_point_size)
            robot_point_size = float(self._config.behavior_robot_point_size)
        else:
            workspace_bounds_min = np.asarray(self._config.droid_bounds_min, dtype=np.float32).copy()
            workspace_bounds_max = np.asarray(self._config.droid_bounds_max, dtype=np.float32).copy()
            scene_point_size = float(self._config.droid_scene_point_size)
            robot_point_size = float(self._config.droid_robot_point_size)

        # Visualization-only alignment to the centered frame used by eval/training
        # data transforms (`center_shift`). We do NOT mutate scene/robot/camera
        # data here; we only translate workspace bounds, which are also used by
        # upsampling-related background filtering.
        bounds_shift = self._compute_bounds_shift_like_training(sample)
        workspace_bounds_min = workspace_bounds_min + bounds_shift
        workspace_bounds_max = workspace_bounds_max + bounds_shift

        cams_input = list(sample.cameras)
        if "behavior" in dom and float(self._config.behavior_camera_upsample_ratio) > 1.0:
            s = float(self._config.behavior_camera_upsample_ratio)
            cams_input = [_upsample_cam(c, s) for c in cams_input]
        # Dynamic workspace bounds from scene flows over all timesteps (world frame)
        exists_mask_all = sample.scene_exists.astype(bool)
        pts_all = np.asarray(sample.scene_ground_truth, dtype=np.float32)
        if pts_all.ndim != 3 or pts_all.shape[-1] != 3:
            raise ValueError("scene_ground_truth must have shape (T, N, 3)")
        if exists_mask_all.shape != pts_all.shape[:2]:
            raise ValueError("scene_exists must match scene_ground_truth shape (T, N)")
        if not np.any(exists_mask_all):
            raise ValueError("No valid scene points to compute dynamic bounds")
        pts_valid = pts_all[exists_mask_all]
        dyn_bounds_min = pts_valid.min(axis=0).astype(np.float32)
        dyn_bounds_max = pts_valid.max(axis=0).astype(np.float32)

        dyn_min_raw = dyn_bounds_min.copy()
        dyn_max_raw = dyn_bounds_max.copy()
        if np.all(dyn_min_raw >= workspace_bounds_min):
            dyn_bounds_min = np.maximum(dyn_bounds_min, workspace_bounds_min)
        if np.all(dyn_max_raw <= workspace_bounds_max):
            dyn_bounds_max = np.minimum(dyn_bounds_max, workspace_bounds_max)

        background_points, background_colors = merge_camera_point_cloud(
            cams_input,
            dyn_bounds_min,
            dyn_bounds_max,
            include_out_of_bounds=self._config.include_out_of_bounds_points,
        )

        num_frames = int(sample.scene_ground_truth.shape[0])
        # Dynamic overlays: include every frame index
        overlay_indices = list(range(num_frames))
        robot_flow, full_flow, clearance_meshes = self._build_robot_flows(sample, overlay_indices)
        if clearance_meshes:
            clearance_meshes = [
                [mesh.copy() for mesh in parts]
                for parts in clearance_meshes
            ]

        world_transform = None
        if sample.world_transform is not None:
            wt = _ensure_float_array(sample.world_transform)
            if wt.shape == (4, 4):
                world_transform = wt.astype(np.float32)
        elif sample.world_shift is not None:
            wt = np.eye(4, dtype=np.float32)
            wt[:3, 3] = -np.asarray(sample.world_shift, dtype=np.float32).reshape(3)
            world_transform = wt

        if world_transform is not None:
            R = world_transform[:3, :3]
            t = world_transform[:3, 3]
            apply_to_trajectories = sample.robot_flows is None or not np.asarray(sample.robot_flows).size
            if apply_to_trajectories and robot_flow.trajectories.size:
                robot_flow.trajectories = (
                    np.einsum("ij,tkj->tki", R, robot_flow.trajectories).astype(np.float32)
                    + t
                )
            if robot_flow.overlay_meshes:
                for parts in robot_flow.overlay_meshes:
                    for mesh in parts:
                        mesh.apply_transform(world_transform)
            if full_flow.overlay_meshes:
                for parts in full_flow.overlay_meshes:
                    for mesh in parts:
                        mesh.apply_transform(world_transform)
            if clearance_meshes:
                for parts in clearance_meshes:
                    for mesh in parts:
                        mesh.apply_transform(world_transform)
        # Enforce overlay opacity behavior (gripper opaque; full arm 50%)
        if robot_flow.overlay_meshes:
            robot_flow.overlay_opacities = [1.0 for _ in robot_flow.overlay_meshes]
        if full_flow.overlay_meshes:
            full_flow.overlay_opacities = [0.5 for _ in full_flow.overlay_meshes]

        if self._config.robot_magenta_blend > 0.0:
            # Apply magenta tint for gripper-only overlays; keep full-body original
            apply_magenta_blend(robot_flow, float(self._config.robot_magenta_blend))

        robot_tracks = None
        if sample.robot_flows is not None and np.asarray(sample.robot_flows).size:
            robot_tracks = _ensure_float_array(sample.robot_flows)
            if robot_tracks.ndim != 3 or robot_tracks.shape[-1] != 3:
                raise ValueError("robot_flows must have shape (T, N, 3)")
            robot_flow.trajectories = robot_tracks.astype(np.float32, copy=False)
        else:
            robot_tracks = robot_flow.trajectories

        # Robot mask: require existence across all timesteps for flow
        if robot_tracks.size:
            if sample.robot_exists.ndim != 2 or sample.robot_exists.shape[0] != robot_tracks.shape[0] or sample.robot_exists.shape[1] != robot_tracks.shape[1]:
                raise ValueError("robot_exists (T,N) mismatch with robot_flows")
            robot_all_mask = sample.robot_exists.all(axis=0)
            if np.any(~robot_all_mask):
                robot_tracks = robot_tracks[:, robot_all_mask, :]
                robot_flow.trajectories = robot_tracks
                robot_exists_current = sample.robot_exists[:, robot_all_mask]
            else:
                robot_exists_current = sample.robot_exists
        else:
            robot_exists_current = sample.robot_exists

        # New background filtering: keep only voxels that contain at least one scene point at t=0
        # Domain-specific bounds choice: droid → dynamic bounds; behavior → config bounds
        p0 = sample.scene_ground_truth[0]
        exists0 = sample.scene_exists[0]
        if "behavior" in dom:
            fb_min = workspace_bounds_min
            fb_max = workspace_bounds_max
        else:
            fb_min = dyn_bounds_min
            fb_max = dyn_bounds_max
        background_points, background_colors = _filter_background_to_point_voxels(
            background_points,
            background_colors,
            p0,
            exists0,
            float(self._config.grid_size),
            fb_min,
            fb_max,
        )

        filter_to_gripper_meshes(robot_flow)

        # Remove robot points from background using clearance meshes (first overlay frame)
        if clearance_meshes and background_points.size:
            parts0 = clearance_meshes[0] if len(clearance_meshes) > 0 else []
            if parts0:
                # Domain-specific robot clearance (configurable)
                eff_clearance = (float(self._config.behavior_robot_point_clearance) if ("behavior" in dom) else float(self._config.droid_robot_point_clearance))
                keep_mask = robot_clearance_mask(
                    background_points,
                    parts0,
                    float(eff_clearance),
                )
                if np.asarray(keep_mask).dtype == bool and keep_mask.shape[0] == background_points.shape[0]:
                    background_points = background_points[keep_mask]
                    background_colors = background_colors[keep_mask]

        # Note: 2D mask-based culling of background points is disabled to match
        # prior droid behavior. We keep only the 3D clearance-based pruning.

        # Build base colors (T, N, 3)
        Tn, Nn = sample.scene_exists.shape
        base_colors_T: Optional[np.ndarray] = None
        if sample.scene_colors is not None:
            cols = np.asarray(sample.scene_colors)
            if cols.ndim == 2:
                if cols.shape[1] != 3:
                    raise ValueError("scene_colors last dim must be 3")
                # Pad or truncate to match Nn, then broadcast across T
                Nc = cols.shape[0]
                if Nc < Nn:
                    pad = np.zeros((Nn - Nc, 3), dtype=cols.dtype)
                    cols2 = np.concatenate([cols, pad], axis=0)
                else:
                    cols2 = cols[:Nn]
                base_colors_T = np.broadcast_to(cols2[None, ...], (Tn, Nn, 3)).copy().astype(np.uint8)
            elif cols.ndim == 3:
                if cols.shape[2] != 3:
                    raise ValueError("scene_colors expected last dim 3 (RGB)")
                Tcols, Nc, _ = cols.shape
                if Tcols == Tn:
                    pass  # good
                elif Tcols == 1:
                    # Broadcast single-frame colors across time
                    cols = np.broadcast_to(cols, (Tn, Nc, 3)).copy()
                else:
                    raise ValueError(f"scene_colors time dimension mismatch; expected T={Tn}, got {Tcols}")
                if Nc == Nn:
                    base_colors_T = cols.astype(np.uint8, copy=False)
                elif Nc < Nn:
                    pad = np.zeros((Tn, Nn - Nc, 3), dtype=cols.dtype)
                    base_colors_T = np.concatenate([cols, pad], axis=1).astype(np.uint8, copy=False)
                else:
                    base_colors_T = cols[:, :Nn, :].astype(np.uint8, copy=False)
            else:
                raise ValueError("scene_colors must be (N,3) or (T,N,3)")
        else:
            base_colors_T = np.zeros((Tn, Nn, 3), dtype=np.uint8)

        # Positions and masks
        exists_mask = sample.scene_exists.astype(bool)
        supervised_mask = sample.scene_supervised_mask.astype(bool)
        pred_positions = sample.scene_prediction.astype(np.float32) if sample.scene_prediction is not None else None
        if pred_positions is not None and pred_positions.shape[:2] != (Tn, Nn):
            raise ValueError("scene_prediction shape mismatch with ground truth")
        gt_positions = sample.scene_ground_truth.astype(np.float32)

        # Green-tinted GT colors for unsupervised at each timestep
        gt_colors_tinted = blend_with_green(
            base_colors_T,
            ~supervised_mask,
            alpha=float(self._config.unsup_green_alpha),
        )

        # Particle timelines
        coarse_pred_tl: Optional[PointTimeline] = None
        if pred_positions is not None:
            coarse_pred_tl = build_point_timeline(pred_positions, base_colors_T, exists_mask)
        coarse_gt_tl = build_point_timeline(gt_positions, gt_colors_tinted, exists_mask)

        # Flow timelines (rainbow for both pred and GT)
        if pred_positions is not None:
            # For behavior, mirror GT logic: only fully supervised points contribute to flows
            if "behavior" in dom:
                gt_fully_supervised_tmp = supervised_mask.all(axis=0)
                if np.any(gt_fully_supervised_tmp):
                    flow_pred_tl = build_rainbow_flow_timeline(
                        pred_positions[:, gt_fully_supervised_tmp, :],
                        exists_mask[:, gt_fully_supervised_tmp],
                        colormap=lambda u: __import__("matplotlib").cm.get_cmap(self._config.flow_colormap)(u),
                        min_brightness=float(self._config.min_flow_brightness),
                    )
                else:
                    flow_pred_tl = FlowTimeline.empty(Tn)
            else:
                flow_pred_tl = build_rainbow_flow_timeline(
                    pred_positions,
                    exists_mask,
                    colormap=lambda u: __import__("matplotlib").cm.get_cmap(self._config.flow_colormap)(u),
                    min_brightness=float(self._config.min_flow_brightness),
                )
        else:
            flow_pred_tl = FlowTimeline.empty(Tn)
        # Use the dataset-provided supervision for both domains: only fully-supervised points render GT flows.
        gt_fully_supervised = supervised_mask.all(axis=0)
        if np.any(gt_fully_supervised):
            flow_gt_tl = build_rainbow_flow_timeline(
                gt_positions[:, gt_fully_supervised, :],
                exists_mask[:, gt_fully_supervised],
                colormap=lambda u: __import__("matplotlib").cm.get_cmap(self._config.flow_colormap)(u),
                min_brightness=float(self._config.min_flow_brightness),
            )
        else:
            flow_gt_tl = FlowTimeline.empty(Tn)

        # Robot flow timeline (constant color) and optional robot point timeline
        robot_flow_tl: FlowTimeline = FlowTimeline.empty(Tn)
        robot_flows_tl: Optional[PointTimeline] = None
        if robot_tracks is not None and robot_tracks.size:
            tracks_for_lines = np.asarray(robot_tracks, dtype=np.float32)
            robot_flow_tl = build_constant_flow_timeline(
                tracks_for_lines,
                robot_exists_current.astype(bool),
                active_mask=None,
                color_rgb=self._config.robot_color_rgb,
                min_brightness=float(self._config.min_robot_transparency),
            )
            Nr = tracks_for_lines.shape[1]
            mag = np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (Nr, 1))
            robot_colors_T = np.broadcast_to(mag[None, ...], (Tn, Nr, 3)).copy()
            robot_flows_tl = build_point_timeline(tracks_for_lines, robot_colors_T, robot_exists_current.astype(bool))

        # Build a single voxel assignment for the filtered background.
        current_assignment = build_voxel_assignment(
            background_points,
            background_colors,
            gt_positions,
            exists_mask,
            grid_size=float(self._config.grid_size),
        )
        upsample_pred_tl: Optional[PointTimeline] = None
        if pred_positions is not None:
            upsample_pred_tl = current_assignment.build_timeline(
                pred_positions,
                exists_mask,
                supervised=None,
            )
        upsample_gt_tl = current_assignment.build_timeline(
            gt_positions,
            exists_mask,
            supervised=supervised_mask,
            tint_alpha=float(self._config.unsup_green_alpha),
        )

        camera_specs = build_camera_specs(
            cams_input,
            reflection_matrix=sample.world_reflection,
        )
        # Domain-specific depth visualization ranges
        if "behavior" in dom:
            depth_min = float(self._config.depth_min_behavior)
            depth_max = float(self._config.depth_max_behavior)
        else:
            depth_min = float(self._config.depth_min_droid)
            depth_max = float(self._config.depth_max_droid)

        scene_builder = SceneBuilder(
            point_size=float(scene_point_size),
            scene_line_width=float(self._config.scene_line_width),
            robot_line_width=float(self._config.robot_line_width),
            depth_min=depth_min,
            depth_max=depth_max,
            background_dimness=float(self._config.background_dimness),
            background_point_cloud_rgb=self._config.background_point_cloud_rgb,
            frustum_scale=float(self._config.frustum_scale),
            frustum_line_width=float(self._config.frustum_line_width),
            camera_media_max_height=int(self._config.gui_media_max_height),
        )

        metadata = {
            "clip_key": sample.clip_key,
            "scene_points": int(background_points.shape[0]),
            "robot_samples": int(robot_flow.num_points),
            "movement_threshold_m": float(self._config.movement_threshold),
        }
        if world_transform is not None:
            metadata["world_transform"] = world_transform

        # Interactive path: skip static build to avoid starting/stopping an extra server.
        live_session_obj: Optional[PredictionVisualizerLiveSession] = None
        if live_session is not None:
            self._destroy_handles(live_session.custom_gui_handles)
            self._destroy_handles(live_session.builder_gui_handles)
            builder_handles = scene_builder.populate_existing(
                live_session.server,
                scene_points=background_points,
                scene_colors=background_colors,
                scene_flow_points=np.empty((0, 2, 3), dtype=np.float32),
                scene_flow_colors=np.empty((0, 2, 3), dtype=np.uint8),
                robot_flow_points=np.empty((0, 2, 3), dtype=np.float32),
                robot_flow_colors=np.empty((0, 2, 3), dtype=np.uint8),
                cameras=(camera_specs if bool(self._config.show_camera_frustums) else []),
                metadata=metadata,
                with_gui=(not bool(self._config.disable_gui)),
                background_visible=False,
            )
            if bool(self._config.disable_gui):
                custom_handles = []
            else:
                custom_handles = self._setup_gui(
                    live_session.server,
                    live_session,
                    sample,
                    workspace_bounds_min,
                    workspace_bounds_max,
                    coarse_pred_tl=coarse_pred_tl,
                    coarse_gt_tl=coarse_gt_tl,
                    upsample_pred_tl=upsample_pred_tl,
                    upsample_gt_tl=upsample_gt_tl,
                    flow_pred_tl=flow_pred_tl,
                    flow_gt_tl=flow_gt_tl,
                    robot_flow_tl=robot_flow_tl,
                    robot_flows_tl=robot_flows_tl,
                    gripper_overlays=robot_flow.overlay_meshes or [],
                    full_overlays=full_flow.overlay_meshes or [],
                    default_gt_checked=(pred_positions is None),
                    exists_mask=exists_mask,
                    pred_positions_orig=pred_positions,
                    scene_point_size=scene_point_size,
                    robot_point_size=robot_point_size,
                )
            live_session.builder = scene_builder
            live_session.builder_gui_handles = builder_handles
            live_session.custom_gui_handles = custom_handles
            live_session_obj = live_session
        elif launch_viewer:
            try:
                live_server, stopper, builder_handles = scene_builder.launch_interactive(
                    scene_points=background_points,
                    scene_colors=background_colors,
                    scene_flow_points=np.empty((0, 2, 3), dtype=np.float32),
                    scene_flow_colors=np.empty((0, 2, 3), dtype=np.uint8),
                    robot_flow_points=np.empty((0, 2, 3), dtype=np.float32),
                    robot_flow_colors=np.empty((0, 2, 3), dtype=np.uint8),
                    cameras=(camera_specs if bool(self._config.show_camera_frustums) else []),
                    metadata=metadata,
                    host=self._config.viewer_host,
                    port=int(self._config.viewer_port),
                    viewer_background_rgb=self._config.viewer_background_rgb,
                    with_gui=(not bool(self._config.disable_gui)),
                    background_visible=False,
                )
                if bool(self._config.disable_gui):
                    custom_handles = []
                else:
                    live_session_obj = PredictionVisualizerLiveSession(
                        server=live_server,
                        stopper=stopper,
                        builder=scene_builder,
                        builder_gui_handles=builder_handles,
                        custom_gui_handles=[],
                    )
                    custom_handles = self._setup_gui(
                        live_server,
                        live_session_obj,
                        sample,
                        workspace_bounds_min,
                        workspace_bounds_max,
                        coarse_pred_tl=coarse_pred_tl,
                        coarse_gt_tl=coarse_gt_tl,
                        upsample_pred_tl=upsample_pred_tl,
                        upsample_gt_tl=upsample_gt_tl,
                        flow_pred_tl=flow_pred_tl,
                        flow_gt_tl=flow_gt_tl,
                        robot_flow_tl=robot_flow_tl,
                        robot_flows_tl=robot_flows_tl,
                        gripper_overlays=robot_flow.overlay_meshes or [],
                        full_overlays=full_flow.overlay_meshes or [],
                        default_gt_checked=(pred_positions is None),
                        exists_mask=exists_mask,
                        pred_positions_orig=pred_positions,
                        scene_point_size=scene_point_size,
                        robot_point_size=robot_point_size,
                    )
                if live_session_obj is None:
                    live_session_obj = PredictionVisualizerLiveSession(
                        server=live_server,
                        stopper=stopper,
                        builder=scene_builder,
                        builder_gui_handles=builder_handles,
                        custom_gui_handles=custom_handles,
                    )
                else:
                    live_session_obj.custom_gui_handles = custom_handles
            except OSError as exc:  # pragma: no cover - viewer optional
                print(f"[warning] Failed to launch viewer: {exc}")

        result: Dict[str, object] = {}
        if live_session_obj is not None:
            result["live_session"] = live_session_obj
        return result

    def _setup_gui(
        self,
        server,
        live_session: PredictionVisualizerLiveSession,
        sample: PredictionVisualizerSample,
        workspace_bounds_min: np.ndarray,
        workspace_bounds_max: np.ndarray,
        *,
        coarse_pred_tl: Optional[PointTimeline],
        coarse_gt_tl: PointTimeline,
        upsample_pred_tl: Optional[PointTimeline],
        upsample_gt_tl: PointTimeline,
        flow_pred_tl: FlowTimeline,
        flow_gt_tl: FlowTimeline,
        robot_flow_tl: FlowTimeline,
        robot_flows_tl: Optional[PointTimeline],
        gripper_overlays: Sequence[Sequence[object]],
        full_overlays: Sequence[Sequence[object]],
        default_gt_checked: bool,
        exists_mask: np.ndarray,
        pred_positions_orig: Optional[np.ndarray],
        scene_point_size: float = 0.004,
        robot_point_size: float = 0.004,
    ) -> List[object]:
        # Domain flag helpers
        is_behavior = isinstance(sample.domain, str) and ("behavior" in sample.domain.lower())

        bbox_handle = None
        bbox_segments = _build_workspace_bounding_box(workspace_bounds_min, workspace_bounds_max)
        if bbox_segments.size:
            bbox_color = np.array([0, 255, 255], dtype=np.uint8)
            bbox_colors = np.broadcast_to(bbox_color, (bbox_segments.shape[0], 2, 3)).copy()
            try:
                bbox_handle = server.scene.add_line_segments(
                    "workspace/bounds",
                    points=bbox_segments.astype(np.float32, copy=False),
                    colors=bbox_colors,
                    line_width=float(max(self._config.scene_line_width, 1.0)),
                )
                bbox_handle.visible = False
            except Exception:
                bbox_handle = None

        created: List[object] = []

        control_folder = server.gui.add_folder("prediction controls", expand_by_default=True)
        created.append(control_folder)
        with control_folder:
            if bbox_handle is not None:
                bounds_checkbox = server.gui.add_checkbox(
                    label="Show workspace bounds",
                    initial_value=True,
                )

                def _bounds_cb(event):
                    vis = bool(event.target.value)
                    bbox_handle.visible = vis
                    state["workspace_bounds"] = vis

                bounds_checkbox.on_update(_bounds_cb)

            # Removed: robot point cloud overlay per user request

            # Removed: initial full robot overlay toggle to simplify UI

            # We manage GT/Pred flows in dynamic timeline section below

        # Viewer camera pose display (polling-only for remote reliability).
        # Reflection (for future maintainers):
        # - Direct client.camera.on_update can be flaky over SSH port forwarding
        #   (intermittent camera message delivery), but polling server.get_clients()
        #   reliably observes the latest camera state across remote/browser setups.
        # - Updating GuiMarkdownHandle must use `.content = ...` (which updates the
        #   underlying `_markdown` prop). Assigning to a non-existent `.markdown`
        #   attribute silently does nothing and the panel will appear stuck.
        # We therefore keep a single polling path that drives the UI markdown. If a
        # future version of Viser fixes on_update under tunneling, this can be
        # revisited to reduce the poll frequency.
        pose_md = server.gui.add_markdown("Waiting for viewer camera pose updates...")
        created.append(pose_md)
        live_session.camera_pose_markdown = pose_md
        self._ensure_camera_pose_poller(live_session)

        # -----------------------------
        # Dynamic timeline visuals setup
        # -----------------------------
        T = coarse_gt_tl.num_frames

        # Init state
        state = {
            "frame": 0,
            "use_gt": bool(default_gt_checked),
            "upsample": True,
            "workspace_bounds": True,
            "full_overlay_opacity": 0.5,
            "scene_flow_density": float(self._config.scene_flow_density_default),
            "robot_flow_density": float(self._config.robot_flow_density_default),
            "scene_flow_thickness": float(self._config.scene_line_width),
            "robot_flow_thickness": float(self._config.robot_line_width),
            "scene_point_size": float(scene_point_size),
        }
        if bbox_handle is not None:
            bbox_handle.visible = bool(state["workspace_bounds"])
        pred_state = {"coarse": coarse_pred_tl, "flow": flow_pred_tl, "upsample": upsample_pred_tl}
        has_prediction = pred_state["coarse"] is not None
        if not has_prediction:
            state["use_gt"] = True

        # Dynamic point and upsampled point cloud handles
        init_pts, init_cols = (coarse_gt_tl.frame(0) if (state["use_gt"] or pred_state["coarse"] is None) else pred_state["coarse"].frame(0))
        dyn_pc = server.scene.add_point_cloud(
            "scene/dynamic_points",
            points=init_pts.astype(np.float32, copy=False),
            colors=init_cols.astype(np.uint8, copy=False),
            point_size=float(scene_point_size),
            point_shape="rounded",
            precision="float32",
        )
        up_pts, up_cols = (upsample_gt_tl.frame(0) if (state["use_gt"] or pred_state["upsample"] is None) else pred_state["upsample"].frame(0))
        up_pc = server.scene.add_point_cloud(
            "scene/upsampled_cloud",
            points=up_pts.astype(np.float32, copy=False),
            colors=up_cols.astype(np.uint8, copy=False),
            point_size=float(scene_point_size),
            point_shape="rounded",
            precision="float32",
        )
        up_pc.visible = True
        dyn_pc.visible = False

        # Flow handles (pred and GT; visibility controlled by toggle + density)
        fp_pts, fp_cols = flow_pred_tl.slice_for_frame(0)
        fg_pts, fg_cols = flow_gt_tl.slice_for_frame(0)
        flow_pred_handle = server.scene.add_line_segments(
            "scene/flow/pred", points=fp_pts, colors=fp_cols, line_width=float(state["scene_flow_thickness"])
        )
        flow_gt_handle = server.scene.add_line_segments(
            "scene/flow/gt", points=fg_pts, colors=fg_cols, line_width=float(state["scene_flow_thickness"])
        )
        flow_gt_handle.visible = state["use_gt"] and (state["scene_flow_density"] > 0.0)
        flow_pred_handle.visible = has_prediction and (not state["use_gt"]) and (state["scene_flow_density"] > 0.0)

        # Robot points (optional)
        robot_part_handle = None
        if robot_flows_tl is not None:
            rp0, rc0 = robot_flows_tl.frame(0)
            robot_part_handle = server.scene.add_point_cloud(
                "robot/points",
                points=rp0.astype(np.float32, copy=False),
                colors=rc0.astype(np.uint8, copy=False),
                point_size=float(robot_point_size),
                point_shape="rounded",
                precision="float32",
            )
            robot_part_handle.visible = True

        # Robot flow handle (init to frame 0; we update on slider)
        rf_pts, rf_cols = robot_flow_tl.slice_for_frame(0)
        robot_flow_handle = server.scene.add_line_segments(
            "robot/flow", points=rf_pts.astype(np.float32), colors=rf_cols.astype(np.uint8), line_width=float(state["robot_flow_thickness"])
        )
        robot_flow_handle.visible = (state["robot_flow_density"] > 0.0)

        def _apply_point_size(ps: float) -> None:
            ps = float(np.clip(ps, 1e-4, 1.0))
            state["scene_point_size"] = ps
            for h in (dyn_pc, up_pc):
                try:
                    if h is not None and hasattr(h, "point_size"):
                        setattr(h, "point_size", float(ps))
                except Exception:
                    pass

        def _apply_scene_flow_thickness(w: float) -> None:
            w = float(np.clip(float(w), 1.0, 50.0))
            state["scene_flow_thickness"] = w
            for h in (flow_gt_handle, flow_pred_handle):
                if h is None:
                    continue
                if not hasattr(h, "line_width"):
                    raise AttributeError("Viser line segment handle missing 'line_width' attribute")
                h.line_width = float(w)

        def _apply_robot_flow_thickness(w: float) -> None:
            w = float(np.clip(float(w), 1.0, 50.0))
            state["robot_flow_thickness"] = w
            if not hasattr(robot_flow_handle, "line_width"):
                raise AttributeError("Viser line segment handle missing 'line_width' attribute")
            robot_flow_handle.line_width = float(w)

        # Ensure point sizes reflect the configured defaults on startup.
        _apply_point_size(float(scene_point_size))
        _apply_scene_flow_thickness(float(state["scene_flow_thickness"]))
        _apply_robot_flow_thickness(float(state["robot_flow_thickness"]))

        # Per-frame robot overlays
        gripper_handles_per_t: list[list[object]] = []
        full_handles_per_t: list[list[object]] = []
        for ti in range(T):
            ghs: list[object] = []
            if ti < len(gripper_overlays):
                for mi, mesh in enumerate(gripper_overlays[ti]):
                    name = f"robot/gripper_overlay/{ti:03d}/{mi:02d}"
                    # Gripper overlay with configurable opacity and magenta tint blending
                    base_color = np.array([200, 200, 200], dtype=np.float32)
                    visual = getattr(mesh, "visual", None)
                    material = getattr(visual, "material", None) if visual is not None else None
                    if material is not None and hasattr(material, "baseColorFactor"):
                        base = np.asarray(material.baseColorFactor, dtype=np.float32)
                        if base.size >= 3:
                            scale = 255.0 if np.max(base) > 1.0 else 1.0
                            rgb = np.clip(base[:3] / scale, 0.0, 1.0)
                            base_color = (rgb * 255.0).astype(np.float32)
                    # Apply magenta blend on top of base color (per‑mesh solid color)
                    blend = float(self._config.robot_magenta_blend)
                    magenta = np.array([255.0, 0.0, 255.0], dtype=np.float32)
                    if is_behavior:
                        base_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    color_f = np.clip((1.0 - blend) * base_color + blend * magenta, 0.0, 255.0)
                    color = (int(color_f[0]), int(color_f[1]), int(color_f[2]))
                    h = server.scene.add_mesh_simple(
                        name,
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        color=color,
                        opacity=float(self._config.gripper_overlay_opacity),
                        wireframe=False,
                        flat_shading=False,
                        cast_shadow=True,
                        receive_shadow=True,
                    )
                    h.visible = (ti == 0)
                    ghs.append(h)
            gripper_handles_per_t.append(ghs)

            fhs: list[object] = []
            if ti < len(full_overlays):
                for mi, mesh in enumerate(full_overlays[ti]):
                    name = f"robot/full_overlay/{ti:03d}/{mi:02d}"
                    # Use original mesh base color (no magenta blend), opacity 0.5
                    base_color = np.array([200, 200, 200], dtype=np.float32)
                    visual = getattr(mesh, "visual", None)
                    material = getattr(visual, "material", None) if visual is not None else None
                    if material is not None and hasattr(material, "baseColorFactor"):
                        base = np.asarray(material.baseColorFactor, dtype=np.float32)
                        if base.size >= 3:
                            scale = 255.0 if np.max(base) > 1.0 else 1.0
                            rgb = np.clip(base[:3] / scale, 0.0, 1.0)
                            base_color = (rgb * 255.0).astype(np.float32)
                    # If this mesh is a gripper part, overwrite to full black for full-body overlay
                    try:
                        mid = get_mesh_stable_id(mesh).lower()
                        if any(tok in mid for tok in ("finger", "knuckle", "gripper", "robotiq")):
                            base_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    except Exception:
                        pass
                    color = (int(base_color[0]), int(base_color[1]), int(base_color[2]))
                    h = server.scene.add_mesh_simple(
                        name,
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        color=color,
                        opacity=float(state["full_overlay_opacity"]),
                        wireframe=False,
                        flat_shading=False,
                        cast_shadow=True,
                        receive_shadow=True,
                    )
                    h.visible = (ti == 0)
                    fhs.append(h)
            full_handles_per_t.append(fhs)

        # Update helpers
        def _set_overlay_visibility(old_t: int, new_t: int) -> None:
            if 0 <= old_t < T:
                for h in gripper_handles_per_t[old_t]:
                    h.visible = False
                for h in full_handles_per_t[old_t]:
                    h.visible = False
            if 0 <= new_t < T:
                for h in gripper_handles_per_t[new_t]:
                    h.visible = True
                for h in full_handles_per_t[new_t]:
                    h.visible = True

        def _apply_full_overlay_opacity(opacity: float) -> None:
            opacity = float(np.clip(float(opacity), 0.0, 1.0))
            state["full_overlay_opacity"] = float(opacity)
            for handles in full_handles_per_t:
                for h in handles:
                    h.opacity = float(opacity)

        def _apply_frame(t: int) -> None:
            t = int(np.clip(t, 0, max(T - 1, 0)))
            robot_t = int(t)
            scene_t = int(t)
            using_gt = True if not has_prediction else bool(state["use_gt"])
            using_upsampled_view = state["upsample"] and (using_gt or pred_state["upsample"] is not None)
            # Particles / upsampled
            if using_gt:
                pts, cols = (upsample_gt_tl.frame(scene_t) if using_upsampled_view else coarse_gt_tl.frame(scene_t))
            else:
                if using_upsampled_view and pred_state["upsample"] is not None:
                    pts, cols = pred_state["upsample"].frame(scene_t)
                else:
                    pts, cols = pred_state["coarse"].frame(scene_t)
            cols = cols.astype(np.uint8, copy=False)
            if state["upsample"]:
                up_pc.points = pts.astype(np.float32)
                up_pc.colors = cols.astype(np.uint8)
            else:
                dyn_pc.points = pts.astype(np.float32)
                dyn_pc.colors = cols.astype(np.uint8)

            # Flows
            sp, sc = (flow_gt_tl.slice_for_frame(scene_t) if using_gt else pred_state["flow"].slice_for_frame(scene_t))
            flow_gt_handle.visible = state["use_gt"] and (state["scene_flow_density"] > 0.0)
            flow_pred_handle.visible = has_prediction and (not state["use_gt"]) and (state["scene_flow_density"] > 0.0)
            if using_gt:
                flow_gt_handle.points = sp.astype(np.float32)
                flow_gt_handle.colors = sc.astype(np.uint8)
            else:
                flow_pred_handle.points = sp.astype(np.float32)
                flow_pred_handle.colors = sc.astype(np.uint8)

            # Robot flow
            rfp, rfc = robot_flow_tl.slice_for_frame(robot_t)
            robot_flow_handle.points = rfp.astype(np.float32)
            robot_flow_handle.colors = rfc.astype(np.uint8)
            robot_flow_handle.visible = state["robot_flow_density"] > 0.0

            # Robot points
            if robot_flows_tl is not None and robot_part_handle is not None and robot_part_handle.visible:
                rpp, rpc = robot_flows_tl.frame(robot_t)
                robot_part_handle.points = rpp.astype(np.float32)
                robot_part_handle.colors = rpc.astype(np.uint8)

        # Unified controls (same folder): sliders and toggles
        gt_toggle = None
        with control_folder:
            slider = server.gui.add_slider("Frame", min=0, max=max(T - 1, 0), step=1, initial_value=0)
            if has_prediction:
                gt_toggle = server.gui.add_checkbox(label="Ground-truth", initial_value=bool(default_gt_checked))
            up_toggle = server.gui.add_checkbox(label="Upsample", initial_value=True)
            full_opacity_slider = server.gui.add_slider(
                "Full overlay opacity",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=float(state["full_overlay_opacity"]),
            )
            # Flow density sliders
            scene_density_slider = server.gui.add_slider(
                "Scene flow density", min=0.0, max=1.0, step=0.01,
                initial_value=float(self._config.scene_flow_density_default)
            )
            robot_density_slider = server.gui.add_slider(
                "Robot flow density", min=0.0, max=1.0, step=0.01,
                initial_value=float(self._config.robot_flow_density_default)
            )
            scene_thickness_slider = server.gui.add_slider(
                "Scene Flow Thickness",
                min=1.0,
                max=50.0,
                step=0.5,
                initial_value=float(state["scene_flow_thickness"]),
            )
            robot_thickness_slider = server.gui.add_slider(
                "Robot Flow Thickness",
                min=1.0,
                max=50.0,
                step=0.5,
                initial_value=float(state["robot_flow_thickness"]),
            )
            point_size_slider = server.gui.add_slider(
                "Point size",
                min=0.0005,
                max=0.02,
                step=0.0005,
                initial_value=float(scene_point_size),
            )

        # Store t=0 positions for density control
        r0_full: Optional[np.ndarray] = None
        if robot_flows_tl is not None:
            r0, _ = robot_flows_tl.frame(0)
            if r0.size:
                r0_full = r0.astype(np.float32, copy=False)

        # Helper: farthest point sampling order
        # Helper: select FPS indices via Open3D

        # FPS orders for scene flows

        # Rebuild functions
        def _rebuild_scene_flow_timelines() -> None:
            # Scene flows depend on branch (GT/Pred), density, and large-step suppression
            frac = float(state["scene_flow_density"]) if "scene_flow_density" in state else 1.0
            # Helper: compute per-point large-step mask (True if any consecutive step > threshold)
            def _compute_large_step_mask(positions_all: np.ndarray, exists_all: np.ndarray) -> np.ndarray:
                pts = np.asarray(positions_all, dtype=np.float32)
                ex = np.asarray(exists_all, dtype=bool)
                if pts.ndim != 3 or pts.shape[-1] != 3:
                    raise ValueError("positions_all must be (T,N,3)")
                if ex.shape != pts.shape[:2]:
                    raise ValueError("exists_all must be (T,N)")
                Tloc = pts.shape[0]
                Nloc = pts.shape[1]
                if Tloc < 2 or Nloc == 0:
                    return np.zeros((Nloc,), dtype=bool)
                dpos = pts[1:] - pts[:-1]  # (T-1,N,3)
                step = np.linalg.norm(dpos, axis=-1)  # (T-1,N)
                valid = ex[:-1] & ex[1:]
                # Ignore invalid steps by setting them to 0
                step_masked = np.where(valid, step, 0.0)
                thr_step = float(self._config.flow_max_step_m)
                too_fast = (step_masked > thr_step).any(axis=0)
                return too_fast.astype(bool)

            # Always rebuild both GT and Pred flow timelines to ensure correctness on branch toggles
            nonlocal flow_gt_tl, pred_state
            
            def _build_flow_for_branch(positions_all: np.ndarray, keep_mask: np.ndarray) -> FlowTimeline:
                if frac <= 0.0 or not np.any(keep_mask):
                    return FlowTimeline.empty(T)
                idx = np.nonzero(keep_mask)[0]
                p0 = positions_all[0, keep_mask, :]
                k = max(1, int(round(frac * p0.shape[0])))
                sel_local = _fps_select_indices_o3d(p0, k)
                sel = idx[sel_local]
                pos_sel = positions_all[:, sel, :]
                exists_sel = exists_mask[:, sel]
                return build_rainbow_flow_timeline(
                    pos_sel,
                    exists_sel,
                    colormap=lambda u: __import__("matplotlib").cm.get_cmap(self._config.flow_colormap)(u),
                    min_brightness=float(self._config.min_flow_brightness),
                )

            exists0 = exists_mask[0]
            fully_supervised = sample.scene_supervised_mask.astype(bool).all(axis=0)
            if is_behavior:
                gt_too_fast = np.zeros_like(fully_supervised, dtype=bool)
                pred_too_fast = np.zeros_like(fully_supervised, dtype=bool)
            else:
                gt_valid_mask = exists_mask & sample.scene_supervised_mask.astype(bool)
                gt_too_fast = _compute_large_step_mask(sample.scene_ground_truth, gt_valid_mask)
                pred_too_fast = (
                    _compute_large_step_mask(pred_positions_orig, exists_mask)
                    if pred_positions_orig is not None
                    else np.ones_like(fully_supervised, dtype=bool)
                )

            # GT keep mask
            keep_gt = exists0 & fully_supervised & (~gt_too_fast)
            flow_gt_tl = _build_flow_for_branch(sample.scene_ground_truth, keep_gt)

            # Pred keep mask
            if pred_positions_orig is None:
                pred_state["flow"] = FlowTimeline.empty(T)
            else:
                if is_behavior:
                    # Mirror GT logic exactly for behavior
                    keep_pred = exists0 & fully_supervised & (~pred_too_fast)
                else:
                    keep_pred = exists0 & (~pred_too_fast)
                pred_state["flow"] = _build_flow_for_branch(pred_positions_orig, keep_pred)

        def _rebuild_robot_flow_timeline() -> None:
            nonlocal robot_flow_tl
            frac = float(state["robot_flow_density"]) if "robot_flow_density" in state else 1.0
            if frac <= 0.0:
                # Empty timeline
                robot_flow_tl = FlowTimeline.empty(T)
                return
            if robot_flows_tl is None or r0_full is None:
                return
            if r0_full.size == 0:
                return
            k = max(1, int(round(frac * r0_full.shape[0])))
            sel = _fps_select_indices_o3d(r0_full, k)
            # Reconstruct a positions array for selected robots from point timeline
            positions = [robot_flows_tl.frame(t)[0][sel] for t in range(T)]
            positions = np.stack(positions, axis=0)
            exists_sel = np.ones((T, sel.shape[0]), dtype=bool)
            robot_flow_tl = build_constant_flow_timeline(
                positions,
                exists_sel,
                active_mask=None,
                color_rgb=self._config.robot_color_rgb,
                min_brightness=float(self._config.min_robot_transparency),
            )

        # Apply initial density defaults
        try:
            _rebuild_scene_flow_timelines()
            _rebuild_robot_flow_timeline()
            rfp0, rfc0 = robot_flow_tl.slice_for_frame(0)
            robot_flow_handle.points = rfp0.astype(np.float32)
            robot_flow_handle.colors = rfc0.astype(np.uint8)
            robot_flow_handle.visible = (state["robot_flow_density"] > 0.0)
        except Exception:
            pass

        # Bind callbacks
        def _slider_cb(event):
            new_t = int(event.target.value)
            old_t = int(state["frame"])
            state["frame"] = new_t
            _set_overlay_visibility(old_t, new_t)
            _apply_frame(new_t)

        def _gt_cb(event):
            if not has_prediction:
                return
            state["use_gt"] = bool(event.target.value)
            # Rebuild flows to ensure GT branch respects current heuristics
            _rebuild_scene_flow_timelines()
            _apply_frame(int(state["frame"]))

        def _up_cb(event):
            state["upsample"] = bool(event.target.value)
            dyn_pc.visible = not state["upsample"]
            up_pc.visible = state["upsample"]
            _apply_frame(int(state["frame"]))

        def _full_opacity_cb(event):
            _apply_full_overlay_opacity(float(event.target.value))

        def _scene_density_cb(event):
            state["scene_flow_density"] = float(event.target.value)
            _rebuild_scene_flow_timelines()
            _apply_frame(int(state["frame"]))

        def _robot_density_cb(event):
            state["robot_flow_density"] = float(event.target.value)
            _rebuild_robot_flow_timeline()
            _apply_frame(int(state["frame"]))

        def _scene_thickness_cb(event):
            _apply_scene_flow_thickness(float(event.target.value))

        def _robot_thickness_cb(event):
            _apply_robot_flow_thickness(float(event.target.value))

        def _point_size_cb(event):
            _apply_point_size(float(event.target.value))

        slider.on_update(_slider_cb)
        if gt_toggle is not None:
            gt_toggle.on_update(_gt_cb)
        up_toggle.on_update(_up_cb)
        full_opacity_slider.on_update(_full_opacity_cb)
        scene_density_slider.on_update(_scene_density_cb)
        robot_density_slider.on_update(_robot_density_cb)
        scene_thickness_slider.on_update(_scene_thickness_cb)
        robot_thickness_slider.on_update(_robot_thickness_cb)
        point_size_slider.on_update(_point_size_cb)
        return created
