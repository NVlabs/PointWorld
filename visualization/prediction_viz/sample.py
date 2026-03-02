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

import dataclasses
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

from ..viser_tools.robot_state_adapter import RobotKinematics, parse_robot_kinematics
from ..viser_tools.visualization_utils import CameraObservation

from .utils import _ensure_float_array


@dataclasses.dataclass(slots=True)
class PredictionVisualizerSample:
    sample_id: str
    domain: str
    clip_key: str
    scene_prediction: Optional[np.ndarray]
    scene_ground_truth: np.ndarray
    scene_colors: Optional[np.ndarray]
    scene_exists: np.ndarray
    scene_supervised_mask: np.ndarray
    # Robot state (generic)
    robot_flows: Optional[np.ndarray]
    robot_exists: np.ndarray
    # For overlays (optional, embodiment-dependent)
    joint_positions: Optional[np.ndarray] = None          # Panda: (T, 7)
    gripper_positions: Optional[np.ndarray] = None        # Panda: (T,)
    joint_names: Optional[List[str]] = None               # R1Pro: list of names matching joint_positions_full
    joint_positions_full: Optional[np.ndarray] = None     # R1Pro: (T, J)
    base_pose: Optional[np.ndarray] = None                # R1Pro: (T, 7) [x,y,z,qx,qy,qz,qw]
    cameras: list[CameraObservation] = dataclasses.field(default_factory=list)
    world_shift: Optional[np.ndarray] = None
    world_transform: Optional[np.ndarray] = None
    world_reflection: Optional[np.ndarray] = None


@dataclasses.dataclass(slots=True)
class PredictionVisualizerLiveSession:
    server: object
    stopper: Callable[[], None]
    builder: object
    builder_gui_handles: List[object] = dataclasses.field(default_factory=list)
    custom_gui_handles: List[object] = dataclasses.field(default_factory=list)
    camera_pose_markdown: Optional[object] = None
    camera_pose_poller_stop: Optional[threading.Event] = None
    camera_pose_poller_thread: Optional[threading.Thread] = None

    def close(self) -> None:
        if self.camera_pose_poller_stop is not None:
            try:
                self.camera_pose_poller_stop.set()
            except Exception:
                pass
        if self.camera_pose_poller_thread is not None:
            try:
                self.camera_pose_poller_thread.join(timeout=1.0)
            except Exception:
                pass
        self.camera_pose_markdown = None
        self.camera_pose_poller_stop = None
        self.camera_pose_poller_thread = None
        for handle in list(self.custom_gui_handles):
            try:
                handle.remove()
            except Exception:
                pass
        self.custom_gui_handles.clear()
        for handle in list(self.builder_gui_handles):
            try:
                handle.remove()
            except Exception:
                pass
        self.builder_gui_handles.clear()
        try:
            self.stopper()
        except Exception:
            pass


def build_sample_from_dictionary(
    *,
    sample_dict: Dict[str, np.ndarray],
    predictions: Optional[Dict[str, np.ndarray]] = None,
) -> PredictionVisualizerSample:
    cameras: list[CameraObservation] = []
    hires_suffix = "_initial_rgb_hires"
    base_suffix = "_initial_rgb"
    prefixes = set()
    for key in sample_dict.keys():
        if key.endswith(hires_suffix):
            prefixes.add(key[: -len(hires_suffix)])
        elif key.endswith(base_suffix):
            prefixes.add(key[: -len(base_suffix)])

    dom = (sample_dict.get("__domain__") or "").lower()
    prefer_hires_rgb_only = "droid" in dom

    for prefix in sorted(prefixes):
        hires_keys = {
            "rgb": f"{prefix}_initial_rgb_hires",
            "depth": f"{prefix}_initial_depth_hires",
            "intr": f"{prefix}_intrinsic_hires",
            "extr": f"{prefix}_extrinsic_hires",
        }
        base_keys = {
            "rgb": f"{prefix}_initial_rgb",
            "depth": f"{prefix}_initial_depth",
            "intr": f"{prefix}_intrinsic",
            "extr": f"{prefix}_extrinsic",
        }
        if prefer_hires_rgb_only:
            rgb_key = hires_keys["rgb"] if hires_keys["rgb"] in sample_dict else base_keys["rgb"]
            keys = {
                "rgb": rgb_key,
                "depth": base_keys["depth"],
                "intr": base_keys["intr"],
                "extr": base_keys["extr"],
            }
        else:
            use_hires = all(k in sample_dict for k in hires_keys.values())
            keys = hires_keys if use_hires else base_keys
        for required in (keys["rgb"], keys["depth"], keys["intr"], keys["extr"]):
            if required not in sample_dict:
                raise KeyError(f"Sample missing required camera field '{required}'")
        cameras.append(
            CameraObservation(
                name=prefix,
                intrinsic=_ensure_float_array(sample_dict[keys["intr"]]),
                extrinsic_world_to_cam=_ensure_float_array(sample_dict[keys["extr"]]),
                rgb=np.asarray(sample_dict[keys["rgb"]]),
                depth=_ensure_float_array(sample_dict[keys["depth"]]),
            )
        )
    if not cameras:
        raise RuntimeError("Sample does not contain any camera RGB/depth data")

    # Normalize robot kinematics across domains (Panda/droid vs R1Pro/behavior)
    kin: RobotKinematics = parse_robot_kinematics(sample_dict)

    gt_scene = _ensure_float_array(sample_dict["gt_scene_flows"]) if "gt_scene_flows" in sample_dict else _ensure_float_array(sample_dict["scene_flows"])
    pred_scene = None
    if predictions is not None and "scene_flows" in predictions:
        pred_scene = _ensure_float_array(predictions["scene_flows"])

    scene_colors: Optional[np.ndarray]
    if "scene_colors" in sample_dict:
        scene_colors_raw = np.asarray(sample_dict["scene_colors"])  # preserve dtype for u8
        if scene_colors_raw.ndim == 2:
            if scene_colors_raw.shape[1] != 3:
                raise ValueError("scene_colors must have shape (N,3) or (T,N,3)")
            scene_colors = scene_colors_raw.astype(np.uint8, copy=False)
        elif scene_colors_raw.ndim == 3:
            if scene_colors_raw.shape[2] != 3:
                raise ValueError("scene_colors must have shape (N,3) or (T,N,3)")
            scene_colors = scene_colors_raw.astype(np.uint8, copy=False)
        else:
            raise ValueError("scene_colors must have shape (N,3) or (T,N,3)")
    else:
        # Fall back to extracting RGB from scene_features channels 3:6
        if "scene_features" not in sample_dict:
            raise KeyError("Sample must include either 'scene_colors' or 'scene_features' (with RGB in channels 3:6)")
        feats = np.asarray(sample_dict["scene_features"])  # keep original dtype to infer range
        if feats.ndim == 2:
            if feats.shape[1] < 6:
                raise ValueError("scene_features last dim < 6; cannot extract RGB at 3:6")
            rgb = feats[:, 3:6]
        elif feats.ndim == 3:
            if feats.shape[2] < 6:
                raise ValueError("scene_features last dim < 6; cannot extract RGB at 3:6")
            rgb = feats[:, :, 3:6]
        else:
            raise ValueError("scene_features must have shape (N,F) or (T,N,F)")
        # Convert to uint8 with explicit range handling
        rgb = np.asarray(rgb)
        maxv = float(np.nanmax(rgb)) if rgb.size else 1.0
        if not np.isfinite(maxv) or maxv <= 0:
            raise ValueError("scene_features RGB channels invalid (non-finite or non-positive)")
        if maxv <= 1.0 + 1e-5:
            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb_u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        scene_colors = rgb_u8

    robot_flows = None
    if "robot_flows" in sample_dict:
        robot_flows = _ensure_float_array(sample_dict["robot_flows"])

    # Required masks from dataset pipeline for faithful filtering
    if "scene_exists" not in sample_dict:
        raise KeyError("Sample must include 'scene_exists' mask from custom_collate_fn")
    if "robot_exists" not in sample_dict:
        raise KeyError("Sample must include 'robot_exists' mask from custom_collate_fn")
    if "scene_supervised_mask" not in sample_dict:
        raise KeyError("Sample must include 'scene_supervised_mask' to drop unsupervised points")

    def _normalize_mask_time(mask_arr: np.ndarray, expected_T: int, expected_N: int, name: str) -> np.ndarray:
        m = np.asarray(mask_arr)
        if m.ndim == 1:
            if m.shape[0] != expected_N:
                raise ValueError(f"{name} expected length {expected_N}, got {m.shape[0]}")
            mm = m.astype(bool)[None, :]
            return np.broadcast_to(mm, (expected_T, expected_N)).copy()
        if m.ndim == 2:
            if m.shape != (expected_T, expected_N):
                raise ValueError(f"{name} expected shape (T,N)=({expected_T},{expected_N}), got {m.shape}")
            return m.astype(bool)
        raise ValueError(f"{name} must be (N,) or (T,N)")

    T = gt_scene.shape[0]
    N = gt_scene.shape[1]
    scene_exists = _normalize_mask_time(sample_dict["scene_exists"], T, N, "scene_exists")
    robot_N = robot_flows.shape[1] if robot_flows is not None else 0
    robot_exists = _normalize_mask_time(sample_dict["robot_exists"], T, robot_N, "robot_exists") if robot_N > 0 else np.zeros((T, 0), dtype=bool)
    scene_supervised_mask = _normalize_mask_time(sample_dict["scene_supervised_mask"], T, N, "scene_supervised_mask")

    shift_amount = None
    if "__shift_amount__" in sample_dict:
        shift_raw = _ensure_float_array(sample_dict["__shift_amount__"]).reshape(-1)
        if shift_raw.size >= 3:
            shift_amount = shift_raw[:3].astype(np.float32)

    reflection_matrix = None
    if "__world_reflection__" in sample_dict:
        refl_raw = _ensure_float_array(sample_dict["__world_reflection__"])
        if refl_raw.shape == (3, 3):
            reflection_matrix = refl_raw.astype(np.float32)

    world_transform = None
    if "__world_transform__" in sample_dict:
        wt_raw = _ensure_float_array(sample_dict["__world_transform__"])
        if wt_raw.shape == (4, 4):
            world_transform = wt_raw.astype(np.float32)

    sample_identifier = sample_dict.get("__key__") or "unknown"
    sample_domain = sample_dict.get("__domain__") or "unknown"
    sample_clip_key = sample_identifier

    # Extract generic URDF state if present
    joint_names: Optional[List[str]] = None
    joint_positions_full: Optional[np.ndarray] = None
    base_pose: Optional[np.ndarray] = None
    if kin.joint_names is not None and kin.joint_positions_full is not None:
        joint_names = list(kin.joint_names)
        joint_positions_full = kin.joint_positions_full.astype(np.float32, copy=False)
    if kin.base_pose is not None:
        base_pose = kin.base_pose.astype(np.float32, copy=False)

    return PredictionVisualizerSample(
        sample_id=str(sample_identifier),
        domain=str(sample_domain),
        clip_key=str(sample_clip_key),
        scene_prediction=pred_scene,
        scene_ground_truth=gt_scene,
        scene_colors=scene_colors,
        scene_exists=scene_exists,
        scene_supervised_mask=scene_supervised_mask,
        robot_flows=robot_flows,
        robot_exists=robot_exists,
        joint_positions=(kin.panda_joint_positions if kin.panda_joint_positions is not None else None),
        gripper_positions=(kin.panda_gripper_positions if kin.panda_gripper_positions is not None else None),
        joint_names=joint_names,
        joint_positions_full=joint_positions_full,
        base_pose=base_pose,
        cameras=cameras,
        world_shift=(None if shift_amount is None else -shift_amount),
        world_transform=world_transform,
        world_reflection=reflection_matrix,
    )


__all__ = [
    "PredictionVisualizerSample",
    "PredictionVisualizerLiveSession",
    "build_sample_from_dictionary",
]
