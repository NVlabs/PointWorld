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
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass(slots=True)
class PredictionVisualizerConfig:
    include_out_of_bounds_points: bool = True
    movement_threshold: float = 0.02
    min_flow_brightness: float = 0.7
    flow_colormap: str = "turbo"
    # Domain-specific point sizes
    behavior_scene_point_size: float = 0.008
    behavior_robot_point_size: float = 0.004
    droid_scene_point_size: float = 0.004
    droid_robot_point_size: float = 0.004
    scene_line_width: float = 2.0
    robot_line_width: float = 2.0
    background_dimness: float = 0.0
    # Optional override for the static scene background point cloud colors (RGB 0..255).
    # When set, replaces the colors of the `scene/background` point cloud, which can be
    # useful for paper-style renders where the reconstructed background should be flat.
    background_point_cloud_rgb: Optional[Tuple[int, int, int]] = None
    frustum_scale: float = 0.07
    frustum_line_width: float = 8.0
    gui_media_max_height: int = 320
    robot_samples: int = 50
    robot_min_samples: int = 60
    min_robot_transparency: float = 0.35
    robot_color_rgb: Tuple[int, int, int] = (255, 0, 255)
    # Domain-specific robot clearance (m) for removing robot points from background
    droid_robot_point_clearance: float = 0.006
    behavior_robot_point_clearance: float = 0.001
    robot_magenta_blend: float = 0.3
    viewer_host: str = "0.0.0.0"
    viewer_port: int = 8080
    # Optional solid background color for the Viser viewer (RGB 0..255).
    # When set, the viewer background is overridden via `server.scene.set_background_image`.
    viewer_background_rgb: Optional[Tuple[int, int, int]] = None
    # If True, do not construct any Viser GUI panels for this session (useful for clean renders).
    disable_gui: bool = False
    # If False, do not add camera frustums to the 3D scene (useful for cinematic renders).
    show_camera_frustums: bool = True
    # Required for voxel upsampling mapping
    grid_size: float = 0.03
    # Opacity for per-frame gripper URDF overlays (0..1)
    gripper_overlay_opacity: float = 0.35
    # Upsampling is voxel-based in release builds.
    # Default flow density values (0..1)
    scene_flow_density_default: float = 0.3
    robot_flow_density_default: float = 0.10
    # Alpha for green tint on unsupervised GT points (0..1) the lower the more subtle the green tint
    unsup_green_alpha: float = 0.3
    # Maximum allowed per-frame step for flow visualization (meters)
    flow_max_step_m: float = 0.1
    # Depth visualization ranges (domain-specific)
    depth_min_droid: float = 0.2
    depth_max_droid: float = 1.3
    depth_min_behavior: float = 0.2
    depth_max_behavior: float = 5.0  # default to 2x droid max
    # Camera upsample ratio for behavior (>=1.0). If >1, upsample RGB/Depth and scale intrinsics.
    behavior_camera_upsample_ratio: float = 1.0
    # Domain-specific workspace bounds and mask tuning
    droid_bounds_min: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([-0.50, -0.40, -0.60], dtype=np.float32)
    )
    droid_bounds_max: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.50, 0.40, 0.60], dtype=np.float32)
    )
    behavior_bounds_min: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([-1.0, -0.8, -0.60], dtype=np.float32)
    )
    behavior_bounds_max: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.6, 0.8, 2.0], dtype=np.float32)
    )

    @classmethod
    def from_args(cls, args: object) -> "PredictionVisualizerConfig":
        cfg = cls()
        if hasattr(args, "viewer_port"):
            cfg.viewer_port = int(getattr(args, "viewer_port"))
        if hasattr(args, "viewer_host"):
            cfg.viewer_host = str(getattr(args, "viewer_host"))
        return cfg


__all__ = ["PredictionVisualizerConfig"]
