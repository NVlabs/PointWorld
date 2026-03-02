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

import atexit
import textwrap
from typing import Callable, Iterable, List, Tuple

import numpy as np
import viser
import cv2
from matplotlib import cm

from .camera import CameraSpec
from ..viser_tools import add_camera_frustums


class SceneBuilder:
    def __init__(
        self,
        point_size: float = 0.003,
        scene_line_width: float = 1.6,
        robot_line_width: float = 2.2,
        depth_min: float = 0.0,
        depth_max: float = 2.0,
        background_dimness: float = 0.5,
        background_point_cloud_rgb: tuple[int, int, int] | None = None,
        frustum_scale: float = 0.12,
        frustum_line_width: float = 1.2,
        camera_media_max_height: int = 320,
    ) -> None:
        self._point_size = float(point_size)
        self._scene_line_width = float(scene_line_width)
        self._robot_line_width = float(robot_line_width)
        self._depth_min = float(depth_min)
        self._depth_max = float(depth_max)
        self._background_dimness = float(np.clip(background_dimness, 0.0, 1.0))
        self._background_point_cloud_rgb = background_point_cloud_rgb
        self._frustum_scale = float(frustum_scale)
        self._frustum_line_width = float(frustum_line_width)
        self._gui_media_max_height = int(camera_media_max_height)
        # Shared viewer up direction for all visualization sessions.
        self._initial_cam_up = (0.0, 0.0, 1.0)

    def _add_scene_background(
        self,
        server: viser.ViserServer,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float,
    ) -> object | None:
        if points.size == 0:
            return None
        if self._background_point_cloud_rgb is not None:
            r, g, b = self._background_point_cloud_rgb
            if not (0 <= int(r) <= 255 and 0 <= int(g) <= 255 and 0 <= int(b) <= 255):
                raise ValueError(f"background_point_cloud_rgb must be uint8 RGB, got {self._background_point_cloud_rgb}")
            if colors.shape[-1] != 3:
                raise ValueError(f"background colors must be RGB (N,3), got shape {colors.shape}")
            # Override background point colors entirely.
            colors = np.empty_like(colors, dtype=np.uint8)
            colors[:, 0] = int(r)
            colors[:, 1] = int(g)
            colors[:, 2] = int(b)
        # Dim background colors to improve contrast with overlays
        if colors.shape[-1] == 3 and self._background_dimness > 0.0:
            dim = np.clip(1.0 - self._background_dimness, 0.0, 1.0)
            colors = (colors.astype(np.float32) * dim).astype(np.uint8)
        return server.scene.add_point_cloud(
            "scene/background",
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            point_size=point_size,
            point_shape="rounded",
            precision="float32",
        )

    @staticmethod
    def _add_scene_flow(
        server: viser.ViserServer,
        points: np.ndarray,
        colors: np.ndarray,
        line_width: float,
    ) -> None:
        if points.size == 0:
            return
        server.scene.add_line_segments(
            "scene/flow",
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            line_width=line_width,
        )

    @staticmethod
    def _add_world_axes(
        server: viser.ViserServer,
        transform: np.ndarray | None,
        *,
        axis_length: float = 0.15,
    ) -> object | None:
        base_segments = np.array(
            [
                [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, axis_length, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]],
            ],
            dtype=np.float32,
        )
        if transform is None:
            T = np.eye(4, dtype=np.float32)
        else:
            T = np.asarray(transform, dtype=np.float32)
            if T.shape != (4, 4):
                T = np.eye(4, dtype=np.float32)
        pts = base_segments.reshape(-1, 3)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)
        transformed = (T @ pts_h.T).T[:, :3]
        segments = transformed.reshape(-1, 2, 3).astype(np.float32, copy=False)
        colors = np.array(
            [
                [[255, 0, 0], [255, 0, 0]],
                [[0, 255, 0], [0, 255, 0]],
                [[0, 0, 255], [0, 0, 255]],
            ],
            dtype=np.uint8,
        )
        try:
            handle = server.scene.add_line_segments(
                "world/axes_transformed",
                points=segments,
                colors=colors,
                line_width=0.01,
            )
        except Exception:
            return None
        return handle

    @staticmethod
    def _add_robot_flow(server: viser.ViserServer, points: np.ndarray, colors: np.ndarray, line_width: float) -> None:
        if points.size == 0:
            return
        server.scene.add_line_segments(
            "robot/flow",
            points=points.astype(np.float32),
            colors=colors.astype(np.uint8),
            line_width=line_width,
        )

    def _add_cameras(
        self,
        server: viser.ViserServer,
        cameras: Iterable[CameraSpec],
    ) -> List[object]:
        camera_list = list(cameras)
        if not camera_list:
            return []
        try:
            handles_map = add_camera_frustums(
                server,
                specs=camera_list,
                name_prefix="cameras",
                scale=self._frustum_scale,
                line_width=self._frustum_line_width,
                visible=True,
            )
        except Exception as exc:  # pragma: no cover - viewer errors surface directly
            raise RuntimeError(f"failed to add camera frustums: {exc}") from exc
        return list(handles_map.values())

    @staticmethod
    def _configure_lighting(server: viser.ViserServer) -> None:
        server.scene.enable_default_lights()
        server.scene.world_axes.visible = False

    def _populate_scene(
        self,
        server: viser.ViserServer,
        scene_points: np.ndarray,
        scene_colors: np.ndarray,
        scene_flow_points: np.ndarray,
        scene_flow_colors: np.ndarray,
        robot_flow_points: np.ndarray,
        robot_flow_colors: np.ndarray,
        cameras: Iterable[CameraSpec],
        metadata: dict,
        *,
        with_gui: bool = True,
        viewer_background_rgb: Tuple[int, int, int] | None = None,
        background_visible: bool = True,
    ) -> List[object]:
        camera_list = list(cameras)
        if viewer_background_rgb is not None:
            r, g, b = viewer_background_rgb
            if not (0 <= int(r) <= 255 and 0 <= int(g) <= 255 and 0 <= int(b) <= 255):
                raise ValueError(f"viewer_background_rgb must be uint8 RGB, got {viewer_background_rgb}")
            bg = np.zeros((2, 2, 3), dtype=np.uint8)
            bg[:, :, 0] = int(r)
            bg[:, :, 1] = int(g)
            bg[:, :, 2] = int(b)
            try:
                # Viser expects (H,W,3) RGB. PNG is lossless (important for flat backgrounds).
                server.scene.set_background_image(bg, format="png")
            except Exception as exc:
                raise RuntimeError(f"Failed to set viewer background color: {viewer_background_rgb}") from exc
        self._configure_lighting(server)
        try:
            server.scene.set_up_direction((self._initial_cam_up[0], self._initial_cam_up[1], self._initial_cam_up[2]))
        except Exception:
            pass
        server.scene.world_axes.visible = False
        gui_handles: List[object] = []
        world_transform = None
        if metadata is not None:
            wt_raw = metadata.get("world_transform")
            if wt_raw is not None:
                wt = np.asarray(wt_raw, dtype=np.float32)
                if wt.shape == (4, 4):
                    world_transform = wt
        axes_handle = self._add_world_axes(server, world_transform)
        if axes_handle is not None:
            gui_handles.append(axes_handle)
        else:
            pass
        background_handle = self._add_scene_background(server, scene_points, scene_colors, self._point_size)
        if background_handle is not None:
            try:
                background_handle.visible = bool(background_visible)
            except Exception:
                pass
        self._add_scene_flow(server, scene_flow_points, scene_flow_colors, self._scene_line_width)
        self._add_robot_flow(server, robot_flow_points, robot_flow_colors, self._robot_line_width)
        gui_handles.extend(self._add_cameras(server, camera_list))

        if with_gui and metadata:
            info_lines = [
                f"- **Clip**: `{metadata.get('clip_key', 'unknown')}`",
                f"- **Scene points**: {metadata.get('scene_points', '?')}",
                f"- **Robot samples**: {metadata.get('robot_samples', '?')}",
                f"- **Threshold**: {metadata.get('movement_threshold_m', '?')} m",
            ]
            summary_handle = server.gui.add_markdown(
                textwrap.dedent(
                    """
                    ### Visualization Summary
                    {lines}
                    """
                ).format(lines="\n".join(info_lines))
            )
            gui_handles.append(summary_handle)

        if with_gui:
            gui_handles.extend(self._add_camera_media(server, camera_list))
        return gui_handles

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        h, w = img.shape[:2]
        max_height = int(self._gui_media_max_height)
        if h <= max_height or max_height <= 0:
            return img
        scale = max_height / float(h)
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_AREA)
        return resized

    def _add_camera_media(self, server: viser.ViserServer, cameras: Iterable[CameraSpec]) -> List[object]:
        handles: List[object] = []
        if not any(cam.image is not None or cam.depth is not None for cam in cameras):
            return handles
        header = server.gui.add_markdown("### Camera Media")
        handles.append(header)
        for camera in cameras:
            if camera.image is not None:
                rgb_small = self._resize_image(np.ascontiguousarray(camera.image))
                handles.append(server.gui.add_image(rgb_small, label=f"{camera.name} RGB"))
            if camera.depth is not None:
                depth_color = self._depth_to_rgba(camera.depth)
                depth_small = self._resize_image(depth_color)
                handles.append(server.gui.add_image(depth_small, label=f"{camera.name} Depth"))
        return handles

    def _depth_to_rgba(self, depth: np.ndarray) -> np.ndarray:
        if depth.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        depth = np.asarray(depth, dtype=np.float32)
        # Compute per-image trimmed range (5th-95th percentiles), then clip to configured range
        finite = np.isfinite(depth)
        if np.any(finite):
            vals = depth[finite]
            p_low, p_high = np.percentile(vals, [5.0, 95.0])
            eff_min = float(np.clip(float(p_low), self._depth_min, self._depth_max))
            eff_max = float(np.clip(float(p_high), self._depth_min, self._depth_max))
        else:
            eff_min = float(self._depth_min)
            eff_max = float(self._depth_max)
        depth = np.nan_to_num(depth, nan=eff_max, posinf=eff_max, neginf=eff_min)
        depth = np.clip(depth, eff_min, eff_max)
        denom = max(eff_max - eff_min, 1e-6)
        depth_norm = (depth - eff_min) / denom
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_color = (cm.get_cmap("jet")(depth_norm)[..., :3] * 255.0).astype(np.uint8)
        return depth_color

    def launch_interactive(
        self,
        scene_points: np.ndarray,
        scene_colors: np.ndarray,
        scene_flow_points: np.ndarray,
        scene_flow_colors: np.ndarray,
        robot_flow_points: np.ndarray,
        robot_flow_colors: np.ndarray,
        cameras: Iterable[CameraSpec],
        metadata: dict,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        verbose: bool = True,
        viewer_background_rgb: Tuple[int, int, int] | None = None,
        with_gui: bool = True,
        background_visible: bool = True,
    ) -> Tuple[viser.ViserServer, Callable[[], None], List[object]]:
        server = viser.ViserServer(host=host, port=port, verbose=verbose)
        stopper = server.stop
        atexit.register(stopper)
        gui_handles = self._populate_scene(
            server,
            scene_points,
            scene_colors,
            scene_flow_points,
            scene_flow_colors,
            robot_flow_points,
            robot_flow_colors,
            cameras,
            metadata,
            viewer_background_rgb=viewer_background_rgb,
            with_gui=bool(with_gui),
            background_visible=bool(background_visible),
        )
        # Print viewer URL
        try:
            display_host = server.get_host()
            display_port = server.get_port()
        except Exception:
            display_host = host
            display_port = port
        base_url = f"http://{display_host}:{display_port}"
        print(f"[viewer] Open: {base_url}")
        return server, stopper, gui_handles

    def populate_existing(
        self,
        server: viser.ViserServer,
        scene_points: np.ndarray,
        scene_colors: np.ndarray,
        scene_flow_points: np.ndarray,
        scene_flow_colors: np.ndarray,
        robot_flow_points: np.ndarray,
        robot_flow_colors: np.ndarray,
        cameras: Iterable[CameraSpec],
        metadata: dict,
        *,
        viewer_background_rgb: Tuple[int, int, int] | None = None,
        with_gui: bool = True,
        background_visible: bool = True,
    ) -> List[object]:
        server.scene.reset()
        return self._populate_scene(
            server,
            scene_points,
            scene_colors,
            scene_flow_points,
            scene_flow_colors,
            robot_flow_points,
            robot_flow_colors,
            cameras,
            metadata,
            viewer_background_rgb=viewer_background_rgb,
            with_gui=bool(with_gui),
            background_visible=bool(background_visible),
        )
