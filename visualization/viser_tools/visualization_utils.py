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
from typing import Iterable, Sequence

import cv2
import numpy as np
import trimesh

from ..viser_flow.camera import CameraSpec, camera_spec
from ..viser_flow.robot_flow import RobotFlow
from ..viser_flow.robot_sampler_lite import get_mesh_stable_id



@dataclasses.dataclass(slots=True)
class CameraObservation:
    """Minimal camera observation used for visualization utilities."""

    name: str
    intrinsic: np.ndarray
    extrinsic_world_to_cam: np.ndarray
    rgb: np.ndarray | None
    depth: np.ndarray | None
    mask: np.ndarray | None = None


def _filter_points_in_bounds(
    points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    return (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )


def project_depth_to_world(
    camera: CameraObservation,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    filter_bounds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    depth = camera.depth
    if depth is None or depth.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    rgb = camera.rgb
    if rgb is None or rgb.size == 0:
        raise ValueError(f"camera '{camera.name}' is missing RGB image for projection")

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"camera '{camera.name}' depth must have shape (H, W)")

    H, W = depth.shape
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[0] != H or rgb.shape[1] != W:
        interp = cv2.INTER_AREA if (rgb.shape[0] >= H and rgb.shape[1] >= W) else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (W, H), interpolation=interp)
    mask = np.isfinite(depth) & (depth > 1e-6)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    ys, xs = np.nonzero(mask)
    depth_vals = depth[ys, xs]

    intrinsic = np.asarray(camera.intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera '{camera.name}' intrinsic must have shape (3, 3)")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if min(fx, fy) <= 0:
        raise ValueError(f"camera '{camera.name}' focal lengths must be positive")

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)

    x_cam = (xs_f - cx) / fx * depth_vals
    y_cam = (ys_f - cy) / fy * depth_vals
    z_cam = depth_vals

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32, copy=False)
    ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
    cam_points_h = np.concatenate([cam_points, ones], axis=1)

    extrinsic = np.asarray(camera.extrinsic_world_to_cam, dtype=np.float32)
    if extrinsic.shape != (4, 4):
        raise ValueError(f"camera '{camera.name}' extrinsic must have shape (4, 4)")

    cam_to_world = np.linalg.inv(extrinsic).astype(np.float32, copy=False)
    world_points_h = cam_points_h @ cam_to_world.T
    world_points = world_points_h[:, :3]

    finite_mask = np.isfinite(world_points).all(axis=1)
    world_points = world_points[finite_mask]
    colors = rgb[ys, xs][finite_mask].astype(np.uint8, copy=False)
    if filter_bounds:
        inside_mask = _filter_points_in_bounds(world_points, bounds_min, bounds_max)
        if not np.any(inside_mask):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        world_points = world_points[inside_mask]
        colors = colors[inside_mask]
    return world_points.astype(np.float32, copy=False), colors


def merge_camera_point_cloud(
    cameras: Iterable[CameraObservation],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    include_out_of_bounds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    inside_points: list[np.ndarray] = []
    inside_colors: list[np.ndarray] = []
    outside_points: list[np.ndarray] = []
    outside_colors: list[np.ndarray] = []

    for camera in cameras:
        points, colors = project_depth_to_world(
            camera,
            bounds_min,
            bounds_max,
            filter_bounds=True,
        )
        if points.size == 0:
            continue
        inside_points.append(points)
        inside_colors.append(colors)

    if include_out_of_bounds:
        for camera in cameras:
            points_all, colors_all = project_depth_to_world(
                camera,
                bounds_min,
                bounds_max,
                filter_bounds=False,
            )
            if points_all.size == 0:
                continue
            outside_mask = np.any(
                (points_all < bounds_min) | (points_all > bounds_max), axis=1
            )
            if np.any(outside_mask):
                outside_points.append(points_all[outside_mask])
                outside_colors.append(colors_all[outside_mask])

    if not inside_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    merged_points = np.concatenate(inside_points, axis=0).astype(np.float32, copy=False)
    merged_colors = np.concatenate(inside_colors, axis=0).astype(np.uint8, copy=False)

    if include_out_of_bounds and outside_points:
        merged_points = np.concatenate([merged_points] + outside_points, axis=0).astype(
            np.float32, copy=False
        )
        merged_colors = np.concatenate([merged_colors] + outside_colors, axis=0).astype(
            np.uint8, copy=False
        )

    return merged_points, merged_colors


def filter_to_gripper_meshes(flow: RobotFlow) -> None:
    overlays = getattr(flow, "overlay_meshes", None)
    if not overlays:
        return
    tokens = ("finger", "knuckle", "gripper", "robotiq")
    filtered: list[list[trimesh.Trimesh]] = []
    alphas: list[float] = []
    opacities = list(getattr(flow, "overlay_opacities", []))
    for idx, parts in enumerate(overlays):
        alpha = opacities[idx] if idx < len(opacities) else 1.0
        keep: list[trimesh.Trimesh] = []
        for mesh in parts:
            mid = get_mesh_stable_id(mesh)
            if any(token in mid.lower() for token in tokens):
                keep.append(mesh)
        if keep:
            filtered.append(keep)
            alphas.append(alpha)
    if filtered:
        flow.overlay_meshes = filtered
        if alphas:
            flow.overlay_opacities = alphas


def robot_clearance_mask(
    points: np.ndarray,
    meshes: Sequence[trimesh.Trimesh],
    clearance: float,
) -> np.ndarray:
    """Return boolean mask for points to keep (True means keep, i.e., not near robot).

    Improvements:
    - Uses densely sampled surface points from meshes (not raw vertices) for robust distance checks.
    - Computes distances with torch.cdist, leveraging GPU if available.
    - Removes slow/scalar fallbacks; fails fast on missing torch.
    """
    if points.size == 0 or clearance <= 0.0 or not meshes:
        return np.ones(points.shape[0], dtype=bool)

    # Surface point sampling parameters (balanced for speed/quality)
    points_per_m2 = 4000.0
    min_points_per_mesh = 512
    max_total_points = 80000

    # Collect per-mesh areas to allocate samples proportionally
    areas = []
    for m in meshes:
        try:
            areas.append(float(m.area))
        except Exception:
            areas.append(0.0)

    sampled_list = []
    for mesh, area in zip(meshes, areas):
        if mesh.is_empty or area <= 0.0:
            continue
        n = int(max(min_points_per_mesh, round(points_per_m2 * area)))
        try:
            pts_s, _ = trimesh.sample.sample_surface(mesh, n)
        except Exception:
            continue
        if pts_s.size:
            sampled_list.append(pts_s.astype(np.float32, copy=False))

    if not sampled_list:
        return np.ones(points.shape[0], dtype=bool)

    sampled = np.concatenate(sampled_list, axis=0).astype(np.float32, copy=False)
    if sampled.shape[0] > max_total_points:
        import numpy as _np
        sel = _np.random.choice(sampled.shape[0], size=max_total_points, replace=False)
        sampled = sampled[sel]

    pts = points.astype(np.float32, copy=False)

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "robot_clearance_mask requires torch for GPU/CPU distance computation"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts_t = torch.from_numpy(pts).to(device)
    samp_t = torch.from_numpy(sampled).to(device)
    keep = torch.ones(pts_t.shape[0], dtype=torch.bool, device=device)
    # Double chunking with matmul-based distances to avoid cdist CUDA kernel issues
    batch_pts = 2048
    batch_samp = 2048
    clr = float(clearance)
    for start in range(0, pts_t.shape[0], batch_pts):
        chunk = pts_t[start : start + batch_pts]
        a2 = (chunk * chunk).sum(dim=1).unsqueeze(1)
        min_d2 = None
        for s0 in range(0, samp_t.shape[0], batch_samp):
            sub = samp_t[s0 : s0 + batch_samp]
            b2 = (sub * sub).sum(dim=1).unsqueeze(0)
            prod = chunk @ sub.T
            d2 = a2 + b2 - 2.0 * prod
            d2 = torch.clamp(d2, min=0.0)
            md2 = d2.min(dim=1).values
            min_d2 = md2 if min_d2 is None else torch.minimum(min_d2, md2)
            del b2, prod, d2, md2
        keep[start : start + batch_pts] = min_d2 > (clr * clr)
        del a2, min_d2
    return keep.cpu().numpy()



def apply_magenta_blend_to_mesh(
    mesh: trimesh.Trimesh,
    blend: float,
    *,
    assume_white_base: bool = False,
) -> None:
    if mesh.is_empty or blend <= 0.0:
        return
    magenta = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and vertex_colors.size:
        colors = np.asarray(vertex_colors)
        scale = 255.0 if np.max(colors) > 1.0 else 1.0
        colors_f = colors.astype(np.float32) / scale
        rgb = colors_f[..., :3]
        if assume_white_base:
            rgb = np.ones_like(rgb)
        alpha = colors_f[..., 3:] if colors_f.shape[-1] > 3 else None
        blended_rgb = np.clip((1.0 - blend) * rgb + blend * magenta, 0.0, 1.0)
        if alpha is not None:
            blended = np.concatenate([blended_rgb, alpha], axis=-1)
        else:
            blended = blended_rgb
        if np.issubdtype(colors.dtype, np.integer):
            visual.vertex_colors = np.round(blended * scale).astype(colors.dtype)
        else:
            visual.vertex_colors = blended.astype(colors.dtype)

    material = getattr(visual, "material", None)
    if material is not None and hasattr(material, "baseColorFactor"):
        base = np.asarray(material.baseColorFactor, dtype=np.float32)
        if base.size >= 3:
            scale = 255.0 if np.max(base) > 1.0 else 1.0
            base = np.clip(base / scale, 0.0, 1.0)
            base_rgb = base[:3] if not assume_white_base else np.ones(3, dtype=np.float32)
            alpha = base[3] if base.size > 3 else 1.0
            blended_rgb = np.clip((1.0 - blend) * base_rgb + blend * magenta, 0.0, 1.0)
            blended = np.concatenate([blended_rgb, [alpha]])
            material.baseColorFactor = (blended * scale).tolist()


def apply_magenta_blend(
    flow: RobotFlow,
    blend: float,
    *,
    assume_white_base: bool = False,
) -> None:
    if blend <= 0.0:
        return
    overlays = getattr(flow, "overlay_meshes", None)
    if not overlays:
        return
    for parts in overlays:
        for mesh in parts:
            apply_magenta_blend_to_mesh(mesh, blend, assume_white_base=assume_white_base)


def build_camera_specs(
    cameras: Iterable[CameraObservation],
    *,
    reflection_matrix: np.ndarray | None = None,
) -> list[CameraSpec]:
    specs: list[CameraSpec] = []
    for cam in cameras:
        specs.append(
            camera_spec(
                cam.name,
                cam.extrinsic_world_to_cam,
                cam.intrinsic,
                cam.rgb,
                cam.depth,
                reflection_matrix=reflection_matrix,
            )
        )
    return specs
