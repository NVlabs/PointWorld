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

from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
from pointworld.urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()
import trimesh
import urdfpy

import transform_utils
from .visualization_utils import apply_magenta_blend, filter_to_gripper_meshes
from ..viser_flow.robot_flow import RobotFlow, RobotFlowBuilder


def _build_panda_flows(
    urdf_path: Path | str,
    *,
    joint_positions: np.ndarray,
    gripper_positions: np.ndarray,
    overlay_indices: Sequence[int],
    total_samples: int,
    min_samples_per_mesh: int,
    gripper_overlay_opacity: float,
    full_overlay_opacity: float,
) -> Tuple[RobotFlow, RobotFlow]:
    joints = np.asarray(joint_positions, dtype=np.float32)
    gripper = np.asarray(gripper_positions, dtype=np.float32)
    indices = [int(i) for i in overlay_indices] if overlay_indices else [max(joints.shape[0] - 1, 0)]

    gripper_builder = RobotFlowBuilder(
        urdf_path,
        total_samples=int(total_samples),
        min_samples_per_mesh=int(min_samples_per_mesh),
        gripper_only=True,
    )
    gripper_flow = gripper_builder.build(
        joints,
        gripper,
        num_overlays=len(indices),
        min_overlay_opacity=float(gripper_overlay_opacity),
        overlay_indices=indices,
    )
    if getattr(gripper_flow, "overlay_meshes", None):
        gripper_flow.overlay_opacities = [float(gripper_overlay_opacity) for _ in gripper_flow.overlay_meshes]

    full_builder = RobotFlowBuilder(
        urdf_path,
        total_samples=int(total_samples),
        min_samples_per_mesh=int(min_samples_per_mesh),
        gripper_only=False,
    )
    full_flow = full_builder.build(
        joints,
        gripper,
        num_overlays=len(indices),
        min_overlay_opacity=float(full_overlay_opacity),
        overlay_indices=indices,
    )
    if getattr(full_flow, "overlay_meshes", None):
        full_flow.overlay_opacities = [float(full_overlay_opacity) for _ in full_flow.overlay_meshes]

    return gripper_flow, full_flow


def _build_generic_urdf_overlays(
    urdf: urdfpy.URDF,
    *,
    joint_names: Sequence[str],
    joint_positions_full: np.ndarray,
    base_pose: Optional[np.ndarray],
    overlay_indices: Sequence[int],
    full_overlay_opacity: float,
    gripper_overlay_opacity: float,
) -> Tuple[RobotFlow, RobotFlow]:
    # Build per-frame overlays from URDF FK using provided joint map and optional base pose
    if joint_positions_full.ndim != 2:
        raise ValueError("joint_positions_full must have shape (T, J)")
    T = joint_positions_full.shape[0]
    indices = [int(i) for i in overlay_indices] if overlay_indices else [max(T - 1, 0)]

    # Prepare name->index map
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    actuated = [j.name for j in urdf.actuated_joints]

    def cfg_for_frame(t: int) -> dict[str, float]:
        cfg: dict[str, float] = {}
        for nm in actuated:
            idx = name_to_idx.get(nm, None)
            val = 0.0 if idx is None else float(joint_positions_full[t, idx])
            cfg[nm] = val
        return cfg

    def apply_T(mesh: trimesh.Trimesh, Tm: np.ndarray) -> trimesh.Trimesh:
        m = mesh.copy()
        m.apply_transform(Tm)
        return m

    # Build full overlays
    full_overlays: list[list[trimesh.Trimesh]] = []
    full_alphas: list[float] = []
    for i, idx in enumerate(indices):
        idx = max(0, min(int(idx), T - 1))
        fk = urdf.visual_trimesh_fk(cfg=cfg_for_frame(idx))
        parts: list[trimesh.Trimesh] = []
        # Apply base pose if available
        T_base = None
        if base_pose is not None:
            bp = np.asarray(base_pose[idx], dtype=np.float32).reshape(-1)
            if bp.size != 7:
                raise ValueError("base_pose must be (T,7)")
            T_base = transform_utils.convert_pose_quat2mat(bp)
        for mesh, Tm in fk.items():
            T_applied = (Tm if T_base is None else (T_base @ Tm))
            parts.append(apply_T(mesh, T_applied))
        full_overlays.append(parts)
        if len(indices) == 1:
            full_alphas.append(1.0)
        else:
            a = float(full_overlay_opacity) + (i / max(len(indices) - 1, 1)) * (1.0 - float(full_overlay_opacity))
            full_alphas.append(float(np.clip(a, 0.0, 1.0)))

    # Gripper-only overlays: start from full overlays and filter
    full_flow = RobotFlow(
        trajectories=np.zeros((T, 0, 3), dtype=np.float32),
        overlay_meshes=full_overlays,
        overlay_opacities=full_alphas,
    )
    gripper_flow = RobotFlow(
        trajectories=np.zeros((T, 0, 3), dtype=np.float32),
        overlay_meshes=[[m.copy() for m in parts] for parts in full_overlays],
        overlay_opacities=[float(gripper_overlay_opacity) for _ in full_overlays],
    )
    # Filter to gripper meshes heuristically
    filter_to_gripper_meshes(gripper_flow)
    return gripper_flow, full_flow


def build_robot_flows(
    urdf_path: Path | str,
    *,
    overlay_indices: Sequence[int],
    total_samples: int,
    min_samples_per_mesh: int,
    magenta_blend: float,
    gripper_overlay_opacity: float,
    full_overlay_opacity: float,
    # Optional states
    joint_positions: Optional[np.ndarray] = None,
    gripper_positions: Optional[np.ndarray] = None,
    joint_names: Optional[Sequence[str]] = None,
    joint_positions_full: Optional[np.ndarray] = None,
    base_pose: Optional[np.ndarray] = None,
    robot_flows: Optional[np.ndarray] = None,
) -> Tuple[RobotFlow, RobotFlow, list[list[trimesh.Trimesh]]]:
    """Unified robot overlay builder for Panda (droid) and R1Pro (behavior).

    - If Panda-style joints are provided (joint_positions & gripper_positions), use the Panda path.
    - Else, fall back to generic URDF FK using (joint_names, joint_positions_full, base_pose).

    The returned RobotFlow objects will use provided robot_flows (if any) for trajectories; otherwise
    trajectories remain empty (overlays still render).
    """
    urdf_p = Path(urdf_path)
    if not urdf_p.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_p}")

    # Decide path
    use_panda = (joint_positions is not None) and (gripper_positions is not None)

    if use_panda:
        g_flow, f_flow = _build_panda_flows(
            urdf_p,
            joint_positions=np.asarray(joint_positions, dtype=np.float32),
            gripper_positions=np.asarray(gripper_positions, dtype=np.float32).reshape(-1),
            overlay_indices=overlay_indices,
            total_samples=total_samples,
            min_samples_per_mesh=min_samples_per_mesh,
            gripper_overlay_opacity=gripper_overlay_opacity,
            full_overlay_opacity=full_overlay_opacity,
        )
    else:
        if joint_names is None or joint_positions_full is None:
            # Cannot build overlays; return empty flows
            T_guess = 0
            if robot_flows is not None and np.asarray(robot_flows).ndim == 3:
                T_guess = int(np.asarray(robot_flows).shape[0])
            empty = RobotFlow(
                trajectories=(np.asarray(robot_flows, dtype=np.float32) if robot_flows is not None else np.zeros((T_guess, 0, 3), dtype=np.float32)),
                overlay_meshes=[],
                overlay_opacities=[],
            )
            return empty, empty, []
        urdf = urdfpy.URDF.load(str(urdf_p))
        g_flow, f_flow = _build_generic_urdf_overlays(
            urdf,
            joint_names=joint_names,
            joint_positions_full=np.asarray(joint_positions_full, dtype=np.float32),
            base_pose=(np.asarray(base_pose, dtype=np.float32) if base_pose is not None else None),
            overlay_indices=overlay_indices,
            full_overlay_opacity=float(full_overlay_opacity),
            gripper_overlay_opacity=float(gripper_overlay_opacity),
        )

    # Attach trajectories if provided
    if robot_flows is not None and np.asarray(robot_flows).size:
        traj = np.asarray(robot_flows, dtype=np.float32)
        g_flow.trajectories = traj
        f_flow.trajectories = traj

    if magenta_blend > 0.0:
        apply_magenta_blend(g_flow, float(magenta_blend))

    overlay_meshes: list[list[trimesh.Trimesh]] = []
    if getattr(f_flow, "overlay_meshes", None):
        for parts in f_flow.overlay_meshes:
            overlay_meshes.append([mesh.copy() for mesh in parts])

    return g_flow, f_flow, overlay_meshes
