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
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from pointworld.urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()
import trimesh
import urdfpy
from .robot_sampler_lite import URDFRealSampler


@dataclasses.dataclass(slots=True)
class RobotFlow:
    trajectories: np.ndarray  # (T, N, 3)
    overlay_meshes: list[list[trimesh.Trimesh]]  # list of overlays; each is list of per-part meshes
    overlay_opacities: list[float]  # per-overlay opacity in [0,1]

    @property
    def num_points(self) -> int:
        return 0 if self.trajectories.size == 0 else self.trajectories.shape[1]


class RobotFlowBuilder:
    def __init__(
        self,
        urdf_path: Path | str,
        total_samples: int = 4000,
        min_samples_per_mesh: int = 50,
        gripper_only: bool = True,
    ) -> None:
        self._urdf_path = Path(urdf_path)
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self._urdf_path}")
        self._urdf = urdfpy.URDF.load(str(self._urdf_path))
        self._total_samples = int(total_samples)
        self._min_samples = int(min_samples_per_mesh)
        self._gripper_only = bool(gripper_only)
        self._sampler = URDFRealSampler(
            urdf_path=str(self._urdf_path),
            gripper_only=self._gripper_only,
            min_samples_per_mesh=self._min_samples,
        )
        self._mesh_colors: Dict[int, np.ndarray | None] = {}
        for link in self._urdf.links:
            for visual in link.visuals:
                mat = getattr(visual, "material", None)
                color = None
                if mat is not None and getattr(mat, "color", None) is not None:
                    color = np.asarray(mat.color, dtype=np.float32)
                for mesh in visual.geometry.meshes:
                    self._mesh_colors[id(mesh)] = color


    def _neutral_cfg(self) -> Dict[str, float]:
        cfg = {joint.name: 0.0 for joint in self._urdf.actuated_joints}
        cfg.setdefault("finger_joint", 0.0)
        return cfg

    def _cfg_from_state(self, joint: np.ndarray, gripper: float) -> Dict[str, float]:
        cfg = self._neutral_cfg()
        expected = [f"panda_joint{i}" for i in range(1, joint.shape[0] + 1)]
        for name, value in zip(expected, joint, strict=True):
            cfg[name] = float(value)
        cfg["finger_joint"] = float(gripper)
        return cfg

    # Note: sampling and FK are delegated to URDFRealSampler.

    def build(
        self,
        joint_positions: np.ndarray,
        gripper_positions: np.ndarray,
        *,
        num_overlays: int = 1,
        min_overlay_opacity: float = 0.5,
        overlay_indices: Sequence[int] | None = None,
    ) -> RobotFlow:
        if joint_positions.ndim != 2:
            raise ValueError("joint_positions must have shape (T, 7)")
        if gripper_positions.ndim != 1:
            raise ValueError("gripper_positions must have shape (T,)")
        num_frames = joint_positions.shape[0]
        if gripper_positions.shape[0] != num_frames:
            raise ValueError("gripper_positions length mismatch")

        # Use local sampler cloned from point-world logic
        self._sampler.presample(self._total_samples, seed=1)
        robot_flows = self._sampler.compute_world_trajectories(joint_positions, gripper_positions)
        # Build overlay meshes at evenly spaced frames
        indices: list[int]
        if overlay_indices is not None:
            indices = [int(i) for i in overlay_indices]
            if not indices:
                indices = [num_frames - 1]
            for idx in indices:
                if idx < 0 or idx >= num_frames:
                    raise ValueError(f"overlay index {idx} out of range [0, {num_frames - 1}]")
            K = len(indices)
        else:
            K = max(1, int(num_overlays))
            if K == 1:
                ts = [1.0]
            else:
                ts = [i / (K - 1) for i in range(K)]
            indices = [int(round(t * (num_frames - 1))) for t in ts]
        alphas: list[float] = []
        overlays: list[list[trimesh.Trimesh]] = []
        for i, idx in enumerate(indices):
            cfg = self._cfg_from_state(joint_positions[idx], gripper_positions[idx])
            fk = self._urdf.visual_trimesh_fk(cfg=cfg)
            parts: list[trimesh.Trimesh] = []
            for mesh, transform in fk.items():
                transformed = mesh.copy()
                transformed.apply_transform(transform)
                color = self._mesh_colors.get(id(mesh))
                if color is not None:
                    rgba = np.asarray(color, dtype=np.float32)
                    if rgba.shape[0] == 3:
                        rgba = np.concatenate([rgba, [1.0]])
                    rgba = np.clip(rgba, 0.0, 1.0)
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba.tolist(),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    visual = transformed.visual
                    if hasattr(visual, "material"):
                        visual.material = material
                        transformed.visual = visual
                    else:
                        transformed.visual = trimesh.visual.TextureVisuals(
                            material=material
                        )
                parts.append(transformed)
            overlays.append(parts)
            if K == 1:
                alphas.append(1.0)
            else:
                # Linear ramp from min_overlay_opacity -> 1.0
                a = float(min_overlay_opacity) + (i / max(K - 1, 1)) * (1.0 - float(min_overlay_opacity))
                alphas.append(float(np.clip(a, 0.0, 1.0)))
        return RobotFlow(
            trajectories=robot_flows.astype(np.float32),
            overlay_meshes=overlays,
            overlay_opacities=alphas,
        )
