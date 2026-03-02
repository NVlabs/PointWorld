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

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from ..urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()
import trimesh
import urdfpy


def get_mesh_stable_id(mesh: trimesh.Trimesh, idx: int | None = None) -> str:
    """Deterministic identifier for a mesh.

    Mirrors point-world/robot_sampler.get_mesh_stable_id logic:
    - prefer source filename; then metadata; then geometric signature
    """
    try:
        src = getattr(mesh, "source", None)
        if src is not None:
            fname = getattr(src, "file_name", None)
            if isinstance(fname, (str, bytes)) and fname:
                base = str(fname).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        metadata = mesh.metadata or {}
        for key in ("name", "file_name"):
            val = metadata.get(key)
            if isinstance(val, (str, bytes)) and val:
                base = str(val).lower()
                return f"{base}_{idx}" if idx is not None else base
    except Exception:
        pass
    try:
        bounds = np.asarray(mesh.bounds).reshape(-1)
        bounds = np.round(bounds, 6)
        vcount = len(getattr(mesh, "vertices", []))
        fcount = len(getattr(mesh, "faces", []))
        base = f"b{','.join(map(str, bounds))}_v{vcount}_f{fcount}"
        return f"{base}_{idx}" if idx is not None else base
    except Exception:
        base = f"unknown_{id(mesh)}"
        return f"{base}_{idx}" if idx is not None else base


def _sample_surface_deterministic(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic surface sampling: returns (points, normals).

    Implements face-area weighted triangle sampling with reproducible RNG.
    """
    if count <= 0 or mesh.area <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.area_faces.astype(np.float64)
    prob = areas / max(areas.sum(), 1e-12)
    idx = rng.choice(len(faces), size=int(count), p=prob)
    tri = faces[idx]  # (K, 3)
    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    u = rng.random(size=(count,)).astype(np.float64)
    v = rng.random(size=(count,)).astype(np.float64)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    pts = (v0 * w0[:, None] + v1 * w1[:, None] + v2 * w2[:, None]).astype(np.float32)
    # Face normals broadcast
    nrm = mesh.face_normals[idx].astype(np.float32)
    return pts, nrm


@dataclass
class URDFRealSampler:
    urdf_path: str
    gripper_only: bool = True
    min_samples_per_mesh: int = 1

    def __post_init__(self) -> None:
        self._urdf = urdfpy.URDF.load(self.urdf_path)
        self._presampled_pts: Dict[str, np.ndarray] = {}
        self._presampled_nrm: Dict[str, np.ndarray] = {}
        self._target_count: int | None = None
        self._rng: np.random.Generator | None = None

    def _neutral_cfg(self) -> Dict[str, float]:
        cfg = {j.name: 0.0 for j in self._urdf.actuated_joints}
        cfg.setdefault("finger_joint", 0.0)
        for i in range(1, 8):
            cfg.setdefault(f"panda_joint{i}", 0.0)
        return cfg

    def presample(self, num_points: int, seed: int | None = None) -> None:
        if num_points <= 0:
            self._presampled_pts = {}
            self._presampled_nrm = {}
            return
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._target_count = int(num_points)
        fk_ref = self._urdf.visual_trimesh_fk(cfg=self._neutral_cfg())

        names, meshes, areas = [], [], []
        for i, (mesh, _T) in enumerate(fk_ref.items()):
            mid = get_mesh_stable_id(mesh, i)
            if self.gripper_only:
                ml = mid.lower()
                if not any(k in ml for k in ("finger", "knuckle", "robotiq", "gripper")):
                    continue
            eff_area = float(mesh.area)
            if "hand_camera_part" in mid.lower():
                eff_area *= 1e-6
            names.append(mid)
            meshes.append(mesh)
            areas.append(max(eff_area, 1e-9))

        if not names:
            # If filtering produced nothing, use all meshes
            for i, (mesh, _T) in enumerate(fk_ref.items()):
                mid = get_mesh_stable_id(mesh, i)
                names.append(mid)
                meshes.append(mesh)
                areas.append(max(float(mesh.area), 1e-9))

        total_area = float(sum(areas))
        counts = []
        allocated = 0
        for i, a in enumerate(areas):
            if i == len(areas) - 1:
                c = max(self.min_samples_per_mesh, num_points - allocated)
            else:
                w = a / total_area if total_area > 0 else 0.0
                c = max(self.min_samples_per_mesh, int(round(w * num_points)))
                allocated += c
            counts.append(int(c))

        pts_dict: Dict[str, np.ndarray] = {}
        nrm_dict: Dict[str, np.ndarray] = {}
        for mid, mesh, c in zip(names, meshes, counts):
            pts, nrm = _sample_surface_deterministic(mesh, int(c), rng)
            pts_dict[mid] = pts
            nrm_dict[mid] = nrm

        # Optional global downsampling to exactly respect target count
        total = sum(arr.shape[0] for arr in pts_dict.values())
        if total > num_points:
            # Flatten indices across meshes, sample without replacement
            mesh_keys = list(pts_dict.keys())
            offsets = np.cumsum([0] + [pts_dict[k].shape[0] for k in mesh_keys])
            sel = rng.choice(total, size=num_points, replace=False)
            sel.sort()
            # Reconstruct per-mesh selections
            new_pts: Dict[str, np.ndarray] = {}
            new_nrm: Dict[str, np.ndarray] = {}
            for i, key in enumerate(mesh_keys):
                start, end = offsets[i], offsets[i+1]
                # take sel indices in [start, end)
                mask = (sel >= start) & (sel < end)
                if not np.any(mask):
                    continue
                local_idx = sel[mask] - start
                new_pts[key] = pts_dict[key][local_idx]
                new_nrm[key] = nrm_dict[key][local_idx]
            pts_dict = new_pts
            nrm_dict = new_nrm

        self._presampled_pts = pts_dict
        self._presampled_nrm = nrm_dict

    def compute_world_trajectories(
        self,
        joint_positions: np.ndarray,  # (T, 7)
        gripper_positions: np.ndarray,  # (T,) or (T,1)
    ) -> np.ndarray:
        jp = np.asarray(joint_positions, dtype=np.float32)
        gp = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
        assert jp.ndim == 2 and jp.shape[1] == 7, "joint_positions must be (T,7)"
        assert gp.shape[0] == jp.shape[0], "gripper_positions length mismatch"
        T = jp.shape[0]

        # Collect transforms per mesh id over time
        transforms: Dict[str, np.ndarray] = {mid: np.zeros((T, 4, 4), dtype=np.float32) for mid in self._presampled_pts.keys()}

        for t in range(T):
            cfg = {f"panda_joint{i+1}": float(jp[t, i]) for i in range(7)}
            cfg["finger_joint"] = float(gp[t])
            fk = self._urdf.visual_trimesh_fk(cfg=cfg)
            for i, (mesh, T_wm) in enumerate(fk.items()):
                mid = get_mesh_stable_id(mesh, i)
                if mid in transforms:
                    transforms[mid][t] = T_wm.astype(np.float32)

        # Apply transforms to presampled points
        traj_list = []
        for mid, pts in self._presampled_pts.items():
            T_wm = transforms.get(mid, None)
            if T_wm is None or T_wm.shape[0] == 0:
                continue
            n = pts.shape[0]
            homo = np.concatenate([pts.astype(np.float32), np.ones((n, 1), dtype=np.float32)], axis=1)  # (N,4)
            world = (T_wm @ homo.T).transpose(0, 2, 1)[..., :3]  # (T,N,3)
            traj_list.append(world)

        if not traj_list:
            return np.zeros((T, 0, 3), dtype=np.float32)
        traj = np.concatenate(traj_list, axis=1)
        # Enforce target count again if needed (safety net)
        if self._target_count is not None and traj.shape[1] > self._target_count:
            rng = self._rng or np.random.default_rng(0)
            idx = rng.choice(traj.shape[1], size=int(self._target_count), replace=False)
            traj = traj[:, idx, :]
        return traj
