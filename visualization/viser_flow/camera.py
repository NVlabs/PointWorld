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
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def _orthonormalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Minimal, deterministic SO(3) projection.

    - Normalize columns
    - If det<0, flip the first column
    - No special handling for reflections or image-axis alignment
    """
    basis = np.asarray(matrix, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError("rotation component requires a 3x3 matrix")

    columns: list[np.ndarray] = []
    for i in range(3):
        col = basis[:, i]
        norm = np.linalg.norm(col)
        if norm < 1e-8:
            col = np.zeros(3, dtype=np.float64)
            col[i] = 1.0
            norm = 1.0
        columns.append(col / norm)

    Rm = np.stack(columns, axis=1)
    if np.linalg.det(Rm) < 0:
        Rm[:, 0] *= -1.0
    return Rm.astype(np.float32)


@dataclasses.dataclass(slots=True)
class CameraSpec:
    name: str
    extrinsic: np.ndarray  # (4, 4) camera-to-world transform
    intrinsic: np.ndarray  # (3, 3)
    image: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    fov_y: float
    aspect: float
    position: np.ndarray  # (3,)
    orientation_wxyz: np.ndarray  # (4,)


def _infer_image_shape(
    image: np.ndarray | None,
    depth: np.ndarray | None,
) -> tuple[int, int]:
    source = image if image is not None else depth
    if source is None:
        raise ValueError("camera requires either RGB or depth image to derive resolution")
    if source.ndim == 2:
        return int(source.shape[0]), int(source.shape[1])
    if source.ndim == 3:
        return int(source.shape[0]), int(source.shape[1])
    raise ValueError("Unsupported image shape for camera media")


def camera_spec(
    name: str,
    extrinsic_world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    image: np.ndarray | None,
    depth: np.ndarray | None,
    *,
    reflection_matrix: np.ndarray | None = None,
) -> CameraSpec:
    extrinsic_world_to_cam = np.asarray(extrinsic_world_to_cam, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if extrinsic_world_to_cam.shape != (4, 4):
        raise ValueError("extrinsic must have shape (4, 4)")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")

    height, width = _infer_image_shape(image, depth)
    fy = intrinsic[1, 1]
    if fy <= 0:
        raise ValueError("intrinsic fy must be positive")
    fov_y = float(2.0 * np.arctan2(height, 2.0 * fy))
    aspect = float(width / height)

    cam_to_world = np.linalg.inv(extrinsic_world_to_cam)
    rotation = cam_to_world[:3, :3]
    # NOTE: do not attempt to "fix" parity or align image axes. Keep it minimal.
    rotation = _orthonormalize_columns(rotation)
    translation = cam_to_world[:3, 3]
    rot = R.from_matrix(rotation)
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    # Warn loudly if reflection bookkeeping indicates a flip – we do not try to
    # make the frustum exact under reflections. Caller should treat frustum as approximate.
    if reflection_matrix is not None:
        refl = np.asarray(reflection_matrix, dtype=np.float32)
        if refl.shape == (3, 3):
            if np.linalg.det(refl) < 0 or np.any(np.diag(refl) < 0):
                print("\033[33m[warn] Camera frustum may be inaccurate due to world flip (reflection detected).\033[0m")

    return CameraSpec(
        name=name,
        extrinsic=cam_to_world,
        intrinsic=intrinsic,
        image=image,
        depth=depth,
        fov_y=fov_y,
        aspect=aspect,
        position=translation.astype(np.float32),
        orientation_wxyz=quat_wxyz,
    )
