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
from typing import Dict, Optional, Sequence

import numpy as np
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass(slots=True)
class RobotKinematics:
    """Normalized robot kinematic description for visualization/overlays.

    Supports both Panda (droid) and R1Pro (behavior) inputs, and is extensible for
    future embodiments. All arrays are numpy float32.
    """

    # Panda (droid): joints + scalar gripper aperture
    panda_joint_positions: Optional[np.ndarray] = None  # (T, 7)
    panda_gripper_positions: Optional[np.ndarray] = None  # (T,)

    # Generic URDF (e.g., R1Pro/behavior): joint name map + full joint vector
    joint_names: Optional[Sequence[str]] = None  # names corresponding to columns of joint_positions_full
    joint_positions_full: Optional[np.ndarray] = None  # (T, J)
    base_pose: Optional[np.ndarray] = None  # (T, 7) [x,y,z,qx,qy,qz,qw]


def _ensure_float32(arr) -> np.ndarray:
    a = np.asarray(arr)
    # Convert torch tensors explicitly
    if torch is not None and isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    elif hasattr(arr, "detach") and hasattr(arr, "cpu"):
        # Generic tensor-like fallback
        try:
            a = arr.detach().cpu().numpy()
        except Exception:
            pass
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    else:
        a = a.astype(np.float32, copy=False)
    return a


def parse_robot_kinematics(sample: Dict[str, object]) -> RobotKinematics:
    """Extract a normalized robot kinematic description from a sample dict.

    Priority order:
    1) Panda-style (droid) if both 'joint_positions' and 'gripper_positions' exist
    2) Generic URDF (e.g., R1Pro/behavior) if 'joint_names' and 'joint_positions' exist
       (optionally 'base_pose')

    Returns a RobotKinematics with the appropriate fields set. All arrays are
    coerced to float32; gripper vectors are flattened to (T,).
    """
    kin = RobotKinematics()

    # Panda path (droid)
    if ("joint_positions" in sample) and (sample.get("joint_positions") is not None) \
       and ("gripper_positions" in sample) and (sample.get("gripper_positions") is not None):
        jp = _ensure_float32(sample["joint_positions"])  # (T, 7) for Panda
        gp_raw = np.asarray(sample["gripper_positions"])  # shape can be (T,) or (T,1)
        if gp_raw.ndim == 2 and gp_raw.shape[1] == 1:
            gp = gp_raw[:, 0].astype(np.float32, copy=False)
        else:
            gp = _ensure_float32(gp_raw).reshape(-1)
        kin.panda_joint_positions = jp
        kin.panda_gripper_positions = gp
        return kin

    # Generic URDF path (e.g., R1Pro/behavior)
    if ("joint_names" in sample) and (sample.get("joint_names") is not None) \
       and ("joint_positions" in sample) and (sample.get("joint_positions") is not None):
        raw_names = sample["joint_names"]
        names_iter = raw_names if isinstance(raw_names, (list, tuple)) else list(raw_names)
        normalized_names: list[str] = []
        for name in names_iter:
            if isinstance(name, (bytes, bytearray, np.bytes_)):
                normalized_names.append(name.decode("utf-8"))
            else:
                normalized_names.append(str(name))
        kin.joint_names = normalized_names
        kin.joint_positions_full = _ensure_float32(sample["joint_positions"])  # (T, J)
        if ("base_pose" in sample) and (sample.get("base_pose") is not None):
            kin.base_pose = _ensure_float32(sample["base_pose"])  # (T, 7)
        return kin

    # Nothing recognized; return empty (no overlays rendered, but pipeline continues)
    return kin
