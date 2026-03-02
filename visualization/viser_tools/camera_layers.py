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

from typing import Dict, Sequence, Tuple

import numpy as np
import viser
from matplotlib import cm

from ..viser_flow.camera import CameraSpec

RgbColor = Tuple[int, int, int]


def _colormap_colors(count: int) -> list[RgbColor]:
    if count <= 0:
        return []
    cmap = cm.get_cmap("jet")
    if count == 1:
        samples = [0.0]
    else:
        samples = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float64)
    colors: list[RgbColor] = []
    for sample in samples:
        rgba = cmap(float(sample))
        colors.append(
            (
                int(np.clip(round(rgba[0] * 255.0), 0, 255)),
                int(np.clip(round(rgba[1] * 255.0), 0, 255)),
                int(np.clip(round(rgba[2] * 255.0), 0, 255)),
            )
        )
    return colors


def add_camera_frustums(
    server: viser.ViserServer,
    *,
    specs: Sequence[CameraSpec],
    name_prefix: str = "cameras",
    scale: float,
    line_width: float,
    visible: bool,
) -> Dict[str, viser.CameraFrustumHandle]:
    """Attach camera frustums to the scene and return a name→handle mapping.

    Args:
        server: Active viser server instance.
        specs: CameraSpec objects defining pose and media.
        name_prefix: Scene tree prefix for the frusta.
        scale: Frustum scale factor.
        line_width: Line width for frustum edges.
        visible: Initial visibility for all frusta.

    Returns:
        Dictionary mapping camera names to their corresponding frustum handles.
    """
    if scale < 0.0:
        raise ValueError("frustum scale must be non-negative")
    if line_width <= 0.0:
        raise ValueError("frustum line width must be positive")

    palette = _colormap_colors(len(specs))
    color_map = {spec.name: palette[idx] for idx, spec in enumerate(specs)}

    handles: Dict[str, viser.CameraFrustumHandle] = {}
    for spec in specs:
        rgb = color_map.get(spec.name)
        if rgb is None:
            raise KeyError(f"Missing color override for camera '{spec.name}'")
        handle = server.scene.add_camera_frustum(
            f"{name_prefix}/{spec.name}",
            fov=spec.fov_y,
            aspect=spec.aspect,
            image=spec.image,
            wxyz=spec.orientation_wxyz,
            position=spec.position,
            scale=float(scale),
            line_width=float(line_width),
            color=rgb,
        )
        handle.visible = visible
        handles[spec.name] = handle
    return handles

__all__ = ["add_camera_frustums"]
