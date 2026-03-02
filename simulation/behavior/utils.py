#!/usr/bin/env python3

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
"""Shared utilities for the Behavior (simulation) extraction pipeline."""

import omnigibson as og
import omnigibson.lazy as lazy

EXCLUDE_MESH_NAMES = [
    "robot",
    "ground_plane",
    "r1pro",
    "toggle",
    "unlabelled",
    "unlabeled",
    "background",
]


def points_in_bounds_mask(points, bounds_min, bounds_max):
    """Return a boolean mask for points within axis-aligned bounds."""
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
    within_bounds = (
        (points[:, 0] > bounds_min[0])
        & (points[:, 0] < bounds_max[0])
        & (points[:, 1] > bounds_min[1])
        & (points[:, 1] < bounds_max[1])
        & (points[:, 2] > bounds_min[2])
        & (points[:, 2] < bounds_max[2])
    )
    return within_bounds


def configure_sim_settings():
    """Apply optimized simulation settings for faster, stable rendering."""
    settings = lazy.carb.settings.get_settings()

    # Use asynchronous rendering for faster performance.
    # NOTE: This gets reset EVERY TIME the sim stops / plays.
    settings.set_bool("/app/asyncRendering", True)
    og.sim.render()
    settings.set_bool("/app/asyncRendering", False)
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)

    # Must ALWAYS be set after sim plays because omni overrides these values.
    settings.set("/app/runLoops/main/rateLimitEnabled", False)
    settings.set("/app/runLoops/main/rateLimitUseBusyLoop", False)

    # Repeat to ensure it takes effect.
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)
    settings.set_bool("/app/asyncRendering", False)
    settings.set_bool("/app/asyncRenderingLowLatency", False)
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)

    # Additional RTX settings.
    settings.set_bool("/rtx-transient/dlssg/enabled", True)

    # Disable fractional cutout opacity for speed.
    lazy.carb.settings.get_settings().set_bool("/rtx/raytracing/fractionalCutoutOpacity", False)

    # See https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/TIME_STEPPING.html
    timeline = lazy.omni.timeline.get_timeline_interface()
    timeline.set_play_every_frame(True)

    settings.set("/app/player/useFastMode", True)
    settings.set("/app/show_developer_preference_section", True)
    settings.set("/app/player/useFixedTimeStepping", True)

    for run_loop in ["present", "main", "rendering_0"]:
        settings.set(f"/app/runLoops/{run_loop}/rateLimitEnabled", False)
        settings.set(f"/app/runLoops/{run_loop}/rateLimitFrequency", 120)
        settings.set(f"/app/runLoops/{run_loop}/rateLimitUseBusyLoop", False)
    settings.set("/exts/omni.kit.renderer.core/present/enabled", True)
    settings.set("/app/vsync", True)
