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
"""
Workspace bounds shared across real (DROID) labeling components.
"""
from __future__ import annotations

import numpy as np

# Define workspace bounds (meters)
WORKSPACE_BOUNDS_MIN = np.array([0.00, -0.40, -0.30], dtype=np.float32)
WORKSPACE_BOUNDS_MAX = np.array([0.70, 0.40, 1.20], dtype=np.float32)

_rng = WORKSPACE_BOUNDS_MAX - WORKSPACE_BOUNDS_MIN
WORKSPACE_BOUNDS_MIN_RELAXED = WORKSPACE_BOUNDS_MIN - 0.5 * _rng
WORKSPACE_BOUNDS_MAX_RELAXED = WORKSPACE_BOUNDS_MAX + 0.5 * _rng
