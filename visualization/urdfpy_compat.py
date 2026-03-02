# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for upstream urdfpy on modern NumPy versions."""

from __future__ import annotations

import numpy as np


def ensure_urdfpy_numpy_compat() -> None:
    """Restore deprecated NumPy aliases used by urdfpy 0.0.22."""
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]
