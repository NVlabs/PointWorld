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

import hashlib
import numpy as np
from numba import njit


def _stable_int_hash(*parts) -> int:
    """Deterministic 32-bit hash from arbitrary parts.
    Avoids Python's randomized hash salt across processes.
    """
    m = hashlib.md5()
    for p in parts:
        m.update(str(p).encode("utf-8"))
    return int.from_bytes(m.digest()[:4], byteorder="little", signed=False)


@njit(cache=True, fastmath=True, nogil=True)
def fnv_hash_vec_nb(arr):
    """
    Numba-accelerated FNV-1A 64-bit hash on a 2D integer array (N, 3).
    This helps us quickly generate hash keys per-voxel.
    """
    # arr should already be nonnegative if you shift by min
    # Convert to 64-bit unsigned integers.
    out = np.full(arr.shape[0], np.uint64(14695981039346656037), dtype=np.uint64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i] = out[i] * np.uint64(1099511628211)
            out[i] = out[i] ^ np.uint64(arr[i, j])
    return out


@njit(cache=True, fastmath=True)
def _rotate_xy(points, cos_a, sin_a):
    """
    In-place rotation of (x,y) around the z axis.
    points: (..., 3) array
    """
    # Flatten trailing dims for tight inner loop
    flat = points.reshape(-1, 3)
    for i in range(flat.shape[0]):
        x, y = flat[i, 0], flat[i, 1]
        flat[i, 0] =  cos_a * x - sin_a * y
        flat[i, 1] =  sin_a * x + cos_a * y
    # z column is untouched


__all__ = ["_stable_int_hash", "fnv_hash_vec_nb", "_rotate_xy"]
