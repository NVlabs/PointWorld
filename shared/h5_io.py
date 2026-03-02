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
H5 I/O utilities for storing and loading RGB/depth data with explicit formats.

These helpers are shared by real (DROID) labeling and simulation (behavior) extraction
pipelines to ensure consistent encoding and metadata.
"""
from __future__ import annotations

import h5py
import numpy as np


def save_rgb_as_jpeg_in_h5(group: h5py.Group, dataset_name: str, rgb_image: np.ndarray, jpeg_quality: int = 95) -> None:
    """
    Save RGB image as JPEG binary data in HDF5 dataset.

    Args:
        group: HDF5 group to save to.
        dataset_name: Name of the dataset.
        rgb_image: RGB image as numpy array (H, W, 3) uint8.
        jpeg_quality: JPEG compression quality (1-100).
    """
    import cv2

    assert rgb_image.dtype == np.uint8, f"Expected uint8 RGB image, got {rgb_image.dtype}"
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, f"Expected RGB image (H, W, 3), got {rgb_image.shape}"
    assert 1 <= jpeg_quality <= 100, f"jpeg_quality must be in [1, 100], got {jpeg_quality}"

    success, encoded_img = cv2.imencode(
        ".jpg", rgb_image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    assert success, "Failed to encode image as JPEG"

    jpeg_data = np.frombuffer(encoded_img.tobytes(), dtype=np.uint8)
    dt = h5py.special_dtype(vlen=np.dtype("uint8"))
    dset = group.create_dataset(dataset_name, (1,), dtype=dt)
    dset[0] = jpeg_data
    dset.attrs["write_complete"] = True
    dset.attrs["format"] = "jpeg"
    dset.attrs["quality"] = jpeg_quality
    dset.attrs["original_shape"] = rgb_image.shape


def load_rgb_from_jpeg_in_h5(dataset: h5py.Dataset) -> np.ndarray:
    """
    Load RGB image from JPEG binary data stored in HDF5 dataset.

    Args:
        dataset: HDF5 dataset containing JPEG data.

    Returns:
        RGB image as numpy array (H, W, 3) uint8.
    """
    import cv2

    jpeg_data = dataset[0]
    decoded_img = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
    if decoded_img is None:
        raise RuntimeError("Failed to decode JPEG data")
    return decoded_img[..., ::-1]


def save_depth_as_uint16_mm(group: h5py.Group, dataset_name: str, depth_image_meters: np.ndarray) -> None:
    """
    Save depth image as uint16 in millimeters scale.

    Args:
        group: HDF5 group to save to.
        dataset_name: Name of the dataset.
        depth_image_meters: Depth image in meters as numpy array (H, W) float.
    """
    assert depth_image_meters.ndim == 2, f"Expected depth image (H, W), got {depth_image_meters.shape}"

    depth_mm = np.clip(depth_image_meters * 1000.0, 0, 65535)
    depth_uint16 = depth_mm.astype(np.uint16)

    dset = group.create_dataset(
        dataset_name,
        data=depth_uint16,
        dtype=np.uint16,
        compression="gzip",
        compression_opts=4,
    )
    dset.attrs["write_complete"] = True
    dset.attrs["format"] = "uint16_mm"
    dset.attrs["scale"] = "millimeters"


def load_depth_from_uint16_mm(dataset: h5py.Dataset) -> np.ndarray:
    """
    Load depth image from uint16 millimeters format and convert to meters.

    Args:
        dataset: HDF5 dataset containing uint16 depth data in mm.

    Returns:
        Depth image in meters as numpy array (H, W) float32.
    """
    depth_uint16 = dataset[()]
    return depth_uint16.astype(np.float32) / 1000.0
