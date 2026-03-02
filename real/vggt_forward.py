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
"""VGGT forward-pass wrapper for extrinsics + optional depth/point prediction."""
import os
import sys
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_ROOT = os.path.join(REPO_ROOT, "third_party", "vggt")
if VGGT_ROOT not in sys.path:
    sys.path.append(VGGT_ROOT)

try:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError as e:
    print("Error: Could not import VGGT utilities. Ensure the vggt submodule is initialized and installed.")
    raise e


@contextmanager
def stage_vggt_images(images):
    """Write RGB images to a temp dir for VGGT and yield file paths."""
    temp_dir = tempfile.mkdtemp(prefix="vggt_frames_")
    paths = []
    try:
        for idx, img in enumerate(images):
            path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            Image.fromarray(img).save(path)
            paths.append(path)
        yield paths
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class VGGTForwardPass:
    """VGGT forward pass for camera extrinsics/intrinsics."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def __call__(self, image_paths):
        """
        Perform VGGT forward pass on the given images.

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary containing:
                - extrinsics_3x4: [N, 3, 4] camera extrinsics relative to first image
                - extrinsics_4x4: [N, 4, 4] camera extrinsics relative to first image
                - intrinsics: [N, 3, 3] camera intrinsics
        """
        # Load and preprocess images
        images_tensor = load_and_preprocess_images(image_paths).to(self.device)
        images_tensor = images_tensor[None]  # [batch=1, N, 3, H, W]

        # Forward pass through aggregator and camera head
        aggregated_tokens_list, _ = self.model.aggregator(images_tensor)
        pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]

        # Convert to extrinsics and intrinsics
        extrinsic_np, intrinsic_np = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
        extrinsic_np = extrinsic_np[0].cpu().numpy()  # [N, 3, 4]
        intrinsic_np = intrinsic_np[0].cpu().numpy()  # [N, 3, 3]

        # Convert 3x4 extrinsics to 4x4 for convenience
        extrinsic_4x4_list = []
        for i in range(extrinsic_np.shape[0]):
            M = np.eye(4, dtype=np.float32)
            M[:3, :4] = extrinsic_np[i]
            extrinsic_4x4_list.append(M)

        return {
            "extrinsics_3x4": extrinsic_np,
            "extrinsics_4x4": np.stack(extrinsic_4x4_list, axis=0),
            "intrinsics": intrinsic_np,
        }
