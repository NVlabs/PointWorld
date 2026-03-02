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
"""Extrinsics optimization using VGGT initialization and robot mesh rendering.
This script takes estimated extrinsics (from VGGT) and optimizes them using
precomputed stereo depth (from FoundationStereo).
"""

import sys
sys.path.append('..')
import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm

from real.extrinsics_pipeline import _print, DEVICE, URDF_PATH, process_single_scene
from compute_extrinsics_utils import RobotMeshRenderer
from vggt_forward import VGGTForwardPass
from vggt.models.vggt import VGGT
from real.gcs_utils import enforce_gcs_cache_policy

def main():
    parser = argparse.ArgumentParser(description="Optimize camera extrinsics using robot mesh rendering")
    parser.add_argument("--output_dir", required=True, help="Output directory for saving results and loading precomputed depth")
    parser.add_argument("--input", default=None, help="Input txt file with scene paths")
    parser.add_argument("--scene", default=None, help="Scene path to process")
    parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed processing")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    parser.add_argument("--debug", action="store_true", help="Debug mode - raise errors immediately")
    
    # Processing parameters
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to use")
    parser.add_argument("--downscale", type=float, default=0.5, help="Image downscaling ratio")
    parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum optimization iterations")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--translation_scale", type=float, default=0.01, help="Translation parameter scale")
    parser.add_argument("--rotation_scale", type=float, default=np.deg2rad(0.05), help="Rotation parameter scale (default: 1 degree)")
    parser.add_argument("--dedup_threshold", type=float, default=0.5, help="Coordinate deduplication threshold")
    parser.add_argument("--min_robot_points", type=int, default=1000, help="Minimum number of robot points required for optimization")
    default_vggt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "vggt", "model.pt")
    parser.add_argument("--vggt_model_path", type=str, default=default_vggt, help="Path to VGGT model checkpoint")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"], 
                        help="Optimizer type for depth-based optimization (adam or lbfgs)")
    parser.add_argument(
        "--allow_gcs_streaming",
        action="store_true",
        help=(
            "Bypass GCS cache enforcement and stream directly from gs:// inputs for this run. "
            "Not recommended for repeated multi-stage processing."
        ),
    )
    
    args = parser.parse_args()

    assert args.scene or args.input, "Either scene or input must be provided"
    
    # Read input paths
    if args.input:
        with open(args.input, 'r') as f:
            all_paths = [line.strip() for line in f if line.strip()]
        
        # shuffle with fixed seed (to improve load balancing and to ensure reproducibility)
        random.seed(42)
        random.shuffle(all_paths)
        
        # Partition paths for distributed processing
        if args.world_size <= 0:
            raise ValueError(f"world_size must be >= 1, got {args.world_size}")
        if not (0 <= args.rank < args.world_size):
            raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}")
        paths_to_process = all_paths[args.rank::args.world_size]
    else:
        all_paths = [args.scene]
        paths_to_process = all_paths

    enforce_gcs_cache_policy(
        all_paths,
        stage_name="compute_extrinsics",
        require_cache=True,
        allow_streaming=args.allow_gcs_streaming,
    )
    
    _print(f"Rank {args.rank}/{args.world_size}: Processing {len(paths_to_process)}/{len(all_paths)} scenes")
    
    # Initialize models (only if we have scenes to process)
    if len(paths_to_process) == 0:
        _print(f"No scenes to process for rank {args.rank}")
        return
    
    # Initialize VGGT model from local file
    vggt_model_path = args.vggt_model_path
    assert os.path.exists(vggt_model_path), f"VGGT model file not found: {vggt_model_path}"
    
    vggt_model = VGGT()
    vggt_model.load_state_dict(torch.load(vggt_model_path, map_location=DEVICE))
    vggt_model.to(DEVICE)
    vggt_model.eval()
    vggt_forward = VGGTForwardPass(vggt_model, device=DEVICE)
    _print(f"VGGT model loaded successfully from {vggt_model_path}")
    
    # Initialize robot renderer
    robot_renderer = RobotMeshRenderer(URDF_PATH, device=DEVICE, total_samples=25000)
    
    # Track statistics
    total_processed = 0
    total_successful = 0
    total_skipped = 0
    
    # Process each scene
    for scene_path in tqdm(paths_to_process, desc=f"Rank {args.rank}/{args.world_size}"):
        total_processed += 1
        
        success = process_single_scene(
            scene_path=scene_path,
            output_dir=args.output_dir,
            vggt_forward=vggt_forward,
            robot_renderer=robot_renderer,
            num_frames=args.num_frames,
            downscale_ratio=args.downscale,
            min_robot_points=args.min_robot_points,
            max_iterations=args.max_iterations,
            lr=args.lr,
            translation_scale=args.translation_scale,
            rotation_scale=args.rotation_scale,
            dedup_threshold=args.dedup_threshold,
            optimizer=args.optimizer,
            debug=args.debug
        )
        
        if success:
            total_successful += 1
        else:
            total_skipped += 1
    
    # Print final statistics
    _print(f"\nProcessing completed!")
    _print(f"  Total scenes processed: {total_processed}")
    _print(f"  Successful: {total_successful}")
    _print(f"  Skipped: {total_skipped}")
    _print(f"  Success rate: {100*total_successful/total_processed:.1f}%")


if __name__ == "__main__":
    main() 
