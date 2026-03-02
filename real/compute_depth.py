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
Compute stereo depth for all available cameras in DROID episodes and store in dedicated H5 files.

This script processes DROID episodes to compute stereo depth maps using FoundationStereo,
storing results in compressed H5 files for later use in the labeling pipeline.
"""

import sys
sys.path.append("..")
import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import random

# Import required utilities
from real.real_utils import get_time_str
from real.droid_utils import gather_data_dict, get_uuid
from real.gcs_utils import enforce_gcs_cache_policy
from depth_estimator import DepthEstimator

# FoundationStereo checkpoint path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEPTH_ESTIMATOR_CKPT_PATH = os.path.join(
    REPO_ROOT,
    "checkpoints",
    "foundationstereo",
    "23-51-11",
    "model_best_bp2.pth",
)
DEPTH_ESTIMATOR_CFG_PATH = os.path.join(
    REPO_ROOT,
    "assets",
    "foundationstereo",
    "23-51-11",
    "cfg.yaml",
)


def check_depth_file_exists(output_dir, uuid):
    """
    Check if depth file exists and has valid structure with write_complete flags.
    
    Args:
        output_dir (str): Output directory for depth files
        uuid (str): Episode UUID
        
    Returns:
        bool: True if valid depth file exists, False otherwise
    """
    h5_path = os.path.join(output_dir, "depth", f"{uuid}_depth.h5")
    
    if not os.path.exists(h5_path):
        return False
    
    with h5py.File(h5_path, 'r') as f:
        # Check metadata
        if 'metadata' not in f:
            raise ValueError(f"Depth file missing metadata group: {h5_path}")
        
        metadata = f['metadata']
        if 'write_complete' not in metadata.attrs or not metadata.attrs['write_complete']:
            raise ValueError(f"Depth file metadata incomplete: {h5_path}")
        
        # Check each camera group
        camera_groups = [key for key in f.keys() if key != 'metadata']
        if not camera_groups:
            raise ValueError(f"Depth file missing camera groups: {h5_path}")
        
        for camera_group_name in camera_groups:
            camera_group = f[camera_group_name]
            
            # Check required datasets
            required_keys = ['depth', 'timestamps']
            for key in required_keys:
                if key not in camera_group:
                    raise ValueError(f"Depth file missing {key} in {camera_group_name}: {h5_path}")
                if 'write_complete' not in camera_group[key].attrs or not camera_group[key].attrs['write_complete']:
                    raise ValueError(f"Depth file {key} incomplete in {camera_group_name}: {h5_path}")
    
    return True


def compute_depth_for_scene(
    scene_path,
    output_dir,
    depth_estimator,
    downscale_ratio=1.0,
    batch_size=32,
):
    """
    Compute stereo depth for all cameras in a single scene.
    
    Args:
        scene_path (str): Path to the scene directory
        output_dir (str): Output directory for depth files
        depth_estimator (DepthEstimator): Initialized depth estimator
        downscale_ratio (float): Downscale ratio for the images
        batch_size (int): Batch size for depth estimation
    
    Returns:
        bool: True if successful
    """
    # Get episode UUID
    uuid = get_uuid(scene_path)
    
    # Check if already processed
    if check_depth_file_exists(output_dir, uuid):
        print(f"[{get_time_str()}] Depth already computed for {uuid}, skipping")
        return True
    
    print(f"[{get_time_str()}] Processing scene: {uuid}")
    
    # Gather data with stereo frames for external cameras
    data_dict = gather_data_dict(
        scene_path,
        downscale_ratio=downscale_ratio,
        include_stereo=True,
        include_depth=False,  # We want to compute our own depth
    )
    # Create output H5 file
    depth_dir = os.path.join(output_dir, "depth")
    h5_path = os.path.join(depth_dir, f"{uuid}_depth.h5")
    os.makedirs(depth_dir, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        # Create metadata group
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['uuid'] = uuid
        metadata_group.attrs['camera_count'] = len(data_dict)

        total_cameras = len(data_dict)
        frame_count = None

        # Process each camera
        for i, (camera_serial, camera_data) in enumerate(data_dict.items()):
            print(f"[{get_time_str()}] Processing camera {i+1}/{total_cameras}: {camera_serial}")

            # Extract required data
            left_frames = camera_data['rgb']  # Left camera frames
            right_frames = camera_data['right_frames']  # Right camera frames
            timestamps = camera_data['timestamps']  # Timestamps
            intrinsic = camera_data['intrinsic']
            baseline = camera_data['baseline']
            if frame_count is None:
                frame_count = len(left_frames)

            # Set camera parameters for depth estimator
            depth_estimator.set_camera_params(baseline, intrinsic)

            # Compute depth using batch inference with smaller batch size for memory efficiency
            print(f"[{get_time_str()}] Computing depth for {len(left_frames)} frames...")
            depth_frames = depth_estimator.infer_depth_batch(left_frames, right_frames, batch_size=batch_size)

            # Convert depth to uint16 millimeters
            depth_mm = (depth_frames * 1000.0).astype(np.float32)
            depth_mm = np.clip(depth_mm, 0, 65535)  # Clip to uint16 range
            depth_uint16 = depth_mm.astype(np.uint16)

            camera_type = "ext"
            camera_group_name = f"{camera_serial}+{camera_type}"

            # Create camera group in H5 file with type suffix
            camera_group = f.create_group(camera_group_name)

            # Add camera metadata to group attributes
            camera_group.attrs['camera_serial'] = camera_serial
            camera_group.attrs['camera_type'] = camera_type

            # Store depth with high compression
            depth_dataset = camera_group.create_dataset(
                'depth',
                data=depth_uint16,
                compression='gzip',
                compression_opts=9,
                shuffle=True,
                chunks=True
            )
            depth_dataset.attrs['write_complete'] = True
            depth_dataset.attrs['units'] = 'millimeters'
            depth_dataset.attrs['dtype'] = 'uint16'

            # Store timestamps
            timestamps_dataset = camera_group.create_dataset(
                'timestamps',
                data=timestamps,
            )
            timestamps_dataset.attrs['write_complete'] = True
            timestamps_dataset.attrs['units'] = 'milliseconds'

            print(f"[{get_time_str()}] Saved depth data for camera {camera_group_name}: {depth_uint16.shape}")

        # Update metadata with frame count
        metadata_group.attrs['frame_count'] = frame_count
        metadata_group.attrs['write_complete'] = True
    
    print(f"[{get_time_str()}] Successfully saved depth data to {h5_path}")
    return True


def main():
    """Main function for depth computation script."""
    parser = argparse.ArgumentParser(description="Compute stereo depth for DROID episodes")
    parser.add_argument("--input", required=True, help="Input txt file with scene paths")
    parser.add_argument("--output_dir", required=True, help="Output directory for depth H5 files")
    parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed processing")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    parser.add_argument("--downscale_ratio", type=float, default=0.5, help="Downscale ratio for the images (raw is 1280 x 720)")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for depth estimation")  # greater than 12 will cause cudnn error due to amp
    parser.add_argument("--foundation_stereo_ckpt", type=str, default=DEPTH_ESTIMATOR_CKPT_PATH,
                        help="Path to FoundationStereo checkpoint")
    parser.add_argument("--foundation_stereo_cfg", type=str, default=DEPTH_ESTIMATOR_CFG_PATH,
                        help="Path to FoundationStereo cfg.yaml (must include vit_size)")
    parser.add_argument(
        "--allow_gcs_streaming",
        action="store_true",
        help=(
            "Bypass GCS cache enforcement and stream directly from gs:// inputs for this run. "
            "Not recommended for repeated multi-stage processing."
        ),
    )
    args = parser.parse_args()
    
    # Read input paths
    with open(args.input, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]

    enforce_gcs_cache_policy(
        all_paths,
        stage_name="compute_depth",
        require_cache=True,
        allow_streaming=args.allow_gcs_streaming,
    )
    
    # shuffle with fixed seed (to improve load balancing and to ensure reproducibility)
    random.seed(42)
    random.shuffle(all_paths)
    
    # Partition paths for distributed processing
    if args.world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}")

    paths_to_process = all_paths[args.rank::args.world_size]
    
    print(f"[{get_time_str()}] Rank {args.rank}/{args.world_size}: Processing {len(paths_to_process)}/{len(all_paths)} scenes")
    
    # Normal processing mode
    print(f"[{get_time_str()}] Initializing depth estimator...")
    if not os.path.exists(args.foundation_stereo_ckpt):
        raise FileNotFoundError(f"FoundationStereo checkpoint not found: {args.foundation_stereo_ckpt}")
    if not os.path.exists(args.foundation_stereo_cfg):
        raise FileNotFoundError(f"FoundationStereo cfg not found: {args.foundation_stereo_cfg}")
    depth_estimator = DepthEstimator(
        args.foundation_stereo_ckpt,
        device='cuda',
        cfg_path=args.foundation_stereo_cfg,
    )
    
    # Track statistics
    total_processed = 0
    total_successful = 0
    
    for scene_path in tqdm(paths_to_process, desc="Processing scenes"):
        total_processed += 1
        success = compute_depth_for_scene(
            scene_path,
            args.output_dir,
            depth_estimator,
            downscale_ratio=args.downscale_ratio,
            batch_size=args.batch_size,
        )
        if success:
            total_successful += 1
    
    # Print final statistics
    print(f"\n[{get_time_str()}] Processing completed!")
    print(f"  Total scenes processed: {total_processed}")
    print(f"  Successful: {total_successful}")
    print(f"  Success rate: {100*total_successful/total_processed:.1f}%")


if __name__ == "__main__":
    main()
