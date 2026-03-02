#!/usr/bin/env python

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
import os
import sys
import argparse
import subprocess
from tqdm import tqdm
import multiprocessing as mp

# Import the get_metadata function from droid_utils.py
from real.droid_utils import get_metadata


def download_calibration(serial, save_dir):
    """
    Download camera calibration file for a given serial number
    
    Args:
        serial (str): Camera serial number
        save_dir (str): Directory to save calibration file
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    output_file = os.path.join(save_dir, f"SN{serial}.conf")
    
    # Skip if the file already exists
    if os.path.exists(output_file):
        return True
    
    calibration_url = f"http://calib.stereolabs.com/?SN={serial}"
    
    subprocess.run(
        ["wget", calibration_url, "-O", output_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return True


def process_scene(scene_path):
    """
    Process a single scene and extract camera serials
    
    Args:
        scene_path (str): Path to the scene
        
    Returns:
        set: Set of camera serials found in the scene
    """
    scene_serials = set()
    # Get metadata
    metadata = get_metadata(scene_path)

    # Extract camera serials
    camera_names = ['wrist', 'ext1', 'ext2']

    for camera_name in camera_names:
        serial_key = f'{camera_name}_cam_serial'

        if serial_key in metadata:
            camera_serial = str(metadata[serial_key])
            scene_serials.add(camera_serial)
    
    return scene_serials


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download camera calibration files from scene paths")
    parser.add_argument("--scene_list", help="Path to text file containing scene paths (one per line)")
    parser.add_argument("--save_dir", help="Directory to save calibration files")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="Number of worker processes to use (default: number of CPU cores)")
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.scene_list):
        print(f"Error: Scene list file not found: {args.scene_list}")
        sys.exit(1)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Read scene paths from the input file
    with open(args.scene_list, 'r') as f:
        scene_paths = [line.strip() for line in f if line.strip()]
    
    if not scene_paths:
        print("Error: No scene paths found in the input file")
        sys.exit(1)
    
    # Use multiprocessing to extract camera serials in parallel
    print(f"Processing {len(scene_paths)} scene paths using {args.num_workers} workers...")
    
    # Create a pool of workers
    pool = mp.Pool(processes=args.num_workers)
    
    # Process scenes in parallel and collect results
    all_serials = set()
    for scene_serials in tqdm(
        pool.imap_unordered(process_scene, scene_paths),
        total=len(scene_paths),
        desc="Processing scenes"
    ):
        all_serials.update(scene_serials)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Download calibration files for all unique serials (sequentially as requested)
    print(f"Downloading calibration files for {len(all_serials)} unique camera serials...")
    
    successful_downloads = 0
    for serial in tqdm(all_serials, desc="Downloading calibrations"):
        if download_calibration(serial, args.save_dir):
            successful_downloads += 1
    
    print(f"Successfully downloaded {successful_downloads} out of {len(all_serials)} calibration files")


if __name__ == "__main__":
    main()
