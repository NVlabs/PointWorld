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
import os
import numpy as np
import json
import glob
from tqdm import tqdm
import argparse
from transform_utils import (
    convert_pose_euler2mat,
    convert_pose_quat2mat,
    convert_pose_mat2quat,
    mat2quat,
)
from real.droid_utils import get_metadata, gather_trajectory


def compute_gripper2wrist_transform(scene_path):
    """
    Compute the gripper to wrist camera transformation for a given scene.
    
    Args:
        scene_path (str): Path to the scene directory
        
    Returns:
        dict: Dictionary containing the computed transformation and metadata
    """
    # Load metadata to get wrist camera extrinsics
    metadata = get_metadata(scene_path)
    
    # Check if wrist camera exists in this scene
    if 'wrist_cam_serial' not in metadata or 'wrist_cam_extrinsics' not in metadata:
        return None
    
    # Get wrist camera extrinsics (6DOF format)
    wrist_cam_extrinsics_6dof = np.array(metadata['wrist_cam_extrinsics'])
    
    # Convert to matrix format
    wrist_cam_extrinsics_mat = convert_pose_euler2mat(wrist_cam_extrinsics_6dof)
    
    # Get gripper pose at the first frame
    proprio_dict = gather_trajectory(scene_path)
    if len(proprio_dict['gripper_pose']) == 0:
        return None
    
    # Get the first frame gripper pose (gripper to world)
    gripper_pose_7dof = proprio_dict['gripper_pose'][0]  # [x,y,z,qx,qy,qz,qw]
    gripper_pose_mat = convert_pose_quat2mat(gripper_pose_7dof)
    
    # Compute gripper to wrist camera transformation
    # T_gripper2wrist = T_wrist2world^(-1) * T_gripper2world
    wrist2world_mat = wrist_cam_extrinsics_mat
    world2wrist_mat = np.linalg.inv(wrist2world_mat)
    gripper2world_mat = gripper_pose_mat
    gripper2wrist_mat = world2wrist_mat @ gripper2world_mat
    
    # Convert to quaternion format
    gripper2wrist_quat = convert_pose_mat2quat(gripper2wrist_mat)
    
    # Extract scene metadata
    scene_id = metadata['scene_id']
    uuid = metadata['uuid']
    lab = metadata['lab']
    success = metadata['success']
    robot_serial = metadata['robot_serial']
    wrist_cam_serial = metadata['wrist_cam_serial']
    
    return {
        'gripper2wrist_mat': gripper2wrist_mat.tolist(),
        'gripper2wrist_quat': gripper2wrist_quat.tolist(),
        'scene_id': scene_id,
        'uuid': uuid,
        'lab': lab,
        'success': success,
        'robot_serial': robot_serial,
        'wrist_cam_serial': wrist_cam_serial,
    }

def process_scenes(scene_paths, output_dir, rank, world_size):
    """
    Process a list of scenes, computing and saving transforms for each.
    
    Args:
        scene_paths (list): List of paths to scene directories
        output_dir (str): Directory to save the computed transformations
        rank (int): Worker rank
        world_size (int): Total number of workers
    
    Returns:
        int: Number of successfully processed scenes
    """
    # Each worker processes a disjoint subset of scenes
    worker_scenes = scene_paths[rank::world_size]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create one output file per worker
    output_file = os.path.join(output_dir, f"transforms_worker{rank}.json")
    
    # Dictionary to store all transformations, with UUID as key
    all_transforms = {}
    
    # Check if the file already exists (for resuming)
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_transforms = json.load(f)
        print(f"Worker {rank}: Loaded existing transforms file with {len(all_transforms)} entries")
    
    success_count = 0
    save_interval = 50  # Save every 50 successful processes to prevent data loss
    
    for i, scene_path in enumerate(tqdm(worker_scenes, desc=f"Worker {rank} processing")):
        transform_data = compute_gripper2wrist_transform(scene_path)
        if transform_data:
            # Use UUID as key in the dictionary
            uuid = transform_data['uuid']
            all_transforms[uuid] = transform_data
            success_count += 1

            # Periodically save results to prevent data loss
            if success_count % save_interval == 0:
                with open(output_file, 'w') as f:
                    json.dump(all_transforms, f, indent=2)
                print(f"Worker {rank}: Saved {len(all_transforms)} transforms (intermediate save)")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(all_transforms, f, indent=2)
    
    print(f"Worker {rank}: Saved {len(all_transforms)} transforms to {output_file}")
    
    return success_count

def analyze_transforms(json_dir, output_file, outlier_threshold=2.0):
    """
    Analyze the computed transformations across multiple scenes, grouped by robot_serial.
    This function is meant to be run after all the JSON files have been created by workers.
    
    Args:
        json_dir (str): Directory containing the JSON files with transformations
        output_file (str): File to save the analysis results
        outlier_threshold (float): Number of standard deviations to use for outlier rejection
        
    Returns:
        dict: Dictionary containing analysis results by robot serial
    """
    # Find all worker JSON files in the directory
    json_files = glob.glob(os.path.join(json_dir, "transforms_worker*.json"))
    if not json_files:
        print(f"No worker JSON files found in {json_dir}")
        return None
    
    print(f"Found {len(json_files)} worker JSON files for analysis")
    
    # Group transformations by robot_serial
    transforms_by_robot = {}
    total_transforms = 0
    
    for json_file in tqdm(json_files, desc="Reading JSON files"):
        with open(json_file, 'r') as f:
            worker_data = json.load(f)

        print(f"Processing {json_file} with {len(worker_data)} transforms")

        # Iterate through all transforms in this worker's file
        for uuid, data in worker_data.items():
            robot_serial = data['robot_serial']
            if robot_serial not in transforms_by_robot:
                transforms_by_robot[robot_serial] = []

            # Convert lists back to numpy arrays
            data['gripper2wrist_mat'] = np.array(data['gripper2wrist_mat'])
            data['gripper2wrist_quat'] = np.array(data['gripper2wrist_quat'])

            transforms_by_robot[robot_serial].append(data)
            total_transforms += 1
    
    print(f"Total transforms found: {total_transforms}")
    print(f"Transforms grouped by {len(transforms_by_robot)} robot serials")
    
    # Process each robot serial group
    results_by_robot = {}
    for robot_serial, transforms in transforms_by_robot.items():
        if len(transforms) < 3:
            print(f"Skipping robot {robot_serial} with only {len(transforms)} samples")
            continue
            
        print(f"Processing robot {robot_serial} with {len(transforms)} samples")
        
        # Extract transformation matrices and quaternions
        mats = np.array([t['gripper2wrist_mat'] for t in transforms])
        quats = np.array([t['gripper2wrist_quat'] for t in transforms])
        
        # Step 1: Filter outliers in translation
        translations = quats[:, :3]
        translation_mean = np.mean(translations, axis=0)
        translation_std = np.std(translations, axis=0)
        
        # Calculate Mahalanobis distance for translation (simplified)
        translation_dists = np.sqrt(np.sum(((translations - translation_mean) / translation_std) ** 2, axis=1))
        translation_mask = translation_dists < outlier_threshold
        
        # Step 2: Filter outliers in rotation
        # For rotation, we'll use angular deviation from the mean
        angular_deviations = []
        
        # First compute a mean rotation (temporary, before filtering)
        rot_mats = np.array([mat[:3, :3] for mat in mats])
        mean_rot_mat = np.mean(rot_mats, axis=0)
        u, _, vh = np.linalg.svd(mean_rot_mat)
        mean_rot_mat = u @ vh
        
        # Calculate angular deviations
        for rot_mat in rot_mats:
            rel_rot = rot_mat @ mean_rot_mat.T
            rel_rot_quat = mat2quat(rel_rot)
            angle = 2 * np.arccos(np.clip(np.abs(rel_rot_quat[3]), 0, 1))  # Extract angle from w component
            angular_deviations.append(angle)
        
        angular_deviations = np.array(angular_deviations)
        angular_mean = np.mean(angular_deviations)
        angular_std = np.std(angular_deviations)
        
        # Special case: If all matrices are nearly identical (std close to 0),
        # skip the rotation filtering by accepting all samples
        if angular_std < 1e-10:
            print(f"  All rotation matrices are nearly identical (angular std={angular_std})")
            print(f"  Skipping rotation filtering")
            angular_mask = np.ones_like(angular_deviations, dtype=bool)
        else:
            angular_mask = angular_deviations < (angular_mean + outlier_threshold * angular_std)
        
        # Combine masks for final filtering
        combined_mask = translation_mask & angular_mask
        
        # Log statistics about outlier rejection
        print(f"  Total samples: {len(transforms)}")
        print(f"  Samples after translation filter: {np.sum(translation_mask)}")
        print(f"  Samples after rotation filter: {np.sum(angular_mask)}")
        print(f"  Final samples after filtering: {np.sum(combined_mask)}")
        
        if np.sum(combined_mask) < 3:
            print(f"  Too few samples remaining after filtering ({np.sum(combined_mask)}), skipping robot {robot_serial}")
            import pdb; pdb.set_trace()
            continue
        
        # Apply filter
        filtered_mats = mats[combined_mask]
        filtered_quats = quats[combined_mask]
        
        # Re-compute mean with filtered data
        filtered_translations = filtered_quats[:, :3]
        mean_translation = np.mean(filtered_translations, axis=0)
        
        filtered_rot_mats = np.array([mat[:3, :3] for mat in filtered_mats])
        mean_rot_mat = np.mean(filtered_rot_mats, axis=0)
        u, _, vh = np.linalg.svd(mean_rot_mat)
        mean_rot_mat = u @ vh
        
        # Create mean transformation matrix
        mean_mat = np.eye(4)
        mean_mat[:3, :3] = mean_rot_mat
        mean_mat[:3, 3] = mean_translation
        
        # Convert to quaternion
        mean_quat = np.concatenate([mean_translation, mat2quat(mean_rot_mat)])
        
        # Calculate statistics for the filtered data
        translation_std = np.std(filtered_translations, axis=0)
        
        # Recompute angular deviations with the final mean
        angular_deviations = []
        for rot_mat in filtered_rot_mats:
            rel_rot = rot_mat @ mean_rot_mat.T
            rel_rot_quat = mat2quat(rel_rot)
            angle = 2 * np.arccos(np.clip(np.abs(rel_rot_quat[3]), 0, 1))
            angular_deviations.append(angle)
        
        angular_deviations = np.array(angular_deviations)
        
        # Store results
        results_by_robot[robot_serial] = {
            'mean_mat': mean_mat.tolist(),
            'mean_quat': mean_quat.tolist(),
            'translation_std': translation_std.tolist(),
            'angular_deviation_mean_rad': float(np.mean(angular_deviations)),
            'angular_deviation_std_rad': float(np.std(angular_deviations)),
            'angular_deviation_mean_deg': float(np.rad2deg(np.mean(angular_deviations))),
            'angular_deviation_std_deg': float(np.rad2deg(np.std(angular_deviations))),
            'num_samples_before_filtering': len(transforms),
            'num_samples_after_filtering': int(np.sum(combined_mask)),
            'wrist_cam_serial': transforms[0]['wrist_cam_serial'],
        }
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results_by_robot, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary for each robot
    print("\nSummary by Robot Serial:")
    for robot_serial, results in results_by_robot.items():
        print(f"\nRobot Serial: {robot_serial}")
        print(f"Wrist Camera Serial: {results['wrist_cam_serial']}")
        print(f"Number of samples: {results['num_samples_after_filtering']} (filtered from {results['num_samples_before_filtering']})")
        
        mean_translation = np.array(results['mean_quat'][:3])
        translation_std = np.array(results['translation_std'])
        
        print("\nMean Translation (meters):")
        print(f"X: {mean_translation[0]:.6f} ± {translation_std[0]:.6f}")
        print(f"Y: {mean_translation[1]:.6f} ± {translation_std[1]:.6f}")
        print(f"Z: {mean_translation[2]:.6f} ± {translation_std[2]:.6f}")
        
        print("\nRotation Statistics:")
        print(f"Mean Angular Deviation: {results['angular_deviation_mean_deg']:.2f}° ± {results['angular_deviation_std_deg']:.2f}°")
        
        # # Print the mean transformation matrix
        # print("\nMean Gripper to Wrist Camera Transformation Matrix (4x4):")
        # np.set_printoptions(precision=6, suppress=True)
        # print(np.array(results['mean_mat']))
    
    return results_by_robot

def main():
    parser = argparse.ArgumentParser(description='Compute and analyze gripper to wrist camera transformations')
    parser.add_argument('--scenes_file', type=str, default=None,
                        help='Path to file containing list of valid scenes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save transformation data')
    parser.add_argument('--max_scenes', type=int, default=-1,
                        help='Maximum number of scenes to process')
    parser.add_argument('--rank', type=int, default=0,
                        help='Worker rank (0 to world_size-1)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of parallel workers')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Skip processing scenes and only analyze existing JSON files')
    parser.add_argument('--analysis_output', type=str, default=None,
                        help='Output file for analysis results (only used with --analyze_only)')
    parser.add_argument('--outlier_threshold', type=float, default=2.0,
                        help='Number of std deviations to use for outlier rejection')
    args = parser.parse_args()
    
    if args.analyze_only:
        if args.analysis_output is None:
            args.analysis_output = os.path.join(args.output_dir, "gripper2wrist_transforms.json")
        
        print(f"Running analysis only on JSON files in {args.output_dir}")
        analyze_transforms(args.output_dir, args.analysis_output, args.outlier_threshold)
        return
    else:
        assert args.scenes_file is not None, "Must provide a list of scenes to process"

    # Load list of valid scenes
    if '.txt' in args.scenes_file:
        with open(args.scenes_file, 'r') as f:
            scene_paths = [line.strip() for line in f if line.strip()]
    else:
        scene_paths = glob.glob(os.path.join(args.scenes_file, "*/*/*/*"))
        scene_paths = sorted(scene_paths)
    
    if args.max_scenes > 0:
        scene_paths = scene_paths[:args.max_scenes]
    
    if args.world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}")

    print(f"Worker {args.rank} will process {len(scene_paths) // args.world_size + (1 if args.rank < len(scene_paths) % args.world_size else 0)} scenes")
    
    # Process scenes
    success_count = process_scenes(scene_paths, args.output_dir, args.rank, args.world_size)
    
    print(f"Worker {args.rank} finished. Successfully processed {success_count} scenes.")
    
    # Note: Analysis part is handled by a separate function, which will typically
    # be run after all workers have completed. Uncomment to run the analysis.
    """
    if args.rank == 0:  # Only the master rank should do the analysis
        print("All workers have completed. Running final analysis...")
        analysis_output = os.path.join(args.output_dir, "gripper2wrist_transforms.json")
        analyze_transforms(args.output_dir, analysis_output)
    """

if __name__ == "__main__":
    main()
