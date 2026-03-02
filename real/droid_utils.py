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
# point-world imports: ensure project root is importable regardless of CWD
import sys as _sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)
import cv2
from tqdm import tqdm
import json
import h5py
import glob
import transform_utils
import tempfile
import shutil
from real.gcs_utils import is_gcs_path, get_local_path, list_gcs_files
try:
    import pyzed.sl as sl
except ImportError as exc:
    sl = None
    _PYZED_IMPORT_ERROR = exc
else:
    _PYZED_IMPORT_ERROR = None


def _require_pyzed():
    if sl is None:
        raise ImportError("pyzed is required for ZED SVO processing") from _PYZED_IMPORT_ERROR


def _binary_search_latest_range(arr, left: int, right: int, target):
    if arr[right] <= target or right == left:
        return arr[right]
    mid = ((left + right) >> 1) + 1
    if arr[mid] <= target:
        return _binary_search_latest_range(arr, mid, right, target)
    return _binary_search_latest_range(arr, left, mid - 1, target)


def _binary_search_latest(arr, target):
    if len(arr) <= 0:
        raise ValueError("input array should contain at least one element")
    return _binary_search_latest_range(arr, 0, len(arr) - 1, target)


def _binary_search_closest(arr, target):
    """adapted from rh20t_api"""
    if target in arr:
        return target
    prev_idx = arr.index(_binary_search_latest(arr, target))
    if prev_idx == len(arr) - 1:
        return arr[prev_idx]
    prev_val = arr[prev_idx]
    next_val = arr[prev_idx + 1]
    return prev_val if abs(prev_val - target) < abs(next_val - target) else next_val


def _resolve_svo_path(scene_path, svo_path):
    if not svo_path.startswith('/') and not is_gcs_path(svo_path):
        return os.path.join(scene_path, *svo_path.split('/')[-3:])
    return svo_path


def _init_camera_entries(metadata, include_wrist_cam):
    camera_names = ['wrist', 'ext1', 'ext2'] if include_wrist_cam else ['ext1', 'ext2']
    data_dict = {}
    camera_specs = []
    for camera_name in camera_names:
        serial_key = f'{camera_name}_cam_serial'
        if serial_key not in metadata:
            raise KeyError(f"Camera {camera_name} not found in metadata")
        camera_serial = metadata[serial_key]
        data_dict[camera_serial] = {}
        if camera_name == 'wrist':
            data_dict[camera_serial]['extrinsic'] = np.eye(4)
        else:
            extrinsic_key = f'{camera_name}_cam_extrinsics'
            if extrinsic_key not in metadata:
                raise KeyError(f"Extrinsics for camera {camera_name} not found in metadata")
            extrinsic_6 = np.array(metadata[extrinsic_key])
            extrinsic_4x4 = transform_utils.convert_pose_euler2mat(extrinsic_6[None])[0]
            extrinsic_4x4 = np.linalg.inv(extrinsic_4x4)  # cam2world -> world2cam
            data_dict[camera_serial]['extrinsic'] = extrinsic_4x4
        camera_specs.append((camera_name, camera_serial))
    return data_dict, camera_specs


def _open_svo_camera(local_svo_path, include_depth):
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(local_svo_path)
    init_params.svo_real_time_mode = False
    if include_depth:
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    else:
        init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ValueError(f"Error opening SVO file {local_svo_path}: {err}")
    camera_info = zed.get_camera_information()
    return zed, camera_info


def _extract_intrinsics_and_baseline(calibration_params, downscale_ratio):
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy

    intrinsic = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )

    baseline = calibration_params.stereo_transform.get_translation().get()[0]

    if downscale_ratio != 1.0:
        intrinsic[0, 0] *= downscale_ratio
        intrinsic[1, 1] *= downscale_ratio
        intrinsic[0, 2] *= downscale_ratio
        intrinsic[1, 2] *= downscale_ratio

    return intrinsic, baseline


def _extract_svo_frames(zed, camera_name, include_stereo, include_depth, downscale_ratio, max_frames):
    rgb_frames = []
    right_frames = [] if include_stereo else None
    depth_frames = [] if include_depth else None

    left_image = sl.Mat()
    right_image = sl.Mat() if include_stereo else None
    depth_image = sl.Mat() if include_depth else None
    runtime_params = sl.RuntimeParameters()

    nb_frames = zed.get_svo_number_of_frames()
    if max_frames > 0:
        nb_frames = min(nb_frames, max_frames)

    print(f"Extracting {nb_frames} frames from SVO file for camera {camera_name}...")

    timestamps = []
    frame_count = 0
    with tqdm(total=nb_frames, desc=f"Processing {camera_name} camera") as pbar:
        while frame_count < nb_frames:
            err = zed.grab(runtime_params)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                rgb = left_image.get_data().copy()
                if rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
                if downscale_ratio != 1.0:
                    rgb = cv2.resize(
                        rgb,
                        None,
                        fx=downscale_ratio,
                        fy=downscale_ratio,
                        interpolation=cv2.INTER_LINEAR,
                    )
                rgb_frames.append(rgb)

                ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                timestamps.append(ts)

                if include_stereo:
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                    right = right_image.get_data().copy()
                    if right.shape[2] == 4:
                        right = cv2.cvtColor(right, cv2.COLOR_BGRA2RGB)
                    if downscale_ratio != 1.0:
                        right = cv2.resize(
                            right,
                            None,
                            fx=downscale_ratio,
                            fy=downscale_ratio,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    right_frames.append(right)

                if include_depth:
                    zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                    depth = depth_image.get_data().copy()
                    if downscale_ratio != 1.0:
                        depth = cv2.resize(
                            depth,
                            None,
                            fx=downscale_ratio,
                            fy=downscale_ratio,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    depth_frames.append(depth)

                frame_count += 1
                pbar.update(1)
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print(f"End of SVO file reached for camera {camera_name}")
                break
            else:
                raise ValueError(f"Error grabbing frame from SVO for camera {camera_name}: {err}")

    if not rgb_frames:
        raise ValueError(f"No frames extracted for camera {camera_name}")

    payload = {
        'rgb': np.stack(rgb_frames, axis=0),
        'timestamps': np.array(timestamps),
    }

    if include_stereo:
        payload['right_frames'] = np.stack(right_frames, axis=0)
    if include_depth:
        payload['depth'] = np.stack(depth_frames, axis=0)
        payload['depth'] = np.nan_to_num(
            payload['depth'], nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )

    return payload


def filter_by_timestamps(data, canonical_timestamps, scene_path, is_proprio=False, verbose=False):
    """
    Filter data to match canonical timestamps using binary search for closest match.
    
    Args:
        data (dict): Either data_dict or proprio_dict
        canonical_timestamps (np.ndarray): The canonical timestamps to align to
        scene_path (str): Path to the scene directory
        is_proprio (bool): Whether the data is proprioception data
        verbose (bool): Whether to print verbose output
    Returns:
        dict: Filtered data aligned with canonical timestamps
    """
    filtered_data = {}
    
    if is_proprio:
        # Handle proprioception data
        temp_dir = None
        try:
            if is_gcs_path(scene_path):
                temp_dir = tempfile.mkdtemp()
                
            trajectory_path = os.path.join(scene_path, "trajectory.h5")
            local_trajectory_path = get_local_path(trajectory_path, temp_dir)
            
            with h5py.File(local_trajectory_path, 'r') as f:
                # Get proprioception timestamps
                proprio_timestamps = np.array(f['observation']['timestamp']['robot_state']['read_end'])
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # Convert timestamps to list for binary search
        proprio_timestamps_list = proprio_timestamps.tolist()
        
        # Create new dict with aligned data
        filtered_data = {}
        modality_errors = {}
        for modality in data:
            if modality in ['pre_sampled_points']:
                # Copy non-temporal data directly
                filtered_data[modality] = data[modality]
            else:
                assert len(data[modality]) == len(proprio_timestamps_list), f"Length mismatch for {modality}: {len(data[modality])} != {len(proprio_timestamps_list)}"
                # For temporal data, align to canonical timestamps
                aligned_data = []
                errors = []
                for target_ts in canonical_timestamps:
                    # Find closest timestamp
                    closest_ts = _binary_search_closest(proprio_timestamps_list, target_ts)
                    idx = proprio_timestamps_list.index(closest_ts)
                    aligned_data.append(data[modality][idx])
                    if verbose:
                        errors.append(abs(target_ts - closest_ts))
                filtered_data[modality] = np.array(aligned_data)
                if verbose and len(errors) > 0:
                    modality_errors[modality] = np.array(errors) / 1000.0  # Convert to seconds
    else:
        # Handle camera data
        modality_errors = {}
        nontemporal_keys = ['intrinsic', 'extrinsic', 'baseline']
        for camera_serial in data:
            filtered_data[camera_serial] = {}
            
            # Copy non-temporal data directly
            for key in nontemporal_keys:
                if key in data[camera_serial]:
                    filtered_data[camera_serial][key] = data[camera_serial][key]
            
            camera_timestamps_list = data[camera_serial]['timestamps'].tolist()
            
            # For temporal data, align to canonical timestamps
            for key in data[camera_serial]:
                if key in nontemporal_keys:
                    continue
                assert len(data[camera_serial][key]) == len(camera_timestamps_list), f"Length mismatch for {key}: {len(data[camera_serial][key])} != {len(camera_timestamps_list)}"
                aligned_data = []
                errors = []
                for target_ts in canonical_timestamps:
                    # Find closest timestamp
                    closest_ts = _binary_search_closest(camera_timestamps_list, target_ts)
                    idx = camera_timestamps_list.index(closest_ts)
                    aligned_data.append(data[camera_serial][key][idx])
                    if verbose:
                        errors.append(abs(target_ts - closest_ts))
                filtered_data[camera_serial][key] = np.array(aligned_data)
                if verbose and len(errors) > 0:
                    modality_errors[f"{camera_serial}_{key}"] = np.array(errors) / 1000.0  # Convert to seconds
    
    if verbose:
        for modality, errors in modality_errors.items():
            print(f"{modality} alignment errors (seconds):")
            print(f"  Average: {np.mean(errors):.3f}")
            print(f"  Max: {np.max(errors):.3f}")
            print(f"  Min: {np.min(errors):.3f}")

    return filtered_data

def gather_data_dict(scene_path, downscale_ratio=1.0, include_stereo=False, include_depth=True, include_wrist_cam=False, max_frames=-1):
    """
    Gather data from DROID dataset.
    
    Args:
        scene_path (str): Path to the scene directory
        downscale_ratio (float): Ratio to downscale images (must be <= 1.0)
        include_stereo (bool): Whether to include right stereo frames and baseline for depth estimation
        include_depth (bool): Whether to include depth frames from SVO
        include_wrist_cam (bool): Whether to include the wrist camera data
        max_frames (int): Maximum number of frames to extract (-1 for all frames)
            
    Returns:
        dict: Dictionary with the following structure:
        {
            camera_serial_1: {
                'intrinsic': np.array,  # (3, 3)
                'extrinsic': np.array,  # (4, 4)
                'rgb': np.array,  # (T, H, W, 3)
                'depth': np.array,  # (T, H, W) - Only if include_depth=True
                'right_frames': np.array,  # (T, H, W, 3) - Only if include_stereo=True
                'baseline': float,  # Stereo camera baseline in meters - Only if include_stereo=True
                'timestamps': np.array,  # (T,)
            },
            camera_serial_2: ...
        }
    """
    _require_pyzed()
    data_dict = {}
    temp_dir = None
    
    try:
        # Create a temporary directory if working with GCS paths
        if is_gcs_path(scene_path):
            temp_dir = tempfile.mkdtemp()
            
        # Load metadata files to get camera information
        metadata = get_metadata(scene_path)

        data_dict, camera_specs = _init_camera_entries(metadata, include_wrist_cam)

        # Process each SVO file
        for camera_name, camera_serial in camera_specs:
            if f'{camera_name}_svo_path' not in metadata:
                raise KeyError(f"SVO path for camera {camera_name} not found in metadata")
            svo_path = metadata[f'{camera_name}_svo_path']
            resolved_svo_path = _resolve_svo_path(scene_path, svo_path)
            local_svo_path = get_local_path(resolved_svo_path, temp_dir)

            print(f"Processing SVO file for camera {camera_name}: {local_svo_path}")

            zed, camera_info = _open_svo_camera(local_svo_path, include_depth)
            try:
                calibration_params = camera_info.camera_configuration.calibration_parameters
                intrinsic, baseline = _extract_intrinsics_and_baseline(
                    calibration_params, downscale_ratio
                )
                data_dict[camera_serial]['intrinsic'] = intrinsic
                if include_stereo:
                    data_dict[camera_serial]['baseline'] = baseline

                width = camera_info.camera_configuration.resolution.width
                height = camera_info.camera_configuration.resolution.height
                print(f"Camera {camera_name} resolution: {width}x{height}")

                frames_payload = _extract_svo_frames(
                    zed,
                    camera_name,
                    include_stereo=include_stereo,
                    include_depth=include_depth,
                    downscale_ratio=downscale_ratio,
                    max_frames=max_frames,
                )
                data_dict[camera_serial].update(frames_payload)
            finally:
                zed.close()

            print(f"Extracted {data_dict[camera_serial]['rgb'].shape[0]} frames for camera {camera_serial}")
    finally:
        # Clean up temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return data_dict

def gather_trajectory(scene_path, max_frames=-1):
    temp_dir = None
    try:
        if is_gcs_path(scene_path):
            temp_dir = tempfile.mkdtemp()
            
        trajectory_path = os.path.join(scene_path, "trajectory.h5")
        local_trajectory_path = get_local_path(trajectory_path, temp_dir)
        
        proprio_dict = {}
        with h5py.File(local_trajectory_path, 'r') as f:
            joint_positions = np.array(f['observation']['robot_state']['joint_positions'])
            joint_velocities = np.array(f['observation']['robot_state']['joint_velocities'])
            joint_torques = np.array(f['observation']['robot_state']['joint_torques_computed'])
            gripper_positions = np.array(f['observation']['robot_state']['gripper_position'])  # [0, 1]
            gripper_pose_6 = np.array(f['observation']['robot_state']['cartesian_position'])
            gripper_pose_7 = transform_utils.convert_pose_euler2quat(gripper_pose_6)
            # Get timestamps
            proprio_timestamps = np.array(f['observation']['timestamp']['robot_state']['read_end'])
            
            # Limit frames if max_frames is specified
            if max_frames > 0:
                joint_positions = joint_positions[:max_frames]
                joint_velocities = joint_velocities[:max_frames]
                joint_torques = joint_torques[:max_frames]
                gripper_positions = gripper_positions[:max_frames]
                gripper_pose_7 = gripper_pose_7[:max_frames]
                proprio_timestamps = proprio_timestamps[:max_frames]
            
            proprio_dict['joint_positions'] = joint_positions
            proprio_dict['joint_velocities'] = joint_velocities
            proprio_dict['joint_torques'] = joint_torques
            proprio_dict['gripper_positions'] = gripper_positions * 0.725  # Map normalized gripper position (0-1) to finger_joint angle (0-0.725) for robotiq 2f85 gripper -- 0 means open, 1 means closed
            proprio_dict['gripper_pose'] = gripper_pose_7
            proprio_dict['timestamps'] = proprio_timestamps

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return proprio_dict

def get_uuid(scene_path):
    """Extract UUID directly from metadata filename without downloading the file."""
    if is_gcs_path(scene_path):
        # List metadata files to get the filename
        metadata_files = list_gcs_files(scene_path, "metadata_*.json")

        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found in {scene_path}")

        # Extract UUID from filename: metadata_<uuid>.json
        metadata_filename = os.path.basename(metadata_files[0])
        if not metadata_filename.startswith("metadata_") or not metadata_filename.endswith(".json"):
            raise ValueError(f"Unexpected metadata filename format: {metadata_filename}")

        uuid = metadata_filename[9:-5]  # Remove "metadata_" prefix and ".json" suffix
        return uuid

    # Local file handling
    metadata_files = glob.glob(os.path.join(scene_path, "metadata_*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {scene_path}")

    # Extract UUID from filename: metadata_<uuid>.json
    metadata_filename = os.path.basename(metadata_files[0])
    if not metadata_filename.startswith("metadata_") or not metadata_filename.endswith(".json"):
        raise ValueError(f"Unexpected metadata filename format: {metadata_filename}")

    uuid = metadata_filename[9:-5]  # Remove "metadata_" prefix and ".json" suffix
    return uuid

def get_metadata(scene_path):
    temp_dir = None
    try:
        if is_gcs_path(scene_path):
            temp_dir = tempfile.mkdtemp()
            
            # List all metadata files in the GCS path
            metadata_files = list_gcs_files(scene_path, "metadata_*.json")
            
            if not metadata_files:
                raise FileNotFoundError(f"No metadata files found in {scene_path}")
                
            # Download the first metadata file
            local_metadata_path = get_local_path(metadata_files[0], temp_dir)
            
            # Load the metadata
            with open(local_metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Local file handling
            metadata_files = glob.glob(os.path.join(scene_path, "metadata_*.json"))
            if not metadata_files:
                raise FileNotFoundError(f"No metadata files found in {scene_path}")
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
                
        return metadata
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def load_robot_transforms(transforms_file):
    """
    Load the robot-specific gripper to wrist camera transformations from the analysis file.
    
    Returns:
        dict: Dictionary mapping robot_serial to the corresponding transformation matrix
    """
    robot_transforms = {}
    if os.path.exists(transforms_file):
        with open(transforms_file, 'r') as f:
            transforms_data = json.load(f)
        
        # Convert the stored transforms to numpy arrays
        for robot_serial, data in transforms_data.items():
            robot_transforms[robot_serial] = np.array(data['mean_mat'])
        
        print(f"Loaded gripper2wrist transforms for {len(robot_transforms)} robots.")
    else:
        raise FileNotFoundError(f"Transform file not found: {transforms_file}")
    
    return robot_transforms

def get_robot_serial(scene_path):
    metadata = get_metadata(scene_path)
    return metadata['robot_serial']
