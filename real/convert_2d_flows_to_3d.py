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
import json
import time
import h5py
import numpy as np
import cv2
from tqdm import tqdm

from real.real_utils import (
    get_time_str,
    generate_and_project_workspace_boundary,
    project_points_to_image,
    get_mesh_name as _lu_get_mesh_name,
)
from real.gcs_utils import enforce_gcs_cache_policy
from real.droid_utils import get_metadata, get_uuid
from real.workspace import (
    WORKSPACE_BOUNDS_MIN,
    WORKSPACE_BOUNDS_MAX,
    WORKSPACE_BOUNDS_MIN_RELAXED,
    WORKSPACE_BOUNDS_MAX_RELAXED,
)
from robot_sampler import RobotSampler

# CPU-only constants
MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 4.0
URDF_PATH = "../assets/franka_description/franka_panda_robotiq_2f85.urdf"
from real.flow_postprocessing import remove_outlier_flows
from shared.data_contract import EXPECTED_CAMERA_PAYLOAD_SHAPES
from shared.h5_io import load_rgb_from_jpeg_in_h5, save_rgb_as_jpeg_in_h5, save_depth_as_uint16_mm

QUANTIZED_NORMALS_DTYPE = np.int8


def _quantize_unit_normals_to_int8(normals: np.ndarray) -> np.ndarray:
    """Quantize normals from [-1, 1] float space into int8 [-127, 127]."""
    normals_f32 = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(normals_f32).all():
        raise ValueError("scene_normals contains non-finite values")
    clipped = np.clip(normals_f32, -1.0, 1.0)
    return np.rint(clipped * 127.0).astype(QUANTIZED_NORMALS_DTYPE)


def _resize_initial_depth_to_contract(depth_image_meters: np.ndarray) -> np.ndarray:
    """Resize depth image to the release contract resolution if needed."""
    target_h, target_w = EXPECTED_CAMERA_PAYLOAD_SHAPES["initial_depth"]
    depth_f32 = np.asarray(depth_image_meters, dtype=np.float32)
    if depth_f32.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H,W), got {depth_f32.shape}")
    if depth_f32.shape == (target_h, target_w):
        return depth_f32
    return cv2.resize(depth_f32, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _resize_initial_rgb_to_contract(rgb_image: np.ndarray) -> np.ndarray:
    """Resize RGB image to the release contract resolution if needed."""
    target_h, target_w = EXPECTED_CAMERA_PAYLOAD_SHAPES["initial_depth"]
    rgb_u8 = np.asarray(rgb_image)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {rgb_u8.shape}")
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    if rgb_u8.shape[:2] == (target_h, target_w):
        return rgb_u8
    return cv2.resize(rgb_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _scale_intrinsic_for_resolution(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Scale pinhole intrinsics from source image grid to target image grid."""
    intr = np.asarray(intrinsic, dtype=np.float32)
    if intr.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3,3), got {intr.shape}")

    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    dst_h, dst_w = int(target_hw[0]), int(target_hw[1])
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Invalid source/target resolution for intrinsic scaling: "
            f"source={source_hw}, target={target_hw}"
        )
    if (src_h, src_w) == (dst_h, dst_w):
        return intr

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    intr_scaled = intr.copy()
    intr_scaled[0, 0] *= sx
    intr_scaled[0, 2] *= sx
    intr_scaled[1, 1] *= sy
    intr_scaled[1, 2] *= sy
    return intr_scaled


class Flow2DTo3DConverter:
    """
    CPU-only converter that loads cached 2D flows and metadata, re-applies
    masks post hoc using specified extrinsics+depth, lifts to 3D, and writes
    the same H5 schema consumed by the training pipeline.

    Important:
    - No GPU smoothing or optimization here.
    - Robot mask logic mirrors the legacy pipeline.
    - Masks are applied post hoc; we use frame-0 mask parity (seed frame policy).
    """

    def __init__(self, urdf_path: str = None, robot_mask_seed: int | None = None):
        # Use RobotSampler for FK and mesh rasterization (CPU-friendly)
        if urdf_path is None:
            urdf_path = URDF_PATH
        # Resolve relative URDF path w.r.t. this file to be robust to CWD
        if not os.path.isabs(urdf_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.normpath(os.path.join(base_dir, urdf_path))
        self.robot_sampler = RobotSampler(domain='droid', urdf_path=urdf_path)
        self.robot_mask_seed = robot_mask_seed
        self.mesh_presampled_points = {}
        self.mesh_presampled_ready = False
        self.domain = 'droid'

    def _load_2d_flows(self, cache_path):
        with h5py.File(cache_path, 'r') as f:
            if 'write_complete' not in f.attrs:
                raise RuntimeError(f"2d flows cache missing write_complete attr: {cache_path}")
            if not bool(f.attrs['write_complete']):
                raise RuntimeError(f"2d flows cache is incomplete: {cache_path}")
            uuid = f.attrs['uuid']
            scene_path = f.attrs['scene_path']
            canonical_timestamps = np.array(json.loads(f.attrs['canonical_timestamps']))
            frames_per_clip = int(f.attrs['frames_per_clip'])
            skip_every = int(f.attrs['skip_every'])
            downscale_ratio = float(f.attrs['downscale_ratio'])
            space_skip_ratio = int(f.attrs['space_skip_ratio'])
            tracker_type = f.attrs['tracker_type']
            def _as_str(v):
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8")
                return str(v)

            tracking_mask_mode = _as_str(f.attrs.get('tracking_mask_mode', 'none'))
            tracking_mask_extrinsics_source = _as_str(
                f.attrs.get('tracking_mask_extrinsics_source', 'unknown')
            )
            # thresholds (must be present; do not fall back silently)
            required = [
                'ee_pos_threshold',
                'ee_rot_threshold',
                'gripper_closed_ee_pos_threshold',
                'gripper_closed_ee_rot_threshold',
            ]
            missing = [k for k in required if k not in f.attrs]
            if missing:
                raise RuntimeError(
                    "2d flows cache is missing required threshold attributes: "
                    + ", ".join(missing)
                    + ". Please regenerate the 2d flows cache with the updated runner so thresholds are saved."
                )
            ee_pos_threshold = float(f.attrs['ee_pos_threshold'])
            ee_rot_threshold = float(f.attrs['ee_rot_threshold'])
            gripper_closed_ee_pos_threshold = float(f.attrs['gripper_closed_ee_pos_threshold'])
            gripper_closed_ee_rot_threshold = float(f.attrs['gripper_closed_ee_rot_threshold'])
            cams = {}
            for cam_key in f['cameras'].keys():
                g_cam = f['cameras'][cam_key]
                intrinsic = np.array(g_cam['intrinsic'])
                clips = {}
                for clip_key in g_cam['clips'].keys():
                    g_clip = g_cam['clips'][clip_key]
                    initial_rgb = load_rgb_from_jpeg_in_h5(g_clip['initial_rgb'])
                    clip = {
                        'H': int(g_clip.attrs['H']),
                        'W': int(g_clip.attrs['W']),
                        'flows_2d_xy': np.array(g_clip['flows_2d_xy']),
                        'flows_2d_visibility': np.array(g_clip['flows_2d_visibility']).astype(bool),
                        'flow_colors': np.array(g_clip['flow_colors']).astype(np.uint8),
                        'initial_rgb': initial_rgb.astype(np.uint8),
                    }
                    clips[clip_key] = clip
                cams[cam_key] = {'intrinsic': intrinsic, 'clips': clips}

            proprio = {}
            for clip_key in f['proprio'].keys():
                g_p = f['proprio'][clip_key]
                proprio[clip_key] = {
                    'joint_positions': np.array(g_p['joint_positions']),
                    'joint_velocities': np.array(g_p['joint_velocities']),
                    'joint_torques': np.array(g_p['joint_torques']),
                    'gripper_positions': np.array(g_p['gripper_positions']),
                    'gripper_pose': np.array(g_p['gripper_pose']),
                }

        return {
            'uuid': uuid,
            'scene_path': scene_path,
            'canonical_timestamps': canonical_timestamps,
            'frames_per_clip': frames_per_clip,
            'skip_every': skip_every,
            'downscale_ratio': downscale_ratio,
            'space_skip_ratio': space_skip_ratio,
            'tracker_type': tracker_type,
            'tracking_mask_mode': tracking_mask_mode,
            'tracking_mask_extrinsics_source': tracking_mask_extrinsics_source,
            'ee_pos_threshold': ee_pos_threshold,
            'ee_rot_threshold': ee_rot_threshold,
            'gripper_closed_ee_pos_threshold': gripper_closed_ee_pos_threshold,
            'gripper_closed_ee_rot_threshold': gripper_closed_ee_rot_threshold,
            'cameras': cams,
            'proprio': proprio,
        }

    def _load_extrinsics_json(self, cameras_json_path, camera_serials, source: str):
        with open(cameras_json_path, 'r') as f:
            camera_data = json.load(f)
        key = f"{source}_extrinsics"
        world2cam = {}
        for serial in camera_serials:
            assert serial in camera_data, f"Camera {serial} not in {cameras_json_path}"
            assert key in camera_data[serial], f"{key} missing for {serial} in {cameras_json_path}"
            world2cam[serial] = np.array(camera_data[serial][key], dtype=np.float32)
        return world2cam

    def _load_extrinsics(self, output_dir, uuid, camera_serials, extrinsics_source: str):
        assert extrinsics_source in ("vggt", "optimized"), "Invalid extrinsics_source"
        cameras_json_path = os.path.join(output_dir, 'cameras', f'{uuid}_cameras.json')
        assert os.path.exists(cameras_json_path), f"Cameras JSON not found: {cameras_json_path}"
        return self._load_extrinsics_json(cameras_json_path, camera_serials, extrinsics_source)

    def _load_depth_for_camera(self, depth_h5_path, cam_group_name, target_timestamps):
        """
        Depth loader for cases that require dense grids (full-frame operations):
        reads only frames needed to align with target_timestamps and returns
        (T,H,W) float32 in meters.
        """
        with h5py.File(depth_h5_path, 'r') as f:
            assert cam_group_name in f, f"Camera {cam_group_name} not found in depth file"
            g = f[cam_group_name]
            depth_ds = g['depth']            # HDF5 dataset: (N, H, W), uint16 in mm
            depth_ts = np.array(g['timestamps'])

            # Compute nearest-neighbor indices for each target timestamp
            idxs = []
            for ts in target_timestamps:
                idxs.append(int(np.argmin(np.abs(depth_ts - ts))))
            idxs = np.asarray(idxs, dtype=np.int64)

            # Fancy-index read only the required frames, convert to meters
            depth_uint16 = depth_ds[idxs, ...]
            depth_m = depth_uint16.astype(np.float32) / 1000.0
            return depth_m  # (T, H, W)

    def _sample_depth_values_for_tracks(self, depth_ds, depth_ts, target_timestamps, x_int, y_int):
        """
        Stream depth per-frame and sample only at track pixel locations to avoid
        allocating an entire (T,H,W) volume in memory.

        Args:
            depth_ds: h5py Dataset of shape (N, H, W), dtype=uint16 in millimeters
            depth_ts: numpy array of length N with timestamps
            target_timestamps: numpy array of length T (canonical timestamps for clip)
            x_int, y_int: integer pixel coords at depth resolution, shape (T, N_tracks)

        Returns:
            z_meters: (T, N_tracks) float32 sampled depth in meters
            depth_valid: (T, N_tracks) bool validity mask (0 < z <= MAX_DEPTH_M)
            Hd, Wd: ints, depth resolution
            first_frame_depth_sanitized: (1, H, W) float32 sanitized depth grid for the first aligned frame
        """
        Hd, Wd = depth_ds.shape[1:]
        T, N_tracks = x_int.shape

        # Map each target timestamp to nearest depth index once
        idxs = np.empty(T, dtype=np.int64)
        for i, ts in enumerate(target_timestamps):
            idxs[i] = int(np.argmin(np.abs(depth_ts - ts)))

        # Allocate outputs
        z = np.zeros((T, N_tracks), dtype=np.float32)
        depth_valid = np.zeros((T, N_tracks), dtype=bool)

        first_frame_depth_sanitized = None
        max_mm = int(MAX_DEPTH_M * 1000.0 + 0.5)

        # Stream per-frame to keep memory low
        for ti in range(T):
            di = int(idxs[ti])
            frame_mm = depth_ds[di, ...]  # (H,W) uint16
            # Sample at track locations in uint16, convert to meters after
            xi = x_int[ti]
            yi = y_int[ti]
            vals_mm = frame_mm[yi, xi]

            valid = (vals_mm > 0) & (vals_mm <= max_mm)
            depth_valid[ti] = valid
            z[ti] = (vals_mm.astype(np.float32) / 1000.0)

            # Save sanitized full grid for the first frame only
            if ti == 0:
                frame_m = frame_mm.astype(np.float32) / 1000.0
                frame_valid = (frame_mm > 0) & (frame_mm <= max_mm)
                first_frame_depth_sanitized = np.where(frame_valid, frame_m, 0.0)[None, ...].astype(np.float32)

        return z, depth_valid, Hd, Wd, first_frame_depth_sanitized

    def _compute_workspace_mask(self, intrinsic, extrinsic, height, width):
        from scipy.spatial import ConvexHull
        from skimage.draw import polygon

        mask = np.zeros((height, width), dtype=bool)
        face_density = 100
        pts_px = generate_and_project_workspace_boundary(
            WORKSPACE_BOUNDS_MIN.astype(np.float32),
            WORKSPACE_BOUNDS_MAX.astype(np.float32),
            face_density,
            extrinsic.astype(np.float32),
            intrinsic.astype(np.float32),
            width,
            height,
        )
        if len(pts_px) < 3:
            return np.ones_like(mask, dtype=bool)

        hull = ConvexHull(pts_px)
        vertices = np.round(pts_px[hull.vertices]).astype(int)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=(height, width))
        mask[rr, cc] = True
        return mask

    def _presample_mesh_points(self, fk_result, total_samples=100000, gripper_multiplier=2.0):
        """Pre-sample points per mesh (close to legacy sampling logic)."""
        self.mesh_presampled_points = {}
        mesh_names, mesh_objs, mesh_areas = [], [], []
        for i, mesh in enumerate(fk_result):
            name = _lu_get_mesh_name(mesh, i)
            if mesh.area <= 0:
                continue
            eff_area = mesh.area
            if 'hand_camera_part' in name.lower():
                eff_area *= 1e-6
            mesh_names.append(name)
            mesh_objs.append(mesh)
            mesh_areas.append(eff_area)
        if not mesh_names:
            self.mesh_presampled_ready = True
            return
        total_area = float(np.sum(mesh_areas))
        min_per_mesh = 500
        rng_state = None
        if self.robot_mask_seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(self.robot_mask_seed) % (2**32 - 1))
        try:
            for name, mesh, area in zip(mesh_names, mesh_objs, mesh_areas):
                frac = 0.0 if total_area <= 0 else area / total_area
                n = int(total_samples * frac)
                if any(k in name.lower() for k in ['finger', 'knuckle', 'robotiq']):
                    n = int(n * gripper_multiplier)
                n = max(min_per_mesh, n)
                try:
                    pts = mesh.sample(n)
                except Exception:
                    import trimesh
                    pts, _ = trimesh.sample.sample_surface_even(mesh, n)
                self.mesh_presampled_points[name] = pts.astype(np.float32)
        finally:
            if rng_state is not None:
                np.random.set_state(rng_state)
        self.mesh_presampled_ready = True

    def _get_mesh_name(self, mesh, idx):
        return _lu_get_mesh_name(mesh, idx)

    def _build_robot_mask_frame0(self, H, W, intrinsic, world2cam, joint_positions, gripper_position, downscale_ratio):
        cfg = {'finger_joint': float(gripper_position)}
        for ji in range(7):
            cfg[f'panda_joint{ji + 1}'] = float(joint_positions[ji])
        fk_result = self.robot_sampler.forward_kinematics(cfg)

        # Presample mesh points once
        if not self.mesh_presampled_ready:
            self._presample_mesh_points(fk_result)

        robot_mask = np.zeros((H, W), dtype=np.uint8)
        standard_circle_radius = max(1, int(14 * downscale_ratio))
        gripper_circle_radius = max(1, int(8 * downscale_ratio))
        gripper_keywords = ['finger', 'knuckle', 'robotiq']

        for i, mesh in enumerate(fk_result):
            mesh_name = self._get_mesh_name(mesh, i)
            is_gripper_part = any(k in mesh_name.lower() for k in gripper_keywords)
            circle_radius = gripper_circle_radius if is_gripper_part else standard_circle_radius
            if mesh.area <= 0:
                continue

            # Transform and project presampled points
            mesh_key = mesh_name
            if mesh_key not in self.mesh_presampled_points:
                continue
            points_3d = self.mesh_presampled_points[mesh_key]
            transform_matrix = fk_result[mesh].astype(np.float32)
            pts2d = project_points_to_image(
                points_3d,
                transform_matrix,
                world2cam.astype(np.float32),
                intrinsic.astype(np.float32),
                W,
                H,
            )
            if len(pts2d) == 0:
                continue
            pts2d = np.round(pts2d).astype(int)
            # Draw circles
            for (x, y) in pts2d:
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(robot_mask, (x, y), circle_radius, color=1, thickness=-1)

        # Morphological closing
        kernel_size = max(1, int(20 * downscale_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_CLOSE, kernel)
        return robot_mask.astype(bool)

    def _backproject_and_lift(self, x_int, y_int, depth, K, cam2world):
        # x_int, y_int: (T,N) integer pixel coords in depth frame scale
        # depth: (T,H,W) in meters
        T, N = x_int.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth[np.arange(T)[:, None], y_int, x_int]  # (T,N)
        X = (x_int - cx) / fx * z
        Y = (y_int - cy) / fy * z
        ones = np.ones_like(z)
        pts_cam = np.stack([X, Y, z, ones], axis=-1)  # (T,N,4)
        # Apply cam2world 4x4
        pts_world = pts_cam @ cam2world.T  # broadcast (T,N,4)
        return pts_world[..., :3]

    def _enforce_types(self, data_dict):
        for clip_key in data_dict:
            if ':' not in clip_key:
                continue
            data_dict[clip_key]['joint_positions'] = data_dict[clip_key]['joint_positions'].astype(np.float32)
            data_dict[clip_key]['gripper_open'] = data_dict[clip_key]['gripper_open'].astype(bool)
            data_dict[clip_key]['gripper_positions'] = data_dict[clip_key]['gripper_positions'].astype(np.float32)
            data_dict[clip_key]['gripper_pose'] = data_dict[clip_key]['gripper_pose'].astype(np.float32)
            data_dict[clip_key]['joint_velocities'] = data_dict[clip_key]['joint_velocities'].astype(np.float32)
            data_dict[clip_key]['joint_torques'] = data_dict[clip_key]['joint_torques'].astype(np.float32)
            for key in data_dict[clip_key]:
                if '_scene_flows' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(np.float16)
                elif '_scene_normals' in key:
                    data_dict[clip_key][key] = _quantize_unit_normals_to_int8(data_dict[clip_key][key])
                elif '_scene_colors' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(np.uint8)
                elif '_scene_visibility' in key or '_scene_depth_valid_mask' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(bool)
                elif '_rgb' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(np.uint8)
                elif '_depth' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(np.float32)
                elif '_intrinsic' in key or '_extrinsic' in key:
                    data_dict[clip_key][key] = data_dict[clip_key][key].astype(np.float32)
        return data_dict

    def _write_h5(self, processed_data_dict, output_path, ee_pos_threshold, ee_rot_threshold,
                  clip_pos_magnitudes=None, clip_rot_magnitudes=None, clip_gripper_movements=None,
                  gripper_closed_ee_pos_threshold=0.002, gripper_closed_ee_rot_threshold=0.05):
        with h5py.File(output_path, 'w') as f:
            f.attrs['uuid'] = processed_data_dict['uuid']
            f.attrs['creation_time'] = time.strftime("%Y%m%d_%H%M%S")
            f.attrs['domain'] = self.domain
            f.attrs['scene_path'] = self.scene_path
            metadata = get_metadata(self.scene_path)
            for key in metadata:
                f.attrs[key] = metadata[key]
            f.attrs['canonical_timestamps'] = str(processed_data_dict['canonical_timestamps'].tolist())
            f.attrs['ee_pos_threshold'] = ee_pos_threshold
            f.attrs['ee_rot_threshold'] = ee_rot_threshold
            f.attrs['gripper_closed_ee_pos_threshold'] = gripper_closed_ee_pos_threshold
            f.attrs['gripper_closed_ee_rot_threshold'] = gripper_closed_ee_rot_threshold
            for clip_key, clip_data in processed_data_dict.items():
                if ':' not in clip_key:
                    continue
                clip_group = f.create_group(f"{clip_key}")
                if 'joint_positions' in clip_data:
                    T = clip_data['joint_positions'].shape[0]
                    clip_group.attrs['demo_length'] = T
                robot_keys = ['joint_positions', 'joint_velocities', 'joint_torques',
                              'gripper_open', 'gripper_positions', 'gripper_pose']
                for key in robot_keys:
                    if key in clip_data:
                        data = clip_data[key]
                        if key == 'gripper_open':
                            dset = clip_group.create_dataset(key, data=data.astype(np.bool_), compression=None)
                        else:
                            dset = clip_group.create_dataset(key, data=data.astype(np.float32), compression=None)
                        dset.attrs['write_complete'] = True
                camera_prefixes = set()
                for key in clip_data.keys():
                    if key.endswith('_scene_flows'):
                        camera_prefixes.add(key.split('_scene_flows')[0])
                for camera_prefix in camera_prefixes:
                    camera_group = clip_group.create_group(f'camera_{camera_prefix}')
                    for key in ['scene_flows', 'scene_colors', 'scene_normals', 'scene_visibility', 'scene_depth_valid_mask']:
                        full_key = f'{camera_prefix}_{key}'
                        data = clip_data[full_key]
                        if key == 'scene_flows':
                            dset = camera_group.create_dataset(key, data=data.astype(np.float16), compression=None)
                        elif key == 'scene_normals':
                            dset = camera_group.create_dataset(key, data=data.astype(QUANTIZED_NORMALS_DTYPE), compression=None)
                        elif key == 'scene_colors':
                            dset = camera_group.create_dataset(key, data=data.astype(np.uint8), compression=None)
                        else:
                            dset = camera_group.create_dataset(key, data=data.astype(np.bool_), compression=None)
                        dset.attrs['write_complete'] = True

                    initial_rgb_src = np.asarray(clip_data[f'{camera_prefix}_rgb'][0])
                    initial_depth_src = np.asarray(clip_data[f'{camera_prefix}_depth'][0], dtype=np.float32)
                    if initial_depth_src.ndim != 2:
                        raise ValueError(
                            f"Expected {camera_prefix}_depth first frame with shape (H,W), "
                            f"got {initial_depth_src.shape}"
                        )
                    if initial_rgb_src.ndim != 3 or initial_rgb_src.shape[2] != 3:
                        raise ValueError(
                            f"Expected {camera_prefix}_rgb first frame with shape (H,W,3), "
                            f"got {initial_rgb_src.shape}"
                        )
                    if initial_rgb_src.shape[:2] != initial_depth_src.shape:
                        raise ValueError(
                            f"RGB/depth source resolution mismatch for {camera_prefix}: "
                            f"rgb={initial_rgb_src.shape[:2]}, depth={initial_depth_src.shape}"
                        )

                    initial_rgb = _resize_initial_rgb_to_contract(initial_rgb_src)
                    initial_depth = _resize_initial_depth_to_contract(initial_depth_src)
                    intrinsic_src = np.asarray(clip_data[f'{camera_prefix}_intrinsic'], dtype=np.float32)
                    intrinsic = _scale_intrinsic_for_resolution(
                        intrinsic_src,
                        source_hw=initial_depth_src.shape,
                        target_hw=initial_depth.shape,
                    )
                    extrinsic = np.asarray(clip_data[f'{camera_prefix}_extrinsic'], dtype=np.float32)

                    dset = camera_group.create_dataset('intrinsic', data=intrinsic.astype(np.float32), compression=None)
                    dset.attrs['write_complete'] = True
                    dset = camera_group.create_dataset('extrinsic', data=extrinsic.astype(np.float32), compression=None)
                    dset.attrs['write_complete'] = True

                    save_rgb_as_jpeg_in_h5(camera_group, 'initial_rgb', initial_rgb)
                    save_depth_as_uint16_mm(camera_group, 'initial_depth', initial_depth)
    
    def _estimate_normals(self, pts_world, world2cam):
        """
        CPU normal estimation per frame using Open3D; orient toward camera.

        Args:
            pts_world: (T,N,3) float32
            world2cam: (4,4) float32

        Returns:
            (T,N,3) float32 normals
        """
        import open3d as o3d
        T, N, _ = pts_world.shape
        normals = np.zeros_like(pts_world, dtype=np.float32)

        # Camera position from world2cam
        R = world2cam[:3, :3]
        t = world2cam[:3, 3]
        cam_pos = -R.T @ t

        for ti in range(T):
            pts = pts_world[ti]
            if pts.shape[0] == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_towards_camera_location(cam_pos.astype(np.float64))
            n = np.asarray(pcd.normals).astype(np.float32)
            n = np.where(np.isfinite(n), n, 0.0)
            normals[ti] = n

        # Final sign check to point toward camera
        for ti in range(T):
            to_cam = cam_pos[None, :] - pts_world[ti]
            denom = np.linalg.norm(to_cam, axis=1, keepdims=True)
            denom = np.clip(denom, 1e-10, None)
            to_cam_unit = to_cam / denom
            dots = (normals[ti] * to_cam_unit).sum(axis=1)
            flip = dots < 0
            normals[ti, flip] *= -1.0

        return normals

    def convert_scene(self, cache_path, output_dir, extrinsics_source='optimized',
                      outlier_eps_values=(0.02, 0.05), outlier_min_points=5, outlier_frames_ratio=0.2,
                      outlier_scope='global', scale_intrinsics_to_depth=False,
                      seed_mask_mode='auto'):
        assert seed_mask_mode in ('auto', 'workspace_and_not_robot', 'workspace_only', 'none'), (
            f"Invalid seed_mask_mode: {seed_mask_mode}"
        )
        cache = self._load_2d_flows(cache_path)
        uuid = cache['uuid']
        scene_path = cache['scene_path']
        timestamps = cache['canonical_timestamps']
        downscale_ratio = cache['downscale_ratio']
        tracking_mask_mode = cache.get('tracking_mask_mode', 'none')
        if seed_mask_mode == 'auto':
            # If 2D tracking already applied masks at tracker-time, skip post-hoc
            # seed masking to avoid double-filtering.
            effective_seed_mask_mode = (
                'none' if tracking_mask_mode != 'none' else 'workspace_and_not_robot'
            )
        else:
            effective_seed_mask_mode = seed_mask_mode
        print(
            f"[{get_time_str()}] Converting {uuid}: tracking_mask_mode={tracking_mask_mode}, "
            f"seed_mask_mode={seed_mask_mode} -> effective_seed_mask_mode={effective_seed_mask_mode}"
        )

        depth_h5_path = os.path.join(output_dir, 'depth', f'{uuid}_depth.h5')
        assert os.path.exists(depth_h5_path), f"Depth file not found: {depth_h5_path}"

        # Build mappings for extrinsics by camera serial
        camera_serials = [k.split('+')[0] for k in cache['cameras'].keys()]
        world2cam_map = self._load_extrinsics(output_dir, uuid, camera_serials, extrinsics_source)

        # Depth will be loaded per camera from H5 and aligned to canonical timestamps

        # Assemble processed_data_dict matching legacy flow schema
        processed = {
            'uuid': uuid,
            'canonical_timestamps': timestamps,
        }

        # Iterate cameras
        for cam_key, cam_payload in cache['cameras'].items():
            camera_serial, cam_type = cam_key.split('+')
            if cam_type != 'ext':
                raise ValueError(f"Unsupported camera type '{cam_type}' in 2D flows cache (expected 'ext')")
            K = cam_payload['intrinsic']
            world2cam = world2cam_map[camera_serial]
            cam2world = np.linalg.inv(world2cam)

            # Static workspace mask for the camera at depth resolution will be computed per-clip (in case K differs across cache)
            # Open depth file once per camera for efficiency
            with h5py.File(depth_h5_path, 'r') as _f_depth:
                _g_cam = _f_depth[cam_key]
                _depth_ds = _g_cam['depth']
                _depth_ts = np.array(_g_cam['timestamps'])
                Hd_all, Wd_all = _depth_ds.shape[1:]

                for clip_key, c in cam_payload['clips'].items():
                    # Parse start:end
                    start, end = [int(x) for x in clip_key.split(':')]
                    # Load only the needed depth frames for this clip to reduce memory
                    clip_ts = timestamps[start:end]
                    Hd, Wd = Hd_all, Wd_all

                    H_rgb, W_rgb = c['H'], c['W']
                    flows_2d = c['flows_2d_xy']  # (T,N,2)
                    vis = c['flows_2d_visibility']  # (T,N)
                    colors = c['flow_colors']  # (T,N,3)

                    # If depth resolution != RGB resolution, rescale pixel coords to depth resolution
                    sx = Wd / float(W_rgb)
                    sy = Hd / float(H_rgb)
                    K_lift = K.copy()
                    if scale_intrinsics_to_depth and (abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6):
                        # Keep projection model consistent with resized pixel coordinates.
                        K_lift[0, 0] *= sx
                        K_lift[0, 2] *= sx
                        K_lift[1, 1] *= sy
                        K_lift[1, 2] *= sy
                    tracks_depth = flows_2d.copy()
                    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                        tracks_depth[..., 0] = tracks_depth[..., 0] * sx
                        tracks_depth[..., 1] = tracks_depth[..., 1] * sy

                    # Rounding & clamping before sampling depth
                    tracks_rounded = np.round(tracks_depth).astype(int)
                    out_of_bounds = (
                        (tracks_depth[..., 0] < 0)
                        | (tracks_depth[..., 0] >= Wd)
                        | (tracks_depth[..., 1] < 0)
                        | (tracks_depth[..., 1] >= Hd)
                    )
                    vis = vis.copy()
                    vis[out_of_bounds] = False
                    x_int = np.clip(tracks_rounded[..., 0], 0, Wd - 1)
                    y_int = np.clip(tracks_rounded[..., 1], 0, Hd - 1)

                    # Ensure timestamps and coords align to T_clip by repeating last if needed
                    T_clip = flows_2d.shape[0]
                    if clip_ts.shape[0] < T_clip:
                        pad_k = T_clip - clip_ts.shape[0]
                        clip_ts = np.concatenate([clip_ts, np.repeat(clip_ts[-1], pad_k)])
                        x_int = np.concatenate([x_int, np.repeat(x_int[-1:, :], pad_k, axis=0)], axis=0)
                        y_int = np.concatenate([y_int, np.repeat(y_int[-1:, :], pad_k, axis=0)], axis=0)

                    # Sample depth per-frame at track pixels (no (T,H,W) allocation)
                    z_tn, depth_valid, Hd, Wd, first_depth_grid = self._sample_depth_values_for_tracks(
                        _depth_ds, _depth_ts, clip_ts, x_int, y_int
                    )

                    # Posthoc masks at frame 0 only (seed policy)
                    workspace_mask = self._compute_workspace_mask(K_lift, world2cam, Hd, Wd)
                    # For robot FK, use proprio of this clip at frame 0
                    jp = cache['proprio'][clip_key]['joint_positions'][0]
                    gp = cache['proprio'][clip_key]['gripper_positions'][0]
                    robot_mask = self._build_robot_mask_frame0(
                        Hd, Wd, K_lift, world2cam, jp, gp, downscale_ratio
                    )
                    if effective_seed_mask_mode == 'workspace_and_not_robot':
                        keep_mask0 = workspace_mask & (~robot_mask)
                    elif effective_seed_mask_mode == 'workspace_only':
                        keep_mask0 = workspace_mask
                    else:
                        keep_mask0 = np.ones((Hd, Wd), dtype=bool)

                    # Evaluate keep mask for seed-frame locations
                    x0 = x_int[0]
                    y0 = y_int[0]
                    keep = keep_mask0[y0, x0].astype(bool)  # (N,)

                    # Depth validity already computed at track locations

                    # Legacy parity: clamp out-of-range depth to 0 before lifting.
                    # The original monolithic pipeline applied this in process_depth()
                    # before pointmap generation.
                    z_lift = z_tn.copy()
                    invalid_for_lift = (
                        (~np.isfinite(z_lift))
                        | (z_lift < MIN_DEPTH_M)
                        | (z_lift > MAX_DEPTH_M)
                    )
                    z_lift[invalid_for_lift] = 0.0

                    # Lift to 3D using sampled depth values
                    fx, fy = K_lift[0, 0], K_lift[1, 1]
                    cx, cy = K_lift[0, 2], K_lift[1, 2]
                    X = (x_int - cx) / fx * z_lift
                    Y = (y_int - cy) / fy * z_lift
                    ones = np.ones_like(z_lift)
                    pts_cam = np.stack([X, Y, z_lift, ones], axis=-1)
                    pts_world = (pts_cam @ cam2world.T)[..., :3]
                    # Estimate normals on CPU
                    normals = self._estimate_normals(pts_world.astype(np.float32), world2cam.astype(np.float32))

                    # Remove unkept tracks entirely
                    pts_world = pts_world[:, keep]
                    normals = normals[:, keep]
                    colors = colors[:, keep]
                    vis = vis[:, keep]  # tracker visibility only
                    depth_valid = depth_valid[:, keep]

                    # Prepare processed_data_dict entries
                    camera_prefix = f"{camera_serial}_ext"
                    if clip_key not in processed:
                        gripper_open = (cache['proprio'][clip_key]['gripper_positions'] < 0.1).astype(bool)
                        if gripper_open.ndim == 1:
                            gripper_open = gripper_open[:, None]
                        processed[clip_key] = {
                            'joint_positions': cache['proprio'][clip_key]['joint_positions'],
                            'joint_velocities': cache['proprio'][clip_key]['joint_velocities'],
                            'joint_torques': cache['proprio'][clip_key]['joint_torques'],
                            'gripper_positions': cache['proprio'][clip_key]['gripper_positions'],
                            'gripper_open': gripper_open,
                            'gripper_pose': cache['proprio'][clip_key]['gripper_pose'],
                        }

                    processed[clip_key][f'{camera_prefix}_scene_flows'] = pts_world.astype(np.float32)
                    processed[clip_key][f'{camera_prefix}_scene_colors'] = colors.astype(np.uint8)
                    processed[clip_key][f'{camera_prefix}_scene_normals'] = normals.astype(np.float32)
                    processed[clip_key][f'{camera_prefix}_scene_visibility'] = vis.astype(bool)
                    processed[clip_key][f'{camera_prefix}_scene_depth_valid_mask'] = depth_valid.astype(bool)
                    # Minimal rgb/depth (first frame only) for writer compatibility
                    processed[clip_key][f'{camera_prefix}_rgb'] = c['initial_rgb'][None].astype(np.uint8)
                    # Use sanitized first-frame depth grid
                    processed[clip_key][f'{camera_prefix}_depth'] = first_depth_grid.astype(np.float32)
                    processed[clip_key][f'{camera_prefix}_intrinsic'] = K_lift.astype(np.float32)
                    processed[clip_key][f'{camera_prefix}_extrinsic'] = world2cam.astype(np.float32)

        # 3D workspace bounds filter across time (relaxed bounds, parity with original postprocess)
        for clip_key in list(processed.keys()):
            if ':' not in clip_key:
                continue
            camera_prefixes = [k.replace('_scene_flows', '') for k in processed[clip_key].keys() if k.endswith('_scene_flows')]
            for cp in camera_prefixes:
                pts = processed[clip_key][f'{cp}_scene_flows']
                cols = processed[clip_key][f'{cp}_scene_colors']
                vis = processed[clip_key][f'{cp}_scene_visibility']
                dvm = processed[clip_key][f'{cp}_scene_depth_valid_mask']
                norms = processed[clip_key][f'{cp}_scene_normals']
                xin = (pts[...,0] >= WORKSPACE_BOUNDS_MIN_RELAXED[0]) & (pts[...,0] <= WORKSPACE_BOUNDS_MAX_RELAXED[0])
                yin = (pts[...,1] >= WORKSPACE_BOUNDS_MIN_RELAXED[1]) & (pts[...,1] <= WORKSPACE_BOUNDS_MAX_RELAXED[1])
                zin = (pts[...,2] >= WORKSPACE_BOUNDS_MIN_RELAXED[2]) & (pts[...,2] <= WORKSPACE_BOUNDS_MAX_RELAXED[2])
                inb = (xin & yin & zin)
                keep = inb.all(axis=0)
                processed[clip_key][f'{cp}_scene_flows'] = pts[:, keep]
                processed[clip_key][f'{cp}_scene_colors'] = cols[:, keep]
                processed[clip_key][f'{cp}_scene_visibility'] = vis[:, keep]
                processed[clip_key][f'{cp}_scene_depth_valid_mask'] = dvm[:, keep]
                processed[clip_key][f'{cp}_scene_normals'] = norms[:, keep]

        # Outlier removal (CPU)
        # - global: remove across concatenated external cameras (current default behavior)
        # - per_camera: remove independently per camera (legacy postprocess-like behavior)
        # - none: skip outlier removal
        assert outlier_scope in ('global', 'per_camera', 'none'), f"Invalid outlier_scope: {outlier_scope}"
        for clip_key in list(processed.keys()):
            if ':' not in clip_key:
                continue
            cps = []
            for k in processed[clip_key].keys():
                if k.endswith('_scene_flows'):
                    cpi = k.replace('_scene_flows','')
                    cps.append(cpi)
            if not cps:
                continue
            if outlier_scope == 'none':
                continue

            if outlier_scope == 'global':
                trajs = [processed[clip_key][f'{cp}_scene_flows'] for cp in cps]
                viss = [processed[clip_key][f'{cp}_scene_visibility'] for cp in cps]
                dvms = [processed[clip_key][f'{cp}_scene_depth_valid_mask'] for cp in cps]
                counts = [t.shape[1] for t in trajs]
                gtraj = np.concatenate(trajs, axis=1)
                gvis = np.concatenate(viss, axis=1)
                gdvm = np.concatenate(dvms, axis=1)
                _, _, _, out_idx = remove_outlier_flows(
                    gtraj,
                    visibility_mask=gvis,
                    depth_valid_mask=gdvm,
                    eps_values=list(outlier_eps_values),
                    min_points=outlier_min_points,
                    min_outlier_frames_ratio=outlier_frames_ratio,
                )
                if out_idx is not None and len(out_idx)>0:
                    keep = np.ones(gtraj.shape[1], dtype=bool)
                    keep[out_idx] = False
                    s=0
                    for i,cp in enumerate(cps):
                        e=s+counts[i]
                        m=keep[s:e]
                        for suf in ['_scene_flows','_scene_colors','_scene_visibility','_scene_depth_valid_mask','_scene_normals']:
                            processed[clip_key][f'{cp}{suf}'] = processed[clip_key][f'{cp}{suf}'][:, m]
                        s=e
                continue

            # per_camera outlier filtering
            for cp in cps:
                traj = processed[clip_key][f'{cp}_scene_flows']
                vis = processed[clip_key][f'{cp}_scene_visibility']
                dvm = processed[clip_key][f'{cp}_scene_depth_valid_mask']
                _, _, _, out_idx = remove_outlier_flows(
                    traj,
                    visibility_mask=vis,
                    depth_valid_mask=dvm,
                    eps_values=list(outlier_eps_values),
                    min_points=outlier_min_points,
                    min_outlier_frames_ratio=outlier_frames_ratio,
                )
                if out_idx is not None and len(out_idx) > 0:
                    keep = np.ones(traj.shape[1], dtype=bool)
                    keep[out_idx] = False
                    for suf in ['_scene_flows','_scene_colors','_scene_visibility','_scene_depth_valid_mask','_scene_normals']:
                        processed[clip_key][f'{cp}{suf}'] = processed[clip_key][f'{cp}{suf}'][:, keep]

        # Enforce types and write via local writer to match schema exactly
        self.scene_path = scene_path
        self.uuid = uuid
        processed = self._enforce_types(processed)
        flows_dir_name = 'flows-fs-optimize' if extrinsics_source == 'optimized' else f'flows-fs-{extrinsics_source}'
        flows_dir = os.path.join(output_dir, flows_dir_name)
        flows_h5_path = os.path.join(flows_dir, f'{uuid}_flows.h5')
        os.makedirs(os.path.dirname(flows_h5_path), exist_ok=True)
        self._write_h5(
            processed,
            flows_h5_path,
            ee_pos_threshold=float(cache['ee_pos_threshold']),
            ee_rot_threshold=float(cache['ee_rot_threshold']),
            clip_pos_magnitudes=None,
            clip_rot_magnitudes=None,
            clip_gripper_movements=None,
            gripper_closed_ee_pos_threshold=float(cache['gripper_closed_ee_pos_threshold']),
            gripper_closed_ee_rot_threshold=float(cache['gripper_closed_ee_rot_threshold']),
        )
        print(f"[{get_time_str()}] Converted {uuid} cache -> {flows_h5_path}")
        return flows_h5_path


def main():
    import argparse

    def _read_scene_paths(input_txt: str):
        if not os.path.exists(input_txt):
            raise FileNotFoundError(f"Input scene list not found: {input_txt}")
        with open(input_txt, "r") as f:
            scene_paths = [line.strip() for line in f if line.strip()]
        if len(scene_paths) == 0:
            raise ValueError(f"Input scene list is empty: {input_txt}")
        return scene_paths

    def _resolve_cache_files_from_scene_paths(scene_paths, cache_dir: str, output_dir: str):
        pairs = []
        errors = []
        for scene_path in scene_paths:
            try:
                uuid = get_uuid(scene_path)
            except Exception as exc:
                errors.append(f"{scene_path}: failed to extract uuid ({exc})")
                continue

            cache_path = os.path.join(cache_dir, f"{uuid}_2d_flows.h5")
            depth_path = os.path.join(output_dir, "depth", f"{uuid}_depth.h5")
            cameras_json_path = os.path.join(output_dir, "cameras", f"{uuid}_cameras.json")

            missing = []
            if not os.path.exists(cache_path):
                missing.append(cache_path)
            if not os.path.exists(depth_path):
                missing.append(depth_path)
            if not os.path.exists(cameras_json_path):
                missing.append(cameras_json_path)
            if missing:
                errors.append(f"{scene_path} ({uuid}): missing required artifacts -> {', '.join(missing)}")
                continue

            try:
                with h5py.File(cache_path, "r") as f:
                    if "write_complete" not in f.attrs or not bool(f.attrs["write_complete"]):
                        errors.append(
                            f"{scene_path} ({uuid}): cache exists but is incomplete "
                            f"(write_complete={f.attrs.get('write_complete', None)!r})"
                        )
                        continue
            except Exception as exc:
                errors.append(f"{scene_path} ({uuid}): failed to read cache {cache_path} ({exc})")
                continue

            pairs.append((scene_path, cache_path))

        if errors:
            max_show = 20
            shown = errors[:max_show]
            more = len(errors) - len(shown)
            msg = "\n".join(shown)
            if more > 0:
                msg += f"\n... and {more} more error(s)"
            raise RuntimeError(
                "Fail-fast preflight failed while resolving scene list for conversion. "
                f"Found {len(errors)} issue(s):\n{msg}"
            )
        return pairs

    parser = argparse.ArgumentParser(description='Convert cached 2D flows to 3D H5 (CPU-only)')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help=(
            "Optional text file of scene paths. "
            "When set, converter processes only these scenes and fail-fast validates "
            "required artifacts (2d_flows/depth/cameras) before starting."
        ),
    )
    parser.add_argument('--flows_2d_dir', type=str, default=None, help='Directory containing *_2d_flows.h5 files (default: <output_dir>/2d_flows)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output root for flows (and default depth/cameras)')
    parser.add_argument('--extrinsics_source', type=str, required=True, help='Source of extrinsics: one of {vggt, optimized}')
    parser.add_argument('--urdf_path', type=str, default=URDF_PATH, help='URDF path for RobotSampler (droid domain)')
    parser.add_argument('--max_scenes', type=int, default=None, help='Optional limit for batch modes')
    parser.add_argument('--rank', type=int, default=0, help='Worker rank (0..world_size-1)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of workers')
    parser.add_argument('--outlier_eps_values', type=float, nargs='+', default=[0.02, 0.05], help='DBSCAN eps values')
    parser.add_argument('--outlier_min_points', type=int, default=5, help='DBSCAN min_samples')
    parser.add_argument('--outlier_frames_ratio', type=float, default=0.2, help='Outlier frame ratio threshold')
    parser.add_argument(
        '--outlier_scope',
        type=str,
        choices=['global', 'per_camera', 'none'],
        default='global',
        help='Outlier removal scope: global across cameras, per-camera independently, or disabled',
    )
    parser.add_argument('--robot_mask_seed', type=int, default=None, help='Optional RNG seed for robot mask sampling')
    parser.add_argument(
        '--scale_intrinsics_to_depth',
        action='store_true',
        help='When H/W differ between 2D tracks and depth, scale fx/fy/cx/cy to depth resolution before lifting/masking',
    )
    parser.add_argument(
        '--seed_mask_mode',
        type=str,
        choices=['auto', 'workspace_and_not_robot', 'workspace_only', 'none'],
        default='auto',
        help=(
            "Seed-frame keep mask mode before lifting. "
            "'auto' disables post-hoc masking when tracker-time masking was used."
        ),
    )
    parser.add_argument(
        '--allow_gcs_streaming',
        action='store_true',
        help=(
            'Bypass GCS cache enforcement and stream directly from gs:// inputs for this run. '
            'Not recommended for repeated multi-stage processing.'
        ),
    )
    args = parser.parse_args()

    conv = Flow2DTo3DConverter(urdf_path=args.urdf_path, robot_mask_seed=args.robot_mask_seed)

    def run_one(cache_path: str):
        flows_h5 = conv.convert_scene(
            cache_path, args.output_dir,
            extrinsics_source=args.extrinsics_source,
            outlier_eps_values=tuple(args.outlier_eps_values),
            outlier_min_points=args.outlier_min_points,
            outlier_frames_ratio=args.outlier_frames_ratio,
            outlier_scope=args.outlier_scope,
            scale_intrinsics_to_depth=args.scale_intrinsics_to_depth,
            seed_mask_mode=args.seed_mask_mode,
        )
        return flows_h5

    cache_dir = args.flows_2d_dir or os.path.join(args.output_dir, '2d_flows')
    if not os.path.isdir(cache_dir):
        print(f"[{get_time_str()}] ERROR: flows_2d_dir not found: {cache_dir}")
        sys.exit(1)

    # Distribute files/scenes across workers
    if args.world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}")

    if args.input is not None:
        scene_paths_all = _read_scene_paths(args.input)
        if args.max_scenes is not None:
            scene_paths_all = scene_paths_all[:args.max_scenes]

        enforce_gcs_cache_policy(
            scene_paths_all,
            stage_name='convert_2d_flows_to_3d',
            require_cache=True,
            allow_streaming=args.allow_gcs_streaming,
        )

        pairs_all = _resolve_cache_files_from_scene_paths(scene_paths_all, cache_dir, args.output_dir)
        pairs_rank = pairs_all[args.rank::args.world_size]
        scene_paths = [sp for sp, _ in pairs_rank]
        files = [fp for _, fp in pairs_rank]
        print(
            f"Worker {args.rank}/{args.world_size} converting {len(files)} selected caches "
            f"(from --input: {args.input})"
        )
    else:
        files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('_2d_flows.h5')]
        files.sort()
        if args.max_scenes is not None:
            files = files[:args.max_scenes]
        files = files[args.rank::args.world_size]
        print(f"Worker {args.rank}/{args.world_size} converting {len(files)} caches from {cache_dir}")

        scene_paths = []
        for fp in files:
            with h5py.File(fp, 'r') as f:
                scene_path = f.attrs.get('scene_path', '')
                if isinstance(scene_path, (bytes, bytearray)):
                    scene_path = scene_path.decode('utf-8')
                scene_paths.append(str(scene_path))
        enforce_gcs_cache_policy(
            scene_paths,
            stage_name='convert_2d_flows_to_3d',
            require_cache=True,
            allow_streaming=args.allow_gcs_streaming,
        )

    converted = []

    pbar = tqdm(files, total=len(files), desc=f"worker {args.rank} convert", unit="file", dynamic_ncols=True)
    for fp in pbar:
        tqdm.write(f"[{get_time_str()}] Converting: {fp}")
        res = run_one(fp)
        if res:
            converted.append(res)
    print(f"[{get_time_str()}] Converted {len(converted)} file(s)")


if __name__ == '__main__':
    main()
