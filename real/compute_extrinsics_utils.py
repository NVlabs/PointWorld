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
import sys; sys.path.append('..')
import os
import json
import numpy as np
# urdfpy still relies on np.float; restore alias for numpy>=1.24
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
from typing import List
from real.real_utils import get_mesh_name
from real.droid_utils import get_uuid, load_robot_transforms
import torch
import urdfpy
import h5py
# Additional imports for visualization
# sys.path.append('check_extrinsics')
# from check_extrinsics_utils import PyrenderViewer, blend_images, depth2rgb
# from raw import RawScene

REAL_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_GRIPPER2WRIST_TRANSFORMS_PATH = os.path.join(
    REAL_DIR,
    "gripper2wrist_transforms.json",
)

class RobotVisibilityError(Exception):
    """Raised when robot is not visible in cameras during optimization."""
    def __init__(self, message, error_type="no_robot_visible"):
        self.error_type = error_type
        super().__init__(message)

def check_precomputed_depth_exists(scene_path, output_dir, uuid=None):
    """
    Check if precomputed depth file exists for a scene without loading it.
    
    Args:
        scene_path: Path to the DROID scene
        output_dir: Output directory where depth files are stored
        uuid: Scene UUID (if None, will be extracted from scene_path)
        
    Returns:
        tuple: (exists, h5_path, error_message)
            - exists: bool indicating if the depth file exists and is valid
            - h5_path: Path to the depth file
            - error_message: Error message if file doesn't exist or is invalid
    """
    # Get scene UUID
    if uuid is None:
        uuid = get_uuid(scene_path)

    # Path to depth H5 file
    h5_path = os.path.join(output_dir, "depth", f"{uuid}_depth.h5")

    if not os.path.exists(h5_path):
        return False, h5_path, f"Precomputed depth file not found: {h5_path}"

    # Quick check that the file is valid and complete
    with h5py.File(h5_path, 'r') as f:
        # Check metadata
        if 'metadata' not in f:
            raise ValueError(f"Invalid depth file: missing metadata in {h5_path}")

        metadata = f['metadata']
        if 'write_complete' not in metadata.attrs:
            raise ValueError(f"Depth file missing write_complete flag: {h5_path}")
        if not metadata.attrs['write_complete']:
            raise ValueError(f"Depth file not complete: {h5_path}")

        # Check that camera groups exist
        camera_groups = [key for key in f.keys() if key != 'metadata']
        if len(camera_groups) == 0:
            raise ValueError(f"No camera groups found in depth file: {h5_path}")

    return True, h5_path, None

def check_camera_results_exist(scene_path, output_dir, uuid=None):
    """
    Check if camera results JSON file exists for a scene without loading it.
    
    Args:
        scene_path: Path to the DROID scene
        output_dir: Output directory where camera results are stored
        uuid: Scene UUID (if None, will be extracted from scene_path)
        
    Returns:
        tuple: (exists, json_path, error_message)
            - exists: bool indicating if the JSON file exists and is readable
            - json_path: Path to the camera results JSON file
            - error_message: Error message if file doesn't exist or is invalid
    """
    # Get scene UUID
    if uuid is None:
        uuid = get_uuid(scene_path)

    # Path to camera results JSON file
    json_path = os.path.join(output_dir, "cameras", f"{uuid}_cameras.json")

    if not os.path.exists(json_path):
        return False, json_path, f"Camera results file not found: {json_path}"

    # Quick check that the file is valid and readable
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check that it has the expected structure
    if 'uuid' not in data:
        raise ValueError(f"Invalid camera results file: missing uuid in {json_path}")

    if data['uuid'] != uuid:
        raise ValueError(f"UUID mismatch in camera results file: expected {uuid}, got {data['uuid']}")

    # Check that it has at least one camera with the required fields
    camera_count = 0
    required_fields = ['vggt_extrinsics', 'optimized_extrinsics',
                     'measured_intrinsics', 'vggt_intrinsics']

    for key, value in data.items():
        if key not in ['uuid', 'optimization_summary'] and isinstance(value, dict):
            camera_count += 1
            for field in required_fields:
                if field not in value:
                    raise ValueError(f"Missing required field '{field}' for camera in {json_path}")

    if camera_count != 2:  # expect 2 external cameras
        raise ValueError(f"No camera data found in results file: {json_path}")

    return True, json_path, None

def deduplicate_coordinates(xy: torch.Tensor, threshold_pixels: float = 0.5):
    """
    Torch-native replacement for `deduplicate_coordinates`.

    Args
    ----
    xy : Tensor
        • shape (N, 2)  –– single set of image coords, or  
        • shape (B,N,2) –– batch (preferred for large jobs)  
        The tensor may live on either CPU or GPU.

    threshold_pixels : float, default=0.5  
        Side length of the square grid cell, in pixels.

    Returns
    -------
    Tensor | List[Tensor]
        Indices of unique points (first hit per cell).  
        • single-set  → 1-D Long tensor  
        • batched     → list of 1-D Long tensors (length B)
    """
    if xy.numel() == 0:
        return [] if xy.ndim == 3 else torch.empty(0, dtype=torch.long)

    # 1) Quantise to an integer grid
    q = torch.round(xy / threshold_pixels).to(torch.int32)          # [...,2]

    # 2) Collapse (x,y) pairs into a single 1-D code that is guaranteed unique
    #    inside a batch.  Using a stride avoids clashes without knowing max range.
    stride = 131_071                                                 # large prime
    code  = q[..., 0] * stride + q[..., 1]                           # [...,]

    # 3) Handle batched & unbatched paths separately for clarity
    if xy.ndim == 2:  # single point-set -------------------------------------------------
        idx        = torch.arange(code.shape[0], device=xy.device)
        uniq, inv  = torch.unique(code, return_inverse=True, sorted=False)
        # torch>=1.12: scatter_reduce_ gives O(N) first-occurrence indices
        first      = torch.full_like(uniq, fill_value=code.shape[0], dtype=torch.long)
        first.scatter_reduce_(0, inv, idx, reduce="amin")            #  [oai_citation:0‡docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html?utm_source=chatgpt.com)
        return first

    else:        # batched path ----------------------------------------------------------
        B, N      = code.shape[:2]
        batch_idx = torch.arange(N, device=xy.device).expand(B, N)   # [B,N]
        out: List[torch.Tensor] = []
        for b in range(B):                                           # still GPU
            uniq, inv = torch.unique(code[b], return_inverse=True, sorted=False)
            first     = torch.full_like(uniq, fill_value=N, dtype=torch.long)
            first.scatter_reduce_(0, inv, batch_idx[b], reduce="amin")
            out.append(first)
        return out

def sample_depth_with_grid_sample(
    depth: torch.Tensor,          # 2-D  (H,W)  or 3-D  (B,H,W)
    xy: torch.Tensor,             # 2-D  (N,2)  or 3-D  (B,N,2)
    align_corners: bool = True
) -> torch.Tensor:
    """
    Bilinear-interpolated depth lookup via F.grid_sample, batched.

    Returns
    -------
    torch.Tensor
        • shape (N,)  if inputs were 2-D + 2-D  
        • shape (B,N) if inputs were 3-D + 3-D
    """
    # --------------------------- sanity checks ----------------------------
    assert isinstance(depth, torch.Tensor) and isinstance(xy, torch.Tensor), \
        "depth and xy must be torch tensors"
    assert depth.device == xy.device, "depth and xy must be on the same device"
    assert depth.ndim in (2, 3), "depth must be [H,W] or [B,H,W]"
    assert xy.ndim == depth.ndim,  \
        "xy must have same batch rank as depth (either both batched or both un-batched)"
    if depth.ndim == 3:
        B, H, W = depth.shape
        assert xy.shape[:2] == (B, -1)[:2], "xy must be [B,N,2]"  # N is free
    else:
        H, W = depth.shape
        B = 1

    # --------------------- normalise xy to [-1,1] -------------------------
    if depth.ndim == 2:                       # un-batched path
        x, y = xy[:, 0], xy[:, 1]
        grid = torch.stack((2*x/(W-1)-1, 2*y/(H-1)-1), dim=1)       # [N,2]
        grid = grid.unsqueeze(0).unsqueeze(0)                       # [1,1,N,2]
        depth_in = depth.unsqueeze(0).unsqueeze(0)                  # [1,1,H,W]
        out = torch.nn.functional.grid_sample(
            depth_in, grid, mode="bilinear",
            padding_mode="zeros", align_corners=align_corners
        )                                                           # [1,1,1,N]
        return out.view(-1)                                         # [N]

    else:                                       # batched path
        B, N = xy.shape[:2]
        x, y = xy[..., 0], xy[..., 1]                              # [B,N]
        grid = torch.stack((2*x/(W-1)-1, 2*y/(H-1)-1), dim=2)      # [B,N,2]
        grid = grid.unsqueeze(1)                                   # [B,1,N,2]
        depth_in = depth.unsqueeze(1)                              # [B,1,H,W]
        out = torch.nn.functional.grid_sample(
            depth_in, grid, mode="bilinear",
            padding_mode="zeros", align_corners=align_corners
        )                                                          # [B,1,1,N]
        return out.squeeze(1).squeeze(1)                           # [B,N]

def pose_6dof_to_matrix(pose_6dof):
    """Convert 6-DOF pose (x, y, z, roll, pitch, yaw) to 4x4 transformation matrix.
    
    Args:
        pose_6dof: (6,) tensor with [x, y, z, roll, pitch, yaw] in radians
    
    Returns:
        4x4 transformation matrix
    """
    x, y, z, roll, pitch, yaw = pose_6dof
    
    # Create rotation matrix from roll, pitch, yaw (ZYX convention)
    # R = R_z(yaw) * R_y(pitch) * R_x(roll)
    
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)  
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    # Rotation matrix (ZYX Euler angles)
    R = torch.zeros(3, 3, device=pose_6dof.device, dtype=pose_6dof.dtype)
    
    R[0, 0] = cos_y * cos_p
    R[0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    
    R[1, 0] = sin_y * cos_p
    R[1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    
    R[2, 0] = -sin_p
    R[2, 1] = cos_p * sin_r
    R[2, 2] = cos_p * cos_r
    
    # Create 4x4 transformation matrix
    T = torch.eye(4, device=pose_6dof.device, dtype=pose_6dof.dtype)
    T[:3, :3] = R
    T[:3, 3] = torch.stack([x, y, z])
    
    return T

class RobotMeshRenderer:
    """Simplified robot mesh renderer for extrinsics optimization."""
    
    def __init__(self, urdf_path, device="cuda", total_samples=25000):  # Reduced from 50k to 25k for speed
        self.device = device
        self.urdf_path = urdf_path
        self.dtype = torch.float32
        
        # Initialize URDF-based forward kinematics
        assert os.path.exists(urdf_path), f"URDF file not found: {urdf_path}"
        self.robot_urdf = urdfpy.URDF.load(urdf_path)
        print(f"Loaded URDF from: {urdf_path}")
        
        # Mesh points cache
        self.mesh_points = {}  # {mesh_name: torch.Tensor}
        self.total_samples = total_samples
        
        # Cache for FK results to avoid recomputation
        self.fk_cache = {}
        self.world_points_cache = {}
        
    def _get_forward_kinematics(self, joint_positions, gripper_position):
        """Compute forward kinematics for given joint configuration with caching."""
        # Create cache key from joint positions and gripper position
        cache_key = tuple(np.round(joint_positions, 4).tolist() + [round(gripper_position, 4)])
        
        if cache_key in self.fk_cache:
            return self.fk_cache[cache_key]
        
        # Construct configuration dictionary
        cfg = {'finger_joint': float(gripper_position)}
        for ji in range(7):
            cfg[f'panda_joint{ji + 1}'] = float(joint_positions[ji])
        
        # Compute forward kinematics using urdfpy
        fk_result = self.robot_urdf.visual_trimesh_fk(cfg=cfg)
        
        # Cache result
        self.fk_cache[cache_key] = fk_result
        return fk_result
    
    def _sample_mesh_points(self, fk_result):
        """Sample points from each mesh for efficient projection."""
        mesh_names = []
        mesh_objects = []
        mesh_areas = []
        
        # Calculate surface area for each mesh
        for i, mesh in enumerate(fk_result):
            mesh_name = get_mesh_name(mesh, i)
            
            # Skip meshes with zero area
            if mesh.area <= 0:
                continue
            
            # Apply area correction factor for camera mount if needed
            effective_area = mesh.area
            if 'hand_camera_part' in mesh_name.lower():
                effective_area *= 0.000001
            
            mesh_names.append(mesh_name)
            mesh_objects.append(mesh)
            mesh_areas.append(effective_area)
        
        if not mesh_names:
            raise ValueError("No meshes with positive area found for robot sampling.")

        # Calculate total area and allocate points
        total_area = sum(mesh_areas)
        if total_area <= 0:
            raise ValueError("Total mesh area is non-positive; cannot sample robot points.")
        
        # Sample points for each mesh (optimized allocation)
        for name, mesh, area in zip(mesh_names, mesh_objects, mesh_areas):
            ratio = area / total_area
            count = max(200, int(self.total_samples * ratio))  # Reduced min from 500 to 200
            points_3d = mesh.sample(count)
            self.mesh_points[name] = torch.from_numpy(points_3d.astype(np.float32)).to(device=self.device, dtype=self.dtype)
    
    def _get_world_points(self, fk_result):
        """Get world-space coordinates of all sample points with caching."""
        # Create cache key from FK result poses
        fk_poses_key = []
        for i, mesh in enumerate(fk_result):
            mesh_name = get_mesh_name(mesh, i)
            if mesh_name in self.mesh_points or not self.mesh_points:
                pose_flat = fk_result[mesh].flatten()
                fk_poses_key.extend(np.round(pose_flat, 4).tolist())
        cache_key = tuple(fk_poses_key)
        
        if cache_key in self.world_points_cache:
            return self.world_points_cache[cache_key]
        
        # Sample mesh points if not done yet
        if not self.mesh_points:
            self._sample_mesh_points(fk_result)
        
        # Build mapping from mesh_name to FK result pose
        fk_poses = {}
        for i, mesh in enumerate(fk_result):
            mesh_name = get_mesh_name(mesh, i)
            if mesh_name in self.mesh_points:
                fk_poses[mesh_name] = fk_result[mesh]
        
        # Transform all points to world coordinates (vectorized)
        all_world_points = []
        for mesh_name, local_points in self.mesh_points.items():
            if mesh_name not in fk_poses:
                continue
            pose = torch.as_tensor(fk_poses[mesh_name], dtype=self.dtype, device=self.device)
            ones = torch.ones((local_points.shape[0], 1), device=self.device, dtype=self.dtype)
            points_h = torch.cat([local_points, ones], dim=1)  # (N, 4)
            world_pts = torch.mm(points_h, pose.T)[:, :3]  # (N, 3)
            all_world_points.append(world_pts)
        
        assert len(all_world_points) > 0, "No valid robot points found"
        result = torch.cat(all_world_points, dim=0)  # [num_points, 3]
        
        # Cache result
        self.world_points_cache[cache_key] = result
        return result

    def resample(self, total_samples: int | None = None) -> None:
        """Force re-sampling of robot mesh points for subsequent evaluations."""
        if total_samples is not None and total_samples > 0:
            self.total_samples = total_samples
        self.mesh_points.clear()
        self.world_points_cache.clear()

def get_robot_transform(
    robot_serial,
    transforms_file=DEFAULT_GRIPPER2WRIST_TRANSFORMS_PATH,
):
    """Get gripper-to-wrist transform for a robot (no fallback)."""
    all_transforms = load_robot_transforms(transforms_file)
    if robot_serial not in all_transforms:
        available = list(all_transforms.keys())
        raise ValueError(
            f"Robot serial {robot_serial} not found in transforms file {transforms_file}. "
            f"Available robot serials: {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    return np.array(all_transforms[robot_serial])
