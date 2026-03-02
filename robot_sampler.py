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
import trimesh
from typing import Dict, List, Tuple, Optional
from numba import njit
import transform_utils


@njit(cache=True, fastmath=True, nogil=True)
def _transform_points_batch(local_points, poses_mat):
    """Transform a batch of points using transformation matrices"""
    n_points = local_points.shape[0]
    n_frames = poses_mat.shape[0]
    world_points = np.empty((n_frames, n_points, 3), dtype=local_points.dtype)
    
    for t in range(n_frames):
        # Extract transformation matrix for this frame
        T = poses_mat[t]
        for i in range(n_points):
            # Transform point: p_world = T @ [p_local; 1]
            x, y, z = local_points[i, 0], local_points[i, 1], local_points[i, 2]
            world_points[t, i, 0] = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
            world_points[t, i, 1] = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
            world_points[t, i, 2] = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
    
    return world_points


@njit(cache=True, fastmath=True, nogil=True)
def _transform_normals_batch(local_normals, poses_mat):
    """Transform a batch of normals using rotation matrices"""
    n_normals = local_normals.shape[0]
    n_frames = poses_mat.shape[0]
    world_normals = np.empty((n_frames, n_normals, 3), dtype=local_normals.dtype)
    
    for t in range(n_frames):
        # Extract rotation matrix for this frame (3x3 upper-left)
        R = poses_mat[t, :3, :3]
        for i in range(n_normals):
            # Transform normal: n_world = R @ n_local
            x, y, z = local_normals[i, 0], local_normals[i, 1], local_normals[i, 2]
            world_normals[t, i, 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
            world_normals[t, i, 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
            world_normals[t, i, 2] = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z
    
    return world_normals

def get_mesh_stable_id(mesh, idx) -> str:
    try:
        return f'{mesh.source.file_name.lower()}_{idx}'
    except AttributeError:
        return f'{mesh.metadata.get("name", mesh.metadata.get("file_name", f"unknown")).lower()}_{idx}'

def deterministic_sample_surface(mesh, count: int, rng: np.random.RandomState):
    """Deterministically sample points and normals on a mesh surface.

    Args:
        mesh: trimesh.Trimesh
        count: number of samples
        rng: numpy RandomState (caller-controlled)

    Returns:
        points: (count, 3) float32
        normals: (count, 3) float32, unit length
    """
    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.area_faces.astype(np.float64)
    p = areas / areas.sum()
    face_idx = rng.choice(len(faces), size=count, p=p)
    tri = verts[faces[face_idx]]  # (count, 3, 3)
    r = rng.random((count, 2)).astype(np.float64)
    over = (r.sum(axis=1) > 1.0)
    r[over] = 1.0 - r[over]
    u = r[:, 0:1]
    v = r[:, 1:2]
    w = 1.0 - u - v
    points = (w * tri[:, 0] + u * tri[:, 1] + v * tri[:, 2]).astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-9)
    return points, normals

class RobotSampler:
    """
    Handles robot point sampling for both simulation and real-world data.
    
    Simulation mode (langtable, rlbench): Uses mesh-based sampling from .obj files
    Real-world mode (droid): Uses URDF-based sampling with forward kinematics
    """
    
    def __init__(self, domain: str, robot_sampler_dir: str = None, urdf_path: str = None, gripper_only: bool = False):
        """
        Initialize the RobotSampler.

        Args:
            domain (str): Domain name ('langtable', 'rlbench', 'droid')
            robot_sampler_dir (str, optional): Path to mesh directory for sim domains
            urdf_path (str, optional): Path to URDF file for real domains
            gripper_only (bool): If True, only load gripper meshes
        """
        domain = domain.split('_')[0]
        self.domain = domain
        self.gripper_only = gripper_only
        
        # Determine mode based on domain
        sim_domains = ['langtable', 'rlbench']
        real_domains = ['droid']
        
        if domain in sim_domains:
            self.mode = 'sim'
            assert robot_sampler_dir is not None, f"robot_sampler_dir required for sim domain {domain}"
            self._init_sim_mode(robot_sampler_dir)
        elif domain in real_domains:
            self.mode = 'real'
            assert urdf_path is not None, f"urdf_path required for real domain {domain}"
            assert os.path.exists(urdf_path), f"URDF file not found: {urdf_path}"
            self._init_real_mode(urdf_path)
        else:
            raise ValueError(f"Unknown domain: {domain}. Supported: {sim_domains + real_domains}")
        
        # Cache for pre-sampled points and normals
        self._presampled_points: Optional[Dict[str, np.ndarray]] = None
        self._presampled_normals: Optional[Dict[str, np.ndarray]] = None
        
        print(f"RobotSampler initialized in {self.mode} mode for domain {domain}")

    def _init_sim_mode(self, robot_sampler_dir: str):
        """Initialize simulation mode with mesh loading"""
        robot_sampler_dir = os.path.join(robot_sampler_dir, self.domain.split('_')[0])
        assert os.path.isdir(robot_sampler_dir), f"Robot mesh directory not found: {robot_sampler_dir}"

        self.mesh_objects: Dict[str, trimesh.Trimesh] = {}
        self.mesh_areas: Dict[str, float] = {}

        # Load mesh files
        for filename in os.listdir(robot_sampler_dir):
            if filename.endswith(".obj"):
                mesh_name = filename[:-4]  # Remove .obj extension
                
                # Filter for gripper meshes if gripper_only is True
                if self.gripper_only and not self._is_gripper_mesh(mesh_name):
                    continue
                
                mesh_path = os.path.join(robot_sampler_dir, filename)
                mesh = self._load_and_process_mesh(mesh_path, mesh_name)
                
                if mesh is not None:
                    self.mesh_objects[mesh_name] = mesh
                    self.mesh_areas[mesh_name] = mesh.area

        # Add domain-specific primitives
        if self.domain.split('_')[0] == 'langtable':
            self._add_langtable_primitives()

        assert len(self.mesh_objects) > 0, f"No valid robot meshes loaded from {robot_sampler_dir}"

    def _init_real_mode(self, urdf_path: str):
        """Initialize real-world mode with URDF loading"""
        try:
            if not hasattr(np, "float"):
                np.float = float  # type: ignore[attr-defined]
            import urdfpy
        except ImportError:
            raise ImportError("urdfpy is required for real mode. Install with: pip install urdfpy")
        
        self.urdf_path = urdf_path
        self.robot_urdf = urdfpy.URDF.load(urdf_path)

    def _load_and_process_mesh(self, mesh_path: str, mesh_name: str) -> trimesh.Trimesh:
        """Load and process a mesh file, applying domain-specific transformations."""
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Handle non-Trimesh objects
        if isinstance(mesh, list):
            assert len(mesh) > 0 and isinstance(mesh[0], trimesh.Trimesh), f"Invalid mesh list in {mesh_path}"
            mesh = mesh[0]
        elif isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        assert isinstance(mesh, trimesh.Trimesh), f"Could not load valid Trimesh from {mesh_path}"
        
        # Apply domain-specific scaling
        if self.domain.split('_')[0] == 'langtable' and mesh_name == 'gripper_headLink':
            # Apply 0.001 scaling to head.obj as specified in URDF
            mesh.apply_scale(0.001)
        
        assert mesh.area > 1e-6, f"Mesh {mesh_name} has insufficient surface area: {mesh.area}"
        return mesh

    def _add_langtable_primitives(self):
        """Add Language Table specific primitive objects."""
        # Create cylinder primitive for gripper tip as specified in URDF
        # <cylinder length="0.135" radius="0.0127" />
        cylinder_tip = trimesh.primitives.Cylinder(
            radius=0.0127,
            height=0.135,
            sections=32  # reasonable tessellation
        )
        
        # Convert to mesh and store
        self.mesh_objects['gripper_tipLink'] = cylinder_tip
        self.mesh_areas['gripper_tipLink'] = cylinder_tip.area

    def _is_gripper_mesh(self, mesh_name: str) -> bool:
        """Check if a mesh name corresponds to a gripper part."""
        mesh_name_lower = mesh_name.lower()
        if self.domain.split('_')[0] == 'langtable':
            keywords = ['gripper_headLink', 'gripper_tipLink']
        elif self.domain.split('_')[0] == 'rlbench':
            keywords = ['gripper', 'finger']
        return any(keyword in mesh_name_lower for keyword in keywords)

    def presample(self, num_points: int, seed: int | None = None) -> None:
        """
        Pre-sample points and normals from robot meshes, caching internally.
        
        Args:
            num_points (int): Total number of points to sample
        """
        if self.mode == 'sim':
            self._presample_sim_mode(num_points, seed=seed)
        elif self.mode == 'real':
            self._presample_real_mode(num_points, seed=seed)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _presample_sim_mode(self, num_points: int, seed: int | None = None):
        """Pre-sample points for simulation mode"""
        if num_points == 0:
            self._presampled_points = {}
            self._presampled_normals = {}
            return

        # Get available meshes
        available_meshes = list(self.mesh_objects.keys())
        assert len(available_meshes) > 0, "No meshes available for sampling"

        # Calculate point allocation based on surface area
        total_area = sum(self.mesh_areas[name] for name in available_meshes)
        assert total_area > 1e-6, f"Total surface area too small: {total_area}"

        sampled_points = {}
        sampled_normals = {}
        allocated = 0

        rng = np.random.RandomState(int(seed) % (2**32 - 1) if seed is not None else None)

        for i, mesh_name in enumerate(available_meshes):
            mesh = self.mesh_objects[mesh_name]
            
            # Allocate points proportionally (last mesh gets remainder)
            if i == len(available_meshes) - 1:
                count = num_points - allocated
            else:
                ratio = self.mesh_areas[mesh_name] / total_area
                count = int(num_points * ratio)
            
            count = max(0, count)
            allocated += count

            if count > 0:
                pts, nrm = deterministic_sample_surface(mesh, count, rng)
                sampled_points[mesh_name] = pts
                sampled_normals[mesh_name] = nrm
            else:
                sampled_points[mesh_name] = np.zeros((0, 3), dtype=np.float32)
                sampled_normals[mesh_name] = np.zeros((0, 3), dtype=np.float32)

        self._presampled_points = sampled_points
        self._presampled_normals = sampled_normals

    def _presample_real_mode(self, num_points: int, seed: int | None = None):
        """Pre-sample points for real-world mode using URDF"""
        # Get reference configuration
        reference_cfg = {'finger_joint': 0.0}
        for ji in range(7):
            reference_cfg[f'panda_joint{ji + 1}'] = 0.0
        
        # Get visual meshes in reference pose
        fk_ref = self.forward_kinematics(reference_cfg)
        
        # Calculate surface area for each mesh
        mesh_names = []
        mesh_objects = []
        mesh_areas = []
        
        # Define gripper keywords for filtering
        gripper_keywords = ['finger', 'knuckle', 'robotiq']
        
        for i, mesh in enumerate(fk_ref):
            mesh_name = get_mesh_stable_id(mesh, i)
            
            # Filter for gripper meshes if gripper_only is True
            if self.gripper_only:
                is_gripper = any(keyword.lower() in mesh_name.lower() for keyword in gripper_keywords)
                if not is_gripper:
                    continue
            
            mesh_objects.append(mesh)
            
            # Apply area correction for camera mount if needed
            effective_area = mesh.area
            if 'hand_camera_part' in mesh_name.lower():
                effective_area *= 0.000001
            assert effective_area < 1, f'effective_area: {effective_area} for mesh {mesh_name} is too large'
            
            mesh_names.append(mesh_name)
            mesh_areas.append(effective_area)
        
        # Ensure deterministic ordering by sorting by stable mesh id
        if len(mesh_names) > 1:
            order = sorted(range(len(mesh_names)), key=lambda i: mesh_names[i])
            mesh_names  = [mesh_names[i] for i in order]
            mesh_objects= [mesh_objects[i] for i in order]
            mesh_areas  = [mesh_areas[i] for i in order]

        total_area = sum(mesh_areas)
        
        # Allocate points based on surface area
        mesh_point_counts = {}
        for i, (name, area) in enumerate(zip(mesh_names, mesh_areas)):
            count = max(1, int(num_points * (area / total_area)))
            if i == len(mesh_names) - 1:  # Last mesh gets remaining points
                allocated_so_far = sum(mesh_point_counts.values())
                count = num_points - allocated_so_far
            mesh_point_counts[name] = count
        
        # Pre-sample points from each mesh (deterministic)
        pre_sampled_points = {}
        pre_sampled_normals = {}
        rng = np.random.RandomState(int(seed) % (2**32 - 1) if seed is not None else None)
        for name, mesh, count in zip(mesh_names, mesh_objects, [mesh_point_counts[name] for name in mesh_names]):
            if count > 0:
                pts, nrm = deterministic_sample_surface(mesh, count, rng)
                pre_sampled_points[name] = pts
                pre_sampled_normals[name] = nrm
        
        self._presampled_points = pre_sampled_points
        self._presampled_normals = pre_sampled_normals

    def compute_points(self, input_data: Dict, num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute robot points, normals, and colors for all timesteps.
        
        Args:
            input_data (Dict): Input data containing:
                - For sim mode: {'mesh_trajectories': Dict[str, np.ndarray]}
                - For real mode: {'joint_positions': np.ndarray, 'gripper_positions': np.ndarray}
            num_frames (int): Number of frames
            
        Returns:
            Tuple of (robot_points, robot_normals, robot_colors):
                - robot_points: (T, N, 3) np.float32
                - robot_normals: (T, N, 3) np.float32  
                - robot_colors: (T, N, 3) np.uint8 (magenta)
        """
        assert self._presampled_points is not None, "Must call presample() first"
        
        if self.mode == 'sim':
            return self._compute_points_sim_mode(input_data, num_frames)
        elif self.mode == 'real':
            return self._compute_points_real_mode(input_data, num_frames)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _compute_points_sim_mode(self, input_data: Dict, num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute points for simulation mode"""
        assert 'mesh_trajectories' in input_data, "mesh_trajectories required for sim mode"
        robot_trajectories = input_data['mesh_trajectories']
        
        # Count total robot points
        robot_count = sum(p.shape[0] for p in self._presampled_points.values())
        
        # Create arrays for robot data
        robot_points_array = np.zeros((num_frames, robot_count, 3), dtype=np.float32)
        robot_normals_array = np.zeros((num_frames, robot_count, 3), dtype=np.float32)
        robot_colors_array = np.zeros((num_frames, robot_count, 3), dtype=np.uint8)
        robot_colors_array[:, :, 0] = 255  # Red
        robot_colors_array[:, :, 1] = 0    # Green  
        robot_colors_array[:, :, 2] = 255  # Blue (Magenta)
        
        # Transform points using optimized batch operations
        point_index = 0
        for mesh_name in self._presampled_points.keys():
            local_points = self._presampled_points[mesh_name]
            local_normals = self._presampled_normals[mesh_name]
            
            if local_points.shape[0] == 0:
                continue
            assert mesh_name in robot_trajectories, f"Trajectory not found for mesh {mesh_name}"
            trajectory = robot_trajectories[mesh_name]
            n_points = local_points.shape[0]
            
            # Convert poses to transformation matrices
            poses_mat = transform_utils.convert_pose_quat2mat(trajectory)
            
            # Transform points and normals
            world_points = _transform_points_batch(local_points, poses_mat)
            world_normals = _transform_normals_batch(local_normals, poses_mat)
            
            # Add to arrays
            robot_points_array[:, point_index:point_index + n_points] = world_points
            robot_normals_array[:, point_index:point_index + n_points] = world_normals
            
            point_index += n_points
        
        return robot_points_array, robot_normals_array, robot_colors_array

    def _compute_points_real_mode(self, input_data: Dict, num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute points for real-world mode"""
        assert 'joint_positions' in input_data, "joint_positions required for real mode"
        assert 'gripper_positions' in input_data, "gripper_positions required for real mode"
        
        joint_positions = input_data['joint_positions']  # (T, 7)
        gripper_positions = input_data['gripper_positions']  # (T, 1)
        
        assert joint_positions.shape[0] == num_frames, f"joint_positions length {joint_positions.shape[0]} != num_frames {num_frames}"
        assert gripper_positions.shape[0] == num_frames, f"gripper_positions length {gripper_positions.shape[0]} != num_frames {num_frames}"
        
        # Count total robot points
        robot_count = sum(p.shape[0] for p in self._presampled_points.values())
        
        # Create arrays for robot data  
        robot_points_array = np.zeros((num_frames, robot_count, 3), dtype=np.float32)
        robot_normals_array = np.zeros((num_frames, robot_count, 3), dtype=np.float32)
        robot_colors_array = np.zeros((num_frames, robot_count, 3), dtype=np.uint8)
        robot_colors_array[:, :, 0] = 255  # Red
        robot_colors_array[:, :, 1] = 0    # Green
        robot_colors_array[:, :, 2] = 255  # Blue (Magenta)
        
        # Pre-compute FK results for all timesteps
        mesh_transforms = {}  # Store transformation matrices for each mesh across time
        
        # First pass: compute FK for all timesteps and organize by mesh
        for t in range(num_frames):
            # Construct configuration dictionary
            cfg = {'finger_joint': float(gripper_positions[t])}
            for ji in range(7):
                cfg[f'panda_joint{ji + 1}'] = float(joint_positions[t][ji])
            
            # Compute forward kinematics
            fk_result = self.robot_urdf.visual_trimesh_fk(cfg=cfg)
            
            # Store transforms for each mesh
            for i, mesh in enumerate(fk_result):
                mesh_name = get_mesh_stable_id(mesh, i)
                
                if mesh_name not in self._presampled_points:
                    continue
                
                # Initialize array for this mesh if first time
                if mesh_name not in mesh_transforms:
                    mesh_transforms[mesh_name] = np.zeros((num_frames, 4, 4), dtype=np.float32)
                
                # Store transformation matrix
                mesh_transforms[mesh_name][t] = fk_result[mesh]
        
        # Second pass: use optimized batch transformation functions
        point_index = 0
        for mesh_name in self._presampled_points.keys():
            if mesh_name not in mesh_transforms:
                continue
                
            local_points = self._presampled_points[mesh_name]
            local_normals = self._presampled_normals[mesh_name]
            transforms = mesh_transforms[mesh_name]  # (T, 4, 4)
            
            n_points = local_points.shape[0]
            
            # Transform points and normals using optimized functions
            world_points = _transform_points_batch(local_points, transforms)
            world_normals = _transform_normals_batch(local_normals, transforms)
            
            # Add to arrays
            robot_points_array[:, point_index:point_index + n_points] = world_points
            robot_normals_array[:, point_index:point_index + n_points] = world_normals
            
            point_index += n_points
        
        return robot_points_array, robot_normals_array, robot_colors_array

    def forward_kinematics(self, cfg: Dict) -> Dict:
        """
        Compute forward kinematics for a given joint configuration.
        Only works in 'real' mode.
        
        Args:
            cfg (Dict): Joint configuration dictionary.
            
        Returns:
            Dict: A dictionary mapping visual mesh objects to their poses.
        """
        assert self.mode == 'real', "forward_kinematics is only supported in 'real' mode"
        return self.robot_urdf.visual_trimesh_fk(cfg=cfg)
