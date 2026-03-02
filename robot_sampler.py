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
import torch
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from pointworld.urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()
import urdfpy
from utils import build_pk_chain_from_urdf

# Global gripper keywords for different robot types
GRIPPER_KEYWORDS = ['finger', 'knuckle', 'robotiq', 'gripper']
# panda hand mesh names:
GRIPPER_KEYWORDS += ['Part__Feature002_011', 'Part__Feature005_005', 'Part__Feature_011', 'Part__Feature001_011', 'Part__Feature005_000']
R1PRO_INERTIAL_FRAME_OFFSET = torch.tensor([0.049262, 0.000088054, -0.18255], dtype=torch.float32)
ROBOTOIQ_MIMIC_JOINTS = (
    ('finger_joint', 1.0),
    ('left_inner_knuckle_joint', 1.0),
    ('left_inner_finger_joint', -1.0),
    ('right_inner_knuckle_joint', -1.0),
    ('right_inner_finger_joint', 1.0),
    ('right_outer_knuckle_joint', -1.0),
)

def build_robotiq_joint_dict(
    finger_joint: torch.Tensor,
    joint_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Expand a single Robotiq finger joint position into its mimic joints.

    Uses the same sign conventions as the deploy PK mapping (see deploy/robots.py).
    Expects finger_joint to be a tensor with batch dimension (B,) or (B,1).
    """
    missing = [name for name, _ in ROBOTOIQ_MIMIC_JOINTS if name not in joint_names]
    if missing:
        raise KeyError(f"Missing Robotiq joints in URDF: {missing}")
    fj = finger_joint.reshape(-1)
    return {name: (fj * scale) for name, scale in ROBOTOIQ_MIMIC_JOINTS}

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


def is_gripper(mesh_name: str) -> bool:
    """Check if a mesh is a gripper."""
    return any(keyword.lower() in mesh_name.lower() for keyword in GRIPPER_KEYWORDS)

def is_gripper_filtered(mesh_name: str, gripper_filter: str) -> bool:
    """
    Check if a mesh should be included based on gripper filter.
    
    Args:
        mesh_name: Name of the mesh
        gripper_filter: Filter type ('both', 'left', 'right')
        
    Returns:
        bool: True if mesh should be included
    """
    if not is_gripper(mesh_name):
        return True  # Non-gripper meshes are always included
    
    mesh_name_lower = mesh_name.lower()
    
    if gripper_filter == 'both':
        return True
    elif gripper_filter == 'left':
        # Include left gripper meshes and exclude right gripper meshes
        return 'left' in mesh_name_lower and 'right' not in mesh_name_lower
    elif gripper_filter == 'right':
        # Include right gripper meshes and exclude left gripper meshes
        return 'right' in mesh_name_lower and 'left' not in mesh_name_lower
    else:
        raise ValueError(f"Invalid gripper_filter: {gripper_filter}. Must be 'both', 'left', or 'right'")

def convert_joints_to_dict(joint_tensor: torch.Tensor, actuated_joints: List[str]) -> Tuple[Dict[str, torch.Tensor], Tuple]:
    """
    Convert joint tensor to dictionary format for RobotSampler.fk().
    
    This is a class-independent helper function that can be used across different classes.
    
    Args:
        joint_tensor: (..., n_joints) tensor of joint positions
        actuated_joints: List of joint names in the same order as joint_tensor
        
    Returns:
        Tuple of (joint_dict, original_shape):
            - joint_dict: Dictionary mapping joint names to flattened tensors
            - original_shape: Original shape of joint_tensor (all dimensions except last)
    """
    assert joint_tensor.shape[-1] == len(actuated_joints), f"joint_tensor.shape[-1] ({joint_tensor.shape[-1]}) != len(actuated_joints) ({len(actuated_joints)})"
    joint_dict = {}
    original_shape = joint_tensor.shape[:-1]  # All dimensions except last
    
    for i, joint_name in enumerate(actuated_joints):
        # Extract joint values and preserve batch dimensions
        joint_values = joint_tensor[..., i]  # (...,)
        # Flatten all batch dimensions for fk(), then we'll reshape after
        joint_dict[joint_name] = joint_values.reshape(-1)  # (batch_size,)
        
    return joint_dict, original_shape


def get_mesh_name(mesh, idx):
    """Get mesh name from trimesh object."""
    try:
        return f'{mesh.source.file_name.lower()}_{idx}'
    except AttributeError:
        return f'{mesh.metadata.get("name", mesh.metadata.get("file_name", f"unknown")).lower()}_{idx}'


class RobotSampler:
    """
    GPU-accelerated robot point sampler using pytorch_kinematics for batched forward kinematics.
    
    This class handles presampling of robot mesh points and efficient batched transformation
    using PyTorch tensors for GPU acceleration.
    """
    
    def __init__(
        self,
        urdf_path: str,
        gripper_only: bool = True,
        device: str = 'cuda',
        apply_r1pro_inertial_frame_offset: bool = False,
        link_whitelist: Optional[List[str]] = None,
    ):
        """
        Initialize the GPU-accelerated RobotSampler.
        
        Args:
            urdf_path: Path to the robot URDF file
            gripper_only: If True, only sample points from gripper-related meshes
            device: Device to use for computation ('cuda' or 'cpu')
            apply_r1pro_inertial_frame_offset: Whether to apply R1Pro-specific frame offset
            link_whitelist: Optional list of link names to keep when sampling meshes
        """
        self.urdf_path = urdf_path
        self.gripper_only = gripper_only
        self.apply_r1pro_inertial_frame_offset = apply_r1pro_inertial_frame_offset
        self._link_whitelist = set(link_whitelist) if link_whitelist is not None else None

        # Set device (prefer GPU if available)
        self.device = torch.device(device)
        self.dtype = torch.float32
        
        # Validate URDF path
        assert os.path.exists(urdf_path), f"URDF file not found: {urdf_path}"
        
        # Initialize pytorch_kinematics chain
        self._init_kinematics()
        
        # Cache for presampled data
        self._presampled_points: Optional[Dict[str, torch.Tensor]] = None
        self._presampled_normals: Optional[Dict[str, torch.Tensor]] = None
        self._mesh_names: Optional[List[str]] = None
        # Precomputed visual offsets (mesh frame -> parent link frame)
        self._mesh_offsets: Optional[Dict[str, torch.Tensor]] = None
        # Mapping from mesh name to its parent link name
        self._mesh_to_link: Optional[Dict[str, str]] = None
        
        # Initialize mesh mappings to enable fk() calls without point presampling
        self._initialize_mesh_mappings()
        
    def _init_kinematics(self):
        """Initialize pytorch_kinematics chain from URDF."""
        # Load URDF for mesh extraction
        self.robot_urdf = urdfpy.URDF.load(self.urdf_path)
        
        # Load URDF for pytorch_kinematics
        with open(self.urdf_path, "rb") as f:
            urdf_string = f.read()
        
        # Build kinematic chain
        self.chain = build_pk_chain_from_urdf(urdf_string)
        self.chain = self.chain.to(device=self.device, dtype=self.dtype)
        
        # Extract joint information
        self.joint_names = []
        self.joint_defaults = {}
        # Joint metadata ------------------------------------------------------
        self.joint_limits = {}  # Dict mapping joint names to (lower, upper) limits

        joints_list = self.chain.get_joints()
        # pytorch_kinematics provides convenient bulk limits extraction; we
        # prefer that to manual per-joint parsing to faithfully reproduce the
        # values that were previously obtained via the legacy `Kinematics` API.
        # pytorch_kinematics returns joint limits as a 2 × N list where the first
        # row contains *all* lower bounds and the second row all upper bounds.
        # Transpose it once so we can index by joint.
        raw_limits = self.chain.get_joint_limits()  # [[lower_i...], [upper_i...]]
        bulk_limits = list(zip(*raw_limits))  # [(lower, upper), ...] length == N

        for idx, joint in enumerate(joints_list):
            # Skip fixed joints – we do not actuate them nor expose them to the
            # MPPI controller.
            if getattr(joint, "joint_type", "fixed") == "fixed":
                continue

            joint_name = joint.name
            self.joint_names.append(joint_name)
            self.joint_defaults[joint_name] = 0.0

            # Use limits returned by pytorch_kinematics whenever available.
            lower, upper = bulk_limits[idx]
            assert lower is not None and upper is not None, f"Joint {joint_name} has no limits"
            self.joint_limits[joint_name] = (float(lower), float(upper))
        
        print(f"Initialized RobotSampler with {len(self.joint_names)} movable joints on {self.device}")
    
    def _initialize_mesh_mappings(self) -> None:
        """
        Initialize mesh-to-link mappings and visual offsets without point sampling.
        
        This is called during initialization to enable fk() calls without requiring
        actual point presampling.
        """
        # Get reference configuration (all joints at 0)
        reference_cfg = self.joint_defaults.copy()
        
        # Get visual meshes in reference pose using urdfpy
        fk_ref = self.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
        
        # Initialize mappings
        self._mesh_offsets = {}
        self._mesh_to_link = {}
        self._mesh_names = []
        
        # Build zero joint configuration for reference link transforms
        zero_cfg = {jn: torch.zeros(1, device=self.device, dtype=self.dtype) for jn in self.joint_names}
        link_tf_ref = self.chain.forward_kinematics(zero_cfg)

        for i, mesh in enumerate(fk_ref):
            mesh_name = get_mesh_name(mesh, i)
            
            self._mesh_names.append(mesh_name)

            # Compute mesh world transform (numpy -> torch)
            mesh_T_np = fk_ref[mesh]  # numpy (4,4)
            mesh_T = torch.from_numpy(mesh_T_np.astype(np.float32)).to(self.device, dtype=self.dtype)

            # Find best matching link by comparing transforms directly
            matched_link_name = None
            min_err = float('inf')
            
            for link_name, link_tf in link_tf_ref.items():
                link_mat = link_tf.get_matrix()[0].to(self.device, dtype=self.dtype)  # (4,4)
                # Compute Frobenius norm difference
                err = torch.norm(link_mat - mesh_T, p='fro').item()
                if err < min_err:
                    min_err = err
                    matched_link_name = link_name
            assert matched_link_name is not None, f"No link found for mesh '{mesh_name}'"

            # Extract link transform (4×4)
            link_T = link_tf_ref[matched_link_name].get_matrix()[0].to(self.device, dtype=self.dtype)

            # Offset = link_T^{-1} * mesh_T
            offset_T = torch.matmul(torch.inverse(link_T), mesh_T)
            self._mesh_offsets[mesh_name] = offset_T.detach()
            self._mesh_to_link[mesh_name] = matched_link_name

        # Tri-gripper disambiguation: ensure two tri finger meshes bind to distinct links
        if ("panda_finger_joint1" in self.joint_names) and ("panda_finger_joint2" in self.joint_names):
            tri_meshes = sorted([mn for mn in self._mesh_names if "tri_finger.obj" in mn])
            if len(tri_meshes) >= 2 and ("panda_rightfinger" in link_tf_ref) and ("panda_leftfinger" in link_tf_ref):
                self._mesh_to_link[tri_meshes[0]] = "panda_rightfinger"
                self._mesh_to_link[tri_meshes[1]] = "panda_leftfinger"
        
        print(f"Initialized mesh mappings for {len(self._mesh_names)} meshes")
    
    def presample(self, num_points: int, gripper_filter: str = 'both', seed: int | None = None) -> None:
        """
        Pre-sample points and normals from robot visual meshes.
        
        Uses area-proportional allocation and applies special handling for hand_camera_part.
        
        Args:
            num_points: Total number of points to sample across all meshes
            gripper_filter: Filter for which grippers to include ('both', 'left', 'right')
        """
        assert num_points > 0, "num_points must be greater than 0"      
        # Get reference configuration (all joints at 0)
        reference_cfg = self.joint_defaults.copy()
        
        # Get visual meshes in reference pose using urdfpy
        fk_ref = self.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
        
        # Filter and collect mesh information
        mesh_data = []
        for i, mesh in enumerate(fk_ref):
            mesh_name = get_mesh_name(mesh, i)

            # Filter for gripper meshes if needed
            if self.gripper_only and not is_gripper(mesh_name):
                continue
            
            # Apply new gripper filter
            if self.gripper_only:
                if not is_gripper_filtered(mesh_name, gripper_filter):
                    continue
            
            # Only include meshes that were initialized in mesh mappings
            if mesh_name not in self._mesh_names:
                continue

            if self._link_whitelist is not None:
                link_name = self._mesh_to_link.get(mesh_name, None)
                if link_name not in self._link_whitelist:
                    continue
            
            # Calculate effective area with special handling for hand_camera_part
            effective_area = mesh.area
            if 'hand_camera_part' in mesh_name.lower():
                effective_area *= 0.000001  # Reduce camera mount influence
                
            mesh_data.append({
                'name': mesh_name,
                'mesh': mesh,
                'area': effective_area
            })
        
        if not mesh_data:
            raise ValueError("No valid meshes found for sampling")

        # Calculate area-proportional point allocation
        total_area = sum(data['area'] for data in mesh_data)
        
        sampled_points = {}
        sampled_normals = {}
        rng = np.random.RandomState(int(seed) % (2**32 - 1) if seed is not None else None)
        allocated = 0
        
        for i, data in enumerate(mesh_data):
            mesh_name = data['name']
            mesh = data['mesh']
            
            # Allocate points (last mesh gets remainder)
            if i == len(mesh_data) - 1:
                count = num_points - allocated
            else:
                ratio = data['area'] / total_area
                count = int(num_points * ratio)
            
            count = max(0, count)
            allocated += count
            
            if count > 0:
                # Deterministic surface sampling using shared helper
                points, normals = deterministic_sample_surface(mesh, count, rng)
                sampled_points[mesh_name] = torch.from_numpy(points).to(self.device)
                sampled_normals[mesh_name] = torch.from_numpy(normals).to(self.device)
            else:
                # Empty tensors for meshes with 0 points
                sampled_points[mesh_name] = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
                sampled_normals[mesh_name] = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        
        # Cache presampled data
        self._presampled_points = sampled_points
        self._presampled_normals = sampled_normals
        
        total_sampled = sum(points.shape[0] for points in sampled_points.values())
        # print(f"Presampled {total_sampled} points from {len(mesh_data)} meshes")
    
    def fk(
        self,
        joint_values: Dict[str, torch.Tensor],
        link_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform batched forward kinematics and return transforms for the *specified* links only.

        Args:
            joint_values: Dictionary mapping joint names to batched joint values. Each tensor can
                          have shape (batch_size,) or (batch_size, 1).
            link_names: Optional list of link (frame) names for which to compute transforms.  
                        When None, only the set of links referenced by the internal mesh mapping
                        (i.e. `self._mesh_to_link`) are evaluated.  Supplying a minimal list here
                        can drastically speed-up FK since the underlying pytorch_kinematics
                        implementation avoids traversing the entire kinematic tree.
        Returns:
            Dictionary mapping each requested link name to its batched 4×4 homogeneous transform
            matrix with shape (batch_size, 4, 4).
        """

        assert self._mesh_to_link is not None, "Mesh mappings not initialized"
        
        # Prepare complete joint configuration with defaults
        complete_joint_values = {}
        
        # Infer batch size from joint values
        batch_size = None
        for joint_name, values in joint_values.items():
            if joint_name in self.joint_names:
                if isinstance(values, (list, np.ndarray)):
                    values = torch.tensor(values, dtype=self.dtype, device=self.device)
                elif not isinstance(values, torch.Tensor):
                    values = torch.tensor([values], dtype=self.dtype, device=self.device)
                
                values = values.to(self.device, dtype=self.dtype)
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                
                if batch_size is None:
                    batch_size = values.shape[0]
                elif values.shape[0] != batch_size:
                    raise ValueError(f"Inconsistent batch sizes in joint_values")
                
                complete_joint_values[joint_name] = values
        
        assert batch_size is not None, "No valid joint values provided"
        
        # Fill in missing joints with defaults
        for joint_name in self.joint_names:
            if joint_name not in complete_joint_values:
                default_value = torch.full((batch_size,), self.joint_defaults[joint_name], 
                                         dtype=self.dtype, device=self.device)
                complete_joint_values[joint_name] = default_value
        
        # ------------------------------------------------------------------
        # Determine which links we actually need FK for
        # ------------------------------------------------------------------
        if link_names is None or len(link_names) == 0:
            # Default to links referenced by the mesh→link mapping
            link_names = list(dict.fromkeys(self._mesh_to_link.values()))  # dedup while preserving order

        # Convert link names to frame indices (cached internally by pytorch_kinematics)
        frame_indices = self.chain.get_frame_indices(*link_names)

        # Perform batched FK *only* for the requested frames
        transform_dict = self.chain.forward_kinematics(
            complete_joint_values,
            frame_indices=frame_indices,
        )

        # ------------------------------------------------------------------
        # Assemble output dictionary – convert Transform3d → tensor
        # ------------------------------------------------------------------
        link_transforms = {ln: transform_dict[ln].get_matrix() for ln in link_names}

        # Apply R1Pro base link offset as a left-multiplied translation in the robot frame
        if self.apply_r1pro_inertial_frame_offset:
            T_offset = torch.eye(4, dtype=self.dtype, device=self.device)
            T_offset[:3, 3] = R1PRO_INERTIAL_FRAME_OFFSET
            for ln in link_transforms.keys():
                T_link = link_transforms[ln]  # (B,4,4)
                # Pre-multiply: T' = T_offset @ T_link
                link_transforms[ln] = torch.matmul(T_offset, T_link)

        return link_transforms
  
    def compute_points(
        self,
        joint_values: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute robot points, normals, and colors for all timesteps using batched operations.
        
        Args:
            joint_values: Dictionary mapping joint names to values with batch dimension
                         Each tensor should have shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Tuple of (points, normals, colors):
                - points: (batch_size, num_points, 3) torch.Tensor
                - colors: (batch_size, num_points, 3) torch.Tensor (magenta)
                - normals: (batch_size, num_points, 3) torch.Tensor  
        """
        # Get link transforms using the fk function
        link_transforms = self.fk(joint_values)
        
        # Infer batch size from transforms
        batch_size = next(iter(link_transforms.values())).shape[0] if link_transforms else 1
        # Count total points
        total_points = sum(points.shape[0] for points in self._presampled_points.values())
        
        # Initialize output tensors
        all_points = torch.zeros((batch_size, total_points, 3), dtype=self.dtype, device=self.device)
        all_normals = torch.zeros((batch_size, total_points, 3), dtype=self.dtype, device=self.device)
        all_colors = torch.zeros((batch_size, total_points, 3), dtype=torch.uint8, device=self.device)
        
        # Set magenta color (255, 0, 255)
        all_colors[:, :, 0] = 255  # Red
        all_colors[:, :, 2] = 255  # Blue
        
        # Transform points for each mesh
        point_idx = 0
        for mesh_name in self._presampled_points.keys():
            local_points = self._presampled_points[mesh_name]  # (n_points, 3)
            local_normals = self._presampled_normals[mesh_name]      # (n_points, 3)
            
            n_points = local_points.shape[0]
            if n_points == 0:
                continue
            
            # Get mesh transform by looking up its parent link and applying visual offset
            if self._mesh_to_link is not None:
                link_name = self._mesh_to_link.get(mesh_name, None)
                if link_name is not None and link_name in link_transforms:
                    # Get link transform and apply visual offset
                    T_link = link_transforms[link_name]  # (batch_size, 4, 4)
                    offset_T = self._mesh_offsets[mesh_name]  # (4, 4)
                    # Apply offset: T_mesh = T_link @ offset_T
                    T = torch.matmul(T_link, offset_T)
                else:
                    continue  # Skip this mesh if no link transform available
            else:
                continue  # Skip if mesh-to-link mapping not available
            
            # Transform points: homogeneous coordinates
            local_points_homo = torch.cat([
                local_points, 
                torch.ones(n_points, 1, dtype=self.dtype, device=self.device)
            ], dim=1)  # (n_points, 4)
            
            # Batched matrix multiplication: (batch_size, 4, 4) @ (4, n_points) -> (batch_size, 4, n_points)
            world_points_homo = torch.bmm(T, local_points_homo.T.unsqueeze(0).repeat(batch_size, 1, 1))
            world_points = world_points_homo[:, :3, :].transpose(1, 2)  # (batch_size, n_points, 3)
            
            # Transform normals: use rotation part only
            R = T[:, :3, :3]  # (batch_size, 3, 3)
            world_normals = torch.bmm(R, local_normals.T.unsqueeze(0).repeat(batch_size, 1, 1))
            world_normals = world_normals.transpose(1, 2)  # (batch_size, n_points, 3)
            
            # Add to output arrays
            all_points[:, point_idx:point_idx + n_points] = world_points
            all_normals[:, point_idx:point_idx + n_points] = world_normals
            
            point_idx += n_points

        return all_points, all_colors, all_normals
    
    def get_joint_limits(self) -> torch.Tensor:
        """
        Get joint limits for all movable joints.
        
        Returns:
            torch.Tensor: (n_joints, 2) tensor where each row is [lower, upper] limits for that joint
        """
        limits_list = []
        for joint_name in self.joint_names:
            lower, upper = self.joint_limits[joint_name]
            limits_list.append([lower, upper])
        
        return torch.tensor(limits_list, dtype=self.dtype, device=self.device)
