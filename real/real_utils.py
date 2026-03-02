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
import numpy as np
from datetime import datetime
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def project_points_to_image(points_3d, transform_matrix, extrinsic, intrinsic, image_width, image_height):
    """
    Fused numba kernel to project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in local mesh coordinates
        transform_matrix: (4, 4) transformation matrix from mesh local to world coordinates
        extrinsic: (4, 4) world to camera transformation matrix
        intrinsic: (3, 3) camera intrinsic matrix
        image_width: image width in pixels
        image_height: image height in pixels
        
    Returns:
        projected_points: (M, 2) array of valid 2D points in image coordinates (subset of input points)
    """
    n_points = points_3d.shape[0]
    
    # Pre-allocate for projected points (worst case: all points are valid)
    projected_points = np.empty((n_points, 2), dtype=np.float32)
    valid_count = 0
    
    # Process each point
    for i in range(n_points):
        # Convert to homogeneous coordinates
        local_point = np.array([points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 1.0], dtype=np.float32)
        
        # Transform from local mesh coordinates to world coordinates
        world_point = np.zeros(4, dtype=np.float32)
        for j in range(4):
            world_point[j] = (transform_matrix[j, 0] * local_point[0] + 
                             transform_matrix[j, 1] * local_point[1] + 
                             transform_matrix[j, 2] * local_point[2] + 
                             transform_matrix[j, 3] * local_point[3])
        
        # Transform from world to camera coordinates
        cam_point = np.zeros(4, dtype=np.float32)
        for j in range(4):
            cam_point[j] = (extrinsic[j, 0] * world_point[0] + 
                           extrinsic[j, 1] * world_point[1] + 
                           extrinsic[j, 2] * world_point[2] + 
                           extrinsic[j, 3] * world_point[3])
        
        # Check if point is behind camera
        if cam_point[2] <= 0:
            continue
            
        # Project to image coordinates
        img_x = (intrinsic[0, 0] * cam_point[0] + intrinsic[0, 2] * cam_point[2]) / cam_point[2]
        img_y = (intrinsic[1, 1] * cam_point[1] + intrinsic[1, 2] * cam_point[2]) / cam_point[2]
        
        # Check if point is within image bounds
        if img_x >= 0 and img_x < image_width and img_y >= 0 and img_y < image_height:
            projected_points[valid_count, 0] = img_x
            projected_points[valid_count, 1] = img_y
            valid_count += 1
    
    # Return only the valid points
    return projected_points[:valid_count].copy()

@njit(cache=True, fastmath=True, nogil=True)
def generate_and_project_workspace_boundary(workspace_bounds_min, workspace_bounds_max, face_density, 
                                          extrinsic, intrinsic, image_width, image_height):
    """
    Fully fused numba kernel that generates workspace boundary points and projects them to image coordinates.
    
    Args:
        workspace_bounds_min: (3,) array of minimum workspace bounds [x_min, y_min, z_min]
        workspace_bounds_max: (3,) array of maximum workspace bounds [x_max, y_max, z_max]
        face_density: number of points per face edge
        extrinsic: (4, 4) world to camera transformation matrix
        intrinsic: (3, 3) camera intrinsic matrix
        image_width: image width in pixels
        image_height: image height in pixels
        
    Returns:
        projected_points: (M, 2) array of valid 2D points in image coordinates
    """
    # Estimate maximum number of points (6 faces * face_density^2, but we use conservative estimate)
    max_points = 6 * face_density * face_density
    projected_points = np.empty((max_points, 2), dtype=np.float32)
    valid_count = 0
    
    # Extract bounds
    x_min, y_min, z_min = workspace_bounds_min[0], workspace_bounds_min[1], workspace_bounds_min[2]
    x_max, y_max, z_max = workspace_bounds_max[0], workspace_bounds_max[1], workspace_bounds_max[2]
    
    # Generate step sizes
    y_step = (y_max - y_min) / (face_density - 1) if face_density > 1 else 0.0
    z_step = (z_max - z_min) / (face_density - 1) if face_density > 1 else 0.0
    x_step = (x_max - x_min) / (face_density - 1) if face_density > 1 else 0.0
    
    # X-constant faces (YZ planes)
    for x_face in (x_min, x_max):
        for y_idx in range(face_density):
            y = y_min + y_idx * y_step
            for z_idx in range(face_density):
                z = z_min + z_idx * z_step
                
                # Transform to homogeneous coordinates
                world_point = np.array([x_face, y, z, 1.0], dtype=np.float32)
                
                # Transform from world to camera coordinates
                cam_point = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    cam_point[j] = (extrinsic[j, 0] * world_point[0] + 
                                   extrinsic[j, 1] * world_point[1] + 
                                   extrinsic[j, 2] * world_point[2] + 
                                   extrinsic[j, 3] * world_point[3])
                
                # Check if point is behind camera
                if cam_point[2] <= 0:
                    continue
                    
                # Project to image coordinates
                img_x = (intrinsic[0, 0] * cam_point[0] + intrinsic[0, 2] * cam_point[2]) / cam_point[2]
                img_y = (intrinsic[1, 1] * cam_point[1] + intrinsic[1, 2] * cam_point[2]) / cam_point[2]
                
                # Check if point is within image bounds
                if img_x >= 0 and img_x < image_width and img_y >= 0 and img_y < image_height:
                    projected_points[valid_count, 0] = img_x
                    projected_points[valid_count, 1] = img_y
                    valid_count += 1
    
    # Y-constant faces (XZ planes)
    for y_face in (y_min, y_max):
        for x_idx in range(face_density):
            x = x_min + x_idx * x_step
            for z_idx in range(face_density):
                z = z_min + z_idx * z_step
                
                # Transform to homogeneous coordinates
                world_point = np.array([x, y_face, z, 1.0], dtype=np.float32)
                
                # Transform from world to camera coordinates
                cam_point = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    cam_point[j] = (extrinsic[j, 0] * world_point[0] + 
                                   extrinsic[j, 1] * world_point[1] + 
                                   extrinsic[j, 2] * world_point[2] + 
                                   extrinsic[j, 3] * world_point[3])
                
                # Check if point is behind camera
                if cam_point[2] <= 0:
                    continue
                    
                # Project to image coordinates
                img_x = (intrinsic[0, 0] * cam_point[0] + intrinsic[0, 2] * cam_point[2]) / cam_point[2]
                img_y = (intrinsic[1, 1] * cam_point[1] + intrinsic[1, 2] * cam_point[2]) / cam_point[2]
                
                # Check if point is within image bounds
                if img_x >= 0 and img_x < image_width and img_y >= 0 and img_y < image_height:
                    projected_points[valid_count, 0] = img_x
                    projected_points[valid_count, 1] = img_y
                    valid_count += 1
    
    # Z-constant faces (XY planes)
    for z_face in (z_min, z_max):
        for x_idx in range(face_density):
            x = x_min + x_idx * x_step
            for y_idx in range(face_density):
                y = y_min + y_idx * y_step
                
                # Transform to homogeneous coordinates
                world_point = np.array([x, y, z_face, 1.0], dtype=np.float32)
                
                # Transform from world to camera coordinates
                cam_point = np.zeros(4, dtype=np.float32)
                for j in range(4):
                    cam_point[j] = (extrinsic[j, 0] * world_point[0] + 
                                   extrinsic[j, 1] * world_point[1] + 
                                   extrinsic[j, 2] * world_point[2] + 
                                   extrinsic[j, 3] * world_point[3])
                
                # Check if point is behind camera
                if cam_point[2] <= 0:
                    continue
                    
                # Project to image coordinates
                img_x = (intrinsic[0, 0] * cam_point[0] + intrinsic[0, 2] * cam_point[2]) / cam_point[2]
                img_y = (intrinsic[1, 1] * cam_point[1] + intrinsic[1, 2] * cam_point[2]) / cam_point[2]
                
                # Check if point is within image bounds
                if img_x >= 0 and img_x < image_width and img_y >= 0 and img_y < image_height:
                    projected_points[valid_count, 0] = img_x
                    projected_points[valid_count, 1] = img_y
                    valid_count += 1
    
    # Return only the valid points
    return projected_points[:valid_count].copy()

def get_mesh_name(mesh, idx):
    try:
        return f'{mesh.source.file_name.lower()}_{idx}'
    except AttributeError:
        return f'{mesh.metadata.get("name", mesh.metadata.get("file_name", f"unknown")).lower()}_{idx}'

def get_time_str():
    """
    Return current time in the format: YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
