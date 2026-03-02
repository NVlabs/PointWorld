#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize one generated H5 clip directly (not WDS).

This script migrates the release visualization stack for data-branch use:
- --h5_dir is required.
- --h5_name and --clip_key are optional.
- If either optional arg is missing, a random choice is made from the selected scope.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np

import transform_utils

SUPPORTED_DOMAINS = {"droid", "behavior"}
DOMAIN_TO_URDF = {
    "droid": "assets/franka_description/franka_panda_robotiq_2f85.urdf",
    "behavior": "assets/r1pro/urdf/r1pro.urdf",
}


def _discover_h5_files(h5_dir: Path) -> list[Path]:
    files = sorted(
        [p for p in h5_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}]
    )
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found under: {h5_dir}")
    return files


def _resolve_h5_choice(h5_dir: Path, h5_files: Sequence[Path], h5_name: str | None, rng: random.Random) -> Path:
    if h5_name is None:
        return rng.choice(list(h5_files))

    requested = Path(h5_name)
    if requested.is_absolute():
        chosen = requested.resolve()
    else:
        direct = (h5_dir / requested).resolve()
        if direct.exists():
            chosen = direct
        else:
            basename_matches = [p for p in h5_files if p.name == h5_name]
            if len(basename_matches) == 1:
                chosen = basename_matches[0].resolve()
            elif len(basename_matches) > 1:
                raise ValueError(
                    f"Ambiguous --h5_name '{h5_name}' under {h5_dir}; matched {len(basename_matches)} files."
                )
            else:
                raise FileNotFoundError(f"Requested --h5_name not found under {h5_dir}: {h5_name}")

    if not chosen.exists():
        raise FileNotFoundError(f"Resolved H5 path does not exist: {chosen}")
    if not chosen.is_relative_to(h5_dir.resolve()):
        raise ValueError(f"Resolved H5 path must be under --h5_dir ({h5_dir}): {chosen}")
    return chosen


def _choose_clip_key(clip_group_names: Sequence[str], requested_clip_key: str | None, rng: random.Random) -> str:
    if not clip_group_names:
        raise RuntimeError("Selected H5 file has no top-level clip groups.")
    if requested_clip_key is not None:
        if requested_clip_key not in clip_group_names:
            raise KeyError(
                f"Requested --clip_key '{requested_clip_key}' not found. Available sample: {clip_group_names[:10]}"
            )
        return requested_clip_key
    return rng.choice(list(clip_group_names))


def _decode_initial_rgb(dataset: h5py.Dataset) -> np.ndarray:
    raw_value = dataset[0]

    if isinstance(raw_value, np.ndarray) and raw_value.ndim == 1 and raw_value.dtype == np.uint8:
        jpeg_bytes = raw_value.tobytes()
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode JPEG bytes in initial_rgb.")
        return decoded[..., ::-1]

    if isinstance(raw_value, (bytes, bytearray, memoryview)):
        decoded = cv2.imdecode(np.frombuffer(bytes(raw_value), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode JPEG bytes in initial_rgb.")
        return decoded[..., ::-1]

    arr = np.asarray(raw_value)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    raise ValueError(f"Unsupported initial_rgb payload with shape={arr.shape}, dtype={arr.dtype}")


def _decode_initial_depth(dataset: h5py.Dataset) -> np.ndarray:
    depth = np.asarray(dataset[()])
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    if depth.dtype != np.float32:
        return depth.astype(np.float32)
    return depth


def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors)
    if arr.dtype == np.uint8:
        return arr
    if arr.size == 0:
        return arr.astype(np.uint8)
    arr_f = arr.astype(np.float32)
    max_val = float(np.nanmax(arr_f))
    if max_val <= 1.0 + 1e-6:
        arr_f = arr_f * 255.0
    return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)


def _extract_behavior_scene_from_camera(camera_group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "local_scene_points",
        "local_scene_colors",
        "scene_mesh_trajectories",
    )
    for key in required:
        if key not in camera_group:
            raise KeyError(f"Missing behavior camera payload '{key}' in camera group.")

    local_points_group = camera_group["local_scene_points"]
    local_colors_group = camera_group["local_scene_colors"]
    trajectories_group = camera_group["scene_mesh_trajectories"]

    mesh_names = sorted(
        set(local_points_group.keys())
        & set(local_colors_group.keys())
        & set(trajectories_group.keys())
    )
    if not mesh_names:
        raise RuntimeError("No common mesh names across local_scene_points/local_scene_colors/scene_mesh_trajectories.")

    world_points_per_mesh: list[np.ndarray] = []
    world_colors_per_mesh: list[np.ndarray] = []

    num_frames: int | None = None
    for mesh_name in mesh_names:
        mesh_points = np.asarray(local_points_group[mesh_name][()], dtype=np.float32)
        mesh_colors = _to_uint8_colors(np.asarray(local_colors_group[mesh_name][()]))
        mesh_poses = np.asarray(trajectories_group[mesh_name][()], dtype=np.float32)

        if mesh_points.ndim != 2 or mesh_points.shape[1] != 3:
            raise ValueError(f"local_scene_points[{mesh_name}] must be (N,3), got {mesh_points.shape}")
        if mesh_colors.ndim != 2 or mesh_colors.shape[1] != 3:
            raise ValueError(f"local_scene_colors[{mesh_name}] must be (N,3), got {mesh_colors.shape}")
        if mesh_poses.ndim != 2 or mesh_poses.shape[1] != 7:
            raise ValueError(f"scene_mesh_trajectories[{mesh_name}] must be (T,7), got {mesh_poses.shape}")

        pose_mats = np.asarray(transform_utils.convert_pose_quat2mat(mesh_poses), dtype=np.float32)
        if pose_mats.ndim != 3 or pose_mats.shape[1:] != (4, 4):
            raise ValueError(
                f"Converted pose matrices for mesh '{mesh_name}' must be (T,4,4), got {pose_mats.shape}"
            )

        mesh_num_frames = int(pose_mats.shape[0])
        if num_frames is None:
            num_frames = mesh_num_frames
        elif mesh_num_frames != num_frames:
            raise ValueError(
                f"Inconsistent frame counts in behavior mesh trajectories: {mesh_num_frames} vs {num_frames}"
            )

        rotations = pose_mats[:, :3, :3]
        translations = pose_mats[:, :3, 3]
        world_points = np.einsum("tij,nj->tni", rotations, mesh_points) + translations[:, None, :]

        world_points_per_mesh.append(world_points.astype(np.float32, copy=False))
        world_colors_per_mesh.append(
            np.broadcast_to(mesh_colors[None, :, :], (mesh_num_frames, mesh_colors.shape[0], 3)).copy()
        )

    assert num_frames is not None
    scene_flows = np.concatenate(world_points_per_mesh, axis=1)
    scene_colors = np.concatenate(world_colors_per_mesh, axis=1)
    scene_exists = np.ones((num_frames, scene_flows.shape[1]), dtype=bool)
    return scene_flows, scene_colors, scene_exists


def _extract_droid_scene_from_camera(camera_group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "scene_flows" not in camera_group:
        raise KeyError("Missing droid camera payload 'scene_flows'.")
    if "scene_colors" not in camera_group:
        raise KeyError("Missing droid camera payload 'scene_colors'.")

    scene_flows = np.asarray(camera_group["scene_flows"][()], dtype=np.float32)
    scene_colors = _to_uint8_colors(np.asarray(camera_group["scene_colors"][()]))

    if scene_flows.ndim != 3 or scene_flows.shape[2] != 3:
        raise ValueError(f"scene_flows must be (T,N,3), got {scene_flows.shape}")

    if scene_colors.ndim == 2:
        if scene_colors.shape[1] != 3:
            raise ValueError(f"scene_colors must be (N,3) or (T,N,3), got {scene_colors.shape}")
        scene_colors = np.broadcast_to(
            scene_colors[None, :, :], (scene_flows.shape[0], scene_colors.shape[0], 3)
        ).copy()
    elif scene_colors.ndim == 3:
        if scene_colors.shape[0] != scene_flows.shape[0] or scene_colors.shape[2] != 3:
            raise ValueError(
                f"scene_colors must align with scene_flows time dimension; "
                f"got {scene_colors.shape} vs {scene_flows.shape}"
            )
    else:
        raise ValueError(f"scene_colors must be (N,3) or (T,N,3), got {scene_colors.shape}")

    if "scene_visibility" in camera_group:
        scene_exists = np.asarray(camera_group["scene_visibility"][()]).astype(bool)
    elif "scene_depth_valid_mask" in camera_group:
        scene_exists = np.asarray(camera_group["scene_depth_valid_mask"][()]).astype(bool)
    else:
        scene_exists = np.ones(scene_flows.shape[:2], dtype=bool)

    if scene_exists.ndim == 3 and scene_exists.shape[2] == 1:
        scene_exists = scene_exists[..., 0]
    if scene_exists.shape != scene_flows.shape[:2]:
        raise ValueError(
            f"scene existence mask must be (T,N) matching scene_flows; got {scene_exists.shape} vs {scene_flows.shape[:2]}"
        )

    return scene_flows, scene_colors, scene_exists


def _extract_droid_scene_from_clip(
    clip_group: h5py.Group, camera_keys: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_flows: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_exists: list[np.ndarray] = []
    num_frames: int | None = None

    for camera_key in camera_keys:
        camera_group = clip_group[camera_key]
        flows, colors, exists = _extract_droid_scene_from_camera(camera_group)
        if num_frames is None:
            num_frames = int(flows.shape[0])
        elif int(flows.shape[0]) != num_frames:
            raise ValueError(
                f"Inconsistent droid frame counts across cameras: {camera_key} has {flows.shape[0]}, expected {num_frames}"
            )
        all_flows.append(flows.astype(np.float32, copy=False))
        all_colors.append(colors.astype(np.uint8, copy=False))
        all_exists.append(exists.astype(bool, copy=False))

    if num_frames is None:
        raise RuntimeError("No droid camera scene payloads found.")

    scene_flows = np.concatenate(all_flows, axis=1)
    scene_colors = np.concatenate(all_colors, axis=1)
    scene_exists = np.concatenate(all_exists, axis=1)
    return scene_flows, scene_colors, scene_exists


def _detect_domain_from_clip_schema(clip_group: h5py.Group, camera_keys: Sequence[str]) -> str:
    if not camera_keys:
        raise RuntimeError("No camera_* groups found in clip.")
    first_camera = clip_group[camera_keys[0]]
    if "scene_flows" in first_camera:
        return "droid"
    if "local_scene_points" in first_camera and "scene_mesh_trajectories" in first_camera:
        return "behavior"
    raise RuntimeError(
        f"Unable to infer domain from camera payload keys: {list(first_camera.keys())}"
    )


def _decode_joint_names(joint_names_payload: np.ndarray) -> list[str]:
    joint_names_arr = np.asarray(joint_names_payload).reshape(-1)
    decoded: list[str] = []
    for raw_name in joint_names_arr:
        if isinstance(raw_name, (bytes, bytearray, np.bytes_)):
            decoded.append(raw_name.decode("utf-8"))
        else:
            decoded.append(str(raw_name))
    if not decoded:
        raise ValueError("joint_names is empty.")
    return decoded


def _build_urdf_visual_mesh_link_map(urdf) -> dict[int, str]:
    mesh_id_to_link: dict[int, str] = {}
    for link in urdf.links:
        for visual in (link.visuals or []):
            geometry = getattr(visual, "geometry", None)
            if geometry is None:
                continue
            mesh_wrapper = getattr(geometry, "mesh", None)
            if mesh_wrapper is None:
                continue
            mesh_list = getattr(mesh_wrapper, "meshes", None)
            if mesh_list is None:
                continue
            for mesh in mesh_list:
                mesh_id_to_link[id(mesh)] = str(link.name)
    return mesh_id_to_link


def _sample_surface_deterministic(mesh, count: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or float(mesh.area) <= 0.0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    faces = mesh.faces
    verts = mesh.vertices
    areas = mesh.area_faces.astype(np.float64)
    prob = areas / max(float(areas.sum()), 1e-12)
    face_idx = rng.choice(len(faces), size=int(count), p=prob)
    tri = verts[faces[face_idx]]  # (count, 3, 3)
    bary = rng.random((count, 2)).astype(np.float64)
    over = bary.sum(axis=1) > 1.0
    bary[over] = 1.0 - bary[over]
    u = bary[:, 0:1]
    v = bary[:, 1:2]
    w = 1.0 - u - v
    points = (w * tri[:, 0] + u * tri[:, 1] + v * tri[:, 2]).astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    return points, normals


def _build_droid_robot_flows(
    *,
    joint_positions: np.ndarray,
    gripper_positions: np.ndarray,
    urdf_path: Path,
    max_robot_points: int,
) -> np.ndarray:
    from visualization.viser_flow.robot_sampler_lite import URDFRealSampler

    joints = np.asarray(joint_positions, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 7:
        raise ValueError(f"DROID joint_positions must be (T,7), got {joints.shape}")
    gripper = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
    if gripper.shape[0] != joints.shape[0]:
        raise ValueError(
            f"DROID gripper_positions length mismatch: {gripper.shape[0]} vs {joints.shape[0]}"
        )
    if int(max_robot_points) <= 0:
        raise ValueError(f"max_robot_points must be > 0, got {max_robot_points}")

    sampler = URDFRealSampler(
        urdf_path=str(urdf_path),
        gripper_only=True,
        min_samples_per_mesh=1,
    )
    sampler.presample(int(max_robot_points), seed=1)
    trajectories = sampler.compute_world_trajectories(joints, gripper)

    if trajectories.ndim != 3 or trajectories.shape[0] != joints.shape[0] or trajectories.shape[2] != 3:
        raise ValueError(
            f"DROID robot trajectories must be (T,N,3) with T={joints.shape[0]}, got {trajectories.shape}"
        )
    return trajectories.astype(np.float32, copy=False)


def _build_behavior_robot_flows(
    *,
    joint_positions: np.ndarray,
    joint_names: Sequence[str],
    base_pose: np.ndarray,
    urdf_path: Path,
    max_robot_points: int,
) -> np.ndarray:
    from visualization.urdfpy_compat import ensure_urdfpy_numpy_compat

    ensure_urdfpy_numpy_compat()
    import urdfpy

    joints = np.asarray(joint_positions, dtype=np.float32)
    if joints.ndim != 2:
        raise ValueError(f"BEHAVIOR joint_positions must be (T,J), got {joints.shape}")
    names = [str(name) for name in joint_names]
    if joints.shape[1] != len(names):
        raise ValueError(
            f"BEHAVIOR joint_positions/joint_names mismatch: {joints.shape[1]} vs {len(names)}"
        )
    base_pose_arr = np.asarray(base_pose, dtype=np.float32)
    if base_pose_arr.ndim != 2 or base_pose_arr.shape[1] != 7:
        raise ValueError(f"BEHAVIOR base_pose must be (T,7), got {base_pose_arr.shape}")
    if base_pose_arr.shape[0] != joints.shape[0]:
        raise ValueError(
            f"BEHAVIOR base_pose length mismatch: {base_pose_arr.shape[0]} vs {joints.shape[0]}"
        )
    base_pose_mats = np.asarray(transform_utils.convert_pose_quat2mat(base_pose_arr), dtype=np.float32)
    if base_pose_mats.shape != (joints.shape[0], 4, 4):
        raise ValueError(f"Unexpected base_pose matrix shape: {base_pose_mats.shape}")
    if int(max_robot_points) <= 0:
        raise ValueError(f"max_robot_points must be > 0, got {max_robot_points}")

    urdf = urdfpy.URDF.load(str(urdf_path))
    mesh_id_to_link = _build_urdf_visual_mesh_link_map(urdf)
    actuated = [joint.name for joint in urdf.actuated_joints]
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [joint_name for joint_name in actuated if joint_name not in name_to_idx]
    allowed_missing = {
        "steer_motor_joint1",
        "steer_motor_joint2",
        "steer_motor_joint3",
        "wheel_motor_joint1",
        "wheel_motor_joint2",
        "wheel_motor_joint3",
    }
    unexpected_missing = [joint_name for joint_name in missing if joint_name not in allowed_missing]
    if unexpected_missing:
        raise ValueError(
            f"BEHAVIOR missing required actuated joints for URDF kinematics: {unexpected_missing}"
        )

    cfg_ref = {joint_name: float(joints[0, name_to_idx[joint_name]]) if joint_name in name_to_idx else 0.0 for joint_name in actuated}
    fk_ref = urdf.visual_trimesh_fk(cfg=cfg_ref)
    if not fk_ref:
        raise RuntimeError("URDF visual_trimesh_fk returned no meshes for behavior robot.")

    gripper_link_tokens = ("gripper", "finger", "robotiq", "knuckle")
    mesh_entries = []
    for mesh in fk_ref.keys():
        mesh_id = id(mesh)
        mesh_link_name = mesh_id_to_link.get(mesh_id, "")
        if not any(token in mesh_link_name.lower() for token in gripper_link_tokens):
            continue
        mesh_entries.append((mesh_id, mesh_link_name, mesh, max(float(mesh.area), 0.0)))
    if not mesh_entries:
        raise ValueError(
            "BEHAVIOR gripper-only sampling selected zero meshes. "
            "Expected visual mesh links containing one of "
            f"{gripper_link_tokens}, but no matches were found."
        )

    total_area = sum(area for _, _, _, area in mesh_entries)
    counts = []
    allocated = 0
    for idx, (_, _, _, area) in enumerate(mesh_entries):
        if idx == len(mesh_entries) - 1:
            count = max(0, int(max_robot_points) - allocated)
        else:
            ratio = area / total_area if total_area > 0 else 0.0
            count = int(int(max_robot_points) * ratio)
            allocated += count
        counts.append(int(count))

    rng = np.random.RandomState(1)
    sampled_local_points: dict[int, np.ndarray] = {}
    for (mesh_id, _, mesh, _), count in zip(mesh_entries, counts):
        points, _ = _sample_surface_deterministic(mesh, int(count), rng)
        sampled_local_points[mesh_id] = points

    trajectories = []
    num_frames = joints.shape[0]
    for frame_idx in range(num_frames):
        cfg = {
            joint_name: (
                float(joints[frame_idx, name_to_idx[joint_name]])
                if joint_name in name_to_idx
                else 0.0
            )
            for joint_name in actuated
        }
        fk_frame = urdf.visual_trimesh_fk(cfg=cfg)
        fk_by_id = {id(mesh): transform for mesh, transform in fk_frame.items()}
        tracks_per_mesh = []
        for mesh_id, local_points in sampled_local_points.items():
            if mesh_id not in fk_by_id:
                mesh_link_name = mesh_id_to_link.get(mesh_id, "unknown")
                raise ValueError(
                    f"BEHAVIOR gripper mesh (link='{mesh_link_name}', id={mesh_id}) "
                    f"missing from FK at frame {frame_idx}"
                )
            transform_mesh = np.asarray(fk_by_id[mesh_id], dtype=np.float32)
            transform_world = np.asarray(base_pose_mats[frame_idx] @ transform_mesh, dtype=np.float32)
            if local_points.size == 0:
                continue
            ones = np.ones((local_points.shape[0], 1), dtype=np.float32)
            points_h = np.concatenate([local_points.astype(np.float32, copy=False), ones], axis=1)
            world_points = (transform_world @ points_h.T).T[:, :3]
            tracks_per_mesh.append(world_points.astype(np.float32, copy=False))
        if tracks_per_mesh:
            trajectories.append(np.concatenate(tracks_per_mesh, axis=0))
        else:
            trajectories.append(np.zeros((0, 3), dtype=np.float32))

    return np.stack(trajectories, axis=0).astype(np.float32, copy=False)


def _populate_robot_flows(
    *,
    sample_dict: dict[str, np.ndarray | str],
    domain: str,
    urdf_path: Path,
    max_robot_points: int,
) -> None:
    scene_flows = np.asarray(sample_dict["scene_flows"], dtype=np.float32)
    num_frames = int(scene_flows.shape[0])
    if domain == "droid":
        robot_flows = _build_droid_robot_flows(
            joint_positions=np.asarray(sample_dict["joint_positions"], dtype=np.float32),
            gripper_positions=np.asarray(sample_dict["gripper_positions"], dtype=np.float32),
            urdf_path=urdf_path,
            max_robot_points=int(max_robot_points),
        )
    elif domain == "behavior":
        robot_flows = _build_behavior_robot_flows(
            joint_positions=np.asarray(sample_dict["joint_positions"], dtype=np.float32),
            joint_names=list(sample_dict["joint_names"]),  # type: ignore[arg-type]
            base_pose=np.asarray(sample_dict["base_pose"], dtype=np.float32),
            urdf_path=urdf_path,
            max_robot_points=int(max_robot_points),
        )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    if robot_flows.ndim == 3 and robot_flows.shape[1] > int(max_robot_points):
        rng = np.random.RandomState(1)
        keep = np.sort(rng.choice(robot_flows.shape[1], size=int(max_robot_points), replace=False))
        robot_flows = robot_flows[:, keep, :]

    if robot_flows.ndim != 3 or robot_flows.shape[0] != num_frames or robot_flows.shape[2] != 3:
        raise ValueError(
            f"Generated robot_flows must be (T,N,3) with T={num_frames}, got {robot_flows.shape}"
        )
    sample_dict["robot_flows"] = robot_flows
    sample_dict["robot_exists"] = np.ones(robot_flows.shape[:2], dtype=bool)


def _build_sample_dict(h5_path: Path, clip_key: str) -> tuple[dict[str, np.ndarray | str], str, list[str]]:
    with h5py.File(h5_path, "r") as h5_file:
        if "domain" not in h5_file.attrs:
            raise KeyError(f"Missing root attribute 'domain' in {h5_path}")
        raw_domain = h5_file.attrs["domain"]
        if isinstance(raw_domain, (bytes, bytearray, np.bytes_)):
            domain = raw_domain.decode("utf-8")
        else:
            domain = str(raw_domain)
        domain = domain.strip().lower()
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unsupported root domain attribute '{domain}' in {h5_path}; expected one of {sorted(SUPPORTED_DOMAINS)}"
            )

        if clip_key not in h5_file:
            raise KeyError(f"Clip key '{clip_key}' not found in {h5_path}")
        clip_group = h5_file[clip_key]

        camera_keys = sorted([k for k in clip_group.keys() if k.startswith("camera_")])
        if not camera_keys:
            raise RuntimeError(f"Clip '{clip_key}' has no camera_* groups")

        schema_domain = _detect_domain_from_clip_schema(clip_group, camera_keys)
        if schema_domain != domain:
            raise ValueError(
                f"Domain mismatch in {h5_path}:{clip_key}: root attr domain='{domain}' but clip schema implies '{schema_domain}'"
            )
        sample_dict: dict[str, np.ndarray | str] = {
            "__key__": f"{h5_path.stem}-{clip_key}",
            "__domain__": domain,
        }

        for camera_key in camera_keys:
            camera_group = clip_group[camera_key]
            required_camera_keys = ("initial_rgb", "initial_depth", "intrinsic", "extrinsic")
            for required_key in required_camera_keys:
                if required_key not in camera_group:
                    raise KeyError(
                        f"Missing camera payload '{required_key}' in {h5_path}:{clip_key}:{camera_key}"
                    )

            sample_dict[f"{camera_key}_initial_rgb"] = _decode_initial_rgb(camera_group["initial_rgb"])
            sample_dict[f"{camera_key}_initial_depth"] = _decode_initial_depth(camera_group["initial_depth"])
            sample_dict[f"{camera_key}_intrinsic"] = np.asarray(camera_group["intrinsic"][()], dtype=np.float32)
            sample_dict[f"{camera_key}_extrinsic"] = np.asarray(camera_group["extrinsic"][()], dtype=np.float32)

        if domain == "behavior":
            scene_camera_group = clip_group[camera_keys[0]]
            scene_flows, scene_colors, scene_exists = _extract_behavior_scene_from_camera(scene_camera_group)
            for required_behavior_key in ("joint_positions", "joint_names", "base_pose"):
                if required_behavior_key not in clip_group:
                    raise KeyError(
                        f"Missing behavior clip payload '{required_behavior_key}' in {h5_path}:{clip_key}"
                    )
            sample_dict["joint_positions"] = np.asarray(clip_group["joint_positions"][()], dtype=np.float32)
            sample_dict["joint_names"] = _decode_joint_names(np.asarray(clip_group["joint_names"][()]))
            sample_dict["base_pose"] = np.asarray(clip_group["base_pose"][()], dtype=np.float32)
        else:
            scene_flows, scene_colors, scene_exists = _extract_droid_scene_from_clip(
                clip_group, camera_keys
            )
            for required_droid_key in ("joint_positions", "gripper_positions"):
                if required_droid_key not in clip_group:
                    raise KeyError(
                        f"Missing droid clip payload '{required_droid_key}' in {h5_path}:{clip_key}"
                    )
            sample_dict["joint_positions"] = np.asarray(clip_group["joint_positions"][()], dtype=np.float32)
            sample_dict["gripper_positions"] = np.asarray(clip_group["gripper_positions"][()], dtype=np.float32)

        if scene_flows.ndim != 3 or scene_flows.shape[2] != 3:
            raise ValueError(f"scene_flows must be (T,N,3), got {scene_flows.shape}")
        if scene_exists.shape != scene_flows.shape[:2]:
            raise ValueError(
                f"scene_exists must be (T,N) matching scene_flows; got {scene_exists.shape} vs {scene_flows.shape[:2]}"
            )

        num_frames = int(scene_flows.shape[0])
        sample_dict["scene_flows"] = scene_flows
        sample_dict["scene_colors"] = scene_colors
        sample_dict["scene_exists"] = scene_exists.astype(bool)
        sample_dict["scene_supervised_mask"] = scene_exists.astype(bool)

    return sample_dict, domain, camera_keys


def _prompt_close_viewer() -> str:
    prompt = "Press ENTER to close visualization, or type 'q' then ENTER: "
    if sys.stdin is not None and sys.stdin.isatty():
        return input(prompt)
    try:
        with open("/dev/tty", "r", encoding="utf-8") as tty:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            line = tty.readline()
            if line == "":
                raise RuntimeError(
                    "No interactive TTY input received. Run in an interactive terminal."
                )
            return line
    except OSError as exc:
        raise RuntimeError(
            "Visualization requires an interactive TTY. Run this command from a terminal."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one clip from generated H5 outputs (not final WDS). "
            "--h5_dir is required; --h5_name/--clip_key are optional."
        )
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing generated .h5/.hdf5 files (searched recursively).",
    )
    parser.add_argument(
        "--h5_name",
        type=str,
        default=None,
        help=(
            "Optional target H5 file (basename or path under --h5_dir). "
            "If omitted, one file is selected randomly."
        ),
    )
    parser.add_argument(
        "--clip_key",
        type=str,
        default=None,
        help=(
            "Optional target clip key inside the selected H5 file. "
            "If omitted, one clip is selected randomly."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible H5/clip selection.",
    )
    parser.add_argument("--viewer_host", type=str, default="0.0.0.0", help="Viewer host bind address.")
    parser.add_argument("--viewer_port", type=int, default=8080, help="Viewer port.")
    parser.add_argument(
        "--max_robot_points",
        type=int,
        default=500,
        help="Maximum number of robot points sampled from URDF kinematics for visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from visualization.prediction_viz import (
        PredictionVisualizer,
        PredictionVisualizerConfig,
        build_sample_from_dictionary,
    )

    h5_dir = Path(args.h5_dir).expanduser().resolve()
    if not h5_dir.exists() or not h5_dir.is_dir():
        raise NotADirectoryError(f"--h5_dir must be an existing directory: {h5_dir}")

    rng = random.Random(args.seed)
    h5_files = _discover_h5_files(h5_dir)
    chosen_h5 = _resolve_h5_choice(h5_dir, h5_files, args.h5_name, rng)

    with h5py.File(chosen_h5, "r") as h5_file:
        clip_names = sorted([k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Group)])
    chosen_clip = _choose_clip_key(clip_names, args.clip_key, rng)

    sample_dict, domain, camera_keys = _build_sample_dict(chosen_h5, chosen_clip)

    config = PredictionVisualizerConfig()
    config.viewer_host = str(args.viewer_host)
    config.viewer_port = int(args.viewer_port)

    urdf_path = Path(DOMAIN_TO_URDF[domain]).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found for domain '{domain}': {urdf_path}")

    _populate_robot_flows(
        sample_dict=sample_dict,
        domain=domain,
        urdf_path=urdf_path,
        max_robot_points=int(args.max_robot_points),
    )

    viz_sample = build_sample_from_dictionary(sample_dict=sample_dict, predictions=None)
    visualizer = PredictionVisualizer(config, urdf_path=urdf_path)

    print(f"[viz] domain={domain}")
    print(f"[viz] h5={chosen_h5}")
    print(f"[viz] clip={chosen_clip}")
    print(f"[viz] urdf={urdf_path}")
    print(f"[viz] max_robot_points={int(args.max_robot_points)}")
    if domain == "droid":
        print(f"[viz] scene cameras={len(camera_keys)} (concatenated)")
    else:
        print(f"[viz] scene camera={camera_keys[0]}")

    viz_result = visualizer.visualize(viz_sample, launch_viewer=True, live_session=None)
    if "live_session" not in viz_result or viz_result["live_session"] is None:
        raise RuntimeError("Viewer failed to launch a live session.")

    live_session = viz_result["live_session"]
    display_host, display_port = visualizer.viewer_endpoint()
    if display_host in {"0.0.0.0", "127.0.0.1"}:
        display_host = "localhost"
    print(f"[viz] viewer: http://{display_host}:{display_port}")

    try:
        _prompt_close_viewer()
    finally:
        live_session.close()


if __name__ == "__main__":
    main()
