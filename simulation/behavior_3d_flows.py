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
Behavior 3D Flow Extraction (Simulation).

This module contains the CLI entrypoint and the extractor implementation for
Behavior raw H5 episodes -> extracted clip H5s.
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch as th

# OmniGibson decorates many utility functions with torch.compile(). In our
# headless data-generation flow this can stall at first-run inductor
# compilation for long periods; default to eager mode unless explicitly
# overridden.
if os.environ.get("PW_ENABLE_TORCH_COMPILE", "0").lower() not in {"1", "true", "yes"}:
    def _identity_compile(func=None, *args, **kwargs):
        if func is None:
            return lambda f: f
        return func

    th.compile = _identity_compile

import omnigibson as og
from omnigibson.controllers import IsGraspingState
from omnigibson.envs.data_wrapper import DataPlaybackWrapper
from omnigibson.macros import gm
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import RigidContactAPI

from simulation.behavior.camera import CameraDataExtractor
from simulation.behavior.motion import MotionTracker
from simulation.behavior.raw_input import (
    derive_behavior_output_relpath,
    enforce_behavior_cache_policy,
    get_local_behavior_input_path,
    is_behavior_hf_path,
)
from simulation.behavior.types import ClipCandidate
from simulation.behavior.utils import configure_sim_settings
from simulation.behavior.writer import ClipWriter


class BehaviorFlowExtractor:
    """
    Behavior 3D flow extractor.

    Runs a one-pass playback over a single raw H5 episode and writes per-clip
    outputs (points, trajectories, and metadata) into an aggregated H5.
    """

    def __init__(self, args):
        self.args = args
        self.env = None
        self.robot = None
        self.workspace_bounds_min = np.asarray(self.args.workspace_bounds_min, dtype=np.float32)
        self.workspace_bounds_max = np.asarray(self.args.workspace_bounds_max, dtype=np.float32)

        self.invalid_clips_count = 0
        self.total_scene_points_processed = 0
        self.clips_processed = 0

        self.clip_writer = None
        self.camera_extractor = None
        self.motion_tracker = None
        self._playback_output_path = None

    def _create_environment(self, input_path, playback_output_path):
        """Create OmniGibson environment with integrated setup."""
        gm.ENABLE_TRANSITION_RULES = False
        gm.ENABLE_FLATCACHE = True
        gm.ENABLE_CCD = True
        gm.GUI_VIEWPORT_ONLY = True
        gm.ENABLE_HQ_RENDERING = True

        if self.args.headless:
            gm.HEADLESS = True
            gm.RENDER_VIEWER_CAMERA = False

        external_camera_poses = {
            "left": [[-0.2, 0.6, 2.0], [-0.1930, 0.4163, 0.8062, -0.3734]],
            "right": [[-0.2, -0.6, 2.0], [0.4164, -0.1929, -0.3737, 0.8060]],
        }

        external_sensors_config = []
        for name, (position, orientation) in external_camera_poses.items():
            external_sensors_config.append(
                {
                    "sensor_type": "VisionSensor",
                    "name": f"{name}",
                    "relative_prim_path": f"/controllable__r1pro__robot_r1/base_link/{name}",
                    "modalities": ["rgb", "depth_linear", "normal", "seg_instance_id"],
                    "sensor_kwargs": {
                        "image_height": self.args.image_height,
                        "image_width": self.args.image_width,
                        "horizontal_aperture": 40.0,
                    },
                    "position": th.tensor(position, dtype=th.float32),
                    "orientation": th.tensor(orientation, dtype=th.float32),
                    "pose_frame": "parent",
                }
            )

        external_sensors_config.append(
            {
                "sensor_type": "VisionSensor",
                "name": "head",
                "relative_prim_path": "/controllable__r1pro__robot_r1/zed_link/head",
                "modalities": ["rgb", "depth_linear", "normal", "seg_instance_id"],
                "sensor_kwargs": {
                    "image_height": self.args.image_height,
                    "image_width": self.args.image_width,
                    "horizontal_aperture": 60.0,
                },
                "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
                "orientation": T.quat_multiply(
                    th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
                    T.euler2quat(th.tensor([T.deg2rad(-15.0), 0.0, 0.0], dtype=th.float32)),
                ),
                "pose_frame": "parent",
            }
        )

        self.env = DataPlaybackWrapper.create_from_hdf5(
            input_path=input_path,
            output_path=playback_output_path,
            robot_sensor_config=None,
            external_sensors_config=external_sensors_config,
            exclude_sensor_names=["zed"],
            n_render_iterations=1,
            overwrite=True,
            only_successes=False,
            flush_every_n_traj=1,
            include_env_wrapper=False,
            full_scene_file=None,
            include_task=True,
            include_task_obs=True,
            include_robot_control=False,
            include_contacts=True,
        )
        self._playback_output_path = playback_output_path

        og.sim.add_callback_on_play("optimize_rendering", configure_sim_settings)
        self.robot = self.env.env.robots[0]

    def _get_joint_names(self):
        """Return robot joint names in a consistent order."""
        return [name for name in self.robot.joints.keys() if "base_footprint" not in name]

    def _get_joint_name_to_position_dict(self):
        """Return a dict mapping joint name -> current joint position (float)."""
        name_to_pos = {}
        for name in self._get_joint_names():
            joint_prim = self.robot.joints[name]
            pos, _, _ = joint_prim.get_state()
            name_to_pos[name] = float(pos.item())
        return name_to_pos

    def _get_joint_positions_array(self):
        """Return a float32 numpy array of joint positions aligned with `_get_joint_names()`."""
        name_to_pos = self._get_joint_name_to_position_dict()
        ordered = [name_to_pos[n] for n in self._get_joint_names()]
        return np.asarray(ordered, dtype=np.float32)

    def _start_candidate_if_visible(self, downsample_idx, obs, info):
        """Start a new clip candidate at a downsample index if any meshes are visible."""
        clip_key = f"{downsample_idx}:{downsample_idx + self.args.frames_per_clip}"
        clip_seed = self.camera_extractor.capture_clip_seed(obs, info, self.robot, clip_key)

        if not any(len(cam["mesh_data"]) > 0 for cam in clip_seed["cameras"].values()):
            return None

        return ClipCandidate(
            downsample_idx,
            self.args.frames_per_clip,
            clip_seed["world_to_robot"],
            clip_seed,
            self.robot.arm_names,
            len(self._get_joint_names()),
        )

    def extract_episode(self, episode_id):
        """Process a single episode using one-pass always-render buffered-candidates pipeline."""
        assert self.clip_writer is not None, "clip_writer must be initialized before extract_episode"
        assert self.camera_extractor is not None, "camera_extractor must be initialized before extract_episode"
        assert self.motion_tracker is not None, "motion_tracker must be initialized before extract_episode"
        print(f"Processing episode {episode_id}...")

        data_grp = self.env.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        transitions = json.loads(traj_grp.attrs["transitions"])
        traj_grp = h5py_group_to_torch(traj_grp)
        init_metadata = traj_grp["init_metadata"]
        action = traj_grp["action"]
        state = traj_grp["state"]
        state_size = traj_grp["state_size"]

        self.env.scene.restore(self.env.scene_file, update_initial_file=True)

        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.env.scene.n_objects
            for i, obj in enumerate(self.env.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        self.env.reset()

        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        def _safe_link_paths(links):
            out = []
            for link in links or []:
                prim_path = getattr(link, "prim_path", None)
                if prim_path:
                    out.append(prim_path)
            return out

        trunk_arm_paths = []
        gripper_finger_paths_by_arm = {arm: [] for arm in self.robot.arm_names}

        try:
            trunk_arm_paths.extend(_safe_link_paths(self.robot.trunk_links))
        except Exception as exc:
            print(f"[warn] Could not read robot trunk links from API: {exc}")

        for arm in self.robot.arm_names:
            try:
                trunk_arm_paths.extend(_safe_link_paths(self.robot.arm_links[arm]))
            except Exception:
                pass
            try:
                gripper_finger_paths_by_arm[arm].extend(_safe_link_paths(self.robot.gripper_links[arm]))
            except Exception:
                pass
            try:
                gripper_finger_paths_by_arm[arm].extend(_safe_link_paths(self.robot.finger_links[arm]))
            except Exception:
                pass

        # Fallback for API/version mismatches (e.g., RobotDefinition schema drift):
        # infer arm/trunk/finger links from name patterns in robot.links.
        if not trunk_arm_paths or any(len(paths) == 0 for paths in gripper_finger_paths_by_arm.values()):
            try:
                robot_links = getattr(self.robot, "links", {})
                for link_name, link in robot_links.items():
                    prim_path = getattr(link, "prim_path", None)
                    if not prim_path:
                        continue
                    lname = str(link_name).lower()
                    if "trunk" in lname or "torso" in lname:
                        trunk_arm_paths.append(prim_path)
                    for arm in self.robot.arm_names:
                        arm_token = str(arm).lower()
                        if arm_token not in lname:
                            continue
                        if any(tok in lname for tok in ("arm", "shoulder", "elbow", "wrist", "forearm", "upperarm")):
                            trunk_arm_paths.append(prim_path)
                        if any(tok in lname for tok in ("gripper", "finger")):
                            gripper_finger_paths_by_arm[arm].append(prim_path)
            except Exception as exc:
                print(f"[warn] Failed link-name fallback for collision filtering: {exc}")

        # Deduplicate while preserving order.
        trunk_arm_paths = list(dict.fromkeys(trunk_arm_paths))
        gripper_finger_paths_by_arm = {
            arm: list(dict.fromkeys(paths)) for arm, paths in gripper_finger_paths_by_arm.items()
        }

        if not trunk_arm_paths:
            print("[warn] No trunk/arm links found; trunk/arm collision checks disabled for this episode.")
        for arm, paths in gripper_finger_paths_by_arm.items():
            if not paths:
                print(f"[warn] No gripper/finger links found for arm '{arm}'; finger collision checks disabled.")

        trunk_arm_indices = [RigidContactAPI.get_body_col_idx(p)[1] for p in trunk_arm_paths]
        gripper_finger_indices_by_arm = {
            arm: [RigidContactAPI.get_body_col_idx(p)[1] for p in paths]
            for arm, paths in gripper_finger_paths_by_arm.items()
        }
        self.motion_tracker.set_collision_indices(trunk_arm_indices, gripper_finger_indices_by_arm)

        num_frames = len(action)
        num_downsampled = (num_frames + self.args.time_skip_ratio - 1) // self.args.time_skip_ratio

        transitions_flags = np.zeros(num_frames, dtype=bool)
        for t_str in transitions.keys():
            transition_idx = int(t_str)
            if 0 <= transition_idx < num_frames:
                transitions_flags[transition_idx] = True

        last_start_downsample_idx = max(0, num_downsampled - self.args.frames_per_clip)

        print(f"Episode {episode_id}: {num_frames} original frames -> {num_downsampled} downsampled frames")
        print(
            "One-pass streaming with always-render, overlapping windows "
            f"(max start idx: {last_start_downsample_idx})"
        )

        self.active_candidates = {}
        episode_start_time = time.time()
        valid_count = 0
        total_started = 0

        for frame_idx in range(num_frames):
            if transitions_flags[frame_idx]:
                if str(frame_idx) in transitions:
                    self._execute_transitions(transitions[str(frame_idx)])
            og.sim.load_state(state[frame_idx, : int(state_size[frame_idx])], serialized=True)
            obs, _, _, _, info = self.env.env.step(action=action[frame_idx], n_render_iterations=1)

            if frame_idx % self.args.time_skip_ratio != 0:
                if transitions_flags[frame_idx]:
                    for candidate in self.active_candidates.values():
                        candidate.has_transition = True
                trunk_arm_col, gripper_finger_col_by_arm = self.motion_tracker.get_collision_flags_for_frame(frame_idx)
                if trunk_arm_col or any(gripper_finger_col_by_arm.values()):
                    for candidate in self.active_candidates.values():
                        if trunk_arm_col:
                            candidate.has_trunk_arm_collision = True
                        for arm, collided in gripper_finger_col_by_arm.items():
                            if collided:
                                candidate.gripper_finger_collision[arm] = True
                continue

            downsample_idx = frame_idx // self.args.time_skip_ratio

            if downsample_idx % self.args.skip_every == 0 and downsample_idx <= last_start_downsample_idx:
                candidate = self._start_candidate_if_visible(downsample_idx, obs, info)
                if candidate is None:
                    print(f"Warning: No visible meshes across cameras at idx={downsample_idx}")
                else:
                    self.active_candidates[downsample_idx] = candidate
                    total_started += 1

            for start_downsample_idx, candidate in list(self.active_candidates.items()):
                clip_frame_idx = candidate.frames_written

                self._record_robot_state(candidate.robot_series, clip_frame_idx, candidate.world_to_robot)
                self.camera_extractor.record_mesh_and_camera_poses(candidate.clip_seed, clip_frame_idx)
                self.motion_tracker.update_motion_metrics(candidate, frame_idx, transitions_flags)

                candidate.frames_written += 1

                if candidate.frames_written == candidate.frames_per_clip:
                    if self.motion_tracker.is_candidate_valid(candidate):
                        points_count = self.clip_writer.write_candidate(candidate)
                        self.total_scene_points_processed += points_count
                        self.clips_processed += 1
                        valid_count += 1
                    else:
                        self.invalid_clips_count += 1
                    del self.active_candidates[start_downsample_idx]

            if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
                elapsed = time.time() - episode_start_time
                eta = (
                    elapsed * (num_frames - frame_idx - 1) / max(1, frame_idx + 1)
                    if frame_idx < num_frames - 1
                    else 0
                )
                valid_pct = valid_count / max(1, total_started) * 100
                elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"

                avg_points = self.total_scene_points_processed / max(1, self.clips_processed)
                progress_line = (
                    f"Active: {len(self.active_candidates)} | "
                    f"Valid: {valid_count}/{total_started} ({valid_pct:.1f}%) | "
                    f"Avg points: {avg_points:.0f} | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                )
                print(f"\r{progress_line}", end="", flush=True)

        print()

        incomplete_count = len(self.active_candidates)
        if incomplete_count > 0:
            print(f"Dropping {incomplete_count} incomplete candidates at episode end")
            self.active_candidates.clear()

        final_time = time.time() - episode_start_time
        episode_avg_points = (
            self.total_scene_points_processed / max(1, self.clips_processed)
            if self.clips_processed > 0
            else 0
        )
        print(
            f"Episode {episode_id} complete: {valid_count} valid clips from {total_started} candidates "
            f"({valid_count / max(1, total_started) * 100:.1f}% kept) in {final_time:.1f}s, "
            f"avg_points: {episode_avg_points:.0f}"
        )

    def _record_robot_state(self, robot_series, frame_idx, world_to_robot):
        """Record current robot state into the clip's robot time-series arrays."""
        robot_series["joint_positions"][frame_idx] = self._get_joint_positions_array()

        proprio_dict = self.robot._get_proprioception_dict()

        r_pos, r_quat = self.robot.get_position_orientation()
        h_world = T.pose2mat((r_pos, r_quat))
        w2r_t = th.tensor(world_to_robot, dtype=th.float32)
        h_in_robot0 = w2r_t @ h_world
        r0_pos_t, r0_quat_t = T.mat2pose(h_in_robot0)
        r0_pos = r0_pos_t.cpu().numpy()
        r0_quat = r0_quat_t.cpu().numpy()
        robot_series["base_pose"][frame_idx] = np.concatenate([r0_pos, r0_quat]).astype(np.float32)

        for arm in self.robot.arm_names:
            eef_pos = proprio_dict[f"eef_{arm}_pos"]
            eef_quat = proprio_dict[f"eef_{arm}_quat"]
            h_eef_in_robot = T.pose2mat((eef_pos, eef_quat))
            h_eef_in_robot0 = h_in_robot0 @ h_eef_in_robot
            eef_pos_r0_t, eef_quat_r0_t = T.mat2pose(h_eef_in_robot0)
            eef_pos_r0 = eef_pos_r0_t.cpu().numpy()
            eef_quat_r0 = eef_quat_r0_t.cpu().numpy()
            robot_series[f"{arm}_gripper_pose"][frame_idx] = np.concatenate([eef_pos_r0, eef_quat_r0]).astype(
                np.float32
            )

            gripper_qpos = proprio_dict[f"gripper_{arm}_qpos"]
            gripper_qpos_np = gripper_qpos.cpu().numpy()

            gripper_open = np.mean(gripper_qpos_np) > self.args.gripper_finger_threshold
            grasp_state_int = int(proprio_dict[f"grasp_{arm}"].item())
            is_grasping_bool = grasp_state_int == int(IsGraspingState.TRUE)
            if is_grasping_bool:
                gripper_open = False
            robot_series[f"{arm}_gripper_open"][frame_idx, 0] = gripper_open
            robot_series[f"{arm}_is_grasping"][frame_idx, 0] = is_grasping_bool

    def _execute_transitions(self, transitions_data):
        """Execute scene transitions (object additions/removals)."""
        scene = og.sim.scenes[0]
        for add_sys_name in transitions_data["systems"]["add"]:
            scene.get_system(add_sys_name, force_init=True)
        for remove_sys_name in transitions_data["systems"]["remove"]:
            scene.clear_system(remove_sys_name)
        for remove_obj_name in transitions_data["objects"]["remove"]:
            obj = scene.object_registry("name", remove_obj_name)
            scene.remove_object(obj)
        for j, add_obj_info in enumerate(transitions_data["objects"]["add"]):
            obj = create_object_from_init_info(add_obj_info)
            scene.add_object(obj)
            obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
        og.sim.step()

    def run_extraction(self):
        """Process single dataset file."""
        input_file = self.args.input_path
        output_file = self.args.output_path

        if not os.path.isfile(input_file):
            raise ValueError(f"Input file does not exist: {input_file}")

        if not input_file.endswith(".h5") and not input_file.endswith(".hdf5"):
            raise ValueError(f"Input file must be .h5 or .hdf5 format: {input_file}")

        print(f"Processing file: {input_file}")
        print(f"Output: {output_file}")

        self._process_input_file(input_file, output_file)

        if self.clips_processed == 0:
            raise RuntimeError(f"No valid clips were processed for {input_file}")

        avg_points = self.total_scene_points_processed / self.clips_processed
        print(f"avg_points: {avg_points:.0f}")

        print("File processing completed successfully!")

        if self.invalid_clips_count > 0:
            print(f"\nInvalid clips filtered: {self.invalid_clips_count}")

    def _process_input_file(self, input_file, output_file):
        """Process a single H5 file with per-clip temp outputs, then aggregate to final."""
        self.clip_writer = ClipWriter(output_file, self._get_joint_names, input_file)
        if self.clip_writer.cleanup_if_output_complete():
            print(f"Final output already complete at {output_file}; exiting early.")
            return

        playback_output_path = output_file + ".playback_tmp.h5"
        if os.path.abspath(playback_output_path) == os.path.abspath(output_file):
            raise RuntimeError("Playback output path must be distinct from the extracted output path.")
        self._create_environment(input_file, playback_output_path)
        self.camera_extractor = CameraDataExtractor(
            self.env, self.args, self.workspace_bounds_min, self.workspace_bounds_max
        )
        self.motion_tracker = MotionTracker(self.env, self.robot, self.args, self._get_joint_positions_array)

        n_episodes = self.env.input_hdf5["data"].attrs["n_episodes"]
        assert n_episodes == 1, "Assume only one episode for now"

        for episode_id in range(n_episodes):
            self.extract_episode(episode_id)

        self.clip_writer.aggregate_episode()
        if hasattr(self.env, "hdf5_file") and self.env.hdf5_file is not None:
            self.env.hdf5_file.close()
        if self._playback_output_path and os.path.isfile(self._playback_output_path):
            os.remove(self._playback_output_path)
        print(f"File processing completed! Output saved to {output_file}")


def build_arg_parser():
    """Create argument parser for the extractor."""
    parser = argparse.ArgumentParser(
        description="Behavior 3D flows (HF raw H5 list -> extracted H5 files)"
    )

    # Input/Output
    parser.add_argument(
        "--input_list",
        type=str,
        required=True,
        help=(
            "Text file containing one HF BEHAVIOR input per line "
            "(hf://behavior-1k/2025-challenge-rawdata/...)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root directory for list mode.",
    )
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for list mode")
    parser.add_argument("--world_size", type=int, default=1, help="Total workers for list mode")
    parser.add_argument(
        "--allow_remote_streaming",
        action="store_true",
        help=(
            "Allow remote BEHAVIOR episode streaming without persistent cache. "
            "By default this stage requires POINTWORLD_CACHE_DIR when remote paths are used."
        ),
    )

    # Processing arguments
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    # Image parameters
    parser.add_argument("--image_height", type=int, default=180, help="Image height for external cameras")
    parser.add_argument("--image_width", type=int, default=320, help="Image width for external cameras")

    # Temporal parameters
    parser.add_argument("--skip_every", type=int, default=5, help="Skip frames for clip generation")
    parser.add_argument("--frames_per_clip", type=int, default=11, help="Number of frames per clip")
    parser.add_argument("--time_skip_ratio", type=int, default=6, help="Temporal downsampling ratio")

    # Motion filtering arguments
    parser.add_argument(
        "--object_movement_pos_threshold",
        type=float,
        default=0.05,
        help="Position threshold (meters) for object movement detection",
    )
    parser.add_argument(
        "--object_movement_rot_threshold",
        type=float,
        default=np.deg2rad(45),
        help="Rotation threshold (radians) for object movement detection",
    )
    parser.add_argument(
        "--gripper_finger_threshold",
        type=float,
        default=0.049,
        help="Threshold for gripper open/close state detection",
    )
    parser.add_argument(
        "--ee_pos_threshold",
        type=float,
        default=0.20,
        help="End-effector position motion threshold when gripper is open",
    )
    parser.add_argument(
        "--ee_rot_threshold",
        type=float,
        default=np.deg2rad(90),
        help="End-effector rotation motion threshold when gripper is open",
    )
    parser.add_argument(
        "--gripper_closed_ee_pos_threshold",
        type=float,
        default=0.10,
        help="End-effector position motion threshold when gripper is closed",
    )
    parser.add_argument(
        "--gripper_closed_ee_rot_threshold",
        type=float,
        default=np.deg2rad(45),
        help="End-effector rotation motion threshold when gripper is closed",
    )

    # Joint movement threshold
    parser.add_argument(
        "--joint_movement_threshold",
        type=float,
        default=0.02,
        help="Threshold for joint movement detection",
    )

    # Workspace bounds (robot frame)
    parser.add_argument(
        "--workspace_bounds_min",
        nargs=3,
        type=float,
        default=[0.0, -0.8, -0.05],
        help="Workspace bounds min (robot frame)",
    )
    parser.add_argument(
        "--workspace_bounds_max",
        nargs=3,
        type=float,
        default=[1.3, 0.8, 2.0],
        help="Workspace bounds max (robot frame)",
    )

    return parser


def _validate_rank_world_size(rank: int, world_size: int) -> None:
    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}")


def _load_input_list(input_list_path: str) -> list[str]:
    if not os.path.isfile(input_list_path):
        raise FileNotFoundError(f"input_list does not exist: {input_list_path}")
    with open(input_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    input_paths = []
    for line_idx, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        if not line.startswith("hf://"):
            raise ValueError(
                f"input_list line {line_idx} must use hf:// format; got: {line}"
            )
        if not is_behavior_hf_path(line):
            raise ValueError(
                f"input_list line {line_idx} must point to "
                "hf://behavior-1k/2025-challenge-rawdata/...; got: "
                f"{line}"
            )
        input_paths.append(line)
    if len(input_paths) == 0:
        raise ValueError(f"input_list has no valid entries: {input_list_path}")
    return input_paths


def _derive_output_path(input_path: str, output_root: str) -> str:
    rel = derive_behavior_output_relpath(input_path, local_input_root=None)
    return os.path.join(output_root, rel)


def _run_single_file(args, input_path: str, output_path: str) -> None:
    if not is_behavior_hf_path(input_path):
        raise ValueError(
            "input_path must be an HF BEHAVIOR path under "
            "hf://behavior-1k/2025-challenge-rawdata/..."
        )
    enforce_behavior_cache_policy(
        [input_path],
        stage_name="behavior_3d_flows",
        require_cache=True,
        allow_streaming=args.allow_remote_streaming,
    )

    temp_stream_dir = None
    resolved_input_path = input_path
    try:
        if not os.environ.get("POINTWORLD_CACHE_DIR", "").strip():
            temp_stream_dir = tempfile.mkdtemp(prefix="behavior_raw_stream_")

        resolved_input_path = get_local_behavior_input_path(
            input_path,
            temp_dir=temp_stream_dir,
        )

        if not os.path.exists(resolved_input_path):
            raise FileNotFoundError(f"Input path does not exist: {resolved_input_path}")
        if not os.path.isfile(resolved_input_path):
            raise FileNotFoundError(f"Input path must be a file: {resolved_input_path}")
        if not resolved_input_path.endswith(".h5") and not resolved_input_path.endswith(".hdf5"):
            raise ValueError(f"Input file must be .h5 or .hdf5 format: {resolved_input_path}")

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {input_path} -> {output_path}")
        if resolved_input_path != input_path:
            print(f"Resolved remote input: {input_path} -> {resolved_input_path}")
        print(f"Image resolution: {args.image_height}x{args.image_width}")
        print(f"Skip every {args.skip_every} frames, {args.frames_per_clip} frames per clip (downsampled)")
        print(f"Time skip ratio: {args.time_skip_ratio} (record every {args.time_skip_ratio} original frames)")
        print(
            "Motion filtering: "
            f"object_pos_thresh={args.object_movement_pos_threshold}, "
            f"object_rot_thresh={args.object_movement_rot_threshold}, "
            f"gripper_finger_thresh={args.gripper_finger_threshold}"
        )
        print(
            "                 "
            f"ee_pos_thresh={args.ee_pos_threshold:.3f}m, "
            f"ee_rot_thresh={np.rad2deg(args.ee_rot_threshold):.1f}°"
        )
        print(
            "                 "
            f"gripper_closed_ee_pos_thresh={args.gripper_closed_ee_pos_threshold:.3f}m, "
            f"gripper_closed_ee_rot_thresh={np.rad2deg(args.gripper_closed_ee_rot_threshold):.1f}°"
        )
        print(f"                 joint_thresh={args.joint_movement_threshold}")
        print(
            "Workspace bounds (robot frame): "
            f"min={np.array(args.workspace_bounds_min)}, max={np.array(args.workspace_bounds_max)}"
        )

        run_args = argparse.Namespace(**vars(args))
        run_args.input_path = resolved_input_path
        run_args.output_path = output_path
        extractor = BehaviorFlowExtractor(run_args)
        extractor.run_extraction()
        print(f"Successfully processed data to {output_path}")
    finally:
        if temp_stream_dir and os.path.isdir(temp_stream_dir):
            shutil.rmtree(temp_stream_dir, ignore_errors=True)


def main():
    """Entry point for Behavior H5 extraction."""
    np.random.seed(42)
    th.manual_seed(42)

    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_rank_world_size(args.rank, args.world_size)

    input_paths = _load_input_list(args.input_list)
    enforce_behavior_cache_policy(
        input_paths,
        stage_name="behavior_3d_flows",
        require_cache=True,
        allow_streaming=args.allow_remote_streaming,
    )

    worker_inputs = input_paths[args.rank::args.world_size]
    print(
        f"List mode: rank {args.rank}/{args.world_size} "
        f"processing {len(worker_inputs)}/{len(input_paths)} files"
    )

    if len(worker_inputs) == 0:
        print("No files assigned to this worker.")
        return 0

    for idx, input_path in enumerate(worker_inputs, start=1):
        output_path = _derive_output_path(input_path, args.output_root)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"[{idx}/{len(worker_inputs)}] Processing {input_path} -> {output_path}")
        _run_single_file(args, input_path, output_path)
        og.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
