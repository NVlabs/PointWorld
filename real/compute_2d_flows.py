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
import time
import json
import h5py
import numpy as np
from tqdm import tqdm

from real.real_utils import get_time_str
from real.gcs_utils import enforce_gcs_cache_policy
from real.droid_utils import (
    gather_data_dict,
    gather_trajectory,
    filter_by_timestamps,
    get_uuid,
)
from shared.h5_io import save_rgb_as_jpeg_in_h5
from real.flow_2d import Flow2DTracker, COTRACKER_CKPT_PATH


class TrackCacheRunner(Flow2DTracker):
    """
    Runs maskless 2D tracking over DROID scenes, slices into clips, and writes a
    compact tracks cache per scene for later 3D lifting ablations.

    Notes:
    - Uses the shared 2D flow tracker utilities.
    - Does not apply robot/workspace masks during tracking by default (maskless).
    - Stores per-clip first RGB frame to guarantee color parity in later conversions.
    """

    def __init__(self, cotracker_ckpt_path=None):
        super().__init__(
            cotracker_ckpt_path=cotracker_ckpt_path,
        )

    def process(
        self,
        scene_path,
        save_dir,
        time_skip_ratio=2,
        downscale_ratio=0.5,
        frames_per_clip=11,
        skip_every=5,
        space_skip_ratio=2,
        skip_processed=True,
        ee_pos_threshold=0.20,
        ee_rot_threshold=np.deg2rad(90),
        gripper_threshold=0.1,
        gripper_closed_ee_pos_threshold=0.10,
        gripper_closed_ee_rot_threshold=np.deg2rad(60),
    ):
        self.output_dir = save_dir
        self.scene_path = scene_path
        self.uuid = get_uuid(scene_path)
        self.save_dir = os.path.join(self.output_dir, '2d_flows')
        self.cache_h5_path = os.path.join(self.save_dir, f'{self.uuid}_2d_flows.h5')
        self.cache_h5_tmp_path = self.cache_h5_path + ".tmp"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # Enhanced resume/skip logic: decide based on 2d flows completeness
        def _assess_flows_state(path):
            if not os.path.exists(path):
                return 'absent'
            with h5py.File(path, 'r') as f:
                return 'complete' if bool(f.attrs.get('write_complete', False)) else 'incomplete'

        if skip_processed:
            tracks_state = _assess_flows_state(self.cache_h5_path)
            print(f"Resume check [{self.uuid}]: 2d_flows={tracks_state}")
            if tracks_state == 'complete':
                print(f"Skipping scene {scene_path} - 2d flows complete")
                return True
            else:
                print(f"Will process {scene_path}: (tracks={tracks_state})")

        # Clean up any leftover temp file from a previous interrupted run (tracks cache)
        if os.path.exists(self.cache_h5_tmp_path):
            os.remove(self.cache_h5_tmp_path)
            print(f"Removed stale temp file: {self.cache_h5_tmp_path}")
        self.tracker = self._load_tracker()

        # Collect RGB + intrinsics and optionally depth; proprio loaded separately
        print(f"[{get_time_str()}] Processing scene for tracks cache: {scene_path}")
        data_dict = gather_data_dict(
            scene_path,
            downscale_ratio=downscale_ratio,
            include_stereo=False,
            include_depth=False,
        )
        proprio_dict = gather_trajectory(scene_path)

        # Canonical timestamps from first camera + optional time skip
        first_cam = list(data_dict.keys())[0]
        self.canonical_timestamps = data_dict[first_cam]['timestamps']
        if time_skip_ratio > 1:
            self.canonical_timestamps = self.canonical_timestamps[::time_skip_ratio]

        self.T = len(self.canonical_timestamps)
        data_dict = filter_by_timestamps(
            data_dict, self.canonical_timestamps, scene_path, is_proprio=False, verbose=False
        )
        proprio_dict = filter_by_timestamps(
            proprio_dict, self.canonical_timestamps, scene_path, is_proprio=True, verbose=False
        )

        # Slice into clips and filter by EE motion (same as original pipeline)
        sliced_data_dict, sliced_proprio_dict = self.slice_video_clips(
            data_dict, proprio_dict, frames_per_clip=frames_per_clip, skip_every=skip_every
        )
        sliced_data_dict, sliced_proprio_dict = self.filter_clips_by_ee_motion(
            sliced_data_dict,
            sliced_proprio_dict,
            ee_pos_threshold=ee_pos_threshold,
            ee_rot_threshold=ee_rot_threshold,
            gripper_threshold=gripper_threshold,
            gripper_closed_ee_pos_threshold=gripper_closed_ee_pos_threshold,
            gripper_closed_ee_rot_threshold=gripper_closed_ee_rot_threshold,
        )

        # Do not compute any image-space masks here.
        # We intentionally track without assuming camera pose; masks are applied later in 3D.

        # Run tracker on each clip and collect minimal cache outputs
        print(f"[{get_time_str()}] Running tracker and sampling colors for cache...")
        cache_payload = {
            'uuid': self.uuid,
            'scene_path': self.scene_path,
            'canonical_timestamps': self.canonical_timestamps,
            'frames_per_clip': frames_per_clip,
            'skip_every': skip_every,
            'time_skip_ratio': time_skip_ratio,
            'downscale_ratio': downscale_ratio,
            'space_skip_ratio': space_skip_ratio,
            'tracker_type': 'cotracker',
            'tracker_ckpt': self.cotracker_ckpt_path,
            # Persist thresholds used for clip selection so converter can mirror attrs
            'ee_pos_threshold': float(ee_pos_threshold),
            'ee_rot_threshold': float(ee_rot_threshold),
            'gripper_closed_ee_pos_threshold': float(gripper_closed_ee_pos_threshold),
            'gripper_closed_ee_rot_threshold': float(gripper_closed_ee_rot_threshold),
            'gripper_threshold': float(gripper_threshold),
        }

        # Organize per-camera, per-clip cache
        payload_by_camera = {}
        for camera_serial, camera_data in sliced_data_dict.items():
            intrinsic = camera_data['intrinsic']
            camera_type = 'ext'

            for clip_key, clip_rec in camera_data.items():
                if ':' not in clip_key:
                    continue
                rgb_seq = clip_rec['rgb']  # (T, H, W, 3) uint8
                T_clip, H, W, _ = rgb_seq.shape

                pred_tracks, pred_visibility = self._tracker_inference(
                    rgb_seq,
                    space_skip_ratio=space_skip_ratio,
                )

                # OOB invalidation, then round & clamp (parity with original) for color sampling only
                tracks_rounded = np.round(pred_tracks).astype(int)
                out_of_bounds_mask = (
                    (pred_tracks[..., 0] < 0)
                    | (pred_tracks[..., 0] >= W)
                    | (pred_tracks[..., 1] < 0)
                    | (pred_tracks[..., 1] >= H)
                )
                pred_visibility[out_of_bounds_mask] = 0

                x_coords = np.clip(tracks_rounded[..., 0], 0, W - 1)
                y_coords = np.clip(tracks_rounded[..., 1], 0, H - 1)
                t_idx = np.arange(T_clip)[:, None]
                flow_colors = rgb_seq[t_idx, y_coords, x_coords].astype(np.uint8)  # (T,N,3)

                # First frame RGB for writer compatibility
                first_rgb = rgb_seq[0]

                # Write into payload dict
                cam_key = f"{camera_serial}+{camera_type}"
                if cam_key not in payload_by_camera:
                    payload_by_camera[cam_key] = {
                        'intrinsic': intrinsic,
                        'clips': {},
                    }

                payload_by_camera[cam_key]['clips'][clip_key] = {
                    'H': int(H),
                    'W': int(W),
                    'flows_2d_xy': pred_tracks.astype(np.float32),
                    'flows_2d_visibility': pred_visibility.astype(bool),
                    'flow_colors': flow_colors.astype(np.uint8),
                    'first_frame_rgb': first_rgb.astype(np.uint8),
                }

        # Store proprio per-clip (minimal set for later masks)
        proprio_by_clip = {}
        for clip_key, rec in sliced_proprio_dict.items():
            if ':' not in clip_key:
                continue
            proprio_by_clip[clip_key] = {
                'joint_positions': rec['joint_positions'].astype(np.float32),
                'joint_velocities': rec['joint_velocities'].astype(np.float32),
                'joint_torques': rec['joint_torques'].astype(np.float32),
                'gripper_positions': rec['gripper_positions'].astype(np.float32),
                'gripper_pose': rec['gripper_pose'].astype(np.float32),
            }

        # Persist cache
        self.store_2d_flows(self.cache_h5_path, cache_payload, payload_by_camera, proprio_by_clip)
        print(f"[{get_time_str()}] 2d flows saved to {self.cache_h5_path}")
        return True

    def store_2d_flows(self, output_path, meta, cams_dict, proprio_dict):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"
        # Write to a temp file first for atomicity
        with h5py.File(tmp_path, 'w') as f:
            # Metadata
            f.attrs['uuid'] = meta['uuid']
            f.attrs['scene_path'] = meta['scene_path']
            f.attrs['creation_time'] = time.strftime("%Y%m%d_%H%M%S")
            f.attrs['canonical_timestamps'] = json.dumps(list(map(int, meta['canonical_timestamps'])))
            f.attrs['frames_per_clip'] = int(meta['frames_per_clip'])
            f.attrs['skip_every'] = int(meta['skip_every'])
            f.attrs['time_skip_ratio'] = int(meta['time_skip_ratio'])
            f.attrs['downscale_ratio'] = float(meta['downscale_ratio'])
            f.attrs['space_skip_ratio'] = int(meta['space_skip_ratio'])
            f.attrs['tracker_type'] = meta['tracker_type']
            f.attrs['tracker_ckpt'] = meta['tracker_ckpt']
            # Thresholds used for clip selection
            f.attrs['ee_pos_threshold'] = float(meta['ee_pos_threshold'])
            f.attrs['ee_rot_threshold'] = float(meta['ee_rot_threshold'])
            f.attrs['gripper_closed_ee_pos_threshold'] = float(meta['gripper_closed_ee_pos_threshold'])
            f.attrs['gripper_closed_ee_rot_threshold'] = float(meta['gripper_closed_ee_rot_threshold'])
            f.attrs['gripper_threshold'] = float(meta['gripper_threshold'])
            f.attrs['write_complete'] = False

            # Cameras and clips
            cams_group = f.create_group('cameras')
            for cam_key, cam_payload in cams_dict.items():
                g_cam = cams_group.create_group(cam_key)
                g_cam.create_dataset('intrinsic', data=cam_payload['intrinsic'].astype(np.float32))
                g_clips = g_cam.create_group('clips')
                for clip_key, clip_data in cam_payload['clips'].items():
                    g_clip = g_clips.create_group(clip_key)
                    g_clip.attrs['H'] = int(clip_data['H'])
                    g_clip.attrs['W'] = int(clip_data['W'])
                    g_clip.create_dataset('flows_2d_xy', data=clip_data['flows_2d_xy'].astype(np.float32))
                    g_clip.create_dataset('flows_2d_visibility', data=clip_data['flows_2d_visibility'].astype(np.bool_))
                    g_clip.create_dataset('flow_colors', data=clip_data['flow_colors'].astype(np.uint8))
                    # Also store the first RGB frame for writer compatibility in converter
                    save_rgb_as_jpeg_in_h5(g_clip, 'initial_rgb', clip_data['first_frame_rgb'].astype(np.uint8))

            # Proprio per-clip
            proprio_group = f.create_group('proprio')
            for clip_key, pr in proprio_dict.items():
                g_p = proprio_group.create_group(clip_key)
                for k, v in pr.items():
                    g_p.create_dataset(k, data=v)

            f.attrs['write_complete'] = True
        # Atomically replace/rename into place
        try:
            os.replace(tmp_path, output_path)
        except Exception:
            # Best-effort cleanup; re-raise to surface error
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description='Run tracking and store 2d flows per scene (no masks)')
    parser.add_argument('--input', type=str, required=True, help='Text file of scene paths')
    parser.add_argument('--output_dir', type=str, required=True, help='Output root directory')
    parser.add_argument('--max_scenes', type=int, default=None, help='Max scenes to process')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--cotracker_ckpt', type=str, default=COTRACKER_CKPT_PATH, help='Path to CoTracker checkpoint')
    parser.add_argument('--time_skip_ratio', type=int, default=2)
    parser.add_argument('--downscale_ratio', type=float, default=0.5)
    parser.add_argument('--frames_per_clip', type=int, default=11)
    parser.add_argument('--skip_every', type=int, default=5)
    parser.add_argument('--space_skip_ratio', type=int, default=2)
    parser.add_argument('--skip_processed', action='store_true', default=True)
    parser.add_argument('--no-skip_processed', dest='skip_processed', action='store_false', help='Force rebuild even if cache exists')
    # EE motion thresholds
    parser.add_argument('--ee_pos_threshold', type=float, default=0.20)
    parser.add_argument('--ee_rot_threshold', type=float, default=np.deg2rad(90))
    parser.add_argument('--gripper_threshold', type=float, default=0.1)
    parser.add_argument('--gripper_closed_ee_pos_threshold', type=float, default=0.10)
    parser.add_argument('--gripper_closed_ee_rot_threshold', type=float, default=np.deg2rad(60))
    parser.add_argument(
        '--allow_gcs_streaming',
        action='store_true',
        help=(
            'Bypass GCS cache enforcement and stream directly from gs:// inputs for this run. '
            'Not recommended for repeated multi-stage processing.'
        ),
    )

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        scene_paths = [line.strip() for line in f if line.strip()]

    enforce_gcs_cache_policy(
        scene_paths,
        stage_name='compute_2d_flows',
        require_cache=True,
        allow_streaming=args.allow_gcs_streaming,
    )

    random.seed(args.random_seed)
    random.shuffle(scene_paths)
    if args.max_scenes is not None and args.max_scenes > 0:
        scene_paths = scene_paths[:args.max_scenes]

    if args.world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}")

    worker_scenes = scene_paths[args.rank::args.world_size]
    print(f"Worker {args.rank}/{args.world_size} processing {len(worker_scenes)} scenes for track cache")

    runner = TrackCacheRunner(
        cotracker_ckpt_path=args.cotracker_ckpt,
    )
    pbar = tqdm(worker_scenes, total=len(worker_scenes), desc=f"worker {args.rank} cache", unit="scene", dynamic_ncols=True)
    for scene_path in pbar:
        tqdm.write(f"[{get_time_str()}] Worker {args.rank} processing: {scene_path}")
        ok = runner.process(
            scene_path,
            save_dir=args.output_dir,
            time_skip_ratio=args.time_skip_ratio,
            downscale_ratio=args.downscale_ratio,
            frames_per_clip=args.frames_per_clip,
            skip_every=args.skip_every,
            space_skip_ratio=args.space_skip_ratio,
            skip_processed=args.skip_processed,
            ee_pos_threshold=args.ee_pos_threshold,
            ee_rot_threshold=args.ee_rot_threshold,
            gripper_threshold=args.gripper_threshold,
            gripper_closed_ee_pos_threshold=args.gripper_closed_ee_pos_threshold,
            gripper_closed_ee_rot_threshold=args.gripper_closed_ee_rot_threshold,
        )
        tqdm.write(f"[{get_time_str()}] Done with scene {get_uuid(scene_path)}: status={ok}")


if __name__ == '__main__':
    main()
