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

import copy
import os
import sys
import numpy as np
from argparse import ArgumentParser

LOCAL_DATASET_DIR = os.environ.get('LOCAL_DATASET_DIR', '/dataset')
DOMAIN_TO_DATA_DIR = {
    'behavior': f'{LOCAL_DATASET_DIR}/behavior/wds',
    'droid': f'{LOCAL_DATASET_DIR}/droid/wds',
}

def str_to_bool(value):
    """Convert string representations to boolean values."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")

def str_to_none(value):
    """Convert 'none' or 'None' string to Python None, leave other values unchanged."""
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return value


def parse_optional_int(value, arg_name: str):
    """Parse an optional integer CLI value where None means unset."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str):
        if value.lower() == "none":
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{arg_name} must be an integer or 'none', got {value!r}") from exc
    else:
        raise ValueError(f"{arg_name} must be an integer or 'none', got type {type(value).__name__}")
    if parsed < 1:
        raise ValueError(f"{arg_name} must be >= 1 when specified, got {parsed}")
    return parsed


def _collect_explicit_cli_dests(parser: ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit_options = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        explicit_options.add(token.split("=", 1)[0])
    return {option_to_dest[opt] for opt in explicit_options if opt in option_to_dest}

def parse_args(skip_command_line=False):
    parser = ArgumentParser()
    # general
    parser.add_argument('--seed', '-s', type=int, default=-1)
    parser.add_argument('--deterministic_data', '-det', type=str, default='false', help='Use deterministic data')
    parser.add_argument('--deterministic_train', type=str, default='false', help='Use deterministic data pipeline for training (disables resampling)')
    parser.add_argument('--deterministic_algorithms', type=str, default='false',
                        help='Force torch deterministic algorithms (may error if unsupported ops are used)')
    parser.add_argument('--log_dir', '-ld', type=str, default='train_logs')
    parser.add_argument('--exp_name', '-en', type=str, default=None, help='Experiment name; if not provided, will be auto-generated')
    parser.add_argument('--distributed', '-ddp', type=str, default='false', help='Use DDP')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=22)
    parser.add_argument('--num_epochs', '-ne', type=int, default=200)
    parser.add_argument('--num_workers', '-nw', type=int, default=16)
    parser.add_argument('--eval_num_workers', '-enw', type=int, default=5)
    parser.add_argument('--eval_freq', '-ef', type=int, default=600, help='Evaluate every N batches (equivalent to seconds in previous version)')
    parser.add_argument('--save_freq', '-sf', type=int, default=1800, help='Save checkpoint every N batches (equivalent to seconds in previous version)')
    parser.add_argument('--num_eval_batches', '-ntb', type=int, default=10)
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model checkpoint to load')
    parser.add_argument('--allow_optimizer_reset', type=str, default='false',
                        help='Allow optimizer reset if checkpoint optimizer state is incompatible')
    parser.add_argument('--grad_clip_max_norm', '-gcmn', type=float, default=5.0)
    parser.add_argument('--data_dirs', '-dd', type=str, default=None, help='comma separated list of data directories')
    parser.add_argument('--max_train_steps', type=int, default=-1, help='Stop training after this many train steps (<=0 disables)')
    parser.add_argument(
        '--train_splits',
        type=str,
        default=None,
        help='Optional comma-separated split override for training dataloader (for example: train or test).',
    )
    # robot sampler
    parser.add_argument('--max_robot_points', '-mrp', type=int, default=500)
    # augmentation (fixed defaults; only keep camera/point caps)
    parser.add_argument('--max_scene_points', '-msp', type=int, default=12000)
    parser.add_argument(
        '--train_min_num_cameras',
        type=str,
        default='1',
        help="Minimum number of cameras to sample during training ('none' = auto, clipped by per-sample availability).",
    )
    parser.add_argument(
        '--train_max_num_cameras',
        type=str,
        default='3',
        help="Maximum number of cameras to sample during training ('none' = auto, clipped by per-sample availability).",
    )
    parser.add_argument(
        '--eval_min_num_cameras',
        type=int,
        default=2,
        help='Minimum number of cameras to sample during evaluation.',
    )
    parser.add_argument(
        '--eval_max_num_cameras',
        type=int,
        default=2,
        help='Maximum number of cameras to sample during evaluation.',
    )
    # visualization (train-time visualization removed in release)
    # data
    parser.add_argument('--robot_features', '-rfeat', type=str, default='robot_flows,robot_colors,robot_normals,gripper_open,robot_velocity,robot_acceleration')
    parser.add_argument('--scene_features', '-sfeat', type=str, default='scene_flows,scene_colors,scene_normals,gripper_open,dist2robot')
    parser.add_argument('--domains', '-dom', type=str, default='', required=False, help='comma separated list of domains')
    # optimizer
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # dynamics
    parser.add_argument('--grid_size', '-gs', type=float, default=0.015)
    # normalization
    parser.add_argument('--norm_stats_path', type=str, default='stats/droid', help='Path to folder containing precomputed JSON files with normalization statistics')
    # scene encoder (release: fixed to DINOv3 ViT-L16 multi-layer 2D)
    # compile / performance controls
    parser.add_argument('--disable_compile', type=str, default='false', help='Disable torch.compile for inference-only paths')
    # robot + deployment options: selection handled via deploy/robots.py (ROBOT_TYPE)
    # ----- DINOv3 aggregation options (fixed) -----
    parser.add_argument('--depth_threshold', '-dt', type=float, default=0.003, help='Depth threshold for visibility mask')
    # predictor
    parser.add_argument('--ptv3_size', '-ptv3s', type=str, default='base', help='Size of the ptv3 backbone (small|base|large)')
    parser.add_argument('--ptv3_patch_size', '-ptv3ps', type=int, default=256, help='Patch size for ptv3 backbone')
    parser.add_argument('--predictor_dim', '-pd', type=int, default=256, help='Dimension of predictor')
    # loss
    parser.add_argument('--huber_delta', '-hdl', type=float, default=5.0, help='Delta for huber loss')
    # aleatoric uncertainty
    # confidence threshold for uncertainty-aware metrics / viz
    parser.add_argument('--confidence_thres', '-cth', type=float, default=0.8,
                        help='Confidence threshold (0..1) for uncertainty-aware L2 metrics and filtering')
    # eval
    parser.add_argument('--eval_exp_name', '-ev', type=str, default=None)
    parser.add_argument('--eval_num_batches', '-enb', type=int, default=-1, help='Number of batches to evaluate (-1 for all)')
    parser.add_argument('--run_confidence_annotation', type=str, default='false',
                        help='Run a pass over eval_splits to store expert confidence arrays (B,T,Ns) per sample in H5 files; used later to compute filtered metrics.')
    parser.add_argument('--allow_missing_confidence_mask', type=str, default='false',
                        help='Allow evaluation to proceed if expert confidence masks are missing (skips filtered metrics).')
    parser.add_argument('--eval_viz_num', type=int, default=-1,
                        help='Number of eval samples to visualize (-1 disables visualization).')
    parser.add_argument('--eval_skip_viz', type=str, default='false',
                        help='Disable evaluation visualization even if eval_viz_num > 0.')
    parser.add_argument('--viewer_port', type=int, default=8080,
                        help='Viser viewer port for evaluation visualization.')
    if skip_command_line:
        raw_argv = []
        args = parser.parse_args([])
    else:
        raw_argv = sys.argv[1:]
        args = parser.parse_args()

    # make a deep copy of the args, this will be used for wandb sweeps
    args.og_args = copy.deepcopy(args)
    args._explicit_cli_dests = _collect_explicit_cli_dests(parser, raw_argv)

    # convert string arguments (which are supposed to be booleans or None) to bool or None
    for action in parser._actions:
        dest = action.dest
        if dest == 'help':
            continue
        val  = getattr(args, dest)
        # only care about str‑typed args
        if action.type is str:
            default = action.default
            if isinstance(default, str) and default.lower() in ('true','false'):
                setattr(args, dest, str_to_bool(val))
            elif default is None or (isinstance(default, str) and default.lower() == 'none'):
                setattr(args, dest, str_to_none(val))

    # Release-fixed defaults for trimmed CLI
    args.amp = True

    if args.deterministic_train:
        args.deterministic_data = True
    args.train_min_num_cameras = parse_optional_int(args.train_min_num_cameras, "--train_min_num_cameras")
    args.train_max_num_cameras = parse_optional_int(args.train_max_num_cameras, "--train_max_num_cameras")
    if args.train_min_num_cameras is not None and args.train_max_num_cameras is not None:
        if args.train_min_num_cameras > args.train_max_num_cameras:
            raise ValueError(
                "--train_min_num_cameras must be <= --train_max_num_cameras "
                f"(got {args.train_min_num_cameras} > {args.train_max_num_cameras})"
            )
    if args.eval_min_num_cameras < 1:
        raise ValueError(f"--eval_min_num_cameras must be >= 1, got {args.eval_min_num_cameras}")
    if args.eval_max_num_cameras < 1:
        raise ValueError(f"--eval_max_num_cameras must be >= 1, got {args.eval_max_num_cameras}")
    if args.eval_min_num_cameras > args.eval_max_num_cameras:
        raise ValueError(
            "--eval_min_num_cameras must be <= --eval_max_num_cameras "
            f"(got {args.eval_min_num_cameras} > {args.eval_max_num_cameras})"
        )
    # convert comma separated strings to lists
    args.robot_features = [name.strip() for name in args.robot_features.split(',')]
    args.scene_features = [name.strip() for name in args.scene_features.split(',')]

    expected_robot_features = [
        'robot_flows', 'robot_colors', 'robot_normals',
        'gripper_open', 'robot_velocity', 'robot_acceleration',
    ]
    expected_scene_features = [
        'scene_flows', 'scene_colors', 'scene_normals',
        'gripper_open', 'dist2robot',
    ]
    if args.robot_features != expected_robot_features:
        raise ValueError(
            f"Unsupported robot_features for release: {args.robot_features}. "
            f"Expected: {expected_robot_features}"
        )
    if args.scene_features != expected_scene_features:
        raise ValueError(
            f"Unsupported scene_features for release: {args.scene_features}. "
            f"Expected: {expected_scene_features}"
        )
    # validate and default
    args.domains = [name.strip() for name in args.domains.split(',')] if args.domains else []
    if any(domain.startswith('droid') for domain in args.domains):
        args.dynamics_head_init_scale = 1.0
    else:
        args.dynamics_head_init_scale = 0.0
    # map domains to data dirs if not provided
    if args.data_dirs is None:
        args.data_dirs = []
        for domain in args.domains:
            args.data_dirs.append(DOMAIN_TO_DATA_DIR[domain])
    else:
        args.data_dirs = [name.strip() for name in args.data_dirs.split(',')]
    assert len(args.data_dirs) == len(args.domains), f'expected data_dirs and domains to have one to one mapping, got {len(args.data_dirs)} and {len(args.domains)}'
    # set a random random seed if None
    if args.seed == -1:
        import time
        args.seed = int(time.time_ns() + os.getpid()) % 1000000
    if args.eval_exp_name:
        args.amp = False  # often lead to NaN during eval

    if args.max_train_steps == 0:
        raise ValueError("--max_train_steps must be positive or -1")
    if args.max_train_steps < -1:
        raise ValueError("--max_train_steps must be -1 or a positive integer")

    if args.train_splits is not None:
        train_splits = [split.strip() for split in args.train_splits.split(',') if split.strip()]
        if len(train_splits) == 0:
            raise ValueError("--train_splits must include at least one split name")
        valid_splits = {"train", "test"}
        invalid_splits = [split for split in train_splits if split not in valid_splits]
        if invalid_splits:
            raise ValueError(
                f"--train_splits contains unsupported split(s): {invalid_splits}. "
                f"Supported splits: {sorted(valid_splits)}"
            )
        args.train_splits = train_splits[0] if len(train_splits) == 1 else train_splits
    
    return args
