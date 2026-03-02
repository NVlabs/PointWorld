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

import glob
import json
import os
import random
from functools import partial

import numpy as np
import webdataset as wds

from dataset_components.cameras import sample_cameras
from dataset_components.collate import custom_collate_fn
from dataset_components.constants import (
    RELEASE_GRIPPER_ONLY,
    RELEASE_TRAIN_SPLITS,
)
from dataset_components.decoders import decode_data, build_flow_sample
from dataset_components.pipeline import sample_transform_pipeline
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf


def gather_shard_paths(data_dir, splits, rank=0):
    """Gather WebDataset shard paths for a split or list of splits.

    Args:
        data_dir: Base dataset directory for a domain.
        splits: Either a single split name (str) or a list/tuple of split names.
        rank: Process rank for logging.

    Returns:
        List of shard file paths, or None if nothing found (when not training).
    """
    def _gather_one(split: str):
        base = data_dir
        if not os.path.isabs(base):
            base = os.path.join(os.path.dirname(__file__), base)
        split_dir = os.path.join(base, split)
        paths = []
        if os.path.exists(split_dir):
            for fname in os.listdir(split_dir):
                if fname.endswith('.tar') and f'{split}' in fname:
                    file_path = os.path.join(split_dir, fname)
                    # Only include files >=1MB.
                    if os.path.getsize(file_path) >= 1048576:
                        paths.append(file_path)
        else:
            if rank == 0:
                print(f"[rank={rank}] Warning: split directory does not exist: {split_dir}")
        return paths

    # Single split
    if isinstance(splits, str):
        shard_paths = _gather_one(splits)
        if len(shard_paths) == 0:
            # For non-train splits, allow empty and return None to skip
            assert splits != 'train', "train dataset should always exist"
            if rank == 0:
                print(f"[rank={rank}] Warning: {splits} dataset not found, skipping")
            return None
        return shard_paths

    # Multiple splits
    assert isinstance(splits, (list, tuple)), f"splits must be str or list/tuple, got {type(splits)}"
    combined = []
    for split in splits:
        paths = _gather_one(str(split))
        if len(paths) == 0 and rank == 0:
            print(f"[rank={rank}] Warning: {split} dataset not found for {data_dir}, skipping this split")
        combined.extend(paths)

    if len(combined) == 0:
        # When training with multiple splits, require at least one to exist
        # but do not crash other modes if invoked accidentally.
        if rank == 0:
            print(f"[rank={rank}] Warning: no shards found for splits {splits} under {data_dir}")
        return None
    return combined


def _resolve_requested_splits(mode, args, override_splits=None):
    """Unify logic for selecting dataset splits for this loader.

    - If override_splits is provided, use it (string or list).
    - Else if mode == 'train', use the release train split list.
    - Else use the mode string ('train' | 'test').
    """
    if mode not in ("train", "test"):
        raise ValueError(f"Unsupported mode '{mode}' (expected 'train' or 'test').")
    if override_splits is not None:
        return override_splits
    if mode == 'train':
        return RELEASE_TRAIN_SPLITS
    return mode


def build_dataset(data_dir, domain, mode, args, rank=0, has_bimanual_robot=False, override_splits=None, force_resampled_eval=False):
    # -------------------------------------
    # set up webdataset
    # -------------------------------------
    # Support chaining multiple splits for training
    requested_splits = _resolve_requested_splits(mode, args, override_splits)
    shard_paths = gather_shard_paths(data_dir, requested_splits, rank)
    if shard_paths is None:
        return None

    # Initialize robot sampler once for this dataset (TorchRobotSampler on CPU).
    urdf_path = resolve_robot_urdf(domain)
    if "droid" in domain or "behavior" in domain:
        robot_sampler = TorchRobotSampler(
            urdf_path=urdf_path,
            gripper_only=RELEASE_GRIPPER_ONLY,
            device="cpu",
        )
    else:
        raise ValueError(f"Unsupported domain '{domain}' for robot sampler (expected droid or behavior).")

    # shuffle the shard paths
    random.Random(args.seed + rank).shuffle(shard_paths)
    if rank == 0:
        split_str = requested_splits if isinstance(requested_splits, str) else "+".join(requested_splits)
        split_label = f"{mode}({split_str})"
        print(f"[{domain}] {split_label} num_shards={len(shard_paths)} (data_dir={data_dir})")
    if mode != 'train' and not force_resampled_eval:
        # For eval, enforce deterministic shard order
        shardshuffle = False
        detshuffle = True
    else:
        if args.deterministic_data:
            shardshuffle = False
            detshuffle = True
        else:
            shardshuffle = True
            detshuffle = False
    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=shardshuffle,  # shard order control
        detshuffle=detshuffle,
        resampled=(((mode == 'train') and (not args.deterministic_train)) or (force_resampled_eval and mode != 'train')),
        nodesplitter=(wds.split_by_node if (mode != 'train' and not force_resampled_eval) else None),
        verbose=(rank == 0),
        seed=(args.seed + rank),
        handler=wds.warn_and_continue,
    )

    # -------------------------------------
    # decode from tar files, get T-step sequences, shuffle, and sample cameras
    # -------------------------------------
    use_eval_override = getattr(args, '_eval_override_active', False)
    force_single_arm = False
    if use_eval_override:
        force_single_arm_domains = getattr(args, 'eval_force_single_arm_domains', set())
        force_single_arm = (mode != 'train') and (domain in force_single_arm_domains)
    if mode == 'train':
        min_num_cameras = args.train_min_num_cameras
        max_num_cameras = args.train_max_num_cameras
    else:
        min_num_cameras = args.eval_min_num_cameras
        max_num_cameras = args.eval_max_num_cameras

    dataset = (
        dataset
        # .shuffle(size=32, initial=16, rng=random.Random(args.seed + rank))
        .map(partial(decode_data, domain=domain))  # decode each episode
        .map(partial(build_flow_sample, domain=domain, robot_sampler=robot_sampler, max_robot_points=args.max_robot_points,
                     deterministic=((mode != 'train') or args.deterministic_train), seed=(args.seed + rank), force_single_arm=force_single_arm), handler=wds.warn_and_continue)
        .map(partial(sample_cameras, min_num_cameras=min_num_cameras, max_num_cameras=max_num_cameras,
                     deterministic=((mode != 'train') or args.deterministic_train), seed=(args.seed + rank)), handler=wds.warn_and_continue)
        .map(canonicalize_gripper_keys_and_flags, handler=None)
    )

    # -------------------------------------
    # sample transformations
    # -------------------------------------
    dataset = sample_transform_pipeline(dataset, data_dir, domain, mode, args, has_bimanual_robot, rank=rank)
    return dataset


def build_dataloader(args, mode, rank=0, world_size=1, override_splits=None, force_resampled_eval=False):
    assert isinstance(args.data_dirs, list), f'expected data_dirs to be a list, got {args.data_dirs}'
    assert isinstance(args.domains, list), f'expected domains to be a list, got {args.domains}'
    assert len(args.data_dirs) == len(args.domains), f'expected data_dirs and domains to have one to one mapping, got {len(args.data_dirs)} and {len(args.domains)}'

    use_eval_override = getattr(args, '_eval_override_active', False)
    training_domains = getattr(args, 'train_domains_for_eval', args.domains) if use_eval_override else args.domains
    has_bimanual_robot = any('behavior' in domain for domain in training_domains)

    domain_dir_pairs = list(zip(args.data_dirs, args.domains))
    if use_eval_override:
        whitelist = getattr(args, 'eval_domain_whitelist', None)
        if whitelist and mode != 'train':
            whitelist_set = set(whitelist)
            domain_dir_pairs = [(data_dir, domain) for data_dir, domain in domain_dir_pairs if domain in whitelist_set]

    datasets = []
    num_shards = []
    total_samples = 0

    for data_dir, domain in domain_dir_pairs:
        dataset = build_dataset(data_dir, domain, mode, args, rank, has_bimanual_robot, override_splits=override_splits, force_resampled_eval=(force_resampled_eval and mode != 'train'))
        if dataset is not None:  # could be None if no shards are found for particular split for particular domain
            datasets.append(dataset)
            # Count shards for this dataset
            requested_splits = _resolve_requested_splits(mode, args, override_splits)
            shard_paths = gather_shard_paths(data_dir, requested_splits, rank)
            if shard_paths:
                num_shards.append(len(shard_paths))

            # Aggregate sample count from metadata
            if isinstance(requested_splits, (list, tuple)):
                sample_count = 0
                for split in requested_splits:
                    sample_count += aggregate_dataset_metadata(data_dir, split)
            else:
                sample_count = aggregate_dataset_metadata(data_dir, requested_splits)
            total_samples += sample_count

    if len(datasets) == 0:
        return None, {}
    assert total_samples > 0, f"[{mode}] Total samples must be greater than 0, got {total_samples}"
    min_num_shards = min(num_shards)
    freqs = [1.0 for _ in range(len(datasets))]  # equal frequency for all datasets
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        # Deterministic mixing for eval; random mix for train
        if mode != 'train' or args.deterministic_train:
            dataset = wds.RoundRobin(datasets, longest=False)
        else:
            dataset = wds.RandomMix(datasets, freqs)

    # Set the dataset length based on aggregated metadata
    # Calculate number of batches per rank such that all ranks together cover the full dataset
    total_batches_needed = int(np.ceil(total_samples / args.batch_size))
    batches_per_rank = int(np.ceil(total_batches_needed / world_size))
    assert batches_per_rank > 0, f"[{mode}] Batches per rank must be greater than 0, got {batches_per_rank} for total_batches_needed={total_batches_needed} and world_size={world_size}"

    if rank == 0:
        print(f"[{mode}] Total samples across all datasets: {total_samples}")
        print(f"[{mode}] Total batches needed: {total_batches_needed}")
        print(f"[{mode}] Batches per rank (world_size={world_size}): {batches_per_rank}")
        print(f"[{mode}] Epoch length set to: {batches_per_rank}")

    # -------------------------------------_remove_filtered_gripper_data
    # dataloader
    # -------------------------------------
    num_workers = min(args.num_workers, min_num_shards)
    if mode == 'train' and args.deterministic_train:
        num_workers = 0
    if mode != 'train':
        num_workers = min(num_workers, args.eval_num_workers)  # only use max 5 workers for eval
    dataloader = wds.WebLoader(
        dataset,  # we are recommended to use dataset.batched() but it's not working with random mix
        batch_size=args.batch_size,
        collate_fn=partial(custom_collate_fn, args=args),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # requires num_workers > 0
        # persistent_workers=False,
    ).with_length(batches_per_rank)
    info = {
        'total_samples': int(total_samples),
        'total_batches_needed': int(total_batches_needed),
        'batches_per_rank': int(batches_per_rank),
        'world_size': int(world_size),
    }
    return dataloader, info


def aggregate_dataset_metadata(data_dir: str, split: str) -> int:
    """
    Aggregate metadata from all rank files to get total sample count for a dataset split.

    Args:
        data_dir: Path to the dataset directory
        split: Dataset split ('train' or 'test')

    Returns:
        Total number of samples across all ranks for the given split
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), data_dir)

    # Find all metadata files from different ranks
    metadata_pattern = os.path.join(data_dir, "metadata_rank*.json")
    metadata_files = glob.glob(metadata_pattern)

    assert len(metadata_files) > 0, f"No metadata files found at {metadata_pattern}"

    total_count = 0
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert split in metadata, f"[{split}] Split not found in metadata file {metadata_file}"
        total_count += metadata[split]["processed_count"]

    return total_count
