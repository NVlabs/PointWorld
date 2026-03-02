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
import argparse
import io
import json
import math
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import webdataset as wds
from tqdm import tqdm

from shared.data_contract import (
    BEHAVIOR_CLIP_ATTRIBUTE_KEYS,
    EXPECTED_CAMERA_PAYLOAD_SHAPES,
    IMAGE_KEYS,
    get_wds_data_keys,
    validate_domain,
)

MANIFEST_SCHEMA_VERSION = "wds_manifest.v1"
QUANTIZED_NORMALS_DTYPE = np.int8
QUANTIZED_NORMALS_MIN = -127
QUANTIZED_NORMALS_MAX = 127


def create_output_dirs(output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir


def _extract_behavior_uuid(h5_path: str, input_dir: str) -> str:
    """Derive BEHAVIOR UUID using strict migrated-release convention."""
    del input_dir
    normalized = h5_path.replace("\\", "/")
    match = re.search(r"(task-\d+)/(episode_\d+)\.(h5|hdf5)$", normalized)
    if not match:
        raise AssertionError(
            "BEHAVIOR path does not match expected convention "
            "'.../task-XXXX/episode_YYYYYYYY.(h5|hdf5)': "
            f"{h5_path}"
        )
    return f"{match.group(1)}_{match.group(2)}"


def _uuid_from_h5_path(domain: str, h5_path: str, input_dir: str) -> str:
    if domain == "behavior":
        return _extract_behavior_uuid(h5_path, input_dir)
    return os.path.splitext(os.path.basename(h5_path))[0]


def _clip_id_from_h5_and_clip_key(domain: str, h5_path: str, clip_key: str, input_dir: str) -> str:
    return f"{_uuid_from_h5_path(domain, h5_path, input_dir)}-{clip_key}"


def _serialize_numpy(value: np.ndarray) -> bytes:
    arr_bytes = io.BytesIO()
    np.save(arr_bytes, value)
    return arr_bytes.getvalue()


def _serialize_pickle(value: Any) -> bytes:
    raw = io.BytesIO()
    pickle.dump(value, raw)
    return raw.getvalue()


def _is_normals_modality(modality: str) -> bool:
    return modality in {"scene_normals", "local_scene_normals"}


def _assert_quantized_normals_array(value: np.ndarray, context: str) -> None:
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"Normals payload must be np.ndarray at {context}, got {type(value)}")
    if value.dtype != QUANTIZED_NORMALS_DTYPE:
        raise AssertionError(
            f"Normals payload must be {QUANTIZED_NORMALS_DTYPE} at {context}, got {value.dtype}"
        )
    if value.size > 0:
        min_val = int(value.min())
        max_val = int(value.max())
        if min_val < QUANTIZED_NORMALS_MIN or max_val > QUANTIZED_NORMALS_MAX:
            raise AssertionError(
                f"Normals payload values out of range at {context}: "
                f"[{min_val}, {max_val}] not in [{QUANTIZED_NORMALS_MIN}, {QUANTIZED_NORMALS_MAX}]"
            )


def _assert_camera_payload_contract(dataset: h5py.Dataset, modality: str, h5_path: str, clip_key: str, camera_key: str) -> None:
    if modality == "initial_rgb":
        if dataset.shape != (1,):
            raise AssertionError(
                f"Invalid initial_rgb shape in {h5_path}/{clip_key}/{camera_key}: "
                f"expected (1,), got {dataset.shape}"
            )
        return

    expected_shape = EXPECTED_CAMERA_PAYLOAD_SHAPES[modality]
    if tuple(dataset.shape) != expected_shape:
        raise AssertionError(
            f"Invalid {modality} shape in {h5_path}/{clip_key}/{camera_key}: "
            f"expected {expected_shape}, got {tuple(dataset.shape)}"
        )


def process_single_clip(
    h5_path: str,
    clip_key: str,
    data_keys: List[str],
    shard_writer: wds.ShardWriter,
    domain: str,
    input_dir: str,
) -> None:
    validate_domain(domain)
    with h5py.File(h5_path, "r") as f:
        clip_data = f[clip_key]
        uuid = _uuid_from_h5_path(domain, h5_path, input_dir)

        sample: Dict[str, bytes] = {"__key__": f"{uuid}-{clip_key}"}

        camera_keys = [k for k in clip_data.keys() if k.startswith("camera_")]
        assert len(camera_keys) > 0, f"No camera_* groups in {h5_path}/{clip_key}"

        for modality in data_keys:
            if "scene" in modality:
                for camera_key in camera_keys:
                    camera_data = clip_data[camera_key]
                    assert modality in camera_data, f"Missing {modality} in {h5_path}/{clip_key}/{camera_key}"
                    payload = camera_data[modality]
                    if isinstance(payload, h5py.Dataset):
                        value = payload[:]
                        if _is_normals_modality(modality):
                            _assert_quantized_normals_array(
                                value,
                                context=f"{h5_path}/{clip_key}/{camera_key}/{modality}",
                            )
                        sample[f"{camera_key}_{modality}.npy"] = _serialize_numpy(value)
                    elif isinstance(payload, h5py.Group):
                        group_payload = {}
                        for k, v in payload.items():
                            value = v[:]
                            if _is_normals_modality(modality):
                                _assert_quantized_normals_array(
                                    value,
                                    context=f"{h5_path}/{clip_key}/{camera_key}/{modality}/{k}",
                                )
                            group_payload[k] = value
                        sample[f"{camera_key}_{modality}.pyd"] = _serialize_pickle(group_payload)
                    else:
                        raise TypeError(
                            f"Unsupported scene modality container {type(payload)} in {h5_path}/{clip_key}/{camera_key}/{modality}"
                        )
            elif modality in IMAGE_KEYS:
                for camera_key in camera_keys:
                    camera_data = clip_data[camera_key]
                    assert modality in camera_data, f"Missing {modality} in {h5_path}/{clip_key}/{camera_key}"
                    dataset = camera_data[modality]
                    assert isinstance(dataset, h5py.Dataset), (
                        f"Expected dataset for {modality} in {h5_path}/{clip_key}/{camera_key}, got {type(dataset)}"
                    )
                    _assert_camera_payload_contract(dataset, modality, h5_path, clip_key, camera_key)
                    if modality == "initial_rgb":
                        sample[f"{camera_key}_{modality}.jpg"] = dataset[0].tobytes()
                    elif modality == "initial_depth":
                        sample[f"{camera_key}_{modality}.npy"] = _serialize_numpy(dataset[()])
                    else:
                        sample[f"{camera_key}_{modality}.npy"] = _serialize_numpy(dataset[:])
            else:
                assert modality in clip_data, f"Missing {modality} in {h5_path}/{clip_key}"
                payload = clip_data[modality]
                if isinstance(payload, h5py.Dataset):
                    data = payload[:]
                    if data.dtype == np.object_ or (hasattr(data.dtype, "kind") and data.dtype.kind in {"U", "S", "O"}):
                        sample[f"{modality}.pyd"] = _serialize_pickle(data)
                    else:
                        sample[f"{modality}.npy"] = _serialize_numpy(data)
                elif isinstance(payload, h5py.Group):
                    group_payload = {k: v[:] for k, v in payload.items()}
                    sample[f"{modality}.pyd"] = _serialize_pickle(group_payload)
                else:
                    raise TypeError(f"Unsupported modality container {type(payload)} in {h5_path}/{clip_key}/{modality}")

        if domain == "behavior":
            clip_attributes = {}
            for attr_name in BEHAVIOR_CLIP_ATTRIBUTE_KEYS:
                assert attr_name in clip_data.attrs, f"Missing behavior clip attribute '{attr_name}' in {h5_path}/{clip_key}"
                attr_value = clip_data.attrs[attr_name]
                if hasattr(attr_value, "item"):
                    attr_value = attr_value.item()
                clip_attributes[attr_name] = attr_value
            sample["clip_attributes.pyd"] = _serialize_pickle(clip_attributes)

        has_scene_payload = any(
            ("scene_flows" in key) or ("local_scene_points" in key)
            for key in sample.keys()
        )
        assert has_scene_payload, f"No scene payload found in {h5_path}/{clip_key}"

        shard_writer.write(sample)


def load_integrity_results(integrity_check_file: str) -> List[Tuple[str, str]]:
    print(f"Loading integrity check results from {integrity_check_file}")
    with open(integrity_check_file, "r") as f:
        results = json.load(f)
    assert "valid_clips" in results, f"Missing 'valid_clips' in integrity file: {integrity_check_file}"
    valid_clips = [tuple(clip) for clip in results["valid_clips"]]
    print(f"Loaded {len(valid_clips)} valid clips")
    return valid_clips


def filter_clips_by_uuid_keywords(
    valid_clips: List[Tuple[str, str]],
    only_uuid_keywords: List[str],
    *,
    domain: str,
    input_dir: str,
) -> List[Tuple[str, str]]:
    if not only_uuid_keywords:
        return valid_clips
    filtered: List[Tuple[str, str]] = []
    for h5_path, clip_key in valid_clips:
        uuid_str = _uuid_from_h5_path(domain, h5_path, input_dir)
        if any(keyword in uuid_str for keyword in only_uuid_keywords):
            filtered.append((h5_path, clip_key))
    return filtered


def filter_tasks(valid_clips: List[Tuple[str, str]], tasks: List[str]) -> List[Tuple[str, str]]:
    if not tasks:
        return valid_clips
    lowercase_tasks = [task.lower() for task in tasks]
    filtered_clips = []
    for h5_path, clip_key in valid_clips:
        if any(task in h5_path.lower() for task in lowercase_tasks):
            filtered_clips.append((h5_path, clip_key))
    return filtered_clips


def _load_manifest(manifest_file: str, domain: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version: {schema_version}. "
            f"Expected {MANIFEST_SCHEMA_VERSION}."
        )

    manifest_domain = manifest.get("domain")
    if manifest_domain != domain:
        raise ValueError(
            f"Manifest domain mismatch: manifest={manifest_domain}, requested={domain}."
        )

    if "test_clip_keys" not in manifest:
        raise ValueError("Manifest missing required field: test_clip_keys")
    if not isinstance(manifest["test_clip_keys"], list):
        raise TypeError("Manifest field test_clip_keys must be a list of clip IDs.")

    include_clip_keys = manifest.get("include_clip_keys")
    if include_clip_keys is not None and not isinstance(include_clip_keys, list):
        raise TypeError("Manifest field include_clip_keys must be a list when provided.")

    return manifest


def _index_clips_by_id(
    valid_clips: List[Tuple[str, str]],
    *,
    domain: str,
    input_dir: str,
) -> Dict[str, Tuple[str, str]]:
    clip_map: Dict[str, Tuple[str, str]] = {}
    for h5_path, clip_key in valid_clips:
        clip_id = _clip_id_from_h5_and_clip_key(domain, h5_path, clip_key, input_dir)
        if clip_id in clip_map:
            prev_h5_path, prev_clip_key = clip_map[clip_id]
            raise ValueError(
                "Duplicate canonical clip ID encountered. "
                f"clip_id={clip_id}, first=({prev_h5_path}, {prev_clip_key}), "
                f"second=({h5_path}, {clip_key})"
            )
        clip_map[clip_id] = (h5_path, clip_key)
    return clip_map


def _sample_subset(
    clip_ids: List[str],
    *,
    max_clips: int,
    seed: int,
) -> List[str]:
    if max_clips < 0 or len(clip_ids) <= max_clips:
        return list(clip_ids)
    rng = random.Random(seed)
    sampled = rng.sample(clip_ids, max_clips)
    sampled.sort()
    return sampled


def _assign_splits_from_manifest(
    *,
    all_clip_ids: List[str],
    manifest: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    local_clip_set = set(all_clip_ids)

    include_clip_keys = manifest.get("include_clip_keys")
    if include_clip_keys is None:
        included_set = set(local_clip_set)
    else:
        manifest_include_set = set(include_clip_keys)
        missing_include = sorted(manifest_include_set - local_clip_set)
        if missing_include:
            preview = ", ".join(missing_include[:5])
            raise ValueError(
                "Manifest include_clip_keys has keys missing from local integrity results. "
                f"count={len(missing_include)} example={preview}"
            )
        included_set = manifest_include_set & local_clip_set

    manifest_test_set = set(manifest["test_clip_keys"])
    missing_test = sorted(manifest_test_set - included_set)
    if missing_test:
        preview = ", ".join(missing_test[:5])
        raise ValueError(
            "Manifest test_clip_keys has keys missing from selected include/local universe. "
            f"count={len(missing_test)} example={preview}"
        )

    effective_test_set = manifest_test_set & included_set
    if len(included_set) > 0 and len(effective_test_set) == 0:
        raise ValueError("Manifest produced an empty test split after local intersection.")

    train_clip_ids = [clip_id for clip_id in all_clip_ids if clip_id in included_set and clip_id not in effective_test_set]
    test_clip_ids = [clip_id for clip_id in all_clip_ids if clip_id in effective_test_set]
    return train_clip_ids, test_clip_ids


def _slice_for_worker(clip_ids: List[str], *, rank: int, world_size: int) -> List[str]:
    return clip_ids[rank::world_size]


def preprocess_data(
    domain: str,
    input_dir: str,
    output_dir: str,
    data_keys: List[str],
    maxsize: float = 5e9,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    only_uuid_keywords: Optional[List[str]] = None,
    integrity_check_file: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    max_clips: int = -1,
    train_frac: float = -1.0,
    manifest_file: Optional[str] = None,
) -> str:
    train_dir, test_dir = create_output_dirs(output_dir)

    if integrity_check_file is None:
        integrity_check_file = os.path.join(input_dir, "integrity_check.json")
    assert os.path.exists(integrity_check_file), f"Integrity check file not found: {integrity_check_file}"

    valid_clips = load_integrity_results(integrity_check_file)
    if len(valid_clips) == 0:
        raise AssertionError("Integrity file contains zero valid clips.")

    for h5_path, _ in valid_clips:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"Integrity file points to missing H5 path: {h5_path}. "
                "Regenerate integrity_check.json from this local dataset path."
            )

    valid_clips = sorted(valid_clips)

    only_uuid_keywords = only_uuid_keywords or []
    if only_uuid_keywords:
        print(f"Filtering to UUID keywords: {', '.join(only_uuid_keywords)}")
        valid_clips = filter_clips_by_uuid_keywords(
            valid_clips,
            only_uuid_keywords,
            domain=domain,
            input_dir=input_dir,
        )
        print(f"After UUID keyword filter: {len(valid_clips)} clips")

    tasks = tasks or []
    if tasks:
        print(f"Filtering to tasks: {', '.join(tasks)}")
        valid_clips = filter_tasks(valid_clips, tasks)
        print(f"After task filter: {len(valid_clips)} clips")

    if len(valid_clips) == 0:
        raise AssertionError(
            "No valid clips remain after filtering. "
            "Regenerate integrity_check.json and verify filter keywords/tasks."
        )

    clip_map = _index_clips_by_id(valid_clips, domain=domain, input_dir=input_dir)
    all_clip_ids = sorted(clip_map.keys())

    if max_clips > 0:
        before = len(all_clip_ids)
        all_clip_ids = _sample_subset(all_clip_ids, max_clips=max_clips, seed=seed)
        print(f"Applied max_clips={max_clips}: {before} -> {len(all_clip_ids)}")

    if len(all_clip_ids) == 0:
        raise AssertionError("No clips remain after all pre-split filters.")

    if manifest_file is None:
        raise ValueError(
            "Missing required --manifest. "
            "Create one with make_wds_manifest.py (deterministic seed split), "
            "or pass an existing release manifest path explicitly."
        )

    resolved_manifest_file = os.path.abspath(manifest_file)
    manifest = _load_manifest(resolved_manifest_file, domain=domain)
    train_clip_ids, test_clip_ids = _assign_splits_from_manifest(
        all_clip_ids=all_clip_ids,
        manifest=manifest,
    )
    print(
        f"Using manifest split: train={len(train_clip_ids)}, test={len(test_clip_ids)}, "
        f"manifest={resolved_manifest_file}"
    )

    original_train_count = len(train_clip_ids)
    if train_frac > 0:
        assert 0 < train_frac <= 1, "train_frac must be within (0, 1]"
        if original_train_count > 0:
            target_train = int(math.ceil(original_train_count * train_frac))
            target_train = max(1, min(target_train, original_train_count))
            if target_train < original_train_count:
                train_clip_ids = train_clip_ids[:target_train]
                print(
                    f"Applied train_frac={train_frac}: "
                    f"{original_train_count} -> {len(train_clip_ids)}"
                )

    if len(train_clip_ids) == 0 and len(all_clip_ids) > 0:
        raise AssertionError(
            "train split is empty after split/downsampling; "
            "increase subset size or adjust manifest/train_frac."
        )
    if 0 < len(train_clip_ids) < 5:
        print(
            f"[Worker {rank}] WARNING: tiny train split ({len(train_clip_ids)} clips). "
            "This is suitable only for smoke tests."
        )

    if len(all_clip_ids) < world_size:
        print(
            f"[Worker {rank}] WARNING: only {len(all_clip_ids)} clips for world_size={world_size}; "
            "some workers may receive zero clips."
        )

    worker_train_clip_ids = _slice_for_worker(train_clip_ids, rank=rank, world_size=world_size)
    worker_test_clip_ids = _slice_for_worker(test_clip_ids, rank=rank, world_size=world_size)
    if len(worker_train_clip_ids) == 0 and len(worker_test_clip_ids) == 0:
        print(
            f"[Worker {rank}] WARNING: this worker received zero clips. "
            "Increase subset size or reduce world_size."
        )

    train_clips = [clip_map[clip_id] for clip_id in worker_train_clip_ids]
    test_clips = [clip_map[clip_id] for clip_id in worker_test_clip_ids]

    print(
        f"[Worker {rank}] Assigned clips -> "
        f"train={len(train_clips)}/{len(train_clip_ids)}, "
        f"test={len(test_clips)}/{len(test_clip_ids)}"
    )

    splits: List[Tuple[str, List[Tuple[str, str]], str]] = [
        ("train", train_clips, train_dir),
        ("test", test_clips, test_dir),
    ]

    metadata: Dict[str, Any] = {
        "train": {"processed_count": 0, "global_selected_count": len(train_clip_ids), "worker_assigned_count": len(train_clips)},
        "test": {"processed_count": 0, "global_selected_count": len(test_clip_ids), "worker_assigned_count": len(test_clips)},
    }
    metadata["config"] = {
        "seed": seed,
        "manifest_file": resolved_manifest_file,
        "train_frac": train_frac,
        "max_clips": max_clips,
    }
    source_paths: Dict[str, set] = {
        "train": set(),
        "test": set(),
    }

    for split_name, clip_pairs, split_outdir in splits:
        if len(clip_pairs) == 0:
            continue
        shard_pattern = os.path.join(split_outdir, f"{split_name}-rank{rank:02d}-%06d.tar")
        shard_writer = wds.ShardWriter(shard_pattern, maxsize=maxsize, encoder=False)
        processed_count = 0
        try:
            for h5_path, clip_key in tqdm(clip_pairs, desc=f"Worker {rank} - {split_name}"):
                process_single_clip(
                    h5_path=h5_path,
                    clip_key=clip_key,
                    data_keys=data_keys,
                    shard_writer=shard_writer,
                    domain=domain,
                    input_dir=input_dir,
                )
                processed_count += 1
                source_paths[split_name].add(h5_path)
        finally:
            shard_writer.close()

        metadata[split_name]["processed_count"] = processed_count
        print(f"[Worker {rank}] {split_name}: processed={processed_count}")

    metadata_file = os.path.join(output_dir, f"metadata_rank{rank}.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    for split_name in ["train", "test"]:
        if len(source_paths[split_name]) > 0:
            path_file = os.path.join(output_dir, f"{split_name}_source_paths_rank{rank}.txt")
            with open(path_file, "w") as f:
                for path in sorted(source_paths[split_name]):
                    f.write(f"{path}\n")

    return output_dir


def _parse_csv_arg(value: str) -> List[str]:
    if value.strip() == "":
        return []
    if value.lower() in {"none", "all"}:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert H5 files to WebDataset shards.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source .h5 files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards.")
    parser.add_argument("--domain", type=str, required=True, choices=["droid", "behavior"])
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max shard size in bytes.")
    parser.add_argument("--rank", type=int, default=0, help="Job rank.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--world_size", type=int, default=1, help="Total worker count across jobs.")
    parser.add_argument("--integrity_check_file", type=str, default=None, help="Path to integrity_check.json.")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task filters.")
    parser.add_argument("--max_clips", type=int, default=-1, help="Max number of clips to process (-1 for all).")
    parser.add_argument(
        "--train_frac",
        type=float,
        default=-1.0,
        help="Fraction of post-split train clips to keep in (0,1]; -1 keeps all.",
    )
    parser.add_argument(
        "--only_uuid_keywords",
        type=str,
        default="",
        help="Comma-separated UUID substring filters applied before split.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to split manifest JSON. "
            "Generate with make_wds_manifest.py, or pass an existing release manifest explicitly."
        ),
    )
    args = parser.parse_args()
    validate_domain(args.domain)

    if args.train_frac != -1.0 and not (0 < args.train_frac <= 1):
        raise ValueError("train_frac must be within (0, 1] or -1")

    tasks = _parse_csv_arg(args.tasks)
    only_uuid_keywords = _parse_csv_arg(args.only_uuid_keywords)

    data_keys = get_wds_data_keys(args.domain)

    print(f"Starting conversion with rank={args.rank}, world_size={args.world_size}")
    preprocess_data(
        rank=args.rank,
        world_size=args.world_size,
        domain=args.domain,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        data_keys=data_keys,
        maxsize=args.maxsize,
        seed=args.seed,
        only_uuid_keywords=only_uuid_keywords,
        integrity_check_file=args.integrity_check_file,
        tasks=tasks,
        max_clips=args.max_clips,
        train_frac=args.train_frac,
        manifest_file=args.manifest,
    )


if __name__ == "__main__":
    main()
