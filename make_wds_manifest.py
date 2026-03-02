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
import json
import os
import random
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from shared.data_contract import validate_domain

MANIFEST_SCHEMA_VERSION = "wds_manifest.v1"


def iter_valid_clips(integrity_check_file: str) -> Iterator[Tuple[str, str]]:
    """Stream valid_clips entries from integrity_check.json without loading the whole file."""
    decoder = json.JSONDecoder()
    with open(integrity_check_file, "r", encoding="utf-8") as f:
        buffer = ""
        in_valid_clips_array = False
        found_valid_clips_key = False
        array_finished = False
        while not array_finished:
            chunk = f.read(1 << 20)
            if not chunk and not buffer:
                break
            buffer += chunk
            idx = 0

            if not in_valid_clips_array:
                key_pos = buffer.find('"valid_clips"')
                if key_pos == -1:
                    if len(buffer) > 128:
                        buffer = buffer[-128:]
                    continue
                found_valid_clips_key = True
                open_bracket = buffer.find("[", key_pos)
                if open_bracket == -1:
                    continue
                idx = open_bracket + 1
                in_valid_clips_array = True

            while True:
                n = len(buffer)
                while idx < n and buffer[idx] in " \n\r\t,":
                    idx += 1
                if idx >= n:
                    buffer = ""
                    break

                if buffer[idx] == "]":
                    array_finished = True
                    buffer = ""
                    break

                try:
                    item, end = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    buffer = buffer[idx:]
                    break

                idx = end
                if not isinstance(item, list) or len(item) < 2:
                    raise ValueError(
                        "Unexpected entry inside valid_clips; expected [h5_path, clip_key]. "
                        f"Got: {type(item)}"
                    )
                h5_path, clip_key = item[0], item[1]
                if not isinstance(h5_path, str) or not isinstance(clip_key, str):
                    raise ValueError(
                        "Unexpected valid_clips item types; expected strings for path and clip key."
                    )
                yield h5_path, clip_key

    if not found_valid_clips_key:
        raise ValueError(f"Missing 'valid_clips' in integrity file: {integrity_check_file}")


def _extract_behavior_uuid(h5_path: str, input_dir: str) -> str:
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


def _assign_splits_from_seed(
    all_clip_ids: List[str],
    *,
    seed: int,
    test_percentage: float,
) -> Tuple[List[str], List[str]]:
    shuffled = list(all_clip_ids)
    random.Random(seed).shuffle(shuffled)
    num_test = int(len(shuffled) * test_percentage)
    if num_test == 0:
        test_clip_ids: List[str] = []
        train_clip_ids = shuffled
    else:
        test_clip_ids = sorted(shuffled[-num_test:])
        train_clip_ids = sorted(shuffled[:-num_test])
    return train_clip_ids, test_clip_ids


def _parse_csv_arg(value: str) -> List[str]:
    if value.strip() == "":
        return []
    if value.lower() in {"none", "all"}:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic WDS split manifest from integrity_check.json.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory used for generation/integrity check.")
    parser.add_argument("--domain", type=str, required=True, choices=["droid", "behavior"])
    parser.add_argument("--output_manifest", type=str, required=True, help="Output manifest JSON path.")
    parser.add_argument("--integrity_check_file", type=str, default=None, help="Path to integrity_check.json.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic train/test split.")
    parser.add_argument("--test_percentage", type=float, default=0.1, help="Fraction of selected clips in test split.")
    parser.add_argument("--max_clips", type=int, default=-1, help="Max number of clips to include (-1 for all).")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated task filters applied before deterministic split.",
    )
    parser.add_argument(
        "--only_uuid_keywords",
        type=str,
        default="",
        help="Comma-separated UUID substring filters applied before deterministic split.",
    )
    parser.add_argument(
        "--include_clip_keys",
        action="store_true",
        help="Include include_clip_keys in the manifest (larger file, stricter provenance contract).",
    )
    args = parser.parse_args()
    validate_domain(args.domain)

    if not (0 <= args.test_percentage <= 1):
        raise ValueError(f"test_percentage must be in [0, 1], got {args.test_percentage}")

    integrity_check_file = args.integrity_check_file or os.path.join(args.input_dir, "integrity_check.json")
    if not os.path.exists(integrity_check_file):
        raise FileNotFoundError(f"Integrity check file not found: {integrity_check_file}")

    only_uuid_keywords = _parse_csv_arg(args.only_uuid_keywords)
    tasks = _parse_csv_arg(args.tasks)
    task_filters = [task.lower() for task in tasks]

    clip_ids: set = set()
    total_rows = 0
    selected_rows = 0
    for h5_path, clip_key in iter_valid_clips(integrity_check_file):
        total_rows += 1

        if task_filters and not any(task in h5_path.lower() for task in task_filters):
            continue

        uuid = _uuid_from_h5_path(args.domain, h5_path, args.input_dir)
        if only_uuid_keywords and not any(keyword in uuid for keyword in only_uuid_keywords):
            continue

        clip_ids.add(f"{uuid}-{clip_key}")
        selected_rows += 1

    if total_rows == 0:
        raise AssertionError("Integrity file contains zero valid clips.")
    if len(clip_ids) == 0:
        raise AssertionError("No valid clips remain after filtering.")

    all_clip_ids = sorted(clip_ids)

    if args.max_clips > 0:
        all_clip_ids = _sample_subset(all_clip_ids, max_clips=args.max_clips, seed=args.seed)

    if len(all_clip_ids) == 0:
        raise AssertionError("No clips selected after max_clips/downsampling.")

    _, test_clip_ids = _assign_splits_from_seed(all_clip_ids, seed=args.seed, test_percentage=args.test_percentage)
    if args.test_percentage > 0 and len(test_clip_ids) == 0:
        raise AssertionError(
            "test split is empty; increase selected clip count or test_percentage, "
            "or disable test split with --test_percentage 0."
        )

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "domain": args.domain,
        "seed": args.seed,
        "test_percentage": args.test_percentage,
        "filters": {
            "tasks": tasks,
            "only_uuid_keywords": only_uuid_keywords,
            "max_clips": args.max_clips,
        },
        "stats": {
            "num_selected_total": len(all_clip_ids),
            "num_selected_test": len(test_clip_ids),
        },
        "test_clip_keys": test_clip_ids,
    }
    if args.include_clip_keys:
        manifest["include_clip_keys"] = all_clip_ids

    output_manifest = os.path.abspath(args.output_manifest)
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        "Wrote manifest: "
        f"path={output_manifest}, domain={args.domain}, "
        f"integrity_rows={total_rows}, filtered_rows={selected_rows}, "
        f"selected_total={len(all_clip_ids)}, selected_test={len(test_clip_ids)}"
    )


if __name__ == "__main__":
    main()
