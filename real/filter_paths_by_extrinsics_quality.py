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
"""Filter scene paths by extrinsics optimization quality.

This helper reads a scene list and keeps only scenes with:
  - no `error_info` in cameras/<uuid>_cameras.json
  - optimization_summary.final_loss < threshold

Output is another text file that can be reused by:
  - real/compute_2d_flows.py --input
  - real/convert_2d_flows_to_3d.py --input
"""

import argparse
import json
import os

from real.droid_utils import get_uuid
from real.gcs_utils import enforce_gcs_cache_policy


def _read_scene_paths(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input scene list not found: {path}")
    with open(path, "r") as f:
        scene_paths = [line.strip() for line in f if line.strip()]
    if len(scene_paths) == 0:
        raise ValueError(f"Input scene list is empty: {path}")
    return scene_paths


def _write_scene_paths(path: str, scene_paths: list[str]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        for scene_path in scene_paths:
            f.write(f"{scene_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter scene paths using extrinsics optimization final_loss threshold."
    )
    parser.add_argument("--input", required=True, help="Input scene list text file")
    parser.add_argument("--output", required=True, help="Output filtered scene list text file")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory containing cameras/<uuid>_cameras.json from compute_extrinsics",
    )
    parser.add_argument(
        "--max_final_loss",
        type=float,
        default=0.10,
        help="Keep scene only if optimization_summary.final_loss < this value",
    )
    parser.add_argument(
        "--allow_gcs_streaming",
        action="store_true",
        help=(
            "Bypass GCS cache enforcement and stream directly from gs:// inputs for this run. "
            "Not recommended for repeated multi-stage processing."
        ),
    )
    args = parser.parse_args()

    if args.max_final_loss <= 0:
        raise ValueError(f"--max_final_loss must be > 0, got {args.max_final_loss}")

    scene_paths = _read_scene_paths(args.input)
    enforce_gcs_cache_policy(
        scene_paths,
        stage_name="filter_paths_by_extrinsics_quality",
        require_cache=True,
        allow_streaming=args.allow_gcs_streaming,
    )

    kept_paths: list[str] = []
    rejected_count = 0
    errors: list[str] = []

    for scene_path in scene_paths:
        try:
            uuid = get_uuid(scene_path)
        except Exception as exc:
            errors.append(f"{scene_path}: failed to extract uuid ({exc})")
            continue

        cameras_json_path = os.path.join(args.output_dir, "cameras", f"{uuid}_cameras.json")
        if not os.path.exists(cameras_json_path):
            errors.append(f"{scene_path} ({uuid}): missing cameras JSON: {cameras_json_path}")
            continue

        try:
            with open(cameras_json_path, "r") as f:
                camera_data = json.load(f)
        except Exception as exc:
            errors.append(f"{scene_path} ({uuid}): failed to read JSON ({exc})")
            continue

        if "error_info" in camera_data:
            rejected_count += 1
            continue

        optimization_summary = camera_data.get("optimization_summary")
        if not isinstance(optimization_summary, dict):
            errors.append(f"{scene_path} ({uuid}): missing optimization_summary in {cameras_json_path}")
            continue
        if "final_loss" not in optimization_summary:
            errors.append(f"{scene_path} ({uuid}): missing optimization_summary.final_loss")
            continue

        try:
            final_loss = float(optimization_summary["final_loss"])
        except Exception as exc:
            errors.append(
                f"{scene_path} ({uuid}): invalid optimization_summary.final_loss="
                f"{optimization_summary['final_loss']!r} ({exc})"
            )
            continue

        if final_loss < args.max_final_loss:
            kept_paths.append(scene_path)
        else:
            rejected_count += 1

    if errors:
        max_show = 20
        shown = errors[:max_show]
        more = len(errors) - len(shown)
        msg = "\n".join(shown)
        if more > 0:
            msg += f"\n... and {more} more error(s)"
        raise RuntimeError(
            f"Failed to build filtered scene list due to {len(errors)} error(s):\n{msg}"
        )

    _write_scene_paths(args.output, kept_paths)
    print(
        "Filtered scene list written to "
        f"{args.output}: kept={len(kept_paths)}, rejected={rejected_count}, "
        f"total={len(scene_paths)}, threshold(final_loss < {args.max_final_loss})"
    )


if __name__ == "__main__":
    main()
