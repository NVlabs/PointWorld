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

from dataclasses import dataclass
import json
import os
import numpy as np
import torch

CANONICAL_STATS_FILENAME = "norm_stats.json"
REQUIRED_TOP_LEVEL_KEYS = ("statistics", "per_timestep_statistics")


def _load_canonical_stats_json(stats_path: str) -> dict:
    json_files = sorted([f for f in os.listdir(stats_path) if f.endswith(".json")])
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in stats folder: {stats_path}")
    if CANONICAL_STATS_FILENAME not in json_files:
        raise FileNotFoundError(
            f"Expected canonical stats file '{CANONICAL_STATS_FILENAME}' in {stats_path}; "
            f"found: {json_files}"
        )

    extra_files = [f for f in json_files if f != CANONICAL_STATS_FILENAME]
    if extra_files:
        raise RuntimeError(
            f"Expected only '{CANONICAL_STATS_FILENAME}' in {stats_path}, "
            f"but found extra JSON files: {extra_files}"
        )

    json_path = os.path.join(stats_path, CANONICAL_STATS_FILENAME)
    with open(json_path, "r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Canonical stats file must be a JSON object: {json_path}")
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise KeyError(f"{key} missing in {json_path}")
    return data


def load_stats_from_json_folder(stats_path: str, domains: list[str]) -> tuple[dict, dict]:
    if not stats_path:
        raise ValueError("norm_stats_path must be provided when normalization is enabled")
    if not os.path.isdir(stats_path):
        raise FileNotFoundError(f"Stats path not found or invalid: {stats_path}")

    data = _load_canonical_stats_json(stats_path)
    print(f"Loading canonical statistics from {os.path.join(stats_path, CANONICAL_STATS_FILENAME)}")

    stats_dict = data["statistics"]
    per_step_dict = data["per_timestep_statistics"]
    if not isinstance(stats_dict, dict):
        raise TypeError(f"statistics must be a dict in {CANONICAL_STATS_FILENAME}")
    if not isinstance(per_step_dict, dict):
        raise TypeError(f"per_timestep_statistics must be a dict in {CANONICAL_STATS_FILENAME}")

    for domain in domains:
        if domain not in stats_dict:
            raise KeyError(f"Domain {domain} missing in statistics")
        if domain not in per_step_dict:
            raise KeyError(f"Domain {domain} missing in per_timestep_statistics")

    return stats_dict, per_step_dict


@dataclass
class NormStatsBundle:
    domains: list[str]
    domain_to_index: dict[str, int]
    per_step_mean: torch.Tensor
    per_step_var: torch.Tensor
    robot_mean: torch.Tensor
    robot_var: torch.Tensor
    scene_mean: torch.Tensor
    scene_var: torch.Tensor


def _collect_stat(stats_dict, domain, field, stat):
    dom_stats = stats_dict.get(domain)
    if not isinstance(dom_stats, dict) or field not in dom_stats:
        raise AssertionError(f"Missing {field} stats for domain {domain}")
    field_stats = dom_stats[field]
    if not isinstance(field_stats, dict) or stat not in field_stats:
        raise AssertionError(f"Missing {stat} in {field} stats for domain {domain}")
    return np.array(field_stats[stat], dtype=np.float64)


def _collect_per_timestep_stat(stats_dict, domain, field, timestep_key, stat):
    dom_stats = stats_dict.get(domain)
    if not isinstance(dom_stats, dict) or field not in dom_stats:
        raise AssertionError(f"Missing {field} per-timestep stats for domain {domain}")
    field_stats = dom_stats[field]
    if not isinstance(field_stats, dict):
        raise AssertionError(f"Expected dict for per-timestep field {field} in domain {domain}")
    timestep_stats = field_stats.get(timestep_key)
    if not isinstance(timestep_stats, dict) or stat not in timestep_stats:
        raise AssertionError(
            f"Missing {stat} in {field} {timestep_key} stats for domain {domain}"
        )
    return np.array(timestep_stats[stat], dtype=np.float64)


def _stats_have_field(stats_dict, domains, field):
    for domain in domains:
        dom_stats = stats_dict.get(domain)
        if not isinstance(dom_stats, dict) or field not in dom_stats:
            return False
    return True


def _require_output_stats_key(per_timestep_stats, domains, key: str) -> str:
    if not _stats_have_field(per_timestep_stats, domains, key):
        raise AssertionError(
            f"Missing output stats for domains {domains}; expected '{key}'."
        )
    return key


def load_norm_stats_from_json(args, device, num_timesteps, var_floor: float) -> NormStatsBundle:
    domains = list(args.domains)
    all_stats, all_per_timestep_stats = load_stats_from_json_folder(args.norm_stats_path, domains)

    output_stats_key = _require_output_stats_key(
        all_per_timestep_stats,
        domains,
        "gt_scene_flows_relative",
    )

    # Prepare domain index mapping for runtime selection
    domain_to_index = {d: i for i, d in enumerate(domains)}

    # ============ Output stats per-domain ============
    per_domain_step_means = []  # list of (T,3)
    per_domain_step_vars = []   # list of (T,3)
    for d in domains:
        step_means = []
        step_vars = []
        for t in range(num_timesteps):
            timestep_key = f"timestep_{t}"
            step_mean_np = _collect_per_timestep_stat(
                all_per_timestep_stats, d, output_stats_key, timestep_key, "mean"
            )
            step_var_np = _collect_per_timestep_stat(
                all_per_timestep_stats, d, output_stats_key, timestep_key, "variance"
            )
            step_mean = torch.tensor(step_mean_np, device=device, dtype=torch.float32)
            step_var = torch.tensor(step_var_np, device=device, dtype=torch.float32)
            step_var = step_var.clamp(min=var_floor)
            step_means.append(step_mean)
            step_vars.append(step_var)
        per_domain_step_means.append(torch.stack(step_means))  # (T,3)
        per_domain_step_vars.append(torch.stack(step_vars))    # (T,3)

    per_step_mean = torch.stack(per_domain_step_means)  # (D,T,3)
    per_step_var = torch.stack(per_domain_step_vars)    # (D,T,3)

    # ============ Input features stats per-domain ============
    robot_domain_means = []
    robot_domain_vars = []
    scene_domain_means = []
    scene_domain_vars = []
    for d in domains:
        robot_mean_np = _collect_stat(all_stats, d, "robot_features", "mean")
        robot_var_np = _collect_stat(all_stats, d, "robot_features", "variance")
        scene_mean_np = _collect_stat(all_stats, d, "scene_features", "mean")
        scene_var_np = _collect_stat(all_stats, d, "scene_features", "variance")
        robot_mean = torch.tensor(robot_mean_np, device=device, dtype=torch.float32)
        robot_var = torch.tensor(robot_var_np, device=device, dtype=torch.float32).clamp(min=var_floor)
        scene_mean = torch.tensor(scene_mean_np, device=device, dtype=torch.float32)
        scene_var = torch.tensor(scene_var_np, device=device, dtype=torch.float32).clamp(min=var_floor)
        robot_domain_means.append(robot_mean)
        robot_domain_vars.append(robot_var)
        scene_domain_means.append(scene_mean)
        scene_domain_vars.append(scene_var)

    robot_mean = torch.stack(robot_domain_means)  # (D, Fr)
    robot_var = torch.stack(robot_domain_vars)    # (D, Fr)
    scene_mean = torch.stack(scene_domain_means)  # (D, Ds)
    scene_var = torch.stack(scene_domain_vars)    # (D, Ds)

    return NormStatsBundle(
        domains=domains,
        domain_to_index=domain_to_index,
        per_step_mean=per_step_mean,
        per_step_var=per_step_var,
        robot_mean=robot_mean,
        robot_var=robot_var,
        scene_mean=scene_mean,
        scene_var=scene_var,
    )


def normalize_output(og, dom_idx, per_step_mean, per_step_var):
    assert og.ndim == 4 and og.shape[-1] == 3, f"Expected shape (B, T, Ns, 3), got {og.shape}"
    B, T, _ = og.shape[:3]
    assert dom_idx.dim() == 1 and dom_idx.shape[0] == B, "Domain indices shape mismatch with batch size."
    means = per_step_mean[dom_idx]  # (B,T,3)
    sigmas = per_step_var[dom_idx].sqrt()  # (B,T,3)
    return (og - means.unsqueeze(2)) / sigmas.unsqueeze(2)


def unnormalize_output(normalized, dom_idx, per_step_mean, per_step_var):
    assert normalized.ndim == 4 and normalized.shape[-1] == 3, (
        f"Expected shape (B, T, Ns, 3), got {normalized.shape}"
    )
    B, T, _ = normalized.shape[:3]
    assert dom_idx.dim() == 1 and dom_idx.shape[0] == B, "Domain indices shape mismatch with batch size."
    means = per_step_mean[dom_idx]  # (B,T,3)
    sigmas = per_step_var[dom_idx].sqrt()  # (B,T,3)
    return normalized * sigmas.unsqueeze(2) + means.unsqueeze(2)


def normalize_robot_features(robot_features, dom_idx, robot_mean, robot_var):
    assert robot_features.ndim >= 2, "robot_features must be at least 2D"
    B = robot_features.shape[0]
    assert dom_idx.dim() == 1 and dom_idx.shape[0] == B, "Domain indices shape mismatch with batch size."
    means = robot_mean[dom_idx]  # (B, Fr)
    sigmas = robot_var[dom_idx].sqrt()  # (B, Fr)
    while means.dim() < robot_features.dim():
        means = means.unsqueeze(1)
        sigmas = sigmas.unsqueeze(1)
    return (robot_features - means) / sigmas


def normalize_scene_features(scene_features, dom_idx, scene_mean, scene_var):
    assert scene_features.ndim >= 2, "scene_features must be at least 2D"
    B = scene_features.shape[0]
    assert dom_idx.dim() == 1 and dom_idx.shape[0] == B, "Domain indices shape mismatch with batch size."
    means = scene_mean[dom_idx]  # (B, Ds)
    sigmas = scene_var[dom_idx].sqrt()  # (B, Ds)
    while means.dim() < scene_features.dim():
        means = means.unsqueeze(1)
        sigmas = sigmas.unsqueeze(1)
    return (scene_features - means) / sigmas
