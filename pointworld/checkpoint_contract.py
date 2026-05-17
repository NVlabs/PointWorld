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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION_KEY = "checkpoint_schema_version"
MODEL_CONTRACT_KEY = "model_contract"
DATA_CONTRACT_KEY = "data_contract"

MODEL_CONTRACT_KEYS = (
    "ptv3_size",
    "ptv3_patch_size",
    "predictor_dim",
    "max_scene_points",
    "max_robot_points",
    "grid_size",
    "depth_threshold",
    "norm_stats_path",
    "train_min_num_cameras",
    "train_max_num_cameras",
    "eval_min_num_cameras",
    "eval_max_num_cameras",
)

# Published schema-v1 checkpoints were prepared before these camera-count
# fields were added to the saved model contract. They used these parser defaults.
LEGACY_MODEL_CONTRACT_DEFAULTS = {
    "train_min_num_cameras": 1,
    "train_max_num_cameras": 3,
    "eval_min_num_cameras": 2,
    "eval_max_num_cameras": 2,
}

DATA_CONTRACT_KEYS = (
    "domains",
)

def _cast_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


_MODEL_CASTERS = {
    "ptv3_size": str,
    "ptv3_patch_size": int,
    "predictor_dim": int,
    "max_scene_points": int,
    "max_robot_points": int,
    "grid_size": float,
    "depth_threshold": float,
    "norm_stats_path": str,
    "train_min_num_cameras": _cast_optional_int,
    "train_max_num_cameras": _cast_optional_int,
    "eval_min_num_cameras": int,
    "eval_max_num_cameras": int,
}

_MODEL_ARG_FALLBACKS = {
    "max_scene_points": ("max_scene_particles",),
    "max_robot_points": ("max_robot_particles",),
}

def _args_to_mapping(args: Any, context: str) -> Mapping[str, Any]:
    if args is None:
        raise RuntimeError(f"{context} is missing; cannot build checkpoint contract.")
    if isinstance(args, Mapping):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    raise RuntimeError(
        f"{context} has unsupported type {type(args).__name__}; expected Namespace-like metadata."
    )


def _normalize_domains(value: Any, context: str) -> list[str]:
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        raise RuntimeError(
            f"{context} must be a string/list/tuple, got {type(value).__name__}."
        )
    if not items:
        raise RuntimeError(f"{context} cannot be empty.")
    return items


def _extract_model_value(args_map: Mapping[str, Any], key: str, context: str) -> Any:
    if key in args_map:
        return args_map[key]
    for fallback_key in _MODEL_ARG_FALLBACKS.get(key, ()):
        if fallback_key in args_map:
            return args_map[fallback_key]
    raise RuntimeError(f"{context} missing required model-contract field '{key}'.")


def _validate_model_contract(model_contract: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized_contract = dict(model_contract)
    missing = [k for k in MODEL_CONTRACT_KEYS if k not in normalized_contract]
    if missing:
        unsupported_missing = [k for k in missing if k not in LEGACY_MODEL_CONTRACT_DEFAULTS]
        if unsupported_missing:
            raise RuntimeError(f"{context} model_contract missing keys: {missing}.")
        for key in missing:
            normalized_contract[key] = LEGACY_MODEL_CONTRACT_DEFAULTS[key]
    validated: dict[str, Any] = {}
    for key in MODEL_CONTRACT_KEYS:
        try:
            validated[key] = _MODEL_CASTERS[key](normalized_contract[key])
        except Exception as exc:
            raise RuntimeError(
                f"{context} model_contract field '{key}' has invalid value {normalized_contract[key]!r}."
            ) from exc
    return validated


def _validate_data_contract(data_contract: Mapping[str, Any], context: str) -> dict[str, Any]:
    missing = [k for k in DATA_CONTRACT_KEYS if k not in data_contract]
    if missing:
        raise RuntimeError(f"{context} data_contract missing keys: {missing}.")
    validated = {
        "domains": _normalize_domains(
            data_contract["domains"], f"{context} data_contract['domains']"
        )
    }
    return validated


def build_model_contract_from_args(args: Any, *, context: str = "checkpoint args") -> dict[str, Any]:
    args_map = _args_to_mapping(args, context)
    out: dict[str, Any] = {}
    for key in MODEL_CONTRACT_KEYS:
        raw_value = _extract_model_value(args_map, key, context)
        try:
            out[key] = _MODEL_CASTERS[key](raw_value)
        except Exception as exc:
            raise RuntimeError(
                f"{context} field '{key}' has invalid value {raw_value!r} for model contract."
            ) from exc
    return out


def build_data_contract_from_args(args: Any, *, context: str = "checkpoint args") -> dict[str, Any]:
    args_map = _args_to_mapping(args, context)
    if "domains" not in args_map:
        raise RuntimeError(f"{context} missing required field 'domains' for data contract.")
    return {"domains": _normalize_domains(args_map["domains"], f"{context} 'domains'")}


def attach_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    args: Any = None,
    context: str = "checkpoint",
) -> bool:
    if args is None:
        args = checkpoint.get("args")
    model_contract = build_model_contract_from_args(args, context=f"{context} args")
    data_contract = build_data_contract_from_args(args, context=f"{context} args")
    changed = (
        checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY) != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get(MODEL_CONTRACT_KEY) != model_contract
        or checkpoint.get(DATA_CONTRACT_KEY) != data_contract
    )
    checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CHECKPOINT_SCHEMA_VERSION
    checkpoint[MODEL_CONTRACT_KEY] = model_contract
    checkpoint[DATA_CONTRACT_KEY] = data_contract
    return changed


def read_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    context: str,
    allow_legacy_args_fallback: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    has_contract = MODEL_CONTRACT_KEY in checkpoint and DATA_CONTRACT_KEY in checkpoint
    if has_contract:
        schema_version = checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY)
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                f"{context} has unsupported schema version {schema_version!r}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}."
            )
        model_contract_raw = checkpoint[MODEL_CONTRACT_KEY]
        data_contract_raw = checkpoint[DATA_CONTRACT_KEY]
        if not isinstance(model_contract_raw, Mapping):
            raise RuntimeError(f"{context} model_contract must be a mapping.")
        if not isinstance(data_contract_raw, Mapping):
            raise RuntimeError(f"{context} data_contract must be a mapping.")
        model_contract = _validate_model_contract(model_contract_raw, context)
        data_contract = _validate_data_contract(data_contract_raw, context)
        return model_contract, data_contract

    if allow_legacy_args_fallback:
        args = checkpoint.get("args")
        model_contract = build_model_contract_from_args(args, context=f"{context} args")
        data_contract = build_data_contract_from_args(args, context=f"{context} args")
        return model_contract, data_contract

    raise RuntimeError(
        f"{context} is missing canonical checkpoint contract metadata "
        f"('{CHECKPOINT_SCHEMA_VERSION_KEY}', '{MODEL_CONTRACT_KEY}', '{DATA_CONTRACT_KEY}')."
    )


def apply_model_contract_to_args(
    args: Any,
    model_contract: Mapping[str, Any],
    *,
    context: str,
    explicit_cli_dests: set[str] | None = None,
) -> list[str]:
    validated_contract = _validate_model_contract(model_contract, context)
    if explicit_cli_dests is None:
        explicit_cli_dests = set(getattr(args, "_explicit_cli_dests", set()))
    changed: list[str] = []
    for key, target_value in validated_contract.items():
        current_value = getattr(args, key, None)
        if key in explicit_cli_dests and current_value != target_value:
            raise RuntimeError(
                f"{context} requires {key}={target_value!r}, but explicit CLI set {key}={current_value!r}. "
                f"Remove the explicit --{key} override (or set it to {target_value!r})."
            )
        if current_value != target_value:
            setattr(args, key, target_value)
            changed.append(key)
    return changed


def train_domains_from_data_contract(
    data_contract: Mapping[str, Any], *, context: str
) -> list[str]:
    validated = _validate_data_contract(data_contract, context)
    return validated["domains"]
