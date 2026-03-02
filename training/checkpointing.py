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
import torch
import torch.distributed as dist
from utils import _print
from pointworld.checkpoint_contract import attach_checkpoint_contract


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint_now(trainer, adjusted_batch_count, log_dict=None):
    if trainer.args.distributed:
        dist.barrier()
    if trainer.rank == 0:
        _print("Saving checkpoint")
        save_checkpoint(trainer, log_dict or {})
    if trainer.args.distributed:
        dist.barrier()


def save_checkpoint(trainer, log_dict=None):
    if trainer.rank != 0:
        return

    save_dict = {
        "model": _unwrap_model(trainer.model).state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "args": trainer.args,
        "exp_name": trainer.exp_name,
        "wandb_id": trainer.wandb_id,
        "batch_count": trainer.batch_count,
        "epoch_count": trainer.epoch_count,
        "sample_count": trainer.sample_count,
    }
    attach_checkpoint_contract(save_dict, args=trainer.args, context="training save checkpoint")

    final_local_path = os.path.join(trainer.save_dir, "model-last.pt")
    torch.save(save_dict, final_local_path)

    return trainer.save_dir


def load_checkpoint_from_path(trainer, model_path=None):
    if model_path is None:
        if trainer.rank == 0:
            _print("No checkpoint path provided, starting from scratch")
        return None
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    if trainer.rank == 0:
        _print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "wandb_id" in checkpoint:
        checkpoint.pop("wandb_id")
    if "exp_name" in checkpoint:
        checkpoint.pop("exp_name")
    return checkpoint


def load_checkpoint(trainer, checkpoint):
    model_state_dict = checkpoint["model"]
    expected_keys = set(_unwrap_model(trainer.model).state_dict().keys())
    ckpt_keys = set(model_state_dict.keys())
    if ckpt_keys != expected_keys:
        missing = sorted(expected_keys - ckpt_keys)
        extra = sorted(ckpt_keys - expected_keys)
        missing_suffix = "..." if len(missing) > 10 else ""
        extra_suffix = "..." if len(extra) > 10 else ""
        raise RuntimeError(
            "Checkpoint model keys do not match the current model state_dict. "
            "Ensure the distributed/Non-DDP setting and model config match. "
            f"Missing keys: {missing[:10]}{missing_suffix}. "
            f"Extra keys: {extra[:10]}{extra_suffix}."
        )
    _unwrap_model(trainer.model).load_state_dict(model_state_dict)

    if not trainer.inference_only:
        if "optimizer" not in checkpoint:
            if getattr(trainer.args, "allow_optimizer_reset", False):
                _print("Checkpoint missing optimizer state; resetting optimizer as requested.")
            else:
                raise RuntimeError(
                    "Checkpoint missing optimizer state. "
                    "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                )
        else:
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                if getattr(trainer.args, "allow_optimizer_reset", False):
                    _print(
                        "Optimizer state incompatible with current optimizer; "
                        "resetting optimizer state as requested."
                    )
                else:
                    raise RuntimeError(
                        "Optimizer state incompatible with current optimizer. "
                        "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                    ) from exc
    assert "batch_count" in checkpoint, "Checkpoint missing batch_count."
    assert "epoch_count" in checkpoint, "Checkpoint missing epoch_count."
    assert "sample_count" in checkpoint, "Checkpoint missing sample_count."
    trainer.batch_count = checkpoint["batch_count"]
    trainer.epoch_count = checkpoint["epoch_count"]
    trainer.sample_count = checkpoint["sample_count"]
