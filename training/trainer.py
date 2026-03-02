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

import json
import os
import subprocess
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import wandb
from dataset_components.dataloader import build_dataloader
from tqdm.auto import tqdm
from utils import *
from torch.amp import GradScaler
from arguments import parse_args
from collections import defaultdict
from utils import _print, check_model_parameters_for_nan, check_tensor_for_nan, handle_nan_grad_norm
import random
import sys
import traceback
import numpy as np
import math
from pointworld.base import BaseModel
from pointworld.checkpoint_contract import apply_model_contract_to_args, read_checkpoint_contract
from training import checkpointing as checkpointing_utils

class Trainer:
    def __init__(self, args, inference_only=False, data_info_dict=None):
        self.args = args
        self.inference_only = inference_only
        self.device = None
        self.world_size = 1
        self.rank = 0
        self.setup_distributed()

        # ----------------------------------------------------------------------------------
        # Setup save dir + wandb (simplified: train-from-scratch semantics)
        # ----------------------------------------------------------------------------------
        self.exp_name, self.wandb_id = self.setup_wandb(exp_name=self.args.exp_name)
        self.save_dir = self.setup_save_dir(exp_name=self.exp_name)
        checkpoint = self.load_checkpoint_from_path(self.args.model_path)
        if checkpoint is not None:
            context = f"training checkpoint '{self.args.model_path}'"
            model_contract, _ = read_checkpoint_contract(checkpoint, context=context)
            changed = apply_model_contract_to_args(
                self.args,
                model_contract,
                context=context,
                explicit_cli_dests=set(getattr(self.args, "_explicit_cli_dests", set())),
            )
            if changed and self.rank == 0:
                _print("Applied canonical checkpoint model_contract for model initialization.")

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
        
        # Norm stats are restored from checkpoint on resume; no manifest handling needed.

        # ----------------------------------------------------------------------------------
        # Build dataloaders (with rank/world_size passed in) or infer dims from checkpoint
        # ----------------------------------------------------------------------------------
        if not self.inference_only:
            data_info_dict = self.setup_dataloader()
        else:
            # In inference-only mode, allow data_info_dict to be None and infer from checkpoint
            if data_info_dict is None:
                assert checkpoint is not None, (
                    "inference_only=True with no data_info_dict requires a valid checkpoint"
                )
                # Extract model state dict and infer projection input dims
                state = checkpoint.get('model')
                if state is None:
                    raise KeyError("Checkpoint missing model weights ('model')")
                scene_key = "scene_feature_encoder.scene_raw_feat_proj.weight"
                robot_key = "robot_proj.fc1.weight"
                if scene_key not in state:
                    raise KeyError(f"Checkpoint missing required key '{scene_key}'")
                if robot_key not in state:
                    raise KeyError(f"Checkpoint missing required key '{robot_key}'")
                scene_w = state[scene_key]
                robot_w = state[robot_key]
                data_info_dict = {
                    'scene_features_dim': int(scene_w.shape[1]),
                    'robot_features_dim': int(robot_w.shape[1]),
                }
        assert data_info_dict is not None, "data_info_dict must be present"

        # ----------------------------------------------------------------------------------
        # Build model and optimizer
        # ----------------------------------------------------------------------------------
        self.cpu_pg = None
        if self.args.distributed:
            self.cpu_pg = dist.new_group(backend="gloo")      # reuse everywhere
            self.eval_global_keys = None                           # populated once
        self.model = BaseModel(args, data_info_dict, rank=self.rank, cpu_pg=self.cpu_pg)
        self.model.to(self.device)
        # Wrap in DDP if using multiple processes
        if self.args.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False,
            )
        if self.rank == 0:
            wandb.watch(self.model, log='all')
        # Create optimizer only if training mode
        if not self.inference_only:
            self._create_optimizer()
        else:
            self.optimizer = None
        
        # ----------------------------------------------------------------------------------
        # Load checkpoint if specified
        # ----------------------------------------------------------------------------------
        # initialize counters
        self.epoch_count = 0
        self.batch_count = 0
        self.curr_run_batch_count = 0
        self.sample_count = 0
        # load model state dict and update counters
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        # Recompute epoch_count from samples seen if training
        if not self.inference_only:
            self.epoch_count = self.sample_count / float(self.train_total_samples)

        # ----------------------------------------------------------------------------------
        # Misc setup
        # ----------------------------------------------------------------------------------
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            log_dict = {
                "model/total_params": total_params,
            }
            for k, v in data_info_dict.items():
                log_dict[f'model/{k}'] = v
            for k, v in log_dict.items():
                _print(f'{k:<30}: {v}')
            _print(f'{"Experiment Name":<30}: {self.exp_name}')
            _print(f'{"Wandb ID":<30}: {self.wandb_id}')
            wandb.log(log_dict)
        
        # Initialize consecutive NaN grad norm counter
        self.consecutive_nan_grad_count = 0

    def setup_save_dir(self, exp_name=None):
        self.log_dir = self.args.log_dir
        if self.log_dir.startswith("s3://"):
            raise ValueError(f"S3 log_dir is not supported in release refactor: {self.log_dir}")
        self.save_dir = os.path.join(self.log_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.rank == 0:
            _print(f'Experiment ID: {exp_name}')
            _print(f'Saving to: {self.save_dir}')
        return self.save_dir
    
    def setup_distributed(self):
        """Initialize Torch Distributed if requested and set device."""
        ### DDP CHANGE ###
        if self.args.distributed:
            dist.init_process_group(backend="nccl", init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            self.device = torch.device(self.args.device)
            self.rank = 0
            self.world_size = 1
        if self.args.deterministic_train:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if getattr(self.args, "deterministic_algorithms", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # set seed
        np.random.seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed + self.rank)
        random.seed(self.args.seed + self.rank)
        torch.cuda.manual_seed(self.args.seed + self.rank)
        _print(f'Rank {self.rank + 1}/{self.world_size} using device {self.device}')

    def setup_wandb(self, exp_name=None):
        # Initialize wandb only on rank 0
        if self.rank == 0:
            # Only rank=0 does wandb.init
            disable_wandb = os.environ.get('WANDB_MODE') == 'disabled'
            self.wandb_run = wandb.init(
                project="point-world",
                name=exp_name,
                mode="online" if not disable_wandb else "disabled",
                config=self.args.og_args
            )
            self.exp_name, self.wandb_id = self.wandb_run.name, self.wandb_run.id
        else:
            self.exp_name = None
            self.wandb_id = None

        # Broadcast exp_name from rank 0 to all other ranks
        if self.args.distributed:
            exp_name_list = [self.exp_name]
            dist.broadcast_object_list(exp_name_list, src=0)
            self.exp_name = exp_name_list[0]
            wandb_id_list = [self.wandb_id]
            dist.broadcast_object_list(wandb_id_list, src=0)
            self.wandb_id = wandb_id_list[0]
        
        return self.exp_name, self.wandb_id
    
    def setup_dataloader(self):
        # Create train / test datasets with DDP slicing
        if self.rank == 0:
            _print('Setting up dataloaders...')
        
        start = time.time()
        train_override_splits = getattr(self.args, "train_splits", None)
        if self.rank == 0 and train_override_splits is not None:
            _print(f"Overriding training splits with: {train_override_splits}")
        self.train_dataloader, train_info = build_dataloader(
            args=self.args,
            mode='train',
            rank=self.rank,
            world_size=self.world_size,
            override_splits=train_override_splits,
        )
        if self.rank == 0: _print(f'Train dataset setup took {time.time() - start:.2f}s')
        
        start = time.time()
        self.test_dataloader, _ = build_dataloader(args=self.args, mode='test', rank=self.rank, world_size=self.world_size, force_resampled_eval=True)
        if self.test_dataloader is None:
            if self.args.eval_freq > 0:
                raise RuntimeError("Test dataloader missing but eval_freq > 0; cannot run evaluation.")
            self.test_iter = None
            if self.rank == 0:
                _print("Test dataset missing; test-time eval disabled.")
        else:
            self.test_iter = iter(self.test_dataloader)
            if self.rank == 0: _print(f'Test dataset setup took {time.time() - start:.2f}s')
        
        data_info_dict = {}
        # Sample a batch to inspect dimension info
        sample_iter = self.test_iter if self.test_iter is not None else iter(self.train_dataloader)
        batch = next(sample_iter)
        robot_features = batch['robot_features']  # (B, T, NR, DR)
        scene_features = batch['scene_features']  # (B, T, NS, DS)
        data_info_dict = {
            'robot_features_dim': robot_features.shape[-1],
            'scene_features_dim': scene_features.shape[-1],
            'data_keys': list(batch.keys())
        }

        # Store total samples for epoch estimation
        self.train_total_samples = train_info['total_samples']

        # persistent iterator for *train inference* evaluation
        self.train_iter = iter(self.train_dataloader)
        return data_info_dict

    def _save_checkpoint_now(self, adjusted_batch_count, log_dict=None):
        return checkpointing_utils.save_checkpoint_now(self, adjusted_batch_count, log_dict)

    def save_checkpoint(self, log_dict=None):
        return checkpointing_utils.save_checkpoint(self, log_dict)

    def load_checkpoint_from_path(self, model_path=None):
        return checkpointing_utils.load_checkpoint_from_path(self, model_path)

    def load_checkpoint(self, checkpoint):     
        return checkpointing_utils.load_checkpoint(self, checkpoint)

    @torch.no_grad()
    def eval_step(self, dataloader, data_iter, prefix, log_dict=None):
        """
        • dataloader : the DataLoader you want to sample from  
        • data_iter  : a *persistent* iterator you keep around  
        • prefix     : string put in front of every logged key  
        """
        self.model.eval()
        if log_dict is None:
            log_dict = {}

        device = self.device
        metrics = defaultdict(lambda: torch.tensor(0.0, device=device))
        counts  = defaultdict(lambda: torch.tensor(0,   device=device))
        last_batch = None
        last_outputs = None
        with tqdm(range(self.args.num_eval_batches),
                  desc=f'eval-{prefix}',
                  leave=False,
                  disable=self.rank != 0) as tbar:
            for _ in tbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Exhausted: recreate iterator and continue without interrupting training
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}

                with torch.autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(batch, training=False)
                    _, loss_dict = (
                        self.model.module.loss_fn(outputs, batch, training=False)
                        if self.args.distributed else
                        self.model.loss_fn(outputs, batch, training=False)
                    )
                last_batch = batch
                last_outputs = outputs

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1 and not torch.isnan(v):
                            metrics[k] += v * self.args.batch_size
                            counts[k]  += self.args.batch_size
                    elif isinstance(v, (int, float)) and not math.isnan(v):
                        metrics[k] += v * self.args.batch_size
                        counts[k]  += self.args.batch_size

        # all‑reduce
        if self.args.distributed:
            if self.eval_global_keys is None:
                tmp = [None] * self.world_size
                dist.all_gather_object(tmp, list(metrics.keys()), group=self.cpu_pg)
                self.eval_global_keys = sorted(set().union(*tmp))

            for k in self.eval_global_keys:        # pad missing keys locally
                metrics.setdefault(k, torch.tensor(0.0, device=self.device))
                counts .setdefault(k, torch.tensor(0,   device=self.device))
                dist.all_reduce(metrics[k], group=self.cpu_pg)
                dist.all_reduce(counts[k], group=self.cpu_pg)

        # Write into log_dict with prefix
        for k in metrics.keys():
            if counts[k] > 0:
                log_dict[f'{prefix}/{k}'] = (metrics[k] / counts[k]).item()

        return log_dict, data_iter

    def should_do(self, action_freq, last_batch, curr_batch):
        """Helper function for cadence checks."""
        return action_freq > 0 and (curr_batch - last_batch >= action_freq)

    def train(self):

        # Only call GradScaler if AMP
        scaler = GradScaler('cuda')
        
        last_save_batch = 0 if self.args.save_freq > 0 else -1
        # Initialize cadence tracking with negative values to ensure first iteration triggers
        last_eval_batch = -self.args.eval_freq if self.args.eval_freq > 0 else -1
        loss_dict = {'total_loss': 0.0}
        while self.epoch_count < self.args.num_epochs:
            # Create progress bar for this epoch (rank 0 only)
            if self.rank == 0:
                tepoch = tqdm(total=len(self.train_dataloader),
                              desc=f'Initializing...',  # Will be updated in the loop
                              leave=False)
            
            # Iterate through one epoch using the shared train_iter
            for train_batch_idx in range(len(self.train_dataloader)):
                # Get next batch from shared iterator
                train_batch = next(self.train_iter)
                log_dict = dict()

                # Calculate adjusted batch count for frequency checks (equivalent to elapsed time)
                # Divide by world_size since each process contributes to the global batch count
                # This makes frequency checks work the same in both distributed and non-distributed settings
                adjusted_batch_count = self.curr_run_batch_count // self.world_size

                # eval when batch count since last eval >= eval_freq
                if self.should_do(self.args.eval_freq, last_eval_batch, adjusted_batch_count):
                    if self.args.distributed:
                        dist.barrier()
                    # eval on train set
                    log_dict, self.train_iter = self.eval_step(
                        self.train_dataloader,
                        self.train_iter,
                        prefix='train_eval',
                        log_dict=log_dict
                    )
                    # eval on test set (if available)
                    if self.test_dataloader is not None:
                        log_dict, self.test_iter = self.eval_step(
                            self.test_dataloader,   self.test_iter,
                            prefix='test',
                            log_dict=log_dict
                        )
                    if self.args.distributed:
                        dist.barrier()
                    torch.cuda.empty_cache()
                    last_eval_batch = adjusted_batch_count

                # save checkpoint when batch count since last save >= save_freq
                if self.should_do(self.args.save_freq, last_save_batch, adjusted_batch_count):
                    self._save_checkpoint_now(adjusted_batch_count, log_dict=log_dict)
                    last_save_batch = adjusted_batch_count

                # Update progress bar with current time and stats (rank 0 only)
                if self.rank == 0:
                    global_batch_count = self.batch_count
                    global_sample_count = self.sample_count
                    current_time = datetime.now().strftime("%H:%M:%S")
                    tepoch.set_description(
                        f'[{current_time} {self.exp_name}], Epoch={self.epoch_count+1:.3f}/{self.args.num_epochs}, '
                        f'B={global_batch_count}, S={global_sample_count}, '
                        f'Loss={loss_dict["total_loss"]:.3e}'
                    )
                    tepoch.update(1)  # Update progress bar by 1

                self.model.train()
                train_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in train_batch.items()}

                self.optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        outputs = self.model(train_batch, training=True)

                        # Check if outputs contain NaN and should be skipped
                        if outputs.get('_has_nan_outputs', False):
                            # Skip this batch and continue to next iteration
                            continue

                        total_loss, loss_dict = (
                            self.model.module.loss_fn(outputs, train_batch, training=True)
                            if self.args.distributed else self.model.loss_fn(outputs, train_batch, training=True)
                        )

                    # Check for NaN in total loss before scaling
                    check_tensor_for_nan(total_loss, "total_loss", f"training step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})")

                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.args.grad_clip_max_norm
                    )

                    # Check gradient norm for NaN and skip batch if needed
                    should_skip_batch, self.consecutive_nan_grad_count = handle_nan_grad_norm(
                        grad_norm, self.consecutive_nan_grad_count, context=f"training step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})"
                    )

                    if should_skip_batch:
                        # Even when skipping, we must call scaler.step() and scaler.update()
                        # to reset the scaler state after unscale_() was called
                        scaler.step(self.optimizer)  # This will be a no-op due to NaN gradients
                        scaler.update()
                        continue  # Skip this batch and move to next iteration

                    # Check model parameters for NaN after gradient computation
                    check_model_parameters_for_nan(self.model, f"after backward pass (epoch {self.epoch_count:.3f}, batch {train_batch_idx})")

                    log_dict['grad_norm'] = float(grad_norm)

                    scaler.step(self.optimizer)
                    scaler.update()

                    # Final check for NaN in model parameters after optimizer step
                    check_model_parameters_for_nan(self.model, f"after optimizer step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})")
                
                except NaNDetectionError as nan_error:
                    # Handle NaN detection error
                    if self.rank == 0:
                        _print("\n" + "="*100)
                        _print("🚨 NaN DETECTED 🚨")
                        _print("="*100)
                        _print(f"Error: {nan_error}")
                        _print(f"Context: {nan_error.context}")
                        _print("="*100)

                        # Save current checkpoint as well
                        _print("Saving emergency checkpoint...")
                        save_dir = self.save_checkpoint(log_dict)
                        _print(f"Emergency checkpoint saved to {save_dir}")
                        
                        _print("="*100)
                        _print("Training stopped due to NaN detection.")
                        _print("Debug files saved to ./debug/ directory.")
                        _print("="*100)
                    
                    # Ensure all ranks are synchronized before exiting
                    if self.args.distributed:
                        dist.barrier()
                    
                    # Re-raise the error to stop training
                    raise nan_error

                # For aggregated metrics, you can all_reduce if desired, then average on rank=0.
                for k, v in loss_dict.items():
                    # convert NaN to 0 so reduce works; keep a flag if you still want NaN in logs
                    is_nan = isinstance(v, float) and math.isnan(v)
                    val_tensor = torch.tensor([0.0 if is_nan else float(v)],
                                            device=self.device, dtype=torch.float)

                    if self.args.distributed:
                        dist.all_reduce(val_tensor,
                                        op=dist.ReduceOp.SUM,
                                        group=self.cpu_pg)

                    avg_val = val_tensor.item() / self.world_size
                    log_dict[f"train/{k}"] = float('nan') if is_nan else avg_val
                # Increment counters
                # batch_count is the global count across all processes and runs
                # curr_run_batch_count is reset each time training starts
                # Both are multiplied by world_size to account for distributed training
                self.batch_count += 1 * self.world_size
                self.curr_run_batch_count += 1 * self.world_size
                self.sample_count += self.args.batch_size * self.world_size
                # Update epoch estimate continuously based on samples processed
                self.epoch_count = self.sample_count / float(self.train_total_samples)

                log_dict['batch_count'] = self.batch_count
                log_dict['epoch_count'] = self.epoch_count
                log_dict['curr_run_batch_count'] = self.curr_run_batch_count
                log_dict['sample_count'] = self.sample_count
                curr_steps = self.curr_run_batch_count // self.world_size

                # If we've reached target epochs mid-epoch, save final checkpoint and exit cleanly
                if self.epoch_count >= self.args.num_epochs:
                    self._save_checkpoint_now(adjusted_batch_count, log_dict={})
                    if self.rank == 0:
                        tepoch.close()
                    return 0

                # Rank 0 logs to W&B
                if self.rank == 0:
                    wandb.log(log_dict)
                log_dict.clear()

                if self.args.max_train_steps > 0 and curr_steps >= self.args.max_train_steps:
                    if self.rank == 0:
                        tepoch.close()
                    return 0
            
            # Close progress bar for this epoch (rank 0 only)
            if self.rank == 0:
                tepoch.close()
            
        return 0

    @torch.no_grad()
    def full_eval(self, max_num_batches=1000):
        """
        Evaluate entire train/test sets, aggregated across all data.
        Metrics are computed across all GPUs and aggregated.
        """
        # barrier to ensure all processes have finished training
        if self.args.distributed:
            dist.barrier()
        
        # Switch to eval mode
        self.model.eval()

        # Helper to accumulate metrics across multiple batches and GPUs
        def accumulate_metrics(dataloader, prefix):
            metrics = defaultdict(lambda: torch.tensor(0.0, device=self.device))
            counts = defaultdict(lambda: torch.tensor(0, device=self.device))
            
            # Only show progress bar on rank 0
            iterator = dataloader
            if self.rank == 0:
                iterator = tqdm(dataloader, desc=f'Eval {prefix}', total=max_num_batches, leave=False)
            
            for i, batch in enumerate(iterator):
                if i >= max_num_batches:
                    break
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Forward pass
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(batch, training=False)
                    total_loss, loss_dict = self.model.module.loss_fn(outputs, batch, training=False) if self.args.distributed else self.model.loss_fn(outputs, batch, training=False)
                
                # Accumulate batch metrics
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        if (v.numel() == 1 or v.dim() == 0) and not torch.isnan(v):
                            metrics[k] += v * self.args.batch_size
                            counts[k] += self.args.batch_size
                    elif isinstance(v, (int, float)) and not math.isnan(v):
                        metrics[k] += v * self.args.batch_size
                        counts[k] += self.args.batch_size

            # Aggregate metrics across GPUs if distributed
            if self.args.distributed:
                all_keys = [None] * self.world_size
                dist.all_gather_object(all_keys, list(metrics.keys()), group=self.cpu_pg)
                union_keys = set().union(*all_keys)
                for k in sorted(union_keys):
                    metrics.setdefault(k, torch.tensor(0.0, device=self.device))
                    counts.setdefault(k, torch.tensor(0,   device=self.device))
                    dist.all_reduce(metrics[k],
                                    op=dist.ReduceOp.SUM,
                                    group=self.cpu_pg)
                    dist.all_reduce(counts[k],
                                    op=dist.ReduceOp.SUM,
                                    group=self.cpu_pg)

            # Compute means
            aggregated_metrics = {}
            for k in metrics.keys():
                if counts[k] > 0:
                    aggregated_metrics[f"{prefix}/{k}"] = (metrics[k] / counts[k]).item()
                else:
                    aggregated_metrics[f"{prefix}/{k}"] = 0.0
            return aggregated_metrics

        # 1) Evaluate on train and test set
        train_metrics = accumulate_metrics(self.train_dataloader, prefix='full_eval/train')
        test_metrics = {}
        if self.test_dataloader is not None:
            test_metrics = accumulate_metrics(self.test_dataloader, prefix='full_eval/test')

        # Combine all metrics and log to W&B (only on rank 0)
        all_metrics = {**train_metrics, **test_metrics}
        if self.rank == 0:
            wandb.log(all_metrics)

        return all_metrics

    def _create_optimizer(self):
        """Create a single AdamW optimizer over all trainable parameters."""
        model = self.model.module if self.args.distributed else self.model
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay,
        )
        if self.rank == 0:
            total_count = sum(p.numel() for p in params)
            _print("\nOptimizer parameters:")
            _print(f"  Total: {total_count:,} elements\n")
