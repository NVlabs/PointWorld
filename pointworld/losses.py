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

import math
import torch
from torch.nn import HuberLoss
from pointworld import metrics as metrics_utils
from utils import safe_loss_computation


def compute_single_output_loss(
    args,
    output_norm,
    gt_target_norm,
    weights,
    pred_exists_supervised,
    log_var,
    var_floor: float,
    var_ceiling: float,
    uncertainty_logvar_weight: float,
):
    # Standard continuous prediction loss
    huber = HuberLoss(delta=args.huber_delta, reduction="none")
    error_term = huber(output_norm, gt_target_norm)  # (B,T,NS,3)
    per_point_loss = error_term.mean(dim=-1)  # (B,T,NS)

    # Apply uncertainty re-weighting (always on)
    with torch.autocast("cuda", enabled=False):
        # Clamp log-variance directly to keep NLL bounded
        s_min = math.log(var_floor)
        s_max = math.log(var_ceiling)
        if log_var.shape[-1] == 1:
            log_var_expanded = log_var.expand_as(error_term)
        else:
            log_var_expanded = log_var

        log_var_clamped = log_var_expanded.clamp(min=s_min, max=s_max)
        var = torch.exp(log_var_clamped)  # (B,T,NS,1|3)

        per_dim_loss = 0.5 * (
            error_term / var +
            uncertainty_logvar_weight * log_var_clamped
        )
        per_point_loss = per_dim_loss.mean(dim=-1)  # (B,T,NS)

    # Apply weights and sum
    dynamics_loss = (per_point_loss * weights).sum()
    total_loss = dynamics_loss

    return total_loss, per_point_loss


@torch.no_grad()
def compute_single_output_metrics(
    args,
    output_flows,
    gt_flows,
    per_point_loss,
    weights,
    moved,
    static,
    pred_exists_supervised,
    log_var,
    pred_norm,
    var_floor: float,
    var_ceiling: float,
):
    l2 = torch.norm(output_flows - gt_flows, dim=-1)  # (B,T,NS)

    metrics = metrics_utils.collect_metrics(
        args,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        l2,
        log_var,
        pred_norm,
        var_floor=var_floor,
        var_ceiling=var_ceiling,
    )

    return metrics


def internal_loss_fn(
    model,
    outputs,
    data_dict,
    training: bool,
    var_floor: float,
    var_ceiling: float,
    sim_domain_keywords: list[str],
    sim_var_const: float,
    uncertainty_logvar_weight: float,
):
    # ------------------------------------------------------------------ #
    #  Unpack tensors & masks                                            #
    # ------------------------------------------------------------------ #
    output_scene_flows = outputs["scene_flows"]  # (B,T,NS,3)
    gt_scene_flows = data_dict["gt_scene_flows"]  # (B,T,NS,3)
    weights = data_dict["point_weights"].to(model.device)
    assert torch.all(weights >= 0), "point_weights must be non-negative"
    weights = weights.clamp_min(0)

    gt_target = data_dict["gt_scene_flows_relative"]
    output_norm = outputs["scene_relative_norm"]
    gt_target_norm = model.normalize(gt_target)

    log_var = outputs["log_var"]

    moved = data_dict["scene_moved_mask"].squeeze(-1).bool()
    static = data_dict["scene_static_mask"].squeeze(-1).bool()
    context = data_dict["scene_context_mask"].squeeze(-1).bool()
    exists = data_dict["scene_exists"].squeeze(-1).bool()
    supervised = data_dict["scene_supervised_mask"].squeeze(-1).bool()
    pred = ~context
    pred_exists_supervised = pred & exists & supervised

    # ------------------------------------------------------------------ #
    #  Compute loss and metrics                                          #
    # ------------------------------------------------------------------ #
    def _prepare_log_var(_log_var):
        Bv, Tv, Nsv = _log_var.shape[:3]
        sim_b = torch.tensor(
            [any(k in dom for k in sim_domain_keywords) for dom in data_dict["__domain__"]],
            device=model.device,
            dtype=torch.bool,
        )
        if sim_b.any():
            sim_mask = sim_b.view(Bv, 1, 1, 1).expand_as(_log_var)
            real_b = ~sim_b
            if real_b.any():
                s_min_tmp = math.log(var_floor)
                s_max_tmp = math.log(var_ceiling)
                clamped = _log_var.detach().clamp(min=s_min_tmp, max=s_max_tmp)
                real_logvar = clamped[real_b]
                real_var_mean = real_logvar.exp().mean().clamp(
                    min=float(var_floor), max=float(var_ceiling)
                )
                const_val = float(real_var_mean.log().item())
            else:
                const_val = math.log(sim_var_const)
            const_tensor = torch.full_like(_log_var, fill_value=const_val)
            _log_var = torch.where(sim_mask, const_tensor, _log_var)
        s_min = math.log(var_floor)
        s_max = math.log(var_ceiling)
        return _log_var.clamp(min=s_min, max=s_max)

    log_var = _prepare_log_var(log_var)

    log_dict = {}
    total_loss, per_point_loss = compute_single_output_loss(
        model.args,
        output_norm,
        gt_target_norm,
        weights,
        pred_exists_supervised,
        log_var,
        var_floor=var_floor,
        var_ceiling=var_ceiling,
        uncertainty_logvar_weight=uncertainty_logvar_weight,
    )
    metrics = compute_single_output_metrics(
        model.args,
        output_scene_flows,
        gt_scene_flows,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        log_var,
        outputs["pred"],
        var_floor=var_floor,
        var_ceiling=var_ceiling,
    )
    metric_keys = list(metrics.keys())
    log_dict.update(metrics)

    log_dict = metrics_utils.collect_per_domain_metrics(
        model.args,
        model.device,
        data_dict,
        output_scene_flows,
        gt_scene_flows,
        per_point_loss,
        weights,
        moved,
        static,
        pred_exists_supervised,
        log_var,
        metric_keys,
        log_dict,
        outputs["pred"],
        lambda *args: compute_single_output_metrics(
            model.args,
            *args,
            var_floor=var_floor,
            var_ceiling=var_ceiling,
        ),
    )

    # ------------------------------------------------------------------ #
    #  Filtered metrics (eval only)                                      #
    # ------------------------------------------------------------------ #
    if (not training) and ("scene_filter_mask" in data_dict):
        conf_seed = data_dict.get("confidence_seed", None)
        if conf_seed is not None:
            if isinstance(conf_seed, torch.Tensor):
                if conf_seed.numel() == 1:
                    seed_val = int(conf_seed.item())
                else:
                    unique = torch.unique(conf_seed.detach().view(-1))
                    assert unique.numel() == 1, "Mixed confidence seeds detected in batch"
                    seed_val = int(unique.item())
            else:
                seed_val = int(conf_seed)
            assert seed_val == int(model.args.seed), (
                f"Confidence seed mismatch: file seed={seed_val}, args.seed={model.args.seed}"
            )

        l2_full = torch.norm(output_scene_flows - gt_scene_flows, dim=-1)  # (B,T,Ns)

        filt = data_dict["scene_filter_mask"].to(l2_full.device).bool()
        valid = filt & pred_exists_supervised

        def _masked_mean(x, m):
            if m.any():
                return x[m].mean().item()
            return float("nan")

        log_dict["filtered_l2/mean"] = _masked_mean(l2_full, valid)
        log_dict["filtered_l2_moved/mean"] = _masked_mean(l2_full, valid & moved)
        log_dict["filtered_l2_static/mean"] = _masked_mean(l2_full, valid & static)

    return total_loss, log_dict


def loss_fn(
    model,
    outputs,
    data_dict,
    training: bool,
    var_floor: float,
    var_ceiling: float,
    sim_domain_keywords: list[str],
    sim_var_const: float,
    uncertainty_logvar_weight: float,
):
    def _compute_loss():
        return internal_loss_fn(
            model,
            outputs,
            data_dict,
            training,
            var_floor=var_floor,
            var_ceiling=var_ceiling,
            sim_domain_keywords=sim_domain_keywords,
            sim_var_const=sim_var_const,
            uncertainty_logvar_weight=uncertainty_logvar_weight,
        )

    return safe_loss_computation(
        _compute_loss,
        f"loss computation (training={training})",
    )
