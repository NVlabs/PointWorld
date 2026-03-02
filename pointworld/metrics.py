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

import torch


def make_nan_dict(template_keys):
    return {k: float("nan") for k in template_keys}


@torch.no_grad()
def collect_metrics(
    args,
    per_point_loss,
    weights,
    moved_mask,
    static_mask,
    pred_exists_supervised,
    l2,
    log_var,
    pred_norm,
    var_floor: float,
    var_ceiling: float,
):
    assert pred_norm is not None
    metrics = {}

    # ------------------------------------------------------------------ #
    #  Basic counts                                                      #
    # ------------------------------------------------------------------ #
    num_valid_supervision = pred_exists_supervised.sum().float()
    num_moved = (moved_mask & pred_exists_supervised).sum().float()
    num_static = (static_mask & pred_exists_supervised).sum().float()

    # ------------------------------------------------------------------ #
    #  Loss values                                                       #
    # ------------------------------------------------------------------ #
    dyn_loss = (per_point_loss * weights).sum()

    metrics["dynamics_loss"] = dyn_loss.detach().cpu().item()

    # total loss
    metrics["total_loss"] = dyn_loss.detach().cpu().item()

    # ------------------------------------------------------------------ #
    #  L2 position errors and prediction magnitude statistics            #
    # ------------------------------------------------------------------ #
    def _safe_stat(tensor, mask, name, max_min=False):
        if mask.sum() > 0:
            metrics[f"{name}/mean"] = tensor[mask].mean().item()
            if max_min:
                metrics[f"{name}/max"] = tensor[mask].max().item()
                metrics[f"{name}/min"] = tensor[mask].min().item()
        else:
            metrics[f"{name}/mean"] = float("nan")
            if max_min:
                metrics[f"{name}/max"] = float("nan")
                metrics[f"{name}/min"] = float("nan")

    _safe_stat(l2, moved_mask & pred_exists_supervised, "l2_moved")
    _safe_stat(l2, static_mask & pred_exists_supervised, "l2_static")
    _safe_stat(l2, pred_exists_supervised, "l2")
    _safe_stat(pred_norm, pred_exists_supervised, "pred", max_min=True)
    _safe_stat(pred_norm, pred_exists_supervised & moved_mask, "pred_moved", max_min=True)
    _safe_stat(pred_norm, pred_exists_supervised & static_mask, "pred_static", max_min=True)

    # ------------------------------------------------------------------ #
    #  Uncertainty (always-on)                                           #
    # ------------------------------------------------------------------ #
    var_flat = torch.exp(log_var).view(-1, log_var.shape[-1])
    mask_flat = pred_exists_supervised.view(-1)
    if mask_flat.any():
        valid_var = var_flat[mask_flat]
        metrics["uncertainty/mean"] = valid_var.mean().item()
    else:
        metrics["uncertainty/mean"] = float("nan")

    # Confidence stats (independent of threshold)
    var_scalar = var_flat.mean(dim=-1) if var_flat.shape[-1] > 1 else var_flat.squeeze(-1)
    conf_flat = 1.0 - (var_scalar - var_floor) / max(1e-12, (var_ceiling - var_floor))
    conf_flat = conf_flat.clamp(0.0, 1.0)
    if mask_flat.any():
        valid_conf = conf_flat[mask_flat]
        metrics["confidence/mean"] = valid_conf.mean().item()
        metrics["confidence/std"] = valid_conf.std(unbiased=False).item()
        metrics["confidence/max"] = valid_conf.max().item()
        metrics["confidence/min"] = valid_conf.min().item()
    else:
        metrics["confidence/mean"] = float("nan")
        metrics["confidence/std"] = float("nan")
        metrics["confidence/max"] = float("nan")
        metrics["confidence/min"] = float("nan")

    # Uncertainty-aware L2 metrics: percentile-based mask on confidence
    conf = conf_flat.view_as(pred_exists_supervised)
    if mask_flat.any():
        valid_conf = conf_flat[mask_flat]
        keep_frac = float(args.confidence_thres)
        keep_frac = min(max(keep_frac, 0.0), 1.0)
        thr = torch.quantile(valid_conf, q=max(0.0, 1.0 - keep_frac))
        conf_mask = conf >= thr
    else:
        conf_mask = torch.zeros_like(conf, dtype=torch.bool)

    _safe_stat(l2, moved_mask & pred_exists_supervised & conf_mask, "l2_moved_conf")
    _safe_stat(l2, static_mask & pred_exists_supervised & conf_mask, "l2_static_conf")
    _safe_stat(l2, pred_exists_supervised & conf_mask, "l2_conf")

    # ------------------------------------------------------------------ #
    #  Weight statistics                                                 #
    # ------------------------------------------------------------------ #
    metrics["weights/mean"] = weights.mean().item()
    metrics["weights/sum"] = weights.sum().item()

    #  Basic counts for visibility
    metrics["moved_percentage"] = num_moved.item() / num_valid_supervision.item()
    metrics["static_percentage"] = num_static.item() / num_valid_supervision.item()
    metrics["supervised_percentage"] = num_valid_supervision.item() / l2.numel()

    return metrics


@torch.no_grad()
def collect_per_domain_metrics(
    args,
    device,
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
    pred_norm,
    compute_single_output_metrics_fn,
):
    for d in args.domains:
        mask_b = torch.tensor(
            [dom == d for dom in data_dict["__domain__"]],
            dtype=torch.bool,
            device=device,
        )
        if mask_b.any():
            idx = mask_b.nonzero(as_tuple=True)[0]

            log_var_dom = log_var[idx]
            dom_metrics = compute_single_output_metrics_fn(
                output_scene_flows[idx],
                gt_scene_flows[idx],
                per_point_loss[idx],
                weights[idx],
                moved[idx],
                static[idx],
                pred_exists_supervised[idx],
                log_var_dom,
                pred_norm[idx],
            )
        else:
            dom_metrics = make_nan_dict(metric_keys)

        log_dict.update({f"{d}/{k}": v for k, v in dom_metrics.items()})
    return log_dict
