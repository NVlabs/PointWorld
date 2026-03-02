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
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from ptv3.ptv3 import PointTransformerV3, MLP
from scene_featurizer import SceneFeatureEncoder
from pointworld.embeddings import TemporalEmbedding
from pointworld import norm_stats as norm_stats_utils
from pointworld import losses as losses_utils
from pointworld import metrics as metrics_utils
from utils import handle_nan_outputs

UNCERTAINTY_LOGVAR_WEIGHT = 1.0
SIM_VAR_CONST = 1e-3
CONTEXT_HORIZON = 1
PRED_HORIZON = 10
VAR_FLOOR = 1e-6
VAR_CEILING = 1e2
SIM_DOMAIN_KEYWORDS = ["behavior"]
PTV3_DROP_PATH = 0.3
_PTV3_BLUEPRINT = None


def _load_ptv3_blueprint():
    global _PTV3_BLUEPRINT
    if _PTV3_BLUEPRINT is not None:
        return _PTV3_BLUEPRINT
    blueprint_path = Path(__file__).resolve().parent.parent / "ptv3" / "ptv3_arch.yaml"
    if not blueprint_path.is_file():
        raise FileNotFoundError(f"PTV3 blueprint not found: {blueprint_path}")
    with open(blueprint_path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "sizes" not in data:
        raise ValueError("ptv3_arch.yaml must contain a top-level 'sizes' mapping")
    sizes = data["sizes"]
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("ptv3_arch.yaml 'sizes' must be a non-empty mapping")
    _PTV3_BLUEPRINT = sizes
    return _PTV3_BLUEPRINT


def _prepare_log_var_for_confidence(log_var, domains):
    if log_var is None:
        raise RuntimeError("log_var is required to compute confidence.")
    if not isinstance(domains, (list, tuple)):
        raise TypeError("domains must be a list or tuple of domain strings.")
    if log_var.ndim != 4:
        raise ValueError(f"log_var must be 4D (B,T,Ns,C); got shape {tuple(log_var.shape)}")
    B = log_var.shape[0]
    if len(domains) != B:
        raise ValueError(f"Domain list length ({len(domains)}) must match batch size ({B}).")

    sim_b = torch.tensor(
        [any(k in dom for k in SIM_DOMAIN_KEYWORDS) for dom in domains],
        device=log_var.device,
        dtype=torch.bool,
    )
    if sim_b.any():
        sim_mask = sim_b.view(B, 1, 1, 1).expand_as(log_var)
        real_b = ~sim_b
        if real_b.any():
            s_min_tmp = math.log(VAR_FLOOR)
            s_max_tmp = math.log(VAR_CEILING)
            clamped = log_var.detach().clamp(min=s_min_tmp, max=s_max_tmp)
            real_logvar = clamped[real_b]
            real_var_mean = real_logvar.exp().mean().clamp(
                min=float(VAR_FLOOR), max=float(VAR_CEILING)
            )
            const_val = float(real_var_mean.log().item())
        else:
            const_val = math.log(SIM_VAR_CONST)
        const_tensor = torch.full_like(log_var, fill_value=const_val)
        log_var = torch.where(sim_mask, const_tensor, log_var)

    s_min = math.log(VAR_FLOOR)
    s_max = math.log(VAR_CEILING)
    return log_var.clamp(min=s_min, max=s_max)


def _resolve_arch_config(size: str) -> dict:
    sizes = _load_ptv3_blueprint()
    if size not in sizes:
        raise ValueError(f"Invalid model args.ptv3_size: {size}. Available: {sorted(sizes.keys())}")
    cfg = sizes[size]
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config for size '{size}': expected mapping")
    alias = cfg.get("alias_of")
    if alias is not None:
        if alias not in sizes:
            raise ValueError(f"Invalid alias_of '{alias}' for size '{size}'")
        cfg = sizes[alias]
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid alias config for size '{alias}': expected mapping")
    return cfg


def _require_list(cfg: dict, key: str) -> list:
    if key not in cfg:
        raise KeyError(f"PTV3 blueprint missing required field '{key}'")
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"PTV3 blueprint '{key}' must be a list")
    return value


def _resolve_channels(items: list, channels: int) -> tuple:
    out = []
    for item in items:
        if isinstance(item, str):
            if item != "channels":
                raise ValueError(f"Unsupported placeholder '{item}' in PTV3 blueprint")
            out.append(int(channels))
        else:
            out.append(int(item))
    return tuple(out)


def _resolve_patch_size(value, depth_len: int, patch_size: int) -> tuple:
    if value is None or value == "auto":
        return (int(patch_size),) * depth_len
    if isinstance(value, list):
        if len(value) != depth_len:
            raise ValueError("PTV3 patch size list must match depth length")
        return tuple(int(v) for v in value)
    raise TypeError("PTV3 patch size must be 'auto' or a list of ints")


def _resolve_stride(value, enc_depths: tuple) -> tuple:
    if value is None or value == "auto":
        return (2,) * (len(enc_depths) - 1)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    raise TypeError("PTV3 stride must be 'auto' or a list of ints")


def build_ptv3(channels, args):
    cfg = _resolve_arch_config(args.ptv3_size)
    if "channels_max" in cfg:
        assert channels <= int(cfg["channels_max"]), (
            f"it's recommended to use a decoder dimension of {cfg['channels_max']} or less"
        )
    if "channels_eq" in cfg:
        assert channels == int(cfg["channels_eq"]), (
            f"it's recommended to use a decoder dimension of {cfg['channels_eq']}"
        )

    enc_depths = tuple(_require_list(cfg, "enc_depths"))
    dec_depths = tuple(_require_list(cfg, "dec_depths"))
    enc_channels = _resolve_channels(_require_list(cfg, "enc_channels"), channels)
    dec_channels = _resolve_channels(_require_list(cfg, "dec_channels"), channels)
    enc_num_head = tuple(int(v) for v in _require_list(cfg, "enc_num_head"))
    dec_num_head = tuple(int(v) for v in _require_list(cfg, "dec_num_head"))
    stride = _resolve_stride(cfg.get("stride", "auto"), enc_depths)
    enc_patch_size = _resolve_patch_size(cfg.get("enc_patch_size", "auto"), len(enc_depths), args.ptv3_patch_size)
    dec_patch_size = _resolve_patch_size(cfg.get("dec_patch_size", "auto"), len(dec_depths), args.ptv3_patch_size)

    # Create the PointTransformerV3 with the configured parameters
    # attention registers are disabled in release configuration
    attn_cls = None
    attn_kwargs = {}

    return PointTransformerV3(
        in_channels=channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=enc_depths,
        enc_channels=enc_channels,
        enc_num_head=enc_num_head,
        enc_patch_size=enc_patch_size,
        dec_depths=dec_depths,
        dec_channels=dec_channels,
        dec_num_head=dec_num_head,
        dec_patch_size=dec_patch_size,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0,
        proj_drop=0,
        drop_path=PTV3_DROP_PATH,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        traceable=True,
        mask_token=False,
        freeze_encoder=False,
        enc_mode=False,
        attention_cls=attn_cls,
        attention_kwargs=attn_kwargs,
    )


class DynamicsPredictor(nn.Module):
    def __init__(self, args, in_channels, T):
        super().__init__()
        self.args, self.T, self.in_c = args, T, in_channels

        # Build the dynamics decoder based on size configuration
        self.predictor_model = build_ptv3(self.in_c, args)

        # Only predict T-1 timesteps since first timestep is input
        self.dynamics_head = nn.Sequential(MLP(in_channels, 128, 128), nn.Linear(128, 3 * (T-1)))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.dynamics_head[-1].weight)
            self.dynamics_head[-1].weight.mul_(args.dynamics_head_init_scale)
            self.dynamics_head[-1].bias.zero_()

        # Add heteroscedastic uncertainty head - only for T-1 steps
        self.u_dim = 1
        self.log_var_head = nn.Sequential(
            MLP(in_channels, 128, 128),
            nn.Linear(128, self.u_dim * (T-1))
        )
        with torch.no_grad():
            self.log_var_head[-1].bias.fill_(math.log(0.005 ** 2))

        # Skip connection FiLM parameters for scene features
        self.skip_film_gamma = nn.Parameter(torch.ones(1, 1, in_channels))
        self.skip_film_beta = nn.Parameter(torch.zeros(1, 1, in_channels))

        # Robot global summary FiLM parameters
        self.robot_film_gamma = nn.Parameter(torch.ones(1, 1, in_channels))
        self.robot_film_beta = nn.Parameter(torch.zeros(1, 1, in_channels))

        self.register_buffer("_grid_size", torch.tensor([args.grid_size]), persistent=False)

    def forward(self, scene_coord0, scene_feat0, scene_exists0, robot_coord_seq, robot_feat, robot_exists, normalize_fn=None, unnormalize_fn=None, training=True):
        B, Ns, _ = scene_coord0.shape
        T, Nr = robot_coord_seq.shape[1:3]
        device = scene_coord0.device

        coord = torch.cat([scene_coord0, robot_coord_seq.reshape(B, T*Nr, 3)], dim=1)  # (B, Ns+T*Nr, 3)
        feat  = torch.cat([scene_feat0,  robot_feat.reshape(B, T*Nr, robot_feat.shape[-1])], dim=1)  # (B, Ns+T*Nr, D)
        exists = torch.cat([scene_exists0, robot_exists.reshape(B, T*Nr)], dim=1)  # (B, Ns+T*Nr)
        is_robot = torch.cat([
            torch.zeros(B, Ns, 1, dtype=torch.bool, device=device),
            torch.ones (B, T*Nr,1, dtype=torch.bool, device=device)
        ], dim=1).flatten()  # (B*(Ns+T*Nr),)
        batch = torch.arange(B, device=device).repeat_interleave(Ns + T*Nr)  # (B*(Ns+T*Nr),)

        # Create data dictionary for transformer
        data_dict = {
            "coord": coord[exists],
            "feat" : feat[exists],
            "batch": batch[exists.flatten()],
            "is_robot": is_robot[exists.flatten()],
            "grid_size": self._grid_size,
        }

        point = self.predictor_model(data_dict)

        # Extract features with skip connection and robot global summary
        with torch.autocast('cuda', enabled=False):
            # Extract scene and robot features separately
            scene_mask = ~is_robot[exists.flatten()]
            robot_mask = is_robot[exists.flatten()]

            scene_feat_output = point.feat.new_zeros(B, Ns, self.in_c)  # (B,Ns,D)
            scene_feat_output[scene_exists0] = point.feat[scene_mask]  # (B,Ns,D)

            # Apply skip connection with FiLM modulation for input scene features
            skip_modulated = scene_feat0 * self.skip_film_gamma + self.skip_film_beta

            # Assert that we have robot features when not ignoring robot
            assert robot_mask.any(), "Expected robot features when not ignoring robot, but robot_mask is empty"

            # Create padded robot features tensor and fill with actual features
            padded_robot_features = torch.zeros(B, T*Nr, self.in_c, device=point.feat.device)  # (B, T*Nr, D)
            padded_robot_features[robot_exists.reshape(B, T*Nr)] = point.feat[robot_mask]  # Fill valid positions

            # Reshape to (B, T, Nr, D) for easier processing
            padded_robot_features = padded_robot_features.view(B, T, Nr, self.in_c)

            # Compute global robot summary for each batch
            robot_global_summaries = []
            for b in range(B):
                # Get valid robot features for this batch across all timesteps
                batch_robot_exists = robot_exists[b]  # (T, Nr)
                batch_robot_features = padded_robot_features[b]  # (T, Nr, D)

                # Extract only valid robot features
                valid_robot_features = batch_robot_features[batch_robot_exists]  # (N_valid_robot, D)

                if valid_robot_features.shape[0] > 0:
            # Max pooling across all valid robot points for this batch
                    batch_summary = valid_robot_features.max(dim=0, keepdim=True)[0]  # (1, D)
                else:
                    # If no robot features in this batch, use zero
                    batch_summary = torch.zeros(1, self.in_c, device=point.feat.device)

                robot_global_summaries.append(batch_summary)

            robot_global_summary = torch.stack(robot_global_summaries, dim=0)  # (B, 1, D)
            robot_global_summary = robot_global_summary.expand(-1, Ns, -1)  # (B, Ns, D)
            # Apply FiLM modulation for robot global summary
            robot_modulated = robot_global_summary * self.robot_film_gamma + self.robot_film_beta
            padded_scene_feat = scene_feat_output + skip_modulated + robot_modulated

            # dynamics head - now only for T-1 steps
            dynamics_head_output = self.dynamics_head(padded_scene_feat)  # (B,Ns,3*(T-1))
            dynamics_partial = dynamics_head_output.view(B, Ns, self.T-1, 3)  # (B,Ns,T-1,3)
            dynamics_partial = dynamics_partial.permute(0,2,1,3)  # (B,T-1,Ns,3)

            # Add zeros for first timestep
            zeros_first = torch.zeros(B, 1, Ns, 3, device=dynamics_partial.device, dtype=dynamics_partial.dtype)
            dynamics = torch.cat([zeros_first, dynamics_partial], dim=1)  # (B,T,Ns,3)

            # uncertainty head - now only for T-1 steps
            log_var_output = self.log_var_head(padded_scene_feat)  # (B,Ns,u_dim*(T-1))
            log_var_partial = log_var_output.view(B, Ns, self.T-1, self.u_dim)  # (B,Ns,T-1,1)
            log_var_partial = log_var_partial.permute(0,2,1,3)  # (B,T-1,Ns,1)

            zeros_first_var = torch.zeros(B, 1, Ns, self.u_dim, device=log_var_partial.device, dtype=log_var_partial.dtype)
            log_var = torch.cat([zeros_first_var, log_var_partial], dim=1)  # (B,T,Ns,1)

            outputs = {"pred": dynamics, "log_var": log_var}

        return outputs


class BaseModel(nn.Module):
    def __init__(self, args, data_info_dict, rank: int = 0, cpu_pg=None):
        super().__init__()
        self.args = args
        self.device = args.device
        self.data_info_dict = data_info_dict
        self.T = CONTEXT_HORIZON + PRED_HORIZON
        self.rank = rank
        self.world_size = dist.get_world_size() if args.distributed else 1
        self.channels = args.predictor_dim
        self.cpu_pg = cpu_pg
        # ---------------------------- scene encoder ---------------------------- #
        self.scene_feature_encoder = SceneFeatureEncoder(
            args,
            self.channels,
            data_info_dict,
            rank,
            self.normalize_scene_features,
        )

        self.robot_proj = MLP(data_info_dict['robot_features_dim'], self.channels, self.channels)
        self.time_embed = TemporalEmbedding(self.channels)
        self.robot_type_emb = nn.Parameter(torch.zeros(1, self.channels))
        self.register_buffer("time_steps", torch.linspace(0, 1, self.T))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.robot_type_emb)
            
        # ---------------------------- dynamics predictor ---------------------------- #
        self.dynamics_predictor = DynamicsPredictor(args, self.channels, self.T)

        # ---------------------------- normalization stats ---------------------------- #
        self._init_norm_stats()
        
        # Initialize consecutive NaN outputs counter
        self.consecutive_nan_outputs_count = 0
    
    def _init_norm_stats(self) -> None:
        stats = norm_stats_utils.load_norm_stats_from_json(
            self.args,
            self.device,
            self.T,
            var_floor=VAR_FLOOR,
        )
        self._domains = list(stats.domains)
        self._domain_to_index = dict(stats.domain_to_index)
        self.register_buffer("norm_stats_per_step_mean", stats.per_step_mean)  # (D,T,3)
        self.register_buffer("norm_stats_per_step_var", stats.per_step_var)    # (D,T,3)
        self.register_buffer("robot_norm_mean", stats.robot_mean)              # (D, Fr)
        self.register_buffer("robot_norm_var", stats.robot_var)                # (D, Fr)
        self.register_buffer("scene_norm_mean", stats.scene_mean)              # (D, Ds)
        self.register_buffer("scene_norm_var", stats.scene_var)                # (D, Ds)

    def normalize(self, og):
        return norm_stats_utils.normalize_output(
            og,
            self._current_domain_indices,
            self.norm_stats_per_step_mean,
            self.norm_stats_per_step_var,
        )

    def unnormalize(self, normalized):
        return norm_stats_utils.unnormalize_output(
            normalized,
            self._current_domain_indices,
            self.norm_stats_per_step_mean,
            self.norm_stats_per_step_var,
        )

    def normalize_robot_features(self, robot_features):
        return norm_stats_utils.normalize_robot_features(
            robot_features,
            self._current_domain_indices,
            self.robot_norm_mean,
            self.robot_norm_var,
        )
        
    def normalize_scene_features(self, scene_features):
        return norm_stats_utils.normalize_scene_features(
            scene_features,
            self._current_domain_indices,
            self.scene_norm_mean,
            self.scene_norm_var,
        )
    
    def encode_scene_features(self, data_dict):
        B = data_dict["scene_flows"].shape[0]
        if not hasattr(self, '_current_domain_indices'):
            batch_domains = data_dict['__domain__']
            assert isinstance(batch_domains, (list, tuple)), "data_dict['__domain__'] must be a list of domain strings"
            assert len(batch_domains) == B, f"Number of domains ({len(batch_domains)}) must match batch size ({B})"
            self._current_domain_indices = torch.tensor(
                [self._domain_to_index[dom] for dom in batch_domains],
                device=self.device, dtype=torch.long
            )

        return self.scene_feature_encoder(data_dict)

    # ------------------------------------------------------------------ #
    def forward(self, data_dict, training=True, encoded_scene_feat0=None):
        for name, tensor in data_dict.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), f"{name} has NaNs"
        # ------------------------- unpack data ---------------------------- #
        # unpack data
        scene_coord0 = data_dict["scene_flows"][:, 0]  # (B,Ns,3)
        scene_feat0  = data_dict["scene_features"][:, 0]  # (B,Ns,Ds)
        scene_exists0 = data_dict["scene_exists"][:, 0]  # (B,Ns)
        robot_coord_seq = data_dict["robot_flows"]  # (B,T,Nr,3)
        robot_feat_seq  = data_dict["robot_features"]  # (B,T,Nr,Fr)
        robot_exists = data_dict["robot_exists"]  # (B,T,Nr)
        # shapes
        B, Ns, _ = scene_coord0.shape
        _, T, Nr, _ = robot_coord_seq.shape

        # ---------------------------- set batch domain indices ---------------------------- #
        batch_domains = data_dict['__domain__']
        assert isinstance(batch_domains, (list, tuple)), "data_dict['__domain__'] must be a list of domain strings"
        assert len(batch_domains) == B, f"Number of domains ({len(batch_domains)}) must match batch size ({B})"
        for dom in batch_domains:
            assert dom in self._domain_to_index, f"Unknown domain '{dom}' in batch."
        self._current_domain_indices = torch.tensor(
            [self._domain_to_index[dom] for dom in batch_domains],
            device=self.device, dtype=torch.long
        )
        
        # ---------------------------- scene encoder ---------------------------- #
        if encoded_scene_feat0 is not None:
            scene_feat0 = encoded_scene_feat0
        else:
            scene_feat0 = self.encode_scene_features(data_dict)
        
        # ---------------------------- robot features ---------------------------- #
        robot_feat_seq = self.normalize_robot_features(robot_feat_seq)
        with torch.autocast('cuda', enabled=False):  # TODO: see if this resolves nan issue
            robot_raw = self.robot_proj(robot_feat_seq)  # (B, T, Nr, C)
            time_emb = self.time_embed(self.time_steps.view(1, T)).unsqueeze(2)  # (1, T, 1, C)
            time_emb = time_emb.expand(B, T, Nr, -1)  # (B, T, Nr, C)
            robot_feat = robot_raw + time_emb + self.robot_type_emb.view(1, 1, 1, -1)  # (B, T, Nr, C)

        out = self.dynamics_predictor(
            scene_coord0, scene_feat0, scene_exists0,
            robot_coord_seq, robot_feat, robot_exists,
            normalize_fn=self.normalize,
            unnormalize_fn=self.unnormalize,
            training=training
        )

        pred_norm = out["pred"]
        pred = self.unnormalize(pred_norm)  # (B,T,Ns,3)
        out["scene_relative_norm"] = pred_norm
        out["scene_relative"] = pred
        out["scene_flows"] = scene_coord0.unsqueeze(1) + pred  # (B,T,Ns,3)
        
        # Check outputs for NaN values with consecutive counting logic
        should_skip_batch, self.consecutive_nan_outputs_count = handle_nan_outputs(
            out, 
            self.consecutive_nan_outputs_count, 
            context=f"forward pass (batch size: {B})"
        )
        
        # In distributed training, synchronize the skip decision across all ranks
        # to prevent NCCL timeout due to ranks being out of sync
        if self.args.distributed:
            import torch.distributed as dist
            # Convert skip decision to tensor for all_reduce
            skip_tensor = torch.tensor([1 if should_skip_batch else 0], 
                                     device=self.device, dtype=torch.int)
            # Use all_reduce with MAX to ensure if any rank wants to skip, all skip
            dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX, group=self.cpu_pg)
            should_skip_batch = skip_tensor.item() > 0
        
        # Mark whether this output should be skipped due to NaN
        out['_has_nan_outputs'] = should_skip_batch

        log_var_vis = out.get("log_var")
        if log_var_vis is None:
            raise RuntimeError("Model outputs missing log_var; cannot derive confidence.")
        log_var_vis = _prepare_log_var_for_confidence(log_var_vis, data_dict["__domain__"])
        var = torch.exp(log_var_vis)
        if var.shape[-1] == 1:
            var_scalar = var.squeeze(-1)
        else:
            var_scalar = var.mean(dim=-1)
        conf = 1.0 - (var_scalar - VAR_FLOOR) / max(1e-12, (VAR_CEILING - VAR_FLOOR))
        out["confidence"] = conf

        return out

    # ------------------------------------------------------------------ #
    #  Loss function                                                     #
    # ------------------------------------------------------------------ #
    def _make_nan_dict(self, template_keys):
        return metrics_utils.make_nan_dict(template_keys)

    @torch.no_grad()
    def _collect_metrics(
        self,
        per_point_loss,            # (B,T,NS)
        weights,                      # (B,T,NS)
        moved_mask, static_mask,      # (B,T,NS)
        pred_exists_supervised,       # (B,T,NS)
        l2,                           # (B,T,NS)
        log_var,                      # Tensor (B,T,NS,[1|3])
        pred_norm=None                # Optional tensor (B,T,NS,3)
    ):
        return metrics_utils.collect_metrics(
            self.args,
            per_point_loss,
            weights,
            moved_mask,
            static_mask,
            pred_exists_supervised,
            l2,
            log_var,
            pred_norm,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
        )

    def _compute_single_output_loss(self, output_norm, gt_target_norm, weights, pred_exists_supervised,
                              log_var):
        return losses_utils.compute_single_output_loss(
            self.args,
            output_norm,
            gt_target_norm,
            weights,
            pred_exists_supervised,
            log_var,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )
    
    @torch.no_grad()
    def _compute_single_output_metrics(self, output_flows, gt_flows, per_point_loss,
                                    weights, moved, static, pred_exists_supervised,
                                    log_var, pred_norm=None):
        return losses_utils.compute_single_output_metrics(
            self.args,
            output_flows,
            gt_flows,
            per_point_loss,
            weights,
            moved,
            static,
            pred_exists_supervised,
            log_var,
            pred_norm,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
        )
        
    def loss_fn(self, outputs, data_dict, training=True):
        return losses_utils.loss_fn(
            self,
            outputs,
            data_dict,
            training,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            sim_domain_keywords=SIM_DOMAIN_KEYWORDS,
            sim_var_const=SIM_VAR_CONST,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )
    
    def _internal_loss_fn(self, outputs, data_dict, training=True):
        return losses_utils.internal_loss_fn(
            self,
            outputs,
            data_dict,
            training,
            var_floor=VAR_FLOOR,
            var_ceiling=VAR_CEILING,
            sim_domain_keywords=SIM_DOMAIN_KEYWORDS,
            sim_var_const=SIM_VAR_CONST,
            uncertainty_logvar_weight=UNCERTAINTY_LOGVAR_WEIGHT,
        )
    
    @torch.no_grad()
    def _collect_per_domain_metrics(self, 
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
                                    pred_norm):
        return metrics_utils.collect_per_domain_metrics(
            self.args,
            self.device,
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
            self._compute_single_output_metrics,
        )
