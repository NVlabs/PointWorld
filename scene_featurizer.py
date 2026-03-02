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

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneEncoder2D(nn.Module):
    """DINOv3-based 2-D scene encoder that projects 3-D points to camera views."""
    def __init__(self, args, channels, data_info_dict, rank: int = 0):
        super().__init__()
        self.args = args
        self.rank = rank
        self.device = args.device
        self.channels = channels
        self.repo_root = Path(__file__).resolve().parent
        
        # Fixed DINOv3 ViT-L16 multi-layer configuration for release.
        self.model_name = "dinov3_vitl16"
        repo, source = self._resolve_dinov3_repo()
        self.model_config = {
            'feat_dim': 1024,
            'repo': repo,
            'source': source,
            'weights': self._resolve_dinov3_weights('dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
        }
        self.feat_dim = self.model_config['feat_dim']

        self.selected_layers = [4, 11, 17, 23]
        in_dim = self.feat_dim * len(self.selected_layers)
        
        # ---------- backbone ----------
        self._load_backbone()
        
        # --- projection head (now uses in_dim) ------------------------
        self.feat_proj = nn.Linear(in_dim, channels)
        self._freeze_encoder()
        # Conditionally enable compile on the small wrapper only if not disabled
        if not args.disable_compile:
            self._maybe_no_grad = torch.compile(self._maybe_no_grad)
    
    def _load_backbone(self):
        """Load the fixed DINOv3 backbone."""
        model_name = self.model_name
        config = self.model_config

        if self.args.distributed:
            self._load_model_distributed(lambda: self._load_dinov3_model(config, model_name))
        else:
            self.dinov3 = self._load_dinov3_model(config, model_name)

    # ------------------------------------------------------------------ helpers
    def _resolve_dinov3_repo(self):
        repo_path = self.repo_root / "third_party" / "dinov3"
        if not repo_path.exists():
            raise ValueError(
                "DINOv3 submodule not found at third_party/dinov3. "
                "Run `git submodule update --init --recursive` to fetch it."
            )
        hubconf = repo_path / "hubconf.py"
        if not hubconf.exists():
            raise ValueError(
                f"DINOv3 submodule is missing hubconf.py at {hubconf}. "
                "Reinitialize the submodule."
            )
        return str(repo_path), 'local'

    def _resolve_dinov3_weights(self, checkpoint_filename):
        checkpoints_dir = self.repo_root / "third_party" / "dinov3" / "checkpoints"
        weights_path = checkpoints_dir / checkpoint_filename
        if weights_path.exists():
            return str(weights_path)

        matches = sorted(checkpoints_dir.glob("dinov3_vitl16_pretrain*.pth"))
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise ValueError(
                "Found multiple DINOv3 checkpoints matching 'dinov3_vitl16_pretrain*.pth'. "
                "Keep only one file in third_party/dinov3/checkpoints or rename to "
                f"{checkpoint_filename}."
            )
        raise ValueError(
            "DINOv3 weights not found. Download the checkpoint to "
            f"{weights_path} or place a single file matching 'dinov3_vitl16_pretrain*.pth' "
            "under third_party/dinov3/checkpoints."
        )
    
    def _load_model_distributed(self, load_func):
        """Helper for distributed model loading with synchronization."""
        import torch.distributed as dist
        if self.rank == 0:
            # Only rank 0 loads the model first
            self.dinov3 = load_func()
        # Synchronize all ranks to ensure loading is complete
        dist.barrier()
        if self.rank != 0:
            # Other ranks can now safely load
            self.dinov3 = load_func()
    
    def _load_dinov3_model(self, config, model_name):
        """Load DINOv3 model via torch.hub."""
        return torch.hub.load(
            config['repo'], 
            model_name, 
            source=config['source'],
            weights=config['weights'],
            trust_repo=True
        )
        
    def _freeze_encoder(self):
        for p in self.dinov3.parameters():
            p.requires_grad_(False)
        self.dinov3.eval()

    @staticmethod
    def _stack_camera_tensors(camera_data, key):
        """camera_data[prefix][key] -> (B, C, ...)"""
        cams = sorted(camera_data.keys())        # stable order
        tensors = [camera_data[c][key] for c in cams]  # list[(B, ...)]
        return torch.stack(tensors, dim=1)       # (B, C, ...)

    # ------------------------------------------------------------------ helpers

    # ------------------------------------------------------------------ forward
    # Note: AMP context is controlled by callers (e.g., train.py).
    # Avoid forcing autocast here so we respect the global AMP settings.
    def forward(self, scene_coord, scene_exists, camera_data):
        """
        Optimized forward method that directly samples from patch features without interpolation.
        
        Args
        ----
        scene_coord : (B, Ns, 3)
        scene_exists: (B, Ns) bool
        camera_data : output of _extract_camera_data (same as before)
        Returns
        -------
        scene_features : (B, Ns, channels) - always returned
        """
        B, Ns, _ = scene_coord.shape
        device = self.device
        cams = sorted(camera_data.keys())
        C = len(cams)

        # ============ 1.  Gather camera tensors ============
        rgb   = self._stack_camera_tensors(camera_data, 'rgb')        # (B,C,H,W,3)  uint8
        depth = self._stack_camera_tensors(camera_data, 'depth')      # (B,C,H,W)    float32
        intr  = self._stack_camera_tensors(camera_data, 'intrinsic')  # (B,C,3,3)
        extr  = self._stack_camera_tensors(camera_data, 'extrinsic')  # (B,C,4,4)
        if all('exists' in camera_data[c] for c in cams):
            cam_exists = self._stack_camera_tensors(camera_data, 'exists').bool()  # (B,C)
        else:
            cam_exists = torch.ones((B, C), dtype=torch.bool, device=device)
        _, _, H, W, _ = rgb.shape

        # ============ 2.  Backbone features for all views (no chunking) ============
        # DINOv3 ViT-L16 path (fixed for release).
        rgb_flat = rgb.view(B * C, H, W, 3)  # (B*C, H, W, 3)
        rgb_input = rgb_flat.permute(0, 3, 1, 2).float() / 255.  # (B*C, 3, H, W)
        patch_size = 16
        patch_h = H // patch_size
        patch_w = W // patch_size
        rgb_backbone_input = rgb_input  # keep original resolution
        
        # Apply ImageNet normalization (fixed for release).
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_backbone_input.device, dtype=rgb_backbone_input.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_backbone_input.device, dtype=rgb_backbone_input.dtype).view(1, 3, 1, 1)
        rgb_backbone_input = (rgb_backbone_input - mean) / std
        
        def _extract_features():
            feats = self.dinov3.get_intermediate_layers(
                rgb_backbone_input,              # (B*C, 3, H, W)
                n=self.selected_layers,          # list of ints as keyword arg
                reshape=False,                   # keep (B*C, tokens, C)
                return_class_token=False)        # only patch tokens
            patch_tokens = torch.cat(feats, dim=-1)
            # Guard against register tokens: keep only trailing patch tokens if present
            n_patches = patch_h * patch_w
            if patch_tokens.shape[1] != n_patches:
                patch_tokens = patch_tokens[:, -n_patches:, :]
            return patch_tokens

        features_or_tokens = self._maybe_no_grad(_extract_features)
        
        patch_tokens = features_or_tokens
        assert patch_h * patch_w == patch_tokens.shape[1], \
            f"Token grid mismatch: {patch_h}x{patch_w} != {patch_tokens.shape[1]} tokens"
        patch_tokens_grid = patch_tokens.view(B * C, patch_h, patch_w, patch_tokens.shape[-1])  # (B*C, patch_h, patch_w, feat_dim)

        # ============ 3.  Project 3-D points to each camera ============
        ones = torch.ones((B, Ns, 1), device=device, dtype=scene_coord.dtype)
        pts_h = torch.cat([scene_coord, ones], dim=-1)                      # (B,Ns,4)
        pts_h = pts_h.unsqueeze(1).expand(-1, C, -1, -1)                    # (B,C,Ns,4)

        # world -> cam and cam -> pixels in FP32 to avoid FP16 overflow/inf
        with torch.autocast('cuda', enabled=False):
            world2cam = extr.float()  # (B,C,4,4)
            intr_f    = intr.float()  # (B,C,3,3)
            pts_h_f   = pts_h.float() # (B,C,Ns,4)

            pts_cam = torch.matmul(world2cam, pts_h_f.transpose(-2, -1)).transpose(-2, -1)[..., :3]  # (B,C,Ns,3)
            pix_h   = torch.matmul(intr_f, pts_cam.transpose(-2, -1)).transpose(-2, -1)              # (B,C,Ns,3)

            z = pix_h[..., 2:3]  # (B,C,Ns,1)
            eps = 1e-6
            valid_z = torch.isfinite(z) & (z.abs() > eps)
            # Safe division for pixels; invalid z will be handled by visibility mask
            safe_z = torch.where(valid_z, z, torch.ones_like(z))
            pixels = pix_h[..., :2] / safe_z  # (B,C,Ns,2)
            # Use zero depth for invalid entries to keep proj_depth finite
            z_safe = torch.where(valid_z, z, torch.zeros_like(z))
            proj_depth = z_safe.squeeze(-1)  # (B,C,Ns)
        assert torch.isfinite(proj_depth).all(), "bad depth"

        # ============ 4.  Visibility & depth consistency ============
        # (a) bounds & positive-depth
        in_img = (
            (pixels[..., 0] >= 0) & (pixels[..., 0] < W) &
            (pixels[..., 1] >= 0) & (pixels[..., 1] < H) &
            (proj_depth > 0) &
            valid_z.squeeze(-1)
        )                                                                   # (B,C,Ns)

        # (b) sample depth image at projected pixels
        norm_xy_depth = torch.stack([
            2 * pixels[..., 0] / (W - 1) - 1,
            2 * pixels[..., 1] / (H - 1) - 1
        ], dim=-1)                                                      # (B,C,Ns,2)
        depth_imgs = depth.unsqueeze(2)  # (B,C,1,H,W)
        # Ensure contiguity to avoid cuDNN non-supported errors
        depth_samp = F.grid_sample(
            depth_imgs.view(B * C, 1, H, W),  # (B*C, 1, H, W)
            norm_xy_depth.view(B * C, Ns, 1, 2),  # (B*C, Ns, 1, 2)
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(-1).squeeze(-1)  # (B*C, Ns)
        depth_samp = depth_samp.view(B, C, Ns)

        # two-sided depth threshold with imbalanced margins
        thr_behind = self.args.depth_threshold
        thr_front  = 0.5 * self.args.depth_threshold
        depth_ok = (proj_depth <= depth_samp + thr_behind) & \
                (proj_depth >= depth_samp - thr_front)
        
        visible  = in_img & depth_ok                                        # (B,C,Ns)
        visible = visible & cam_exists.unsqueeze(-1)

        # ============ 5. Feature sampling (simplified with shared normalized grid) ============
        pixels_flat = pixels.view(B * C, Ns, 2)  # (B*C, Ns, 2)
        visible_flat = visible.view(B * C, Ns)    # (B*C, Ns)
        
        # ViT feature grid: build directly in token coordinates (no-resize path)
        ps = float(patch_size)
        # Effective image region that has tokens
        W_eff = float(patch_w * patch_size)
        H_eff = float(patch_h * patch_size)
        # Clamp to effective region to keep grid_sample finite; exclude via mask below
        x_eff = pixels[..., 0].clamp(0.0, max(W_eff - 1.0, 0.0))
        y_eff = pixels[..., 1].clamp(0.0, max(H_eff - 1.0, 0.0))
        # centers-correct mapping (token centers at (i+0.5)*ps)
        u = x_eff / ps - 0.5
        v = y_eff / ps - 0.5
        # Clamp center indices to valid token index domain
        u = u.clamp(0.0, max(patch_w - 1.0, 0.0))
        v = v.clamp(0.0, max(patch_h - 1.0, 0.0))
        # Normalize to [-1,1] with align_corners=True on (patch_h, patch_w)
        norm_x = 2.0 * u / max(patch_w - 1, 1) - 1.0
        norm_y = 2.0 * v / max(patch_h - 1, 1) - 1.0
        norm_xy_feat = torch.stack([norm_x, norm_y], dim=-1)
        grid = norm_xy_feat.view(B * C, Ns, 1, 2).float()  # (B*C, Ns, 1, 2)
        # Sanitize grid to avoid NaNs/Infs propagating through grid_sample.
        # Send invalid or non-visible locations out of bounds so padding returns zeros.
        grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
        # Build a feature-coverage mask for ViT to exclude pixels that fall in cropped bands
        in_feat_x = (pixels[..., 0] >= 0.0) & (pixels[..., 0] < W_eff)
        in_feat_y = (pixels[..., 1] >= 0.0) & (pixels[..., 1] < H_eff)
        in_feat = in_feat_x & in_feat_y  # (B,C,Ns)
        valid_sample = (visible & in_feat)
        valid_sample_flat = valid_sample.view(B * C, Ns)
        if not torch.all(valid_sample_flat):
            grid = torch.where(
                valid_sample_flat.view(B * C, Ns, 1, 1),
                grid,
                torch.full_like(grid, 2.0)
            )

        # ViT path: sample from patch tokens grid using the shared normalized grid
        feats_grid = F.grid_sample(
            patch_tokens_grid.permute(0, 3, 1, 2),  # (B*C, feat_dim, patch_h, patch_w)
            grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # (B*C, Ns, feat_dim)
        feats_grid_masked = feats_grid * valid_sample_flat.unsqueeze(-1)

        # ============ 6.  Aggregate across cameras BEFORE projection (memory efficient & equivalent) ============
        feats_grid_masked_reshaped = feats_grid_masked.view(B, C, Ns, -1)   # (B,C,Ns,in_dim)
        feat_sum_dino = feats_grid_masked_reshaped.sum(dim=1)               # (B,Ns,in_dim)
        cam_counts = valid_sample.sum(dim=1).clamp(min=1).unsqueeze(-1)     # (B,Ns,1) avoid /0
        aggregated_features = feat_sum_dino / cam_counts                    # (B,Ns,in_dim)

        # project down to `channels` AFTER aggregation (linear -> identical to avg of per-cam projections)
        scene_features = self.feat_proj(aggregated_features)                # (B,Ns,channels)

        # zero-out nonexistent points
        scene_features[~scene_exists] = 0.0

        return scene_features

    # --------------------------------------------------------------------------
    def _maybe_no_grad(self, fn):
        with torch.no_grad():
            return fn()
    
    # --------------------------------------------------------------------------
    # API compatibility methods for profiling and debugging
    # --------------------------------------------------------------------------
    def _extract_camera_data(self, data_dict):
        """Extract camera data from data_dict by finding matching keys with standardized prefixes."""
        camera_data = {}
        
        # Find all standardized camera prefixes (cam0, cam1, cam2, etc.)
        rgb_keys = [k for k in data_dict.keys() if k.endswith('_initial_rgb')]
        
        # Extract standardized prefixes (should be cam0, cam1, etc.)
        prefixes = set()
        for k in rgb_keys:
            prefix = k.replace('_initial_rgb', '')
            if prefix.startswith('cam'):  # Only accept standardized camera prefixes
                prefixes.add(prefix)
        
        for prefix in prefixes:
            rgb_key = f"{prefix}_initial_rgb"
            depth_key = f"{prefix}_initial_depth"
            intrinsic_key = f"{prefix}_intrinsic"
            extrinsic_key = f"{prefix}_extrinsic"
            exists_key = f"{prefix}_exists"
            
            if all(k in data_dict for k in [rgb_key, depth_key, intrinsic_key, extrinsic_key]):
                entry = {
                    'rgb': data_dict[rgb_key],
                    'depth': data_dict[depth_key],
                    'intrinsic': data_dict[intrinsic_key],
                    'extrinsic': data_dict[extrinsic_key]
                }
                if exists_key in data_dict:
                    entry['exists'] = data_dict[exists_key]
                camera_data[prefix] = entry
        
        return camera_data
    


class SceneFeatureEncoder(nn.Module):
    def __init__(self, args, channels, data_info_dict, rank, normalize_scene_features_fn):
        super().__init__()
        self.args = args
        self.normalize_scene_features = normalize_scene_features_fn
        self.scene_encoder = SceneEncoder2D(args, channels, data_info_dict, rank)
        self.scene_raw_feat_proj = nn.Linear(data_info_dict["scene_features_dim"], channels)
        self.scene_encoder_norm = nn.LayerNorm(channels)
        self.scene_raw_norm = nn.LayerNorm(channels)
        self.scene_proj = nn.Linear(channels * 2, channels)

    def forward(self, data_dict):
        scene_coord0 = data_dict["scene_flows"][:, 0]  # (B, Ns, 3)
        scene_feat0 = data_dict["scene_features"][:, 0]  # (B, Ns, Ds)
        scene_exists0 = data_dict["scene_exists"][:, 0]  # (B, Ns)

        scene_feat0 = self.normalize_scene_features(scene_feat0)

        camera_data = self.scene_encoder._extract_camera_data(data_dict)
        backbone_scene_feat0 = self.scene_encoder(scene_coord0, scene_exists0, camera_data)
        backbone_scene_feat0 = backbone_scene_feat0.to(scene_feat0.dtype)
        scene_feat0 = torch.cat(
            [
                self.scene_encoder_norm(backbone_scene_feat0),
                self.scene_raw_norm(self.scene_raw_feat_proj(scene_feat0)),
            ],
            dim=-1,
        )
        return self.scene_proj(scene_feat0)
