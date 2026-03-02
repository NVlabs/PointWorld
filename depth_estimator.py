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
import glob
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


class DepthEstimator:
    """
    Standalone depth estimator using Foundation Stereo model for stereo depth estimation.
    """
    
    def __init__(self, ckpt_path, baseline=None, intrinsic=None, device='cuda', cfg_path=None):
        """
        Initialize the depth estimator.
        
        Args:
            ckpt_path (str): Path to the Foundation Stereo checkpoint
            baseline (float, optional): Baseline distance between stereo cameras in meters
            intrinsic (np.ndarray, optional): Camera intrinsic matrix (3x3)
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.ckpt_path = ckpt_path
        self.baseline = baseline
        self.intrinsic = intrinsic
        self.device = device
        self.cfg_path = cfg_path
        self.model = None
        self.is_loaded = False
        
        # Default parameters
        self.use_hierarchical = False
        self.iters = 32
        
        self.load_model()
        print(f"DepthEstimator initialized with checkpoint: {ckpt_path}")
        
    def _resolve_ckpt_and_cfg(self):
        """Resolve checkpoint and cfg paths.

        Supports two env styles:
        - FOUNDATION_STEREO_CKPT points to a .pth file; cfg.yaml is expected next to it
        - FOUNDATION_STEREO_CKPT points to a directory that contains exactly one
          subdirectory with both cfg.yaml and a model .pth file (e.g., 23-51-11/)
        """
        if self.cfg_path is not None:
            if not os.path.exists(self.cfg_path):
                raise FileNotFoundError(f"FoundationStereo config file not found: {self.cfg_path}")
            if os.path.isdir(self.ckpt_path):
                raise ValueError(
                    "cfg_path override requires ckpt_path to be a .pth file, not a directory "
                    f"(got directory: {self.ckpt_path})"
                )
            if not os.path.isfile(self.ckpt_path):
                raise FileNotFoundError(f"FoundationStereo checkpoint not found: {self.ckpt_path}")
            return self.ckpt_path, self.cfg_path

        path = self.ckpt_path
        if os.path.isdir(path):
            # Search immediate subdirectories for (cfg.yaml, *.pth)
            matches = []
            for name in os.listdir(path):
                d = os.path.join(path, name)
                if not os.path.isdir(d):
                    continue
                cfg = os.path.join(d, "cfg.yaml")
                pths = sorted(glob.glob(os.path.join(d, "*.pth")))
                if os.path.exists(cfg) and len(pths) > 0:
                    # prefer model_best*.pth if present
                    best = [p for p in pths if os.path.basename(p).startswith("model_best")]
                    ckpt = best[0] if len(best) > 0 else pths[0]
                    matches.append((ckpt, cfg))
            assert len(matches) == 1, (
                f"FOUNDATION_STEREO_CKPT='{path}' is a directory; expected exactly one subdir with cfg.yaml and a .pth. Found {len(matches)}"
            )
            return matches[0]
        else:
            assert os.path.isfile(path), f"FoundationStereo checkpoint not found: {path}"
            cfg = os.path.join(os.path.dirname(path), "cfg.yaml")
            assert os.path.exists(cfg), f"FoundationStereo config file not found: {cfg}"
            return path, cfg

    def load_model(self):
        """Load the Foundation Stereo model."""
        if self.is_loaded:
            return
            
        # Import here to avoid import errors if FoundationStereo is not available
        import sys, os
        repo_root = os.path.dirname(os.path.abspath(__file__))
        fs_root = os.path.join(repo_root, "third_party", "FoundationStereo")
        if fs_root not in sys.path:
            sys.path.append(fs_root)
        from core.utils.utils import InputPadder
        from core.foundation_stereo import FoundationStereo
        self.InputPadder = InputPadder

        # Resolve checkpoint and configuration
        self.ckpt_path, cfg_path = self._resolve_ckpt_and_cfg()
        
        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            raise ValueError(
                f"FoundationStereo cfg missing vit_size: {cfg_path}. "
                "Provide a cfg with vit_size (e.g., vitl) via --foundation_stereo_cfg."
            )
        
        print(f"Loading FoundationStereo model from {self.ckpt_path}")
        
        # Initialize model with retry logic for rate limiting
        import time
        start_time = time.time()
        while time.time() - start_time < 600:  # Retry for up to 10 minutes
            try:
                self.model = FoundationStereo(cfg)
                break
            except Exception as e:
                if "rate limit exceeded" in str(e).lower() or \
                    "connection without response" in str(e).lower() or \
                    "too many requests" in str(e).lower():
                    print(f"Retrying due to rate limit or connection issue ({str(e)})...")
                    time.sleep(5)
                else:
                    raise e
        else:
            raise RuntimeError("FoundationStereo initialization failed after 1 minute due to rate limiting")
        
        # Load checkpoint
        ckpt = torch.load(self.ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        
        # Move to device and set to eval mode
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        
        self.is_loaded = True
        print(f"FoundationStereo model loaded successfully (step: {ckpt.get('global_step', 'unknown')}, epoch: {ckpt.get('epoch', 'unknown')})")

    
    
    def set_camera_params(self, baseline, intrinsic):
        """
        Set camera parameters for depth conversion.
        
        Args:
            baseline (float): Baseline distance between stereo cameras in meters
            intrinsic (np.ndarray): Camera intrinsic matrix (3x3)
        """
        self.baseline = baseline
        self.intrinsic = intrinsic
        print(f"Camera parameters set - baseline: {baseline:.4f}m, fx: {intrinsic[0,0]:.2f}")
    
    def set_inference_params(self, use_hierarchical=False, iters=32):
        """
        Set inference parameters.
        
        Args:
            use_hierarchical (bool): Whether to use hierarchical inference
            iters (int): Number of iterations for the model
        """
        self.use_hierarchical = use_hierarchical
        self.iters = iters
        print(f"Inference parameters set - hierarchical: {use_hierarchical}, iters: {iters}")
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def infer_depth(self, left_image, right_image):
        """
        Infer depth from stereo images.
        
        Args:
            left_image (np.ndarray): Left stereo image with shape (H, W) or (H, W, 1) for grayscale
            right_image (np.ndarray): Right stereo image with shape (H, W) or (H, W, 1) for grayscale
            
        Returns:
            np.ndarray: Depth map with shape (H, W)
        """
        # Load model if not loaded
        if not self.is_loaded:
            self.load_model()
        
        # Ensure images are in the right format
        left_image = self._prepare_image(left_image)
        right_image = self._prepare_image(right_image)
        
        # Convert to tensors and move to device
        device = next(self.model.parameters()).device
        left_tensor = torch.as_tensor(left_image).to(device).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        right_tensor = torch.as_tensor(right_image).to(device).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # Pad images to be divisible by 32
        padder = self.InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        left_tensor = left_tensor.contiguous()
        right_tensor = right_tensor.contiguous()
        
        # Run inference
        if self.use_hierarchical:
            disp = self.model.run_hierachical(
                left_tensor, right_tensor, iters=self.iters, test_mode=True, small_ratio=0.5
            )
        else:
            disp = self.model.forward(
                left_tensor, right_tensor, iters=self.iters, test_mode=True
            )
        
        # Unpad the result
        disp = padder.unpad(disp.float())  # [1, 1, H, W]
        disp = disp.squeeze()  # [H, W]
        
        # Convert disparity to depth if camera parameters are available
        assert self.baseline is not None and self.intrinsic is not None, "Camera parameters are not set"
        depth_np = self._disparity_to_depth(disp)
        
        return depth_np
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def infer_depth_batch(self, left_images, right_images, batch_size=32):
        """
        Infer depth from batches of stereo images.
        
        Args:
            left_images (list): List of left stereo images, each with shape (H, W, 3)
            right_images (list): List of right stereo images, each with shape (H, W, 3)
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Depth maps with shape (N, H, W) where N is the number of image pairs
        """
        # Load model if not loaded
        if not self.is_loaded:
            self.load_model()
        
        assert self.baseline is not None and self.intrinsic is not None, "Camera parameters are not set"
        
        num_frames = len(left_images)
        assert len(right_images) == num_frames, "Number of left and right images must match"
        
        depth_frames = []
        
        for i in tqdm(range(0, num_frames, batch_size), desc="Inferring depth"):
            batch_end = min(i + batch_size, num_frames)
            current_batch_size = batch_end - i
            
            # Get current batch
            left_batch = left_images[i:batch_end]
            right_batch = right_images[i:batch_end]
            
            # Process batch
            depth_np = self._infer_depth_batch_internal(left_batch, right_batch)
            
            # Append each depth map to the result list
            for j in range(current_batch_size):
                depth_frames.append(depth_np[j])
        
        return np.stack(depth_frames, axis=0)
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast(True)
    def _infer_depth_batch_internal(self, left_batch, right_batch):
        """
        Internal method to process a batch of stereo images.
        
        Args:
            left_batch (list): List of left images
            right_batch (list): List of right images
            
        Returns:
            np.ndarray: Batch of depth maps with shape (B, H, W)
        """
        device = next(self.model.parameters()).device
        right_np = np.stack(right_batch).astype(np.float32, copy=False)
        left_np = np.stack(left_batch).astype(np.float32, copy=False)
        right_tensor = torch.from_numpy(right_np).to(device=device).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        left_tensor = torch.from_numpy(left_np).to(device=device).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        current_batch_size = left_tensor.shape[0]

        # Pad images to be divisible by 32
        padder = self.InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        
        # Run inference on the entire batch
        if self.use_hierarchical:
            disp = self.model.run_hierachical(
                left_tensor, right_tensor, iters=self.iters, test_mode=True, small_ratio=0.5
            )
        else:
            disp = self.model.forward(
                left_tensor, right_tensor, iters=self.iters, test_mode=True
            )
        
        # Unpad the batch result
        disp = padder.unpad(disp.float())  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]
        
        # Process the entire batch on GPU
        # Create coordinate grid for the batch
        h, w = disp.shape[1:]
        xx = torch.arange(w, device=disp.device).view(1, 1, w).expand(current_batch_size, h, w)
        
        # Calculate right image coordinates
        us_right = xx - disp
        
        # Identify invalid points (not visible in both cameras)
        invalid = us_right < 0
        
        # Set invalid disparities to a very large value
        disp_valid = disp.clone()
        disp_valid[invalid] = float('inf')
        
        # Convert disparity to depth
        fx = torch.tensor(self.intrinsic[0, 0], device=disp.device)
        baseline_tensor = torch.tensor(self.baseline, device=disp.device)
        depth = fx * baseline_tensor / disp_valid
        
        # Handle invalid depth values
        depth[torch.isinf(depth) | torch.isnan(depth)] = 0.0
        
        # Transfer to CPU and convert to numpy
        depth_np = depth.cpu().numpy()

        return depth_np
    
    def _prepare_image(self, image):
        """
        Prepare image for inference (convert grayscale to RGB if needed).
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Prepared image with shape (H, W, 3)
        """
        if len(image.shape) == 2:
            # Grayscale image, convert to RGB by repeating channels
            image = np.stack([image, image, image], axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel image, convert to RGB
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Ensure image is in range [0, 255] and uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def _disparity_to_depth(self, disp):
        """
        Convert disparity to depth using camera parameters.
        
        Args:
            disp (torch.Tensor): Disparity map with shape (H, W)
            
        Returns:
            np.ndarray: Depth map with shape (H, W)
        """
        device = disp.device
        h, w = disp.shape
        
        # Create coordinate grid
        xx = torch.arange(w, device=device).view(1, w).expand(h, w)
        
        # Calculate right image coordinates
        us_right = xx - disp
        
        # Identify invalid points (not visible in both cameras)
        invalid = us_right < 0
        
        # Set invalid disparities to a very large value
        disp_valid = disp.clone()
        disp_valid[invalid] = float('inf')
        
        # Convert disparity to depth: depth = (fx * baseline) / disparity
        fx = torch.tensor(self.intrinsic[0, 0], device=device)
        baseline_tensor = torch.tensor(self.baseline, device=device)
        depth = fx * baseline_tensor / disp_valid
        
        # Handle invalid depth values
        depth[torch.isinf(depth) | torch.isnan(depth)] = 0.0
        
        # Transfer to CPU and convert to numpy
        depth_np = depth.cpu().numpy()
        
        return depth_np
