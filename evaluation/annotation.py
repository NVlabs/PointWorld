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
from datetime import datetime
import numpy as np
import torch
import h5py
from tqdm import tqdm


class ConfidenceHelper:
    def __init__(self, args, device, domain_to_dir, sim_keywords, build_eval_loader, model):
        self.args = args
        self.device = device
        self.domain_to_dir = domain_to_dir
        self._sim_keywords = list(sim_keywords)
        self._build_eval_loader = build_eval_loader
        self.model = model
        self._low_conf_cache = {}

    def is_sim_domain(self, domain: str) -> bool:
        candidates = {str(domain), str(domain).split("+", 1)[0]}
        for cand in candidates:
            for kw in self._sim_keywords:
                if kw and kw in cand:
                    return True
        return False

    def open_conf_files(self, split, mode):
        files = {}
        for dom, base in self.domain_to_dir.items():
            split_dir = os.path.join(base, split)
            os.makedirs(split_dir, exist_ok=True)
            path = os.path.join(split_dir, f"expert_confidence-seed={self.args.seed}.h5")
            if mode == "r":
                files[dom] = h5py.File(path, "r") if os.path.exists(path) else None
            else:
                try:
                    files[dom] = h5py.File(path, "a")
                except OSError as e:
                    print(f"Warning: failed to open {path} in append mode ({e}); backing up and recreating.")
                    if os.path.exists(path):
                        import shutil
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup = path + f".broken-{timestamp}"
                        try:
                            shutil.move(path, backup)
                            print(f"Backed up broken file to {backup}")
                        except Exception as be:
                            print(f"Warning: failed to backup broken file {path}: {be}")
                    files[dom] = h5py.File(path, "w")
                files[dom].attrs["seed"] = int(self.args.seed)
                files[dom].attrs["dataset_dir"] = str(base)
                files[dom].attrs["split"] = str(split)
        return files

    def close_conf_files(self, files):
        for f in files.values():
            if f is not None:
                f.close()

    def _resolve_conf_group(self, file_handle, key, create=False):
        if file_handle is None:
            return None
        grp = file_handle
        for part in str(key).split("/"):
            if create:
                grp = grp.require_group(part)
            else:
                if part not in grp:
                    return None
                grp = grp[part]
        return grp

    def _voxelize_world_points(self, points_world):
        if points_world.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(f"Expected points_world to have shape (N,3), got {points_world.shape}")
        grid = float(self.args.grid_size)
        voxel_view_dtype = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))
        if not np.isfinite(grid) or grid <= 0:
            raise ValueError(f"Invalid grid_size={grid}; must be positive and finite.")
        vox = np.floor(points_world / grid).astype(np.int64, copy=False)
        if vox.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        vox = np.unique(vox, axis=0)
        if vox.shape[0] <= 1:
            return vox
        order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
        return vox[order]

    def _load_low_conf_voxels(self, files, domain, key):
        cache_key = (str(domain), str(key))
        if cache_key in self._low_conf_cache:
            return self._low_conf_cache[cache_key]

        file_handle = files.get(domain)
        if file_handle is None:
            return None

        dataset_name = "low_confidence_voxels"
        voxel_view_dtype = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))

        grp = self._resolve_conf_group(file_handle, key, create=False)
        if grp is None or dataset_name not in grp:
            raise RuntimeError(
                f"Missing confidence voxels for key={key} (domain={domain})."
                " Run with --run_confidence_annotation=true first."
            )

        node = grp[dataset_name]
        if not isinstance(node, h5py.Group):
            raise RuntimeError(
                f"Confidence voxels for key={key} (domain={domain}) are malformed;"
                " delete the file and regenerate annotations."
            )

        version = int(grp.attrs.get("mask_version", 0))
        if version != 3:
            raise RuntimeError(
                f"Confidence mask version mismatch for key={key} (domain={domain});"
                f" expected 3, found {version}. Regenerate annotations."
            )

        grid_attr = float(grp.attrs.get("grid_size", float("nan")))
        if not np.isfinite(grid_attr) or abs(grid_attr - float(self.args.grid_size)) > 1e-9:
            raise RuntimeError(
                f"Grid size mismatch for key={key} (domain={domain}); file={grid_attr}, args={self.args.grid_size}."
            )

        num_timesteps = int(grp.attrs.get("num_timesteps", len(node)))
        views = []
        for t in range(num_timesteps):
            ds_name = f"t{t:03d}"
            if ds_name not in node:
                raise RuntimeError(
                    f"Confidence voxels for key={key} (domain={domain}) missing timestep {ds_name}."
                )
            arr = node[ds_name][:].astype(np.int64, copy=False)
            if arr.size == 0:
                view = np.empty((0,), dtype=voxel_view_dtype)
            else:
                view = np.ascontiguousarray(arr).view(voxel_view_dtype)
            views.append(view)

        cached = {"views": views}
        self._low_conf_cache[cache_key] = cached
        return cached

    def inject_confidence_mask(self, batch, files):
        if "__shift_amount__" not in batch:
            raise RuntimeError("Batch missing __shift_amount__; cannot reconstruct world coordinates for mask.")
        if "scene_flows" not in batch:
            raise RuntimeError("Batch missing scene_flows; cannot build mask.")

        device = batch["scene_flows"].device
        B, T, Ns, _ = batch["scene_flows"].shape

        mask = torch.ones((B, T, Ns), dtype=torch.bool, device=device)

        grid = float(self.args.grid_size)
        voxel_view_dtype = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))

        for b in range(B):
            key = str(batch["__key__"][b])
            dom = batch["__domain__"][b]
            if self.is_sim_domain(dom):
                continue
            cached = self._load_low_conf_voxels(files, dom, key)
            if cached is None:
                continue
            views_per_t = cached["views"]

            if not any(view.size for view in views_per_t):
                continue

            shift = batch["__shift_amount__"][b].detach().cpu().numpy()
            scene_np = batch["scene_flows"][b].detach().cpu().numpy()
            if scene_np.ndim != 3 or scene_np.shape[1] != Ns:
                raise RuntimeError(f"Unexpected scene_flows shape for key={key}: {scene_np.shape}")

            coords = np.ascontiguousarray(np.floor((scene_np - shift) / grid).astype(np.int64, copy=False))
            coords_view = coords.reshape(T * Ns, 3).view(voxel_view_dtype).reshape(T, Ns)

            for t in range(T):
                view_t = views_per_t[t]
                if view_t.size == 0:
                    continue
                bad = np.isin(coords_view[t], view_t, assume_unique=True)
                if bad.any():
                    mask[b, t] &= torch.from_numpy(~bad).to(device=device)

        if "scene_exists" in batch and isinstance(batch["scene_exists"], torch.Tensor):
            exists = batch["scene_exists"].to(device=device).bool()
            mask &= exists

        batch["scene_filter_mask"] = mask
        batch["confidence_seed"] = torch.full((mask.shape[0],), int(self.args.seed), dtype=torch.int64, device=device)

    @torch.no_grad()
    def run_confidence_annotation(self, split):
        if not self.domain_to_dir:
            print("[confidence] No real domains detected; skipping annotation run.")
            return
        dl, _ = self._build_eval_loader(split, enable_mask=False)
        files = self.open_conf_files(split, mode="a")
        self.model.eval()
        total = self.args.eval_num_batches if self.args.eval_num_batches > 0 else None
        dataset_name = "low_confidence_voxels"
        prog = tqdm(dl, desc=f"Annotate confidence [{split}]", total=total)
        for i, batch in enumerate(prog):
            if total is not None and i >= total:
                break
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch, training=False)
            conf = outputs.get("confidence", None)
            if conf is None:
                raise RuntimeError("Model did not return confidence. Ensure uncertainty is enabled.")
            conf = conf.detach().cpu().numpy()
            keys = batch["__key__"]
            doms = batch["__domain__"]
            scene_exists = batch.get("scene_exists", None)
            if scene_exists is not None:
                scene_exists = scene_exists.detach().cpu().numpy().astype(bool)
            if "__shift_amount__" not in batch:
                raise RuntimeError("Dataset did not provide __shift_amount__; cannot project to world frame.")
            keep_frac = float(np.clip(self.args.confidence_thres, 0.0, 1.0))
            B, T, Npad = conf.shape
            for b in range(B):
                dom = doms[b]
                if self.is_sim_domain(dom):
                    continue
                key = str(keys[b])
                f = files.get(dom)
                assert f is not None, f"No confidence file open for domain {dom}"
                if scene_exists is not None:
                    valid_mask = scene_exists[b]
                    assert valid_mask.shape[:2] == (T, Npad), "scene_exists shape mismatch with confidence output"
                    per_point = valid_mask.any(axis=0)
                    Ns = int(per_point.sum())
                else:
                    Ns = Npad
                conf_trim = conf[b, :, :Ns]
                grp = self._resolve_conf_group(f, key, create=True)

                shift = batch["__shift_amount__"][b].detach().cpu().numpy()
                scene_np = batch["scene_flows"][b, :, :Ns, :].detach().cpu().numpy()
                if scene_np.ndim != 3 or scene_np.shape[1] != Ns:
                    raise RuntimeError(f"Unexpected scene_flows shape for key {key}: {scene_np.shape}")

                if scene_exists is not None:
                    valid_mask = scene_exists[b, :, :Ns]
                else:
                    valid_mask = np.ones_like(conf_trim, dtype=bool)

                valid_any = bool(valid_mask.any())
                quantile = max(0.0, 1.0 - keep_frac)
                if valid_any:
                    thr = float(np.quantile(conf_trim[valid_mask], quantile))
                    low_mask = valid_mask & (conf_trim < thr)
                else:
                    thr = float("nan")
                    low_mask = np.zeros_like(conf_trim, dtype=bool)

                if dataset_name in grp:
                    del grp[dataset_name]
                vox_group = grp.create_group(dataset_name)

                world_np = scene_np - shift
                low_counts_total = 0
                empty_vox = np.empty((0, 3), dtype=np.int32)
                for t in range(T):
                    low_t = low_mask[t]
                    if not low_t.any():
                        vox_group.create_dataset(f"t{t:03d}", data=empty_vox, dtype="i4")
                        continue
                    low_points = world_np[t][low_t]
                    low_vox = self._voxelize_world_points(low_points).astype(np.int32, copy=False)
                    low_counts_total += low_vox.shape[0]
                    if low_vox.size == 0:
                        vox_group.create_dataset(f"t{t:03d}", data=empty_vox, dtype="i4")
                    else:
                        vox_group.create_dataset(f"t{t:03d}", data=low_vox, compression="gzip", dtype="i4")

                cache_key = (str(dom), key)
                if cache_key in self._low_conf_cache:
                    del self._low_conf_cache[cache_key]
                grp.attrs["mask_version"] = 3
                grp.attrs["grid_size"] = float(self.args.grid_size)
                grp.attrs["confidence_keep_frac"] = keep_frac
                grp.attrs["confidence_threshold"] = float(thr) if np.isfinite(thr) else float("nan")
                grp.attrs["num_low_voxels"] = int(low_counts_total)
                grp.attrs["num_scene_points"] = int(Ns)
                grp.attrs["aggregation"] = "per_timestep"
                grp.attrs["num_timesteps"] = int(T)
        self.close_conf_files(files)

    def check_conf_files(self, split):
        missing = []
        for dom, base in self.domain_to_dir.items():
            path = os.path.join(base, split, f"expert_confidence-seed={self.args.seed}.h5")
            if not os.path.exists(path):
                missing.append(path)
        return missing

    def decode_expert_confidence(self, batch, conf_files) -> list:
        if "__shift_amount__" not in batch:
            raise RuntimeError("Batch missing __shift_amount__; cannot reconstruct world coordinates for mask.")
        if "scene_flows" not in batch:
            raise RuntimeError("Batch missing scene_flows; cannot decode expert confidence.")

        B, T, Ns, _ = batch["scene_flows"].shape
        grid = float(self.args.grid_size)
        voxel_view_dtype = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))

        scene_np_all = batch["scene_flows"].detach().cpu().numpy()
        shift_all = batch["__shift_amount__"].detach().cpu().numpy()

        out = []
        for b in range(B):
            dom = str(batch["__domain__"][b])
            key = str(batch["__key__"][b])
            if self.is_sim_domain(dom):
                out.append(np.ones((T, Ns), dtype=np.float32))
                continue
            cached = self._load_low_conf_voxels(conf_files, dom, key)
            views_per_t = cached["views"]

            conf_bt = np.ones((T, Ns), dtype=np.float32)

            coords = np.floor((scene_np_all[b] - shift_all[b]) / grid).astype(np.int64, copy=False)
            coords_view = coords.reshape(T * Ns, 3).view(voxel_view_dtype).reshape(T, Ns)
            for t in range(T):
                view_t = views_per_t[t]
                if view_t.size == 0:
                    continue
                bad = np.isin(coords_view[t], view_t, assume_unique=True)
                if bad.any():
                    conf_bt[t][bad] = 0.0
            out.append(conf_bt)
        return out
