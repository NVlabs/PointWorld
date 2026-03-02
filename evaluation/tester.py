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
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from collections import Counter, defaultdict
from typing import Dict

from training.trainer import Trainer
from pointworld.checkpoint_contract import (
    apply_model_contract_to_args,
    read_checkpoint_contract,
    train_domains_from_data_contract,
)
from utils import resolve_default_robot_urdf
from arguments import DOMAIN_TO_DATA_DIR
from dataset_components.dataloader import build_dataloader
from dataset_components.constants import RELEASE_EVAL_SPLITS
from evaluation.metrics import _WeightedStat, _metric_base_name, _should_emit_detail
from evaluation.meta import _EvaluationMetaAccumulator
from evaluation.annotation import ConfidenceHelper

from visualization.prediction_viz import (
    PredictionVisualizer,
    PredictionVisualizerConfig,
    build_sample_from_dictionary,
)

class Tester(Trainer):
    def __init__(self, args):
        # Match dev eval numeric settings (train.py sets this in the dev repo).
        torch.set_float32_matmul_precision("high")

        os.environ["WANDB_MODE"] = "disabled"

        self._allow_missing_confidence_mask = bool(args.allow_missing_confidence_mask)
        self._eval_skip_viz = bool(args.eval_skip_viz)
        self._eval_viz_num = int(args.eval_viz_num)

        if not self._allow_missing_confidence_mask:
            print("Overwriting seed for eval")
            args.seed = 42

        if args.distributed:
            print("Warning: eval.py is non-DDP; setting distributed=False")
            args.distributed = False

        os.makedirs("eval_logs", exist_ok=True)
        splits_tag = "+".join(RELEASE_EVAL_SPLITS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(
            "eval_logs",
            f"{timestamp}-{args.eval_exp_name}-split={splits_tag}-seed={args.seed}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Using eval dir: {eval_dir}")

        assert args.model_path is not None or args.eval_exp_name is not None, "Provide --model_path or --eval_exp_name"
        if args.eval_exp_name:
            assert args.model_path is None, "Cannot specify both --eval_exp_name and --model_path"
            local_model_path = os.path.join(args.log_dir, args.eval_exp_name, "model-last.pt")
            assert os.path.exists(local_model_path), f"Model not found at {local_model_path}"
            args.model_path = local_model_path

        eval_domains_requested = list(args.domains)
        if not eval_domains_requested:
            raise ValueError("Evaluation requires at least one domain in --domains.")
        eval_dir_overrides = dict(zip(eval_domains_requested, getattr(args, "data_dirs", [])))

        def _resolve_eval_data_dir(dom: str) -> str:
            if dom in eval_dir_overrides:
                return eval_dir_overrides[dom]
            if dom in DOMAIN_TO_DATA_DIR:
                return DOMAIN_TO_DATA_DIR[dom]
            valid = ", ".join(sorted(DOMAIN_TO_DATA_DIR.keys()))
            raise KeyError(f"No dataset directory mapping for domain {dom}. Valid keys: {valid}")

        self._preloaded_checkpoint = None

        train_domains: list[str] = []
        if args.model_path is not None:
            try:
                checkpoint_meta = torch.load(args.model_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to load checkpoint metadata from {args.model_path}: {exc}") from exc
            self._preloaded_checkpoint = checkpoint_meta
            context = f"evaluation checkpoint '{args.model_path}'"
            model_contract, data_contract = read_checkpoint_contract(
                checkpoint_meta, context=context
            )
            changed = apply_model_contract_to_args(
                args,
                model_contract,
                context=context,
                explicit_cli_dests=set(getattr(args, "_explicit_cli_dests", set())),
            )
            if changed:
                print("Applied canonical checkpoint model_contract for evaluation model initialization.")
            train_domains = train_domains_from_data_contract(data_contract, context=context)

        if not train_domains:
            raise RuntimeError("Checkpoint is missing training domains; cannot evaluate.")

        missing_eval_domains = [dom for dom in eval_domains_requested if dom not in set(train_domains)]
        if missing_eval_domains:
            raise ValueError(
                f"Eval domains {missing_eval_domains} not found in checkpoint domains {train_domains}."
            )

        eval_dir_map: dict[str, str] = {}
        for dom in eval_domains_requested:
            if dom in eval_dir_map:
                raise ValueError(f"Duplicate eval domain requested: {dom}")
            eval_dir_map[dom] = _resolve_eval_data_dir(dom)

        args.domains = list(train_domains)
        args.data_dirs = []
        for dom in args.domains:
            if dom in eval_dir_map:
                args.data_dirs.append(eval_dir_map[dom])
            else:
                if dom not in DOMAIN_TO_DATA_DIR:
                    valid = ", ".join(sorted(DOMAIN_TO_DATA_DIR.keys()))
                    raise KeyError(
                        f"No default dataset directory mapping for checkpoint domain {dom}. Valid keys: {valid}"
                    )
                args.data_dirs.append(DOMAIN_TO_DATA_DIR[dom])
        self._sim_keywords = ["behavior"]

        args.train_domains_for_eval = list(train_domains)
        args.eval_domain_whitelist = list(eval_domains_requested)
        args._eval_override_active = True

        self._eval_domain_pairs = [(dom, eval_dir_map[dom]) for dom in eval_domains_requested]

        self.eval_dir = eval_dir
        super().__init__(args, inference_only=True, data_info_dict=None)

        def _is_sim_domain(domain: str) -> bool:
            dom = str(domain)
            for kw in self._sim_keywords:
                if kw and kw in dom:
                    return True
            return False

        self.domain_to_dir = {
            d: dd for d, dd in self._eval_domain_pairs if not _is_sim_domain(d)
        }
        self._confidence_helper = ConfidenceHelper(
            self.args,
            self.device,
            self.domain_to_dir,
            self._sim_keywords,
            self._build_eval_loader,
            self.model,
        )

    def try_to_get_checkpoint(self, save_dir, model_path=None):
        if self._preloaded_checkpoint is not None:
            checkpoint = self._preloaded_checkpoint
            self._preloaded_checkpoint = None
            return checkpoint
        return super().try_to_get_checkpoint(save_dir, model_path)

    # ----------------------------- helpers ----------------------------- #
    def _build_eval_loader(self, split, enable_mask=False):
        orig_eval_workers = self.args.eval_num_workers
        self.args.eval_num_workers = self.args.num_workers
        dl, info = build_dataloader(self.args, mode="test", rank=self.rank, world_size=self.world_size, override_splits=split)
        self.args.eval_num_workers = orig_eval_workers
        assert dl is not None, f"No dataloader for split {split}"
        return dl, info

    # ----------------------------- evaluation -------------------------- #
    @torch.no_grad()
    def _accumulate_metrics(self, dl, split, apply_confidence_mask=False):
        self.model.eval()
        metric_accumulators: dict[str, _WeightedStat] = defaultdict(_WeightedStat)
        meta_acc = _EvaluationMetaAccumulator()
        domains_set = set(str(d) for d in self.args.domains)
        total = self.args.eval_num_batches if self.args.eval_num_batches > 0 else None
        iterator = tqdm(dl, desc=f"Eval [{split}]", total=total)

        conf_files = self._confidence_helper.open_conf_files(split, mode="r") if apply_confidence_mask else None
        for i, batch in enumerate(iterator):
            if total is not None and i >= total:
                break
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if apply_confidence_mask:
                self._confidence_helper.inject_confidence_mask(batch, conf_files)

            outputs = self.model(batch, training=False)
            total_loss, loss_dict = self.model.loss_fn(outputs, batch, training=False)

            domains_list = [str(d) for d in batch["__domain__"]]
            domain_counts = Counter(domains_list)
            batch_size = len(domains_list)

            def _prepare_mask(tensor_name: str) -> torch.Tensor:
                tensor = batch[tensor_name]
                assert isinstance(tensor, torch.Tensor), f"Expected tensor for {tensor_name}"
                tensor = tensor.to(torch.bool)
                if tensor.dim() == 4 and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
                return tensor

            scene_exists_mask = _prepare_mask("scene_exists")
            robot_exists_mask = _prepare_mask("robot_exists")
            context_mask = _prepare_mask("scene_context_mask")
            supervised_mask = _prepare_mask("scene_supervised_mask")
            pred_exists_mask = (~context_mask) & scene_exists_mask & supervised_mask

            scene_counts_per_frame = scene_exists_mask.sum(dim=2)
            robot_counts_per_frame = robot_exists_mask.sum(dim=2)
            supervised_counts_per_frame = pred_exists_mask.sum(dim=2)

            if apply_confidence_mask:
                filter_mask = batch["scene_filter_mask"].to(torch.bool)
                kept_counts_per_frame = (filter_mask & pred_exists_mask).sum(dim=2)
                kept_counts_np = kept_counts_per_frame.detach().cpu().numpy()
                supervised_counts_np = supervised_counts_per_frame.detach().cpu().numpy()
                conf_total_np = supervised_counts_np
            else:
                kept_counts_np = None
                conf_total_np = None
                supervised_counts_np = supervised_counts_per_frame.detach().cpu().numpy()

            scene_counts_np = scene_counts_per_frame.detach().cpu().numpy()
            robot_counts_np = robot_counts_per_frame.detach().cpu().numpy()

            meta_acc.update(
                domains_list,
                scene_counts_np,
                robot_counts_np,
                supervised_counts_np,
                kept_counts_np,
                conf_total_np,
            )

            for k, v in loss_dict.items():
                value: float | None = None
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        scalar = float(v.item())
                        if math.isfinite(scalar):
                            value = scalar
                elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                    value = float(v)

                if value is None:
                    continue

                weight = float(batch_size)
                parts = k.split("/")
                for part in parts:
                    if part in domain_counts:
                        weight = float(domain_counts[part])
                        break
                    if part in domains_set:
                        weight = float(domain_counts.get(part, batch_size))
                        break

                metric_accumulators[k].update(value, weight)

        aggregated = {}
        detail_emitted = set()
        for metric_name, stat in metric_accumulators.items():
            summary = stat.to_summary()
            aggregated[f"full_eval/{split}/{metric_name}"] = summary["mean"]
            base_name = _metric_base_name(metric_name)
            if _should_emit_detail(metric_name) and base_name not in detail_emitted:
                aggregated[f"full_eval/{split}/{base_name}/count"] = summary["count"]
                aggregated[f"full_eval/{split}/{base_name}/std"] = summary["std"]
                aggregated[f"full_eval/{split}/{base_name}/std_err"] = summary["std_err"]
                aggregated[f"full_eval/{split}/{base_name}/min"] = summary["min"]
                aggregated[f"full_eval/{split}/{base_name}/max"] = summary["max"]
                detail_emitted.add(base_name)

        aggregated.update(meta_acc.to_entries(f"full_eval/{split}/_meta"))

        if conf_files is not None:
            self._confidence_helper.close_conf_files(conf_files)
        return aggregated

    def run_evaluation(self):
        results = {}
        use_conf_mask = bool(self.domain_to_dir)
        for split in RELEASE_EVAL_SPLITS:
            if not self._eval_skip_viz and int(self._eval_viz_num) > 0:
                self._visualize_eval_samples(split)
            if self.args.run_confidence_annotation:
                if use_conf_mask:
                    self._confidence_helper.run_confidence_annotation(split)
                    print(f"Saved expert confidences for split {split}")
                else:
                    print(f"[confidence] Skipping annotation for split {split} (simulation-only domains).")
            else:
                if use_conf_mask:
                    missing = self._confidence_helper.check_conf_files(split)
                    if missing:
                        if not self._allow_missing_confidence_mask:
                            missing_str = "\n".join(f"  - {path}" for path in missing)
                            raise FileNotFoundError(
                                "Missing expert confidence mask(s) required for evaluation:\n"
                                f"{missing_str}\n"
                                "Run eval.py with --run_confidence_annotation=true first, or rerun with "
                                "--allow_missing_confidence_mask=true to skip filtered metrics."
                            )
                        else:
                            print("Warning: missing confidence file(s), proceeding without filtered metrics:")
                            for m in missing:
                                print(f"  - {m}")
                dl, _ = self._build_eval_loader(split, enable_mask=use_conf_mask)
                res = self._accumulate_metrics(dl, split, apply_confidence_mask=use_conf_mask)
                results.update(res)

        if results:
            report_path = os.path.join(self.eval_dir, "metrics.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(report_path)

    @torch.no_grad()
    def _visualize_eval_samples(self, split: str) -> None:
        N = int(self._eval_viz_num)
        if N <= 0:
            return
        dl, _ = self._build_eval_loader(split, enable_mask=False)

        viz_config = PredictionVisualizerConfig.from_args(self.args)
        urdf_path = Path(resolve_default_robot_urdf(self.args.domains))
        visualizer = PredictionVisualizer(viz_config, urdf_path=urdf_path)
        saved = 0
        live_session = None
        stop_requested = False
        try:
            for batch in dl:
                if saved >= N or stop_requested:
                    break
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch, training=False)

                batch_np = {
                    key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }

                gt_scene_flows = batch_np["gt_scene_flows"]
                data_keys = [str(k) for k in batch_np["__key__"]]
                domains = [str(d) for d in batch_np["__domain__"]]

                scene_flows_pred = outputs["scene_flows"].detach().cpu().numpy()

                B = gt_scene_flows.shape[0]
                for i in range(B):
                    if saved >= N or stop_requested:
                        break
                    sample_dict: Dict[str, object] = {}
                    for key, value in batch_np.items():
                        if isinstance(value, np.ndarray):
                            if value.ndim == 0:
                                sample_dict[key] = value.item()
                            elif value.shape[0] == B:
                                sample_dict[key] = value[i]
                            else:
                                sample_dict[key] = value
                        elif isinstance(value, (list, tuple)):
                            if len(value) == B:
                                sample_dict[key] = value[i]
                            else:
                                sample_dict[key] = value
                        else:
                            sample_dict[key] = value
                    sample_dict["gt_scene_flows"] = gt_scene_flows[i]
                    sample_dict["__key__"] = data_keys[i]
                    sample_dict["__domain__"] = domains[i]

                    viz_sample = build_sample_from_dictionary(
                        sample_dict=sample_dict,
                        predictions={"scene_flows": scene_flows_pred[i]},
                    )

                    viz_result = visualizer.visualize(
                        viz_sample,
                        launch_viewer=(live_session is None),
                        live_session=live_session,
                    )
                    new_session = viz_result.get("live_session")
                    if new_session is not None and new_session is not live_session:
                        live_session = new_session
                        display_host, display_port = visualizer.viewer_endpoint()
                        if display_host in {"0.0.0.0", "127.0.0.1"}:
                            display_host = "localhost"
                        print(f"[viz] Live viewer running at http://{display_host}:{display_port}")
                    else:
                        live_session = viz_result.get("live_session", live_session)
                    print("[viz] Visualization complete.")
                    saved += 1
                    sys.stdout.flush()
                    user_input = self._prompt_viz_continue()
                    if user_input.strip().lower() in {"q", "quit", "exit"}:
                        stop_requested = True
                        break
        finally:
            if live_session is not None:
                live_session.close()

    @staticmethod
    def _prompt_viz_continue() -> str:
        prompt = "Press ENTER to continue to the next eval sample (type q to quit visualization): "
        if sys.stdin is not None and sys.stdin.isatty():
            return input(prompt)
        try:
            with open("/dev/tty", "r") as tty:
                sys.stdout.write(prompt)
                sys.stdout.flush()
                line = tty.readline()
                if line == "":
                    raise RuntimeError(
                        "No interactive TTY input received. Run in a terminal with an attached "
                        "TTY (e.g., `conda run --no-capture-output ...`) or avoid stdin redirection."
                    )
                return line
        except OSError as exc:
            raise RuntimeError(
                "Visualization requires an interactive TTY. Run from a terminal "
                "or unset stdin redirection."
            ) from exc
