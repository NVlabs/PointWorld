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

import numpy as np


class _EvaluationMetaAccumulator:
    """Accumulate minimal dataset-wide metadata during evaluation."""

    def __init__(self):
        self.total_batches = 0
        self.total_sequences = 0
        self.total_frames = 0
        self.confidence_kept = 0
        self.confidence_total = 0

    def update(
        self,
        domains: list[str],
        scene_counts_per_frame: np.ndarray,
        robot_counts_per_frame: np.ndarray,
        supervised_counts_per_frame: np.ndarray,
        confidence_kept_per_frame: np.ndarray | None,
        confidence_total_per_frame: np.ndarray | None,
    ) -> None:
        if scene_counts_per_frame.ndim != 2:
            raise ValueError("scene_counts_per_frame must have shape (B, T)")
        num_sequences, num_frames = scene_counts_per_frame.shape
        assert len(domains) == num_sequences, "Domain list length must match batch size"

        self.total_batches += 1
        self.total_sequences += num_sequences
        self.total_frames += num_sequences * num_frames

        temporal_lengths = np.full((num_sequences,), float(num_frames), dtype=np.float64)
        if confidence_total_per_frame is not None:
            kept_frame_counts = confidence_kept_per_frame if confidence_kept_per_frame is not None else np.zeros_like(confidence_total_per_frame)
            total_frame_counts = confidence_total_per_frame
            kept_total = int(np.round(kept_frame_counts.sum()))
            total = int(np.round(total_frame_counts.sum()))
            self.confidence_kept += kept_total
            self.confidence_total += total

    def to_entries(self, prefix: str) -> dict[str, float | int]:
        entries: dict[str, float | int] = {}
        entries[f"{prefix}/total_batches"] = int(self.total_batches)
        entries[f"{prefix}/total_sequences"] = int(self.total_sequences)
        entries[f"{prefix}/total_frames"] = int(self.total_frames)
        entries[f"{prefix}/confidence/total_predictions"] = int(self.confidence_total)
        entries[f"{prefix}/confidence/kept_predictions"] = int(self.confidence_kept)
        keep_fraction = (self.confidence_kept / self.confidence_total) if self.confidence_total > 0 else 0.0
        entries[f"{prefix}/confidence/keep_fraction"] = float(keep_fraction)
        return entries
