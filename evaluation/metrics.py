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
import numpy as np
import torch


class _WeightedStat:
    """Track weighted first and second moments for scalar metrics."""

    def __init__(self):
        self.weight = 0.0
        self.weighted_sum = 0.0
        self.weighted_sumsq = 0.0
        self.min = None
        self.max = None

    def update(self, value: float, weight: float) -> None:
        if weight <= 0:
            return
        if value is None or not math.isfinite(value):
            return
        self.weight += float(weight)
        contribution = float(value) * weight
        self.weighted_sum += contribution
        self.weighted_sumsq += float(value) * float(value) * weight
        self.min = float(value) if self.min is None else min(self.min, float(value))
        self.max = float(value) if self.max is None else max(self.max, float(value))

    def mean(self) -> float:
        if self.weight <= 0:
            return 0.0
        return self.weighted_sum / self.weight

    def variance(self) -> float:
        if self.weight <= 0:
            return 0.0
        mean = self.mean()
        var = (self.weighted_sumsq / self.weight) - mean * mean
        return max(var, 0.0)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def std_err(self) -> float:
        if self.weight <= 0:
            return 0.0
        return self.std() / math.sqrt(self.weight)

    def to_summary(self) -> dict[str, float | int]:
        if self.weight <= 0:
            return dict(mean=0.0, std=0.0, std_err=0.0, min=0.0, max=0.0, count=0)
        return dict(
            mean=float(self.mean()),
            std=float(self.std()),
            std_err=float(self.std_err()),
            min=float(self.min if self.min is not None else 0.0),
            max=float(self.max if self.max is not None else 0.0),
            count=int(round(self.weight)),
        )


class _RunningStat:
    """Running unweighted statistics for per-frame counts."""

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

    def update(self, values: np.ndarray | torch.Tensor) -> None:
        if isinstance(values, torch.Tensor):
            arr = values.detach().cpu().numpy().ravel()
        else:
            arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float((arr * arr).sum())
        current_min = float(arr.min())
        current_max = float(arr.max())
        self.min = current_min if self.min is None else min(self.min, current_min)
        self.max = current_max if self.max is None else max(self.max, current_max)

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        var = (self.sumsq / self.count) - mean * mean
        return max(var, 0.0)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def std_err(self) -> float:
        if self.count == 0:
            return 0.0
        return self.std() / math.sqrt(self.count)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return dict(mean=0.0, std=0.0, std_err=0.0, min=0.0, max=0.0, count=0)
        return dict(
            mean=float(self.mean()),
            std=float(self.std()),
            std_err=float(self.std_err()),
            min=float(self.min if self.min is not None else 0.0),
            max=float(self.max if self.max is not None else 0.0),
            count=int(self.count),
        )


def _stats_to_entries(prefix: str, stats: _RunningStat) -> dict[str, float | int]:
    summary = stats.as_dict()
    return {f"{prefix}/{key}": value for key, value in summary.items()}


def _metric_base_name(name: str) -> str:
    return name.rsplit("/", 1)[0] if name.endswith("/mean") else name


def _should_emit_detail(name: str) -> bool:
    tail = name.rsplit("/", 1)[-1]
    return tail not in {"max", "min"}
