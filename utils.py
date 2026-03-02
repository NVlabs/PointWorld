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

import datetime
import numpy as np
import torch

ROBOT_URDF_BY_DOMAIN = {
    "behavior": "assets/r1pro/urdf/r1pro.urdf",
    "droid": "assets/franka_description/franka_panda_robotiq_2f85.urdf",
}

__all__ = [
    "resolve_robot_urdf",
    "resolve_default_robot_urdf",
    "NaNDetectionError",
    "make_soft_selector_labels",
    "soft_classification_metrics",
    "_print",
    "check_tensor_for_nan",
    "check_model_parameters_for_nan",
    "safe_loss_computation",
    "handle_nan_grad_norm",
    "handle_nan_outputs",
    "build_pk_chain_from_urdf",
    "build_pk_serial_chain_from_urdf",
]

_PK_URDF_WARNING_FILTER_INSTALLED = False
_PK_URDF_ERROR_SINK = None
_PK_IGNORED_DYNAMICS_ATTRS = {"D", "K", "mu_coulomb", "mu_viscous"}


def _should_silence_pk_urdf_warning(message: str) -> bool:
    if not isinstance(message, str):
        return False
    if not message.startswith('Unknown attribute "'):
        return False
    if "/dynamics" not in message:
        return False
    for attr_name in _PK_IGNORED_DYNAMICS_ATTRS:
        if f'Unknown attribute "{attr_name}"' in message:
            return True
    return False


def _pk_urdf_on_error(message):
    global _PK_URDF_ERROR_SINK
    if _should_silence_pk_urdf_warning(message):
        return
    _PK_URDF_ERROR_SINK(message)


def _install_pk_urdf_warning_filter() -> None:
    global _PK_URDF_WARNING_FILTER_INSTALLED
    global _PK_URDF_ERROR_SINK
    if _PK_URDF_WARNING_FILTER_INSTALLED:
        return
    from pytorch_kinematics.urdf_parser_py.xml_reflection import core as pk_xml_core
    _PK_URDF_ERROR_SINK = pk_xml_core.on_error
    pk_xml_core.on_error = _pk_urdf_on_error
    _PK_URDF_WARNING_FILTER_INSTALLED = True


def resolve_robot_urdf(domain: str) -> str:
    if not domain:
        raise ValueError("Domain is required to resolve robot URDF.")
    dom = str(domain).lower()
    for key, path in ROBOT_URDF_BY_DOMAIN.items():
        if key in dom:
            return path
    raise ValueError(f"Unsupported domain for robot URDF: {domain}")


def resolve_default_robot_urdf(domains) -> str:
    if not domains:
        raise ValueError("At least one domain is required to resolve robot URDF.")
    doms = [str(d).lower() for d in domains]
    if any("droid" in d for d in doms):
        return ROBOT_URDF_BY_DOMAIN["droid"]
    if any("behavior" in d for d in doms):
        return ROBOT_URDF_BY_DOMAIN["behavior"]
    raise ValueError(f"Unsupported domains for robot URDF: {domains}")


def build_pk_chain_from_urdf(urdf_data):
    if not isinstance(urdf_data, (bytes, bytearray)):
        raise TypeError(f"urdf_data must be bytes or bytearray, got {type(urdf_data)}")
    _install_pk_urdf_warning_filter()
    import pytorch_kinematics as pk
    return pk.build_chain_from_urdf(bytes(urdf_data))


def build_pk_serial_chain_from_urdf(urdf_data, end_link_name: str):
    if not isinstance(urdf_data, (bytes, bytearray)):
        raise TypeError(f"urdf_data must be bytes or bytearray, got {type(urdf_data)}")
    if not isinstance(end_link_name, str) or len(end_link_name) == 0:
        raise ValueError("end_link_name must be a non-empty string")
    _install_pk_urdf_warning_filter()
    import pytorch_kinematics as pk
    return pk.build_serial_chain_from_urdf(bytes(urdf_data), end_link_name=end_link_name)


class NaNDetectionError(Exception):
    """Custom exception raised when NaN values are detected during training."""
    def __init__(self, message, context="", nan_details=None):
        super().__init__(message)
        self.context = context
        self.nan_details = nan_details or {}


def make_soft_selector_labels(delta_norm, tau: float, temp_scale: float = None):
    """
    Create soft labels based on the magnitude of delta vectors.

    Args:
        delta_norm: Tensor or numpy array. Can be batched (B,T,NS) or unbatched (T,NS)
        tau: Threshold for the sigmoid function
        temp_scale: Temperature scale for the sigmoid function

    Returns:
        Soft labels of same type and dimensionality as input (without the last dimension)
    """
    is_torch = isinstance(delta_norm, torch.Tensor)
    temp_scale = 5.0 if temp_scale is None else temp_scale
    k = temp_scale / max(tau, 1e-12)
    if is_torch:
        return torch.sigmoid(k * (delta_norm - tau))
    return 1.0 / (1.0 + np.exp(-k * (delta_norm - tau)))


def soft_classification_metrics(logits, soft_targets, threshold=0.5):
    """
    Computes classification metrics when using soft targets (values between 0 and 1).

    Args:
        logits: Raw logits before sigmoid
        soft_targets: Soft labels with values between 0 and 1
        threshold: Threshold for converting logits to binary predictions

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    assert logits.shape == soft_targets.shape
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()

        accuracy = 1 - ((pred_mask - soft_targets) ** 2).mean()

        tp = (pred_mask * soft_targets).sum()
        fp = (pred_mask * (1 - soft_targets)).sum()
        fn = ((1 - pred_mask) * soft_targets).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _print(*args, **kwargs):
    """wrapper that adds datetime prefix to print"""
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    kwargs["flush"] = True
    print(f"[{time_str}]", *args, **kwargs)


def check_tensor_for_nan(tensor, name, context="", raise_on_nan=True):
    """
    Check if a tensor contains NaN values and optionally raise an error with informative message.

    Args:
        tensor: torch.Tensor to check
        name: str, name of the tensor for error reporting
        context: str, additional context for error reporting
        raise_on_nan: bool, whether to raise error on NaN detection

    Returns:
        bool: True if NaN detected, False otherwise

    Raises:
        NaNDetectionError: If NaN detected and raise_on_nan=True
    """
    if not isinstance(tensor, torch.Tensor):
        return False

    has_nan = torch.isnan(tensor).any().item()
    if has_nan and raise_on_nan:
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype
        tensor_device = tensor.device

        nan_details = {
            "tensor_name": name,
            "nan_count": nan_count,
            "total_elements": total_elements,
            "tensor_shape": tensor_shape,
            "tensor_dtype": str(tensor_dtype),
            "tensor_device": str(tensor_device),
        }

        error_msg = (
            f"NaN detected in tensor '{name}' during {context}. "
            f"NaN count: {nan_count}/{total_elements}, shape: {tensor_shape}"
        )

        raise NaNDetectionError(error_msg, context=context, nan_details=nan_details)

    return has_nan


def check_model_parameters_for_nan(model, context="parameter check"):
    """
    Check all model parameters for NaN values.

    Args:
        model: torch.nn.Module to check
        context: str, context for error reporting

    Returns:
        bool: True if any NaN detected, False otherwise

    Raises:
        NaNDetectionError: If NaN detected in any parameter
    """
    nan_params = []
    for name, param in model.named_parameters():
        if param is not None and torch.isnan(param).any():
            nan_params.append((name, param))

    if nan_params:
        nan_details = {
            "nan_param_count": len(nan_params),
            "nan_params": [],
        }

        _print("\n" + "=" * 100)
        _print("NaN DETECTED IN MODEL PARAMETERS")
        _print("=" * 100)
        for name, param in nan_params:
            nan_count = torch.isnan(param).sum().item()
            total_elements = param.numel()
            param_info = {
                "name": name,
                "shape": tuple(param.shape),
                "nan_count": nan_count,
                "total_elements": total_elements,
                "requires_grad": param.requires_grad,
            }
            nan_details["nan_params"].append(param_info)

            _print(f"Parameter: {name}")
            _print(f"  Shape: {tuple(param.shape)}, NaN count: {nan_count}/{total_elements}")
            _print(f"  Requires grad: {param.requires_grad}")
            if param.grad is not None:
                grad_nan_count = torch.isnan(param.grad).sum().item()
                _print(f"  Gradient NaN count: {grad_nan_count}/{param.grad.numel()}")
        _print("=" * 100)

        error_msg = f"NaN detected in {len(nan_params)} model parameters during {context}"
        raise NaNDetectionError(error_msg, context=context, nan_details=nan_details)

    return len(nan_params) > 0


def safe_loss_computation(loss_fn_call, context="loss computation"):
    """
    Safely execute loss function with NaN checking.

    Args:
        loss_fn_call: callable that returns (total_loss, loss_dict)
        context: str, context for error reporting

    Returns:
        tuple: (total_loss, loss_dict)

    Raises:
        RuntimeError: If NaN detected during loss computation
    """
    try:
        total_loss, loss_dict = loss_fn_call()
        check_tensor_for_nan(total_loss, "total_loss", context)
        return total_loss, loss_dict
    except Exception as e:
        if "NaN detected" in str(e):
            raise
        _print(f"\nUnexpected error during {context}: {e}")
        raise RuntimeError(f"Error during {context}: {e}") from e


def handle_nan_grad_norm(grad_norm, consecutive_nan_count, max_consecutive_nans=100, context=""):
    """
    Handle NaN gradient norms with consecutive counting logic.

    Args:
        grad_norm: torch.Tensor, the gradient norm to check
        consecutive_nan_count: int, current count of consecutive NaN grad norms
        max_consecutive_nans: int, maximum allowed consecutive NaN grad norms before raising error
        context: str, additional context for error reporting

    Returns:
        tuple: (should_skip_batch: bool, new_consecutive_count: int)

    Raises:
        NaNDetectionError: If consecutive NaN count exceeds max_consecutive_nans
    """
    if not torch.isfinite(grad_norm).all():
        consecutive_nan_count += 1
        _print(
            f"⚠️  NaN/inf grad norm detected (consecutive: {consecutive_nan_count}/{max_consecutive_nans}) - skipping batch"
        )

        if consecutive_nan_count >= max_consecutive_nans:
            error_msg = (
                f"Too many consecutive NaN/inf gradient norms: {consecutive_nan_count}/{max_consecutive_nans}"
            )
            nan_details = {
                "consecutive_nan_count": consecutive_nan_count,
                "max_consecutive_nans": max_consecutive_nans,
                "grad_norm_value": grad_norm.item() if grad_norm.numel() == 1 else "multi-element tensor",
            }
            raise NaNDetectionError(error_msg, context=context, nan_details=nan_details)

        return True, consecutive_nan_count

    if consecutive_nan_count > 0:
        _print(f"✅ Grad norm recovered after {consecutive_nan_count} consecutive NaN/inf occurrences")
    return False, 0


def handle_nan_outputs(outputs, consecutive_nan_count, max_consecutive_nans=100, context=""):
    """
    Handle NaN model outputs with consecutive counting logic.

    Args:
        outputs: dict, model outputs to check for NaN values
        consecutive_nan_count: int, current count of consecutive NaN output occurrences
        max_consecutive_nans: int, maximum allowed consecutive NaN outputs before raising error
        context: str, additional context for error reporting

    Returns:
        tuple: (should_skip_batch: bool, new_consecutive_count: int)

    Raises:
        NaNDetectionError: If consecutive NaN count exceeds max_consecutive_nans
    """
    def _has_nan_in_outputs(outputs_dict):
        for key, value in outputs_dict.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    return True, key
            elif isinstance(value, dict):
                has_nan, nan_key = _has_nan_in_outputs(value)
                if has_nan:
                    return True, f"{key}.{nan_key}"
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        has_nan, nan_key = _has_nan_in_outputs(item)
                        if has_nan:
                            return True, f"{key}[{i}].{nan_key}"
        return False, None

    has_nan, nan_key = _has_nan_in_outputs(outputs)

    if has_nan:
        consecutive_nan_count += 1
        _print(
            f"⚠️  NaN in model outputs detected at '{nan_key}' (consecutive: {consecutive_nan_count}/{max_consecutive_nans}) - skipping batch"
        )

        if consecutive_nan_count >= max_consecutive_nans:
            error_msg = (
                f"Too many consecutive NaN model outputs: {consecutive_nan_count}/{max_consecutive_nans}"
            )
            nan_details = {
                "consecutive_nan_count": consecutive_nan_count,
                "max_consecutive_nans": max_consecutive_nans,
                "nan_key": nan_key,
                "context": context,
            }
            raise NaNDetectionError(error_msg, context=context, nan_details=nan_details)

        return True, consecutive_nan_count

    if consecutive_nan_count > 0:
        _print(f"✅ Model outputs recovered after {consecutive_nan_count} consecutive NaN occurrences")
    return False, 0
