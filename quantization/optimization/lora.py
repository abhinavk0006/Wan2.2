"""LoRA checkpoint validation and merge helpers, independent of WanI2V.

These helpers merge a rank-factorized adapter into ordinary ``nn.Linear``
weights before a later quantization step. Safetensors files are read one target
layer at a time so both 1.2 GB expert adapters need not be copied into RAM at
once. Nothing in this module loads a Wan checkpoint or starts video generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

_DOWN_SUFFIX = ".lora_down.weight"
_UP_SUFFIX = ".lora_up.weight"
_ALPHA_SUFFIX = ".alpha"
_COMFY_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class LoRAMergeReport:
    targets_merged: int
    max_rank: int
    strength: float


@dataclass(frozen=True)
class _Target:
    path: str
    down_key: str
    up_key: str
    alpha_key: str
    module: nn.Linear
    rank: int


@dataclass(frozen=True)
class LoRATensorPlan:
    path: str
    down_key: str
    up_key: str
    alpha_key: str
    rank: int


def _module_path_from_lora_key(key: str) -> str:
    if not key.endswith(_DOWN_SUFFIX):
        raise ValueError(f"Not a LoRA down-projection key: {key}")
    path = key[: -len(_DOWN_SUFFIX)]
    if path.startswith(_COMFY_PREFIX):
        path = path[len(_COMFY_PREFIX) :]
    if not path:
        raise ValueError(f"LoRA key has no target module: {key}")
    return path


def _check_factors(
    module: nn.Linear,
    down_shape: tuple[int, ...],
    up_shape: tuple[int, ...],
    alpha_shape: tuple[int, ...],
    path: str,
) -> int:
    if len(alpha_shape) != 0:
        raise ValueError(f"LoRA alpha for {path} must be a scalar")
    if len(down_shape) != 2 or len(up_shape) != 2:
        raise ValueError(f"LoRA factors for {path} must be matrices")
    rank, in_features = down_shape
    out_features, up_rank = up_shape
    if rank < 1 or up_rank != rank:
        raise ValueError(f"LoRA factor ranks do not match for {path}")
    if in_features != module.in_features or out_features != module.out_features:
        raise ValueError(
            f"LoRA dimensions {down_shape}/{up_shape} do not match "
            f"Linear({module.in_features}, {module.out_features}) at {path}"
        )
    if module.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"LoRA merging requires floating Linear weights at {path}")
    return rank


def _plan_targets(
    model: nn.Module,
    keys: set[str],
    shapes: Mapping[str, tuple[int, ...]],
) -> list[_Target]:
    modules = dict(model.named_modules())
    down_keys = {key for key in keys if key.endswith(_DOWN_SUFFIX)}
    up_keys = {key for key in keys if key.endswith(_UP_SUFFIX)}
    alpha_keys = {key for key in keys if key.endswith(_ALPHA_SUFFIX)}
    if not down_keys:
        raise ValueError("LoRA checkpoint contains no .lora_down.weight tensors")

    plan: list[_Target] = []
    expected_keys: set[str] = set()
    for down_key in sorted(down_keys):
        path = _module_path_from_lora_key(down_key)
        up_key = down_key[: -len(_DOWN_SUFFIX)] + _UP_SUFFIX
        alpha_key = down_key[: -len(_DOWN_SUFFIX)] + _ALPHA_SUFFIX
        if up_key not in keys or alpha_key not in keys:
            raise ValueError(f"LoRA target {path} is missing its up factor or alpha")
        module = modules.get(path)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"LoRA target {path} does not resolve to nn.Linear")
        rank = _check_factors(
            module,
            shapes[down_key],
            shapes[up_key],
            shapes[alpha_key],
            path,
        )
        plan.append(_Target(path, down_key, up_key, alpha_key, module, rank))
        expected_keys.update((down_key, up_key, alpha_key))

    if up_keys != {target.up_key for target in plan}:
        raise ValueError("LoRA checkpoint contains orphan or unexpected up factors")
    if alpha_keys != {target.alpha_key for target in plan}:
        raise ValueError("LoRA checkpoint contains orphan or unexpected alpha tensors")
    if expected_keys != keys:
        unexpected = sorted(keys - expected_keys)
        raise ValueError(f"LoRA checkpoint contains unexpected tensor keys: {unexpected[:5]}")
    return plan


def plan_lora_tensors(
    linear_dimensions: Mapping[str, tuple[int, int]],
    keys: set[str],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    expected_targets: set[str] | None = None,
) -> dict[str, LoRATensorPlan]:
    """Validate adapter keys/shapes against ``path -> (in, out)`` dimensions.

    This metadata-only planner also works when base Linear weights have already
    been replaced by quantized modules, such as the streamed Kaggle loader.
    """
    down_keys = {key for key in keys if key.endswith(_DOWN_SUFFIX)}
    up_keys = {key for key in keys if key.endswith(_UP_SUFFIX)}
    alpha_keys = {key for key in keys if key.endswith(_ALPHA_SUFFIX)}
    if not down_keys:
        raise ValueError("LoRA checkpoint contains no .lora_down.weight tensors")

    plans: dict[str, LoRATensorPlan] = {}
    expected_keys: set[str] = set()
    for down_key in sorted(down_keys):
        path = _module_path_from_lora_key(down_key)
        if path in plans:
            raise ValueError(f"LoRA checkpoint contains duplicate target {path}")
        if path not in linear_dimensions:
            raise ValueError(f"LoRA target {path} is not a known Linear module")
        up_key = down_key[: -len(_DOWN_SUFFIX)] + _UP_SUFFIX
        alpha_key = down_key[: -len(_DOWN_SUFFIX)] + _ALPHA_SUFFIX
        if up_key not in keys or alpha_key not in keys:
            raise ValueError(f"LoRA target {path} is missing its up factor or alpha")
        in_features, out_features = linear_dimensions[path]
        down_shape = shapes[down_key]
        up_shape = shapes[up_key]
        if len(shapes[alpha_key]) != 0:
            raise ValueError(f"LoRA alpha for {path} must be a scalar")
        if len(down_shape) != 2 or len(up_shape) != 2:
            raise ValueError(f"LoRA factors for {path} must be matrices")
        rank, down_in = down_shape
        up_out, up_rank = up_shape
        if rank < 1 or up_rank != rank:
            raise ValueError(f"LoRA factor ranks do not match for {path}")
        if (down_in, up_out) != (in_features, out_features):
            raise ValueError(
                f"LoRA dimensions {down_shape}/{up_shape} do not match "
                f"Linear({in_features}, {out_features}) at {path}"
            )
        plans[path] = LoRATensorPlan(path, down_key, up_key, alpha_key, rank)
        expected_keys.update((down_key, up_key, alpha_key))

    if up_keys != {plan.up_key for plan in plans.values()}:
        raise ValueError("LoRA checkpoint contains orphan or unexpected up factors")
    if alpha_keys != {plan.alpha_key for plan in plans.values()}:
        raise ValueError("LoRA checkpoint contains orphan or unexpected alpha tensors")
    if expected_keys != keys:
        unexpected = sorted(keys - expected_keys)
        raise ValueError(f"LoRA checkpoint contains unexpected tensor keys: {unexpected[:5]}")
    if expected_targets is not None and set(plans) != expected_targets:
        missing = sorted(expected_targets - set(plans))
        unexpected = sorted(set(plans) - expected_targets)
        raise ValueError(
            f"LoRA target coverage mismatch; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return plans


def merge_lora_weight(
    weight: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: Tensor | float | int,
    *,
    strength: float = 1.0,
) -> Tensor:
    """Return a floating base matrix with one LoRA delta merged into it."""
    if weight.ndim != 2 or down.ndim != 2 or up.ndim != 2:
        raise ValueError("base weight and LoRA factors must be matrices")
    if not weight.is_floating_point() or not down.is_floating_point() or not up.is_floating_point():
        raise TypeError("base weight and LoRA factors must be floating point")
    rank, in_features = down.shape
    out_features, up_rank = up.shape
    if rank < 1 or up_rank != rank:
        raise ValueError("LoRA factor ranks must match and be greater than zero")
    if tuple(weight.shape) != (out_features, in_features):
        raise ValueError(
            f"LoRA dimensions {tuple(down.shape)}/{tuple(up.shape)} do not match "
            f"base weight {tuple(weight.shape)}"
        )
    alpha_value = float(alpha.item()) if isinstance(alpha, Tensor) else float(alpha)
    if not math.isfinite(alpha_value) or not math.isfinite(strength):
        raise ValueError("LoRA alpha and strength must be finite")
    weight32 = weight.to(dtype=torch.float32)
    down32 = down.to(device=weight.device, dtype=torch.float32)
    up32 = up.to(device=weight.device, dtype=torch.float32)
    delta = torch.matmul(up32, down32)
    delta.mul_(alpha_value * strength / rank)
    if not torch.isfinite(delta).all():
        raise ValueError("LoRA update contains non-finite values")
    return weight32 + delta


def _merge_one(
    target: _Target,
    down: Tensor,
    up: Tensor,
    alpha: Tensor | float | int,
    strength: float,
) -> None:
    if not down.is_floating_point() or not up.is_floating_point():
        raise TypeError(f"LoRA factors for {target.path} must be floating point")
    if tuple(down.shape) != (target.rank, target.module.in_features):
        raise ValueError(f"LoRA down factor shape changed for {target.path}")
    if tuple(up.shape) != (target.module.out_features, target.rank):
        raise ValueError(f"LoRA up factor shape changed for {target.path}")
    alpha_value = float(alpha.item()) if isinstance(alpha, Tensor) else float(alpha)
    if not math.isfinite(alpha_value) or not math.isfinite(strength):
        raise ValueError("LoRA alpha and strength must be finite")

    # Accumulate the update in fp32, then cast once to the base Linear dtype.
    merged = merge_lora_weight(
        target.module.weight,
        down,
        up,
        alpha_value,
        strength=strength,
    )
    with torch.no_grad():
        target.module.weight.copy_(merged.to(dtype=target.module.weight.dtype))


def merge_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    strength: float = 1.0,
) -> LoRAMergeReport:
    """Merge a Comfy-style or Wan-style LoRA state mapping into Linear layers."""
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    keys = set(state_dict)
    shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
    plan = _plan_targets(model, keys, shapes)
    for target in plan:
        _merge_one(
            target,
            state_dict[target.down_key],
            state_dict[target.up_key],
            state_dict[target.alpha_key],
            strength,
        )
    return LoRAMergeReport(
        targets_merged=len(plan),
        max_rank=max(target.rank for target in plan),
        strength=strength,
    )


def merge_lora_safetensors(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    strength: float = 1.0,
) -> LoRAMergeReport:
    """Stream and merge LoRA layers from a safetensors file.

    ``safetensors`` is imported only when this file-backed loader is used. It
    reads one target's factors at a time, keeping peak adapter tensor memory
    bounded to a pair of matrices rather than the complete 1.2 GB checkpoint.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Reading a LoRA safetensors file requires the safetensors package"
        ) from exc

    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        shapes = {key: tuple(reader.get_slice(key).get_shape()) for key in keys}
        plan = _plan_targets(model, keys, shapes)
        for target in plan:
            _merge_one(
                target,
                reader.get_tensor(target.down_key),
                reader.get_tensor(target.up_key),
                reader.get_tensor(target.alpha_key),
                strength,
            )
    return LoRAMergeReport(
        targets_merged=len(plan),
        max_rank=max(target.rank for target in plan),
        strength=strength,
    )
