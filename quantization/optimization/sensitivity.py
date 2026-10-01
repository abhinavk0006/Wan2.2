"""Layer/block output error metrics for controlled precision experiments.

This module only compares forward outputs. It does not run video generation,
select a quantization, or perform a sensitivity sweep automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ErrorMetrics:
    elements: int
    max_absolute_error: float
    mean_absolute_error: float
    root_mean_square_error: float
    relative_l2_error: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_tensor_tree(value: Any) -> tuple[list[torch.Tensor], Any]:
    if torch.is_tensor(value):
        return [value.detach().float().reshape(-1)], ("tensor", tuple(value.shape))
    if isinstance(value, (tuple, list)):
        children = [_as_tensor_tree(item) for item in value]
        tensors = [tensor for child, _ in children for tensor in child]
        return tensors, (type(value).__name__, tuple(shape for _, shape in children))
    if isinstance(value, dict):
        keys = sorted(value)
        children = {key: _as_tensor_tree(value[key]) for key in keys}
        tensors = [
            tensor for key in keys for tensor in children[key][0]
        ]
        return tensors, ("dict", tuple((key, children[key][1]) for key in keys))
    raise TypeError(f"Unsupported module output type: {type(value).__name__}")


def compare_outputs(reference: Any, candidate: Any) -> ErrorMetrics:
    """Compare corresponding tensors from a layer or block output tree."""
    refs, ref_structure = _as_tensor_tree(reference)
    cands, candidate_structure = _as_tensor_tree(candidate)
    if ref_structure != candidate_structure:
        raise ValueError("Reference and candidate output structures/shapes do not match")
    if not refs:
        raise ValueError("Outputs must contain at least one tensor")
    ref = torch.cat(refs)
    cand = torch.cat([value.to(device=ref.device) for value in cands])
    delta = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    cand_norm = torch.linalg.vector_norm(cand)
    denom = ref_norm.clamp_min(torch.finfo(ref.dtype).eps)
    cosine_denom = (ref_norm * cand_norm).clamp_min(torch.finfo(ref.dtype).eps)
    return ErrorMetrics(
        elements=ref.numel(),
        max_absolute_error=float(delta.abs().max().item()) if delta.numel() else 0.0,
        mean_absolute_error=float(delta.abs().mean().item()) if delta.numel() else 0.0,
        root_mean_square_error=float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0,
        relative_l2_error=float((torch.linalg.vector_norm(delta) / denom).item()),
        cosine_similarity=float(((ref * cand).sum() / cosine_denom).item()),
    )


def compare_modules(reference_module, candidate_module, *args, **kwargs) -> ErrorMetrics:
    """Run two layers/blocks on identical inputs and compare their outputs."""
    reference_module.eval()
    candidate_module.eval()
    with torch.inference_mode():
        reference = reference_module(*args, **kwargs)
        candidate = candidate_module(*args, **kwargs)
    return compare_outputs(reference, candidate)
