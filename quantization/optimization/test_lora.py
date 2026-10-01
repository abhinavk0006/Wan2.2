import json
import struct

import pytest
import torch
from torch import nn

from quantization.optimization.lora import (
    merge_lora_safetensors,
    merge_lora_state_dict,
    plan_lora_tensors,
)


def make_model():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].self_attn = nn.Module()
    model.blocks[0].self_attn.q = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.blocks[0].self_attn.q.weight.zero_()
    return model


def comfy_lora(down=None, up=None, alpha=None):
    prefix = "diffusion_model.blocks.0.self_attn.q"
    return {
        f"{prefix}.lora_down.weight": down
        if down is not None
        else torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]),
        f"{prefix}.lora_up.weight": up
        if up is not None
        else torch.eye(2),
        f"{prefix}.alpha": alpha
        if alpha is not None
        else torch.tensor(4, dtype=torch.int64),
    }


def write_safetensors(tensors, path):
    dtype_names = {torch.float32: "F32", torch.int64: "I64"}
    header = {}
    data = bytearray()
    for key, tensor in tensors.items():
        raw = bytes(tensor.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        start = len(data)
        data.extend(raw)
        header[key] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def test_comfy_lora_keys_merge_to_matching_wan_module():
    model = make_model()
    report = merge_lora_state_dict(model, comfy_lora(), strength=0.5)
    expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    assert report.targets_merged == 1
    assert report.max_rank == 2
    assert torch.equal(model.blocks[0].self_attn.q.weight, expected)


def test_lora_merge_applies_alpha_over_rank():
    model = make_model()
    state = comfy_lora(alpha=torch.tensor(8, dtype=torch.int64))
    merge_lora_state_dict(model, state)
    expected = torch.tensor([[4.0, 8.0, 12.0], [8.0, 4.0, 0.0]])
    assert torch.equal(model.blocks[0].self_attn.q.weight, expected)


def test_lora_shape_mismatch_fails_before_mutating_model():
    model = make_model()
    state = comfy_lora(up=torch.ones(3, 2))
    with pytest.raises(ValueError, match="do not match"):
        merge_lora_state_dict(model, state)
    assert torch.count_nonzero(model.blocks[0].self_attn.q.weight) == 0


def test_lora_tensor_plan_validates_quantized_base_dimensions():
    tensors = comfy_lora()
    keys = set(tensors)
    shapes = {key: tuple(value.shape) for key, value in tensors.items()}
    plan = plan_lora_tensors(
        {"blocks.0.self_attn.q": (3, 2)},
        keys,
        shapes,
        expected_targets={"blocks.0.self_attn.q"},
    )
    assert plan["blocks.0.self_attn.q"].rank == 2
    with pytest.raises(ValueError, match="coverage mismatch"):
        plan_lora_tensors(
            {"blocks.0.self_attn.q": (3, 2)},
            keys,
            shapes,
            expected_targets={"blocks.0.self_attn.q", "blocks.0.ffn.0"},
        )


def test_lora_missing_factor_is_rejected_before_mutating_model():
    model = make_model()
    state = comfy_lora()
    state.pop("diffusion_model.blocks.0.self_attn.q.alpha")
    with pytest.raises(ValueError, match="missing its up factor or alpha"):
        merge_lora_state_dict(model, state)
    assert torch.count_nonzero(model.blocks[0].self_attn.q.weight) == 0


def test_streaming_safetensors_merge(tmp_path):
    pytest.importorskip("safetensors")
    checkpoint = tmp_path / "expert_lora.safetensors"
    write_safetensors(comfy_lora(), checkpoint)

    model = make_model()
    report = merge_lora_safetensors(model, checkpoint)
    expected = torch.tensor([[2.0, 4.0, 6.0], [4.0, 2.0, 0.0]])
    assert report.targets_merged == 1
    assert torch.equal(model.blocks[0].self_attn.q.weight, expected)
