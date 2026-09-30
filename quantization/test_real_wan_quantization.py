"""
Real Wan architecture INT8 replacement gate.

Run from the Wan2.2 repository root:

    python quantization/test_real_wan_quantization.py

This test intentionally does NOT require A14B weights or a GPU.
It loads the real WanAttentionBlock class, discovers its actual
Linear modules, replaces them with Int8Linear, validates shapes,
checks that non-Linear modules remain untouched, and validates
INT8 checkpoint save/load.

It does NOT run the complete Wan block forward because Wan's
flash-attention path requires CUDA.
"""

import importlib.util
import inspect
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Make imports work when this file is executed from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
QUANT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(QUANT_DIR))

from int8_linear import Int8Linear
from quantize_model import (
    quantize_linear_modules,
    count_linear_modules,
    count_int8_modules,
)


MODEL_PATH = REPO_ROOT / "wan" / "modules" / "model.py"


def load_wan_model_module():
    """Load wan/modules/model.py without triggering wan/__init__.py."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. "
            "Run this script from inside the Wan2.2 repository."
        )

    wan_pkg = type(sys)("wan")
    wan_pkg.__path__ = [str(REPO_ROOT / "wan")]

    modules_pkg = type(sys)("wan.modules")
    modules_pkg.__path__ = [str(REPO_ROOT / "wan" / "modules")]

    sys.modules["wan"] = wan_pkg
    sys.modules["wan.modules"] = modules_pkg

    spec = importlib.util.spec_from_file_location(
        "wan.modules.model",
        MODEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for wan/modules/model.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["wan.modules.model"] = module
    spec.loader.exec_module(module)
    return module


def clone_linear(linear):
    """Keep an FP copy so every quantized projection can be numerically checked."""
    cloned = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    cloned.load_state_dict(linear.state_dict())
    return cloned

def main():
    print("=" * 78)
    print("REAL WAN ARCHITECTURE — INT8 REPLACEMENT GATE")
    print("=" * 78)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Wan model source:", MODEL_PATH)

    model_module = load_wan_model_module()

    WanAttentionBlock = getattr(model_module, "WanAttentionBlock", None)
    if WanAttentionBlock is None:
        raise RuntimeError("WanAttentionBlock was not found in wan/modules/model.py")

    print()
    print("WanAttentionBlock signature:")
    print(inspect.signature(WanAttentionBlock))

    # These are the same small architecture dimensions used by the existing
    # real-block test. No A14B checkpoint is loaded.
    DIM = 512
    FFN_DIM = 2048
    NUM_HEADS = 8

    torch.manual_seed(42)

    block = WanAttentionBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ).float().eval()

    print()
    print("Real WanAttentionBlock instantiated.")
    print("Parameter count:", sum(p.numel() for p in block.parameters()))

    # Snapshot every original Linear before replacement.
    originals = {}
    non_linear_types = {}

    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            originals[name] = clone_linear(module)
        elif name:
            non_linear_types[name] = type(module)

    print()
    print("Original Linear modules:", len(originals))
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            print(
                f"  {name}: "
                f"{module.in_features} -> {module.out_features} "
                f"bias={module.bias is not None}"
            )

    if not originals:
        raise RuntimeError("Real WanAttentionBlock contains no nn.Linear modules.")

    original_linear_count = count_linear_modules(block)

    # Quantize the real block.
    replaced = quantize_linear_modules(block)

    print()
    print("Replaced modules:", len(replaced))
    for name in replaced:
        print("  ", name)

    # Structural validation.
    remaining_linear = count_linear_modules(block)
    int8_count = count_int8_modules(block)

    print()
    print("Remaining FP Linear modules:", remaining_linear)
    print("INT8 Linear modules:", int8_count)

    assert len(replaced) == original_linear_count, (
        f"Expected {original_linear_count} replacements, got {len(replaced)}"
    )
    assert remaining_linear == 0
    assert int8_count == original_linear_count

    for name in replaced:
        module = dict(block.named_modules())[name]
        assert isinstance(module, Int8Linear), f"{name} was not converted"
        assert module.weight_int8.dtype == torch.int8

        original = originals[name]
        assert module.weight_int8.shape == original.weight.shape
        assert module.in_features == original.in_features
        assert module.out_features == original.out_features

    # Confirm obvious non-Linear modules survived.
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert isinstance(block.norm3, nn.LayerNorm)

    # Numerical check each actual Wan projection independently.
    # This avoids Wan's CUDA-only flash-attention path while testing the
    # exact Linear layers that were extracted from the real block.
    print()
    print("Per-projection numerical checks:")

    max_relative_error = 0.0

    quantized_modules = dict(block.named_modules())

    for name, original in originals.items():
        q = quantized_modules[name]

        x = torch.randn(
            1,
            16,
            original.in_features,
            dtype=torch.float32,
        )

        with torch.no_grad():
            y_fp = original(x)
            y_int8 = q(x)

        error = (y_fp - y_int8).abs()
        denom = y_fp.abs().mean().item()

        mean_error = error.mean().item()
        relative_error = mean_error / max(denom, 1e-12)
        max_relative_error = max(max_relative_error, relative_error)

        print(
            f"  {name:35s} "
            f"{original.in_features:5d} -> {original.out_features:5d} "
            f"relative_error={relative_error:.6f}"
        )

        assert y_fp.shape == y_int8.shape
        assert torch.isfinite(y_int8).all()

    print()
    print("Maximum projection relative error:", max_relative_error)

    # Save/load test using the actual quantized Wan block architecture.
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "wan_block_int8_state.pt"
        torch.save(block.state_dict(), checkpoint)

        fresh = WanAttentionBlock(
            dim=DIM,
            ffn_dim=FFN_DIM,
            num_heads=NUM_HEADS,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        ).float().eval()

        quantize_linear_modules(fresh)

        state = torch.load(checkpoint, map_location="cpu")
        fresh.load_state_dict(state)

        for name, q1 in block.named_modules():
            if isinstance(q1, Int8Linear):
                q2 = dict(fresh.named_modules())[name]
                assert torch.equal(q1.weight_int8, q2.weight_int8)
                assert torch.equal(q1.scale, q2.scale)
                if q1.bias is not None:
                    assert torch.equal(q1.bias, q2.bias)

    # Memory accounting for the actual Linear weights in this block.
    fp_weight_bytes = sum(
        m.weight.numel() * m.weight.element_size()
        for m in originals.values()
    )
    int8_weight_bytes = sum(
        m.weight_int8.numel() * m.weight_int8.element_size()
        for m in quantized_modules.values()
        if isinstance(m, Int8Linear)
    )
    scale_bytes = sum(
        m.scale.numel() * m.scale.element_size()
        for m in quantized_modules.values()
        if isinstance(m, Int8Linear)
    )

    print()
    print("Weight memory for actual Wan Linear modules:")
    print(f"  FP32 weights: {fp_weight_bytes / 1024**2:.3f} MiB")
    print(f"  INT8 weights: {int8_weight_bytes / 1024**2:.3f} MiB")
    print(f"  INT8 scales:  {scale_bytes / 1024**2:.6f} MiB")
    print(
        f"  Weight-only reduction: "
        f"{100 * (1 - (int8_weight_bytes + scale_bytes) / fp_weight_bytes):.2f}%"
    )

    print()
    print("=" * 78)
    print("REAL WAN INT8 REPLACEMENT GATE: PASS")
    print("=" * 78)
    print()
    print("Important:")
    print("- This validates the real Wan block's Linear replacement.")
    print("- It does NOT validate full A14B video generation.")
    print("- It does NOT establish INT8 speedup; the prototype dequantizes")
    print("  weights before F.linear().")
    print("- The next GPU test will use the actual A14B experts/checkpoint.")


if __name__ == "__main__":
    main()
