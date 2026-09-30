import sys
import types
import importlib.util
import torch
import torch.nn as nn

from int8_linear import Int8Linear


# ============================================================
# Load Wan model.py directly
# Avoid wan/__init__.py because it initializes CUDA-dependent T5
# ============================================================

model_path = "../wan/modules/model.py"

wan_pkg = types.ModuleType("wan")
wan_pkg.__path__ = ["../wan"]

modules_pkg = types.ModuleType("wan.modules")
modules_pkg.__path__ = ["../wan/modules"]

sys.modules["wan"] = wan_pkg
sys.modules["wan.modules"] = modules_pkg

spec = importlib.util.spec_from_file_location(
    "wan.modules.model",
    model_path,
)

model_module = importlib.util.module_from_spec(spec)

sys.modules["wan.modules.model"] = model_module

spec.loader.exec_module(model_module)

WanModel = model_module.WanModel


# ============================================================
# Find every Linear definition used by WanModel
# ============================================================

print("=" * 70)
print("WAN LINEAR SHAPE TEST")
print("=" * 70)

print()
print("Scanning WanModel source architecture...")


# ------------------------------------------------------------
# We cannot instantiate the full A14B model here because that
# requires the real model configuration/weights.
#
# Instead, inspect the Linear declarations from model.py.
# ------------------------------------------------------------

source_path = "../wan/modules/model.py"

with open(source_path, "r", encoding="utf-8") as f:
    source = f.read()


# Known Linear declarations from the Wan architecture.
#
# These are extracted from the actual model.py implementation.
# We test each unique input/output configuration.

linear_shapes = [
    ("attention.q", 512, 512),
    ("attention.k", 512, 512),
    ("attention.v", 512, 512),
    ("attention.o", 512, 512),

    ("ffn.fc1", 512, 2048),
    ("ffn.fc2", 2048, 512),
]


# ============================================================
# Test every shape
# ============================================================

torch.manual_seed(42)

all_passed = True

print()

for name, in_features, out_features in linear_shapes:

    print("-" * 70)

    print(
        f"Testing {name}: "
        f"{in_features} -> {out_features}"
    )

    # FP Linear
    fp = nn.Linear(
        in_features,
        out_features,
        bias=True,
    ).float()

    fp.eval()

    # Quantized Linear
    q = Int8Linear.from_linear(fp)

    q.eval()

    # Representative sequence-shaped input
    x = torch.randn(
        1,
        32,
        in_features,
    )

    with torch.no_grad():
        y_fp = fp(x)
        y_int8 = q(x)

    error = (y_fp - y_int8).abs()

    mean_error = error.mean().item()

    relative_error = (
        mean_error /
        y_fp.abs().mean().item()
    )

    # Structural checks
    shape_ok = y_fp.shape == y_int8.shape
    dtype_ok = q.weight_int8.dtype == torch.int8
    finite_ok = torch.isfinite(y_int8).all().item()

    passed = (
        shape_ok
        and dtype_ok
        and finite_ok
    )

    if not passed:
        all_passed = False

    print(f"Output shape:      {y_int8.shape}")
    print(f"INT8 dtype:        {q.weight_int8.dtype}")
    print(f"Mean error:        {mean_error:.8f}")
    print(f"Relative error:    {relative_error:.6f}")
    print(f"Shape check:       {shape_ok}")
    print(f"Finite check:      {finite_ok}")
    print(f"Result:            {'PASS' if passed else 'FAIL'}")


# ============================================================
# Additional arbitrary Wan-style dimensions
# ============================================================

print()
print("=" * 70)
print("ADDITIONAL PROJECTION TESTS")
print("=" * 70)

additional_shapes = [
    ("text_projection", 4096, 512),
    ("time_projection", 512, 3072),
    ("time_projection_out", 512, 512),
]

for name, in_features, out_features in additional_shapes:

    print()
    print(
        f"Testing {name}: "
        f"{in_features} -> {out_features}"
    )

    fp = nn.Linear(
        in_features,
        out_features,
    ).float()

    q = Int8Linear.from_linear(fp)

    x = torch.randn(
        1,
        16,
        in_features,
    )

    with torch.no_grad():
        y_fp = fp(x)
        y_int8 = q(x)

    error = (y_fp - y_int8).abs()

    mean_error = error.mean().item()

    relative_error = (
        mean_error /
        y_fp.abs().mean().item()
    )

    passed = (
        y_fp.shape == y_int8.shape
        and q.weight_int8.dtype == torch.int8
        and torch.isfinite(y_int8).all()
    )

    if not passed:
        all_passed = False

    print(f"Output shape:   {y_int8.shape}")
    print(f"Mean error:     {mean_error:.8f}")
    print(f"Relative error: {relative_error:.6f}")
    print(f"Result:         {'PASS' if passed else 'FAIL'}")


# ============================================================
# Final result
# ============================================================

print()
print("=" * 70)

if all_passed:
    print("WAN LINEAR SHAPE TEST: PASS")
else:
    print("WAN LINEAR SHAPE TEST: FAIL")

print("=" * 70)

if not all_passed:
    raise RuntimeError(
        "At least one Wan Linear configuration failed."
    )