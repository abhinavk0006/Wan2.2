import torch
import torch.nn as nn

from int8_linear import Int8Linear


torch.manual_seed(42)

IN_FEATURES = 4096
OUT_FEATURES = 4096

print("Creating FP32 Linear...")

fp = nn.Linear(
    IN_FEATURES,
    OUT_FEATURES,
).float()

fp.eval()


print("Converting to Int8Linear...")

q = Int8Linear.from_linear(fp)

q.eval()


# Same input
x = torch.randn(
    1,
    256,
    IN_FEATURES,
)


# FP32
with torch.no_grad():
    y_fp = fp(x)


# INT8
with torch.no_grad():
    y_int8 = q(x)


# Compare
error = (y_fp - y_int8).abs()

relative_error = (
    error.mean()
    / y_fp.abs().mean()
)


print()
print("FP output:", y_fp.shape)
print("INT8 output:", y_int8.shape)

print()
print(
    "Stored weight:",
    q.weight_int8.dtype,
    q.weight_int8.shape,
)

print(
    "Scale:",
    q.scale.item(),
)

print(
    "Mean absolute error:",
    error.mean().item(),
)

print(
    "Relative error:",
    relative_error.item(),
)


# Structural checks
assert q.weight_int8.dtype == torch.int8
assert q.weight_int8.shape == fp.weight.shape

assert y_fp.shape == y_int8.shape

assert torch.isfinite(y_int8).all()

print()
print("INT8 MODULE TEST: PASS")