import torch
import torch.nn as nn

print("PyTorch:", torch.__version__)

# Match Wan-style projection
layer = nn.Linear(4096, 4096, bias=True).float()

# Simulate Wan hidden states
x = torch.randn(1, 256, 4096)

with torch.no_grad():
    y_fp32 = layer(x)

# Symmetric per-tensor INT8 weight quantization
weight = layer.weight.data.float()

scale = weight.abs().max() / 127.0

weight_int8 = torch.round(
    weight / scale
).clamp(-128, 127).to(torch.int8)

# Dequantize for computation
weight_dequant = weight_int8.float() * scale

with torch.no_grad():
    y_int8 = torch.nn.functional.linear(
        x,
        weight_dequant,
        layer.bias.float()
    )

# Compare
error = (y_fp32 - y_int8).abs()

relative_error = (
    error.mean() /
    y_fp32.abs().mean()
)

print()
print("Input:", x.shape)
print("FP32 output:", y_fp32.shape)
print("INT8 weights:", weight_int8.shape)
print("INT8 dtype:", weight_int8.dtype)
print("Scale:", scale.item())
print("Mean absolute error:", error.mean().item())
print("Relative error:", relative_error.item())

print()
print("Wan-style INT8 Linear prototype: PASS")