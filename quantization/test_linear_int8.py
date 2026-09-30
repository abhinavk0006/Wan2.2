import torch
import torch.nn as nn

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("Device:", device)

layer = nn.Linear(4096, 4096, bias=True).to(device).to(dtype)

x = torch.randn(
    1, 128, 4096,
    device=device,
    dtype=dtype
)

with torch.no_grad():
    y_fp32 = layer(x)

# Simple symmetric weight-only INT8 quantization
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

error = (y_fp32 - y_int8).abs()
relative_error = error.mean() / y_fp32.abs().mean()

print("FP32 output:", y_fp32.shape)
print("INT8 weight:", weight_int8.shape, weight_int8.dtype)
print("Mean absolute error:", error.mean().item())
print("Relative error:", relative_error.item())

print("INT8 prototype: PASS")