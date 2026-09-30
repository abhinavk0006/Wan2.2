import torch
import torch.nn as nn

from int8_linear import Int8Linear
from quantize_model import (
    quantize_linear_modules,
    count_linear_modules,
    count_int8_modules,
)


# --------------------------------------------------
# Build a Wan-like nested model
# --------------------------------------------------

class FakeWanBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.q = nn.Linear(512, 512)
        self.k = nn.Linear(512, 512)
        self.v = nn.Linear(512, 512)
        self.o = nn.Linear(512, 512)

        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):

        # Simplified attention-like projection path
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # We aren't testing real attention here.
        # Just exercise the quantized Linear modules.
        y = self.o(v)

        # FFN
        y = y + self.ffn(x)

        return y


class FakeWanModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            FakeWanBlock(),
            FakeWanBlock(),
        ])

        self.text_projection = nn.Linear(
            4096,
            512,
        )

        self.norm = nn.LayerNorm(512)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = FakeWanModel()

print("=" * 70)
print("AUTOMATIC LINEAR REPLACEMENT TEST")
print("=" * 70)

print()
print(
    "Original Linear modules:",
    count_linear_modules(model),
)

print(
    "Original INT8 modules:",
    count_int8_modules(model),
)


# --------------------------------------------------
# Preserve a non-Linear reference
# --------------------------------------------------

original_norm_type = type(model.norm)


# --------------------------------------------------
# Quantize
# --------------------------------------------------

replaced = quantize_linear_modules(model)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("Replaced modules:")

for name in replaced:
    print("  ", name)


print()
print(
    "Remaining FP Linear modules:",
    count_linear_modules(model),
)

print(
    "INT8 Linear modules:",
    count_int8_modules(model),
)


# --------------------------------------------------
# Structural validation
# --------------------------------------------------

assert len(replaced) == 13

assert count_linear_modules(model) == 0

assert count_int8_modules(model) == 13

assert type(model.norm) is original_norm_type

assert isinstance(
    model.blocks[0].q,
    Int8Linear,
)

assert isinstance(
    model.blocks[0].ffn[0],
    Int8Linear,
)

assert isinstance(
    model.blocks[0].ffn[1],
    nn.GELU,
)

assert isinstance(
    model.norm,
    nn.LayerNorm,
)


# --------------------------------------------------
# Forward test
# --------------------------------------------------

x = torch.randn(
    1,
    32,
    512,
)

with torch.no_grad():

    y = model.blocks[0](x)


print()
print(
    "Quantized block output:",
    y.shape,
)

assert torch.isfinite(y).all()


print()
print("AUTOMATIC REPLACEMENT TEST: PASS")
print("=" * 70)