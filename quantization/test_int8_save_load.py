import os
import torch
import torch.nn as nn

from int8_linear import Int8Linear
from quantize_model import quantize_linear_modules


# --------------------------------------------------
# Small Wan-like model
# --------------------------------------------------

class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.q = nn.Linear(512, 512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):

        y = self.q(x)
        y = y + self.ffn(x)

        return y


# --------------------------------------------------
# Create deterministic FP model
# --------------------------------------------------

torch.manual_seed(42)

model = TestModel().eval()

x = torch.randn(
    1,
    32,
    512,
)

# --------------------------------------------------
# Convert
# --------------------------------------------------

print("=" * 70)
print("INT8 SAVE / LOAD TEST")
print("=" * 70)

quantize_linear_modules(model)

model.eval()


# --------------------------------------------------
# Output before save
# --------------------------------------------------

with torch.no_grad():
    y_before = model(x)


# --------------------------------------------------
# Save
# --------------------------------------------------

path = "test_int8_checkpoint.pt"

torch.save(
    model.state_dict(),
    path,
)

print()
print("Saved:", path)
print(
    "File size:",
    os.path.getsize(path) / (1024 * 1024),
    "MB",
)


# --------------------------------------------------
# Fresh model
# --------------------------------------------------

fresh_model = TestModel().eval()

# Must construct the same INT8 architecture
quantize_linear_modules(fresh_model)

# Load
state = torch.load(
    path,
    map_location="cpu",
)

fresh_model.load_state_dict(state)

fresh_model.eval()


# --------------------------------------------------
# Output after reload
# --------------------------------------------------

with torch.no_grad():
    y_after = fresh_model(x)


# --------------------------------------------------
# Compare
# --------------------------------------------------

difference = (
    y_before - y_after
).abs()

max_difference = difference.max().item()
mean_difference = difference.mean().item()

print()
print("Max difference:", max_difference)
print("Mean difference:", mean_difference)

print(
    "Output shape:",
    y_after.shape,
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert y_before.shape == y_after.shape

assert torch.isfinite(y_after).all()

assert max_difference < 1e-6

print()
print("INT8 SAVE / LOAD TEST: PASS")
print("=" * 70)