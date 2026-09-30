import sys
import types
import importlib.util
import torch


# --------------------------------------------------
# Load model.py without triggering wan/__init__.py
# --------------------------------------------------

model_path = "wan/modules/model.py"

wan_pkg = types.ModuleType("wan")
wan_pkg.__path__ = ["wan"]

modules_pkg = types.ModuleType("wan.modules")
modules_pkg.__path__ = ["wan/modules"]

sys.modules["wan"] = wan_pkg
sys.modules["wan.modules"] = modules_pkg

spec = importlib.util.spec_from_file_location(
    "wan.modules.model",
    model_path
)

model_module = importlib.util.module_from_spec(spec)
sys.modules["wan.modules.model"] = model_module
spec.loader.exec_module(model_module)

WanAttentionBlock = model_module.WanAttentionBlock


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = "cpu"
DTYPE = torch.float32

DIM = 512
FFN_DIM = 2048
NUM_HEADS = 8

BATCH = 1
SEQ_LEN = 32
TEXT_LEN = 16

# Wan attention block expects a 3D video grid.
# Keep it tiny for the CPU test.
T = 1
H = 4
W = 8

assert T * H * W == SEQ_LEN


# --------------------------------------------------
# Build real Wan block
# --------------------------------------------------

print("Loaded WanAttentionBlock directly.")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

block = WanAttentionBlock(
    dim=DIM,
    ffn_dim=FFN_DIM,
    num_heads=NUM_HEADS,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
).to(DEVICE).to(DTYPE)

block.eval()

print("Block created successfully.")
print(
    "Parameters:",
    sum(p.numel() for p in block.parameters())
)


# --------------------------------------------------
# Inputs
# --------------------------------------------------

x = torch.randn(
    BATCH,
    SEQ_LEN,
    DIM,
    device=DEVICE,
    dtype=DTYPE,
)

# Time/modulation embedding
e = torch.randn(
    BATCH,
    6,
    DIM,
    device=DEVICE,
    dtype=DTYPE,
)

# Text/context
context = torch.randn(
    BATCH,
    TEXT_LEN,
    DIM,
    device=DEVICE,
    dtype=DTYPE,
)

context_lens = [TEXT_LEN]

# Wan uses sequence lengths
seq_lens = [SEQ_LEN]

# Video grid
grid_sizes = torch.tensor(
    [[T, H, W]],
    dtype=torch.long,
    device=DEVICE,
)


# --------------------------------------------------
# RoPE frequencies
# --------------------------------------------------

# For this architecture we need rotary frequencies
# matching the attention head dimension.

head_dim = DIM // NUM_HEADS

freqs = torch.randn(
    T * H * W,
    head_dim // 2,
    device=DEVICE,
    dtype=DTYPE,
)


# --------------------------------------------------
# Forward
# --------------------------------------------------

import inspect

print()
print("Forward signature:")
print(inspect.signature(block.forward))

print()
print("Running real Wan block...")

with torch.no_grad():

    output = block(
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    )


if isinstance(output, tuple):
    y = output[0]
else:
    y = output


print()
print("Output shape:", y.shape)
print("Output dtype:", y.dtype)


# --------------------------------------------------
# Inspect Linear layers
# --------------------------------------------------

print()
print("Linear layers:")

linear_count = 0

for name, module in block.named_modules():

    if isinstance(module, torch.nn.Linear):

        linear_count += 1

        print(
            f"  {name}: "
            f"{module.in_features} -> "
            f"{module.out_features}"
        )


print()
print("Linear count:", linear_count)

print()
print("REAL WAN BLOCK TEST: PASS")