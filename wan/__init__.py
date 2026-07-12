# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules

from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V

# Optional task families pull in extra dependencies such as decord and librosa.
# Keep them lazy so pure image/text generation still works in a minimal runtime.
try:
	from .speech2video import WanS2V
except Exception:  # pragma: no cover - optional dependency path
	WanS2V = None

try:
	from .animate import WanAnimate
except Exception:  # pragma: no cover - optional dependency path
	WanAnimate = None