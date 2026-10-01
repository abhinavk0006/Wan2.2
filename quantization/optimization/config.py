"""Declarative inference precision and distillation settings.

Selecting a mode describes intent; it does not claim that its backend can run a
Wan expert. Call ``resolve_backend`` before applying a quantizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class PrecisionMode(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True)
class PrecisionConfig:
    mode: PrecisionMode = PrecisionMode.INT8
    mixed_precision: "MixedPrecisionPolicy" = field(default_factory=lambda: MixedPrecisionPolicy())

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PrecisionMode):
            object.__setattr__(self, "mode", PrecisionMode(self.mode))


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Dtypes for model storage, normal compute and optional named overrides."""

    parameter_dtype: str = "float16"
    compute_dtype: str = "float16"
    autocast_dtype: str | None = None
    module_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}
        configured = [self.parameter_dtype, self.compute_dtype]
        if self.autocast_dtype is not None:
            configured.append(self.autocast_dtype)
        configured.extend(self.module_overrides.values())
        invalid = sorted(set(configured) - allowed)
        if invalid:
            raise ValueError(f"Unsupported dtype name(s) in mixed precision policy: {invalid}")

    def dtype_for(self, module_name: str) -> str:
        """Return the most specific prefix override, else the compute dtype."""
        matches = [
            (prefix, dtype)
            for prefix, dtype in self.module_overrides.items()
            if module_name == prefix or module_name.startswith(prefix + ".")
        ]
        if not matches:
            return self.compute_dtype
        return max(matches, key=lambda item: len(item[0]))[1]


@dataclass(frozen=True)
class PrecisionSelection:
    mode: PrecisionMode
    backend: str | None
    status: str
    executable: bool
    message: str


@dataclass(frozen=True)
class DistillationConfig:
    """Settings declaring an intended distilled sampling path.

    The current values describe the LightX2V Wan2.2 I2V A14B four-step LoRA
    recipe. This records configuration only; it does not execute a sampler.
    """

    enabled: bool = False
    steps: int = 4
    artifact: str | None = None
    method: str | None = None
    solver: str = "euler"
    sample_shift: float = 5.0
    guide_scale: tuple[float, float] = (1.0, 1.0)

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.steps != 4:
            raise ValueError("This optimization layer currently models 4-step distillation only.")
        if self.enabled and not self.artifact:
            raise ValueError("Enabled 4-step distillation requires an artifact identifier or path.")
        if self.enabled and self.solver.lower() != "euler":
            raise ValueError("The configured Wan2.2 Lightning 4-step LoRA recipe requires the Euler solver.")
        if self.sample_shift <= 0:
            raise ValueError("sample_shift must be positive")
        if len(self.guide_scale) != 2 or any(scale < 0 for scale in self.guide_scale):
            raise ValueError("guide_scale must contain two non-negative expert scales")


@dataclass(frozen=True)
class InferenceConfig:
    sampling_steps: int = 40
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    def validate(self) -> None:
        self.distillation.validate()
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if self.sampling_steps == 4 and not self.distillation.enabled:
            raise ValueError(
                "Four sampling steps require an explicitly configured distilled model; "
                "reducing the step count alone is not distillation."
            )
        if self.distillation.enabled and self.sampling_steps != self.distillation.steps:
            raise ValueError("sampling_steps must match the configured distillation step count")


_BACKENDS = {
    PrecisionMode.FP16: (None, "native", True, "Native floating-point inference mode."),
    PrecisionMode.BF16: (None, "native", True, "Native floating-point inference mode; device support must be checked."),
    PrecisionMode.FP8: ("fp8", "planned", False, "FP8 is selectable, but no FP8 inference quantizer is implemented here."),
    PrecisionMode.INT8: ("int8", "implemented", True, "Uses the existing experimental weight-only Int8Linear backend."),
    PrecisionMode.INT4: ("int4", "planned", False, "INT4 is selectable, but no INT4 inference quantizer is implemented here."),
}


def select_precision(config: PrecisionConfig) -> PrecisionSelection:
    """Resolve the requested mode without loading a model or allocating a GPU."""
    try:
        backend, status, executable, message = _BACKENDS[config.mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported precision mode: {config.mode!r}") from exc
    return PrecisionSelection(config.mode, backend, status, executable, message)
