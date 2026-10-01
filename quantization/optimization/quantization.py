"""Backend resolution kept separate from model loading and placement."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .config import PrecisionConfig, PrecisionMode, PrecisionSelection, select_precision


@dataclass(frozen=True)
class QuantizationBackend:
    name: str
    selection: PrecisionSelection

    def apply(self, model):
        if not self.selection.executable:
            raise NotImplementedError(self.selection.message)
        if self.selection.mode is PrecisionMode.INT8:
            # Reuse the validated implementation unchanged.
            quantization_dir = str(Path(__file__).resolve().parents[1])
            if quantization_dir not in sys.path:
                sys.path.insert(0, quantization_dir)
            from quantize_model import quantize_linear_modules

            return quantize_linear_modules(model)
        raise NotImplementedError(
            f"No module replacement is needed or registered for {self.selection.mode.value}."
        )


def resolve_backend(config: PrecisionConfig) -> QuantizationBackend:
    selection = select_precision(config)
    if not selection.executable or selection.backend is None:
        raise NotImplementedError(selection.message)
    return QuantizationBackend(selection.backend, selection)


def describe_backend(config: PrecisionConfig) -> PrecisionSelection:
    """Return planned or implemented status without requiring execution support."""
    return select_precision(config)
