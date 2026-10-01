"""Model-loader boundary kept independent from precision and swapping policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .instrumentation import Measurement, measure_stage


@dataclass
class LoadedModel:
    name: str
    model: Any
    measurement: Measurement


class ExpertModelLoader:
    """Wrap a model-specific callable and report its load duration.

    The callable owns checkpoint-format details. This class neither casts nor
    quantizes the resulting model, and it does not retain experts; lifecycle
    policy belongs to ``ExpertManager``.
    """

    def __init__(self, load_fn: Callable[[str], Any], device: str = "cpu") -> None:
        self.load_fn = load_fn
        self.device = device

    def load(self, name: str) -> LoadedModel:
        with measure_stage(f"load:{name}", self.device) as measurement:
            model = self.load_fn(name)
        return LoadedModel(name=name, model=model, measurement=measurement)
