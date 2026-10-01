"""Small CPU-safe timing and optional CUDA memory measurements."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class Measurement:
    label: str
    elapsed_seconds: float = 0.0
    device: str = "cpu"
    allocated_before_bytes: int | None = None
    allocated_after_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    reserved_before_bytes: int | None = None
    reserved_after_bytes: int | None = None
    peak_reserved_bytes: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class StageMeasurement:
    def __init__(self, label: str, device: str | torch.device = "cpu") -> None:
        self.result = Measurement(label=label, device=str(device))
        self._device = torch.device(device)
        self._start = 0.0

    def __enter__(self) -> Measurement:
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
            index = (
                torch.cuda.current_device()
                if self._device.index is None
                else self._device.index
            )
            self.result.allocated_before_bytes = torch.cuda.memory_allocated(index)
            self.result.reserved_before_bytes = torch.cuda.memory_reserved(index)
            torch.cuda.reset_peak_memory_stats(index)
        self._start = time.perf_counter()
        return self.result

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
            index = (
                torch.cuda.current_device()
                if self._device.index is None
                else self._device.index
            )
            self.result.allocated_after_bytes = torch.cuda.memory_allocated(index)
            self.result.reserved_after_bytes = torch.cuda.memory_reserved(index)
            self.result.peak_allocated_bytes = torch.cuda.max_memory_allocated(index)
            self.result.peak_reserved_bytes = torch.cuda.max_memory_reserved(index)
        self.result.elapsed_seconds = time.perf_counter() - self._start
        return False


def measure_stage(label: str, device: str | torch.device = "cpu") -> StageMeasurement:
    """Use as ``with measure_stage('load', device) as measurement: ...``."""
    return StageMeasurement(label, device)
