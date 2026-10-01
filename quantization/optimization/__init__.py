"""Composable optimization infrastructure for Wan inference experiments.

Configuration, adapter merging, and sampler components remain independently
testable; Kaggle's experimental runner composes them for video generation.
"""

from .config import (
    DistillationConfig,
    InferenceConfig,
    MixedPrecisionPolicy,
    PrecisionConfig,
    PrecisionMode,
    PrecisionSelection,
    select_precision,
)
from .euler import FlowEulerSchedule, build_flow_euler_schedule
from .experts import ExpertManager
from .instrumentation import Measurement, measure_stage
from .lora import LoRAMergeReport, merge_lora_safetensors, merge_lora_state_dict
from .model_loading import ExpertModelLoader, LoadedModel
from .sensitivity import ErrorMetrics, compare_modules, compare_outputs

__all__ = [
    "DistillationConfig",
    "InferenceConfig",
    "MixedPrecisionPolicy",
    "PrecisionConfig",
    "PrecisionMode",
    "PrecisionSelection",
    "select_precision",
    "FlowEulerSchedule",
    "build_flow_euler_schedule",
    "ExpertManager",
    "Measurement",
    "measure_stage",
    "ExpertModelLoader",
    "LoadedModel",
    "LoRAMergeReport",
    "merge_lora_state_dict",
    "merge_lora_safetensors",
    "ErrorMetrics",
    "compare_modules",
    "compare_outputs",
]
