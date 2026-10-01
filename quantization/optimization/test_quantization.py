import torch
from torch import nn

from quantization.optimization.config import PrecisionConfig, PrecisionMode
from quantization.optimization.quantization import describe_backend, resolve_backend


def test_int8_backend_reuses_existing_replacement_for_small_cpu_module():
    model = nn.Sequential(nn.Linear(3, 2))
    selected = describe_backend(PrecisionConfig(mode=PrecisionMode.INT8))
    assert selected.backend == "int8"
    replaced = resolve_backend(PrecisionConfig(mode=PrecisionMode.INT8)).apply(model)
    assert replaced == ["0"]
    assert model(torch.randn(1, 3)).shape == (1, 2)


def test_unimplemented_quantizers_report_status_and_fail_explicitly():
    for mode in (PrecisionMode.FP8, PrecisionMode.INT4):
        config = PrecisionConfig(mode=mode)
        selected = describe_backend(config)
        assert selected.status == "planned"
        try:
            resolve_backend(config)
        except NotImplementedError as error:
            assert mode.value.upper() in str(error)
        else:
            raise AssertionError(f"{mode.value} backend unexpectedly executable")
