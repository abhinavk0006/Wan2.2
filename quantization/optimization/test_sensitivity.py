import torch
from torch import nn

from quantization.optimization.sensitivity import compare_modules, compare_outputs


def test_layer_output_error_metrics_are_zero_for_equal_outputs():
    value = torch.tensor([1.0, -2.0, 3.0])
    metrics = compare_outputs(value, value.clone())
    assert metrics.elements == 3
    assert metrics.max_absolute_error == 0.0
    assert metrics.mean_absolute_error == 0.0
    assert metrics.relative_l2_error == 0.0
    assert abs(metrics.cosine_similarity - 1.0) < 1e-6


def test_module_comparison_supports_layer_and_block_output_trees():
    reference = nn.Linear(4, 3)
    candidate = nn.Linear(4, 3)
    candidate.load_state_dict(reference.state_dict())
    metrics = compare_modules(reference, candidate, torch.randn(2, 4))
    assert metrics.elements == 6
    assert metrics.root_mean_square_error == 0.0


def test_comparison_rejects_shape_mismatch():
    try:
        compare_outputs(torch.zeros(2, 3), torch.zeros(6))
    except ValueError as error:
        assert "structures/shapes" in str(error)
    else:
        raise AssertionError("expected shape mismatch to fail")
