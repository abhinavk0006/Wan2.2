import pytest
import torch

from quantization.optimization.euler import build_flow_euler_schedule


def test_lightning_four_step_euler_schedule_matches_expected_values():
    schedule = build_flow_euler_schedule(
        train_steps=1000,
        inference_steps=4,
        shift=5.0,
    )
    expected_sigmas = torch.tensor([1.0, 0.9375, 5.0 / 6.0, 0.625, 0.0])
    assert schedule.steps == 4
    assert torch.allclose(schedule.sigmas, expected_sigmas)
    assert torch.allclose(schedule.timesteps, expected_sigmas[:-1] * 1000)


def test_euler_step_applies_flow_update_and_promotes_to_fp32():
    schedule = build_flow_euler_schedule(inference_steps=4, shift=5.0)
    sample = torch.ones((1, 2), dtype=torch.float16)
    velocity = torch.ones_like(sample)
    result = schedule.step(sample, velocity, index=0)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.full((1, 2), 0.9375))


def test_euler_schedule_validates_dimensions_and_indices():
    schedule = build_flow_euler_schedule()
    with pytest.raises(ValueError, match="same shape"):
        schedule.step(torch.zeros(2), torch.zeros(3), index=0)
    with pytest.raises(IndexError):
        schedule.step(torch.zeros(2), torch.zeros(2), index=4)
