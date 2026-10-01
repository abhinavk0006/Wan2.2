"""Standalone flow-matching Euler schedule used by Wan Lightning distillation.

The formulas match the public LightX2V Wan2.2 Lightning Euler scheduler. This
small component deliberately does not call or modify the Wan generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowEulerSchedule:
    """Inference timesteps and sigma values for flow-matching Euler updates."""

    timesteps: Tensor
    sigmas: Tensor

    @property
    def steps(self) -> int:
        return int(self.timesteps.numel())

    def step(self, sample: Tensor, model_output: Tensor, index: int) -> Tensor:
        """Apply one explicit Euler update, returning the next sample in fp32."""
        if index < 0 or index >= self.steps:
            raise IndexError(f"Euler step index {index} is outside [0, {self.steps})")
        if sample.shape != model_output.shape:
            raise ValueError("sample and model_output must have the same shape")
        sigma = self.sigmas[index].to(device=sample.device, dtype=torch.float32)
        sigma_next = self.sigmas[index + 1].to(
            device=sample.device, dtype=torch.float32
        )
        sample32 = sample.to(torch.float32)
        output32 = model_output.to(device=sample.device, dtype=torch.float32)
        return sample32 + (sigma_next - sigma) * output32


def build_flow_euler_schedule(
    *,
    train_steps: int = 1000,
    inference_steps: int = 4,
    shift: float = 5.0,
    device: str | torch.device = "cpu",
) -> FlowEulerSchedule:
    """Build the shifted linear flow-matching schedule without diffusers.

    The returned ``timesteps`` contains one model timestep per inference step;
    ``sigmas`` has one additional final value (zero) for the last Euler update.
    """
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    if inference_steps < 1:
        raise ValueError("inference_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")

    base_timesteps = torch.linspace(
        float(train_steps), 0.0, inference_steps + 1, dtype=torch.float32
    )
    sigmas = base_timesteps / float(train_steps)
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas = sigmas.to(device=device)
    timesteps = (sigmas * float(train_steps))[:-1].clone()
    return FlowEulerSchedule(timesteps=timesteps, sigmas=sigmas)
