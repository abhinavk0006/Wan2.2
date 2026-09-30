import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """
    Weight-only symmetric INT8 Linear.

    The stored weights are int8.
    Computation currently dequantizes the weights to the
    original floating-point dtype before F.linear().
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=torch.float32,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.compute_dtype = dtype

        self.register_buffer(
            "weight_int8",
            torch.empty(
                out_features,
                in_features,
                dtype=torch.int8,
            ),
        )

        self.register_buffer(
            "scale",
            torch.tensor(1.0, dtype=torch.float32),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty(
                    out_features,
                    dtype=dtype,
                ),
            )
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear):
        """
        Convert an existing nn.Linear into Int8Linear.
        """

        result = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )

        weight = linear.weight.detach().float()

        max_abs = weight.abs().max()

        if max_abs == 0:
            scale = torch.tensor(
                1.0,
                dtype=torch.float32,
                device=weight.device,
            )
        else:
            scale = max_abs / 127.0

        weight_int8 = torch.round(
            weight / scale
        ).clamp(-128, 127).to(torch.int8)

        result.weight_int8.copy_(
            weight_int8.cpu()
        )

        result.scale.copy_(
            scale.detach().float().cpu()
        )

        if linear.bias is not None:
            result.bias.copy_(
                linear.bias.detach().cpu()
            )

        return result

    def dequantize_weight(self):
        return (
            self.weight_int8.float()
            * self.scale
        )

    def forward(self, x):

        weight = self.dequantize_weight().to(
            device=x.device,
            dtype=x.dtype,
        )

        bias = None

        if self.bias is not None:
            bias = self.bias.to(
                device=x.device,
                dtype=x.dtype,
            )

        return F.linear(
            x,
            weight,
            bias,
        )