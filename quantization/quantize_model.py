import torch
import torch.nn as nn

from int8_linear import Int8Linear


def quantize_linear_modules(
    module: nn.Module,
    prefix: str = "",
    exclude: tuple[str, ...] = (),
):
    """
    Recursively replace nn.Linear modules with Int8Linear.

    Returns:
        list of replaced module names
    """

    replaced = []

    for name, child in list(module.named_children()):

        full_name = (
            f"{prefix}.{name}"
            if prefix
            else name
        )

        # Skip explicitly excluded modules
        if any(
            full_name.startswith(pattern)
            for pattern in exclude
        ):
            continue

        # Replace Linear
        if isinstance(child, nn.Linear):

            quantized = Int8Linear.from_linear(
                child
            )

            setattr(
                module,
                name,
                quantized,
            )

            replaced.append(full_name)

        else:

            replaced.extend(
                quantize_linear_modules(
                    child,
                    prefix=full_name,
                    exclude=exclude,
                )
            )

    return replaced


def count_linear_modules(module):
    return sum(
        1
        for m in module.modules()
        if isinstance(m, nn.Linear)
    )


def count_int8_modules(module):
    return sum(
        1
        for m in module.modules()
        if isinstance(m, Int8Linear)
    )