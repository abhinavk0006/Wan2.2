"""Load, INT8-replace, and smoke-test one real Wan I2V expert on Kaggle CUDA.

Run from the repository root with one Diffusers expert directory:

    python quantization/kaggle_single_expert.py \
        --expert-dir /kaggle/input/wan22-diffusers/high_noise_model \
        --report /kaggle/working/high_noise_int8_report.json

Or stream an FP8-scaled Comfy Wan checkpoint directly into INT8 Linear layers:

    python quantization/kaggle_single_expert.py \
        --checkpoint-file /kaggle/input/wan-high/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
        --report /kaggle/working/high_noise_int8_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from quantize_model import (  # noqa: E402
    count_int8_modules,
    count_linear_modules,
    quantize_linear_modules,
)
from int8_linear import Int8Linear  # noqa: E402
from quantization.optimization.lora import (  # noqa: E402
    merge_lora_weight,
    plan_lora_tensors,
)


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def tensor_storage_bytes(module: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in module.parameters()) + sum(
        t.numel() * t.element_size() for t in module.buffers()
    )


def save_report(report: dict, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to {path}")


def require_diffusers_directory(path: Path) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        if path.is_file() and path.suffix == ".safetensors":
            raise ValueError(
                "Got a standalone .safetensors checkpoint. Use --checkpoint-file "
                "for a Comfy Wan FP8-scaled checkpoint, or pass a Diffusers "
                "expert directory with --expert-dir."
            )
        raise ValueError(
            f"{path} is not a Diffusers expert directory (missing config.json). "
            "Pass the high_noise_model or low_noise_model directory itself."
        )


def _checkpoint_key_map(keys: list[str]) -> dict[str, str]:
    """Map common Comfy/Wan wrapper prefixes to this repo's WanModel names."""
    result: dict[str, str] = {}
    prefixes = ("model.diffusion_model.", "diffusion_model.", "transformer.")
    for original in keys:
        normalized = original
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        if normalized in result:
            raise ValueError(f"Checkpoint has duplicate normalized key {normalized!r}")
        result[normalized] = original
    return result


def _replace_linears_with_empty_int8(model: torch.nn.Module) -> None:
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                replacement = Int8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    dtype=torch.float16,
                ).to(device="meta")
                setattr(parent, name, replacement)


def _dequantize_fp8_tensor(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        result = tensor.float()
        if scale is not None:
            if scale.numel() != 1:
                raise ValueError(
                    "Only per-tensor FP8 scaling is supported by this checkpoint "
                    f"loader; found scale shape {tuple(scale.shape)}"
                )
            result.mul_(scale.float().reshape(()))
        return result
    if scale is not None:
        raise ValueError(
            "Found scale metadata for a non-FP8 tensor; refusing to guess how "
            "this checkpoint should be decoded."
        )
    return tensor.float()


def load_fp8_scaled_wan_checkpoint(
    checkpoint_file: Path,
    lora_file: Path | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Build a meta Wan I2V model and stream a Comfy FP8-scaled checkpoint into it.

    The old Comfy Wan format stores weights as FP8 with a scalar
    ``<name>.scale_weight``. Linear weights are dequantized one at a time and
    immediately reduced to this project's INT8 representation, avoiding a full
    FP16 expert copy in Kaggle RAM.
    """
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "FP8 checkpoint loading needs accelerate and safetensors. Install "
            "them in the Kaggle setup cell."
        ) from exc

    from wan.modules.model import WanModel

    if lora_file is not None and not lora_file.is_file():
        raise FileNotFoundError(lora_file)

    with ExitStack() as stack:
        handle = stack.enter_context(
            safe_open(str(checkpoint_file), framework="pt", device="cpu")
        )
        adapter = (
            stack.enter_context(
                safe_open(str(lora_file), framework="pt", device="cpu")
            )
            if lora_file is not None
            else None
        )
        raw_keys = list(handle.keys())
        key_map = _checkpoint_key_map(raw_keys)
        required = (
            "patch_embedding.weight",
            "text_embedding.0.weight",
            "blocks.0.ffn.0.weight",
            "head.head.weight",
        )
        missing = [name for name in required if name not in key_map]
        if missing:
            raise ValueError(
                "Checkpoint does not have the expected Wan transformer key layout. "
                f"Missing {missing}; sample keys: {raw_keys[:12]}"
            )

        shape = lambda key: tuple(handle.get_slice(key_map[key]).get_shape())
        patch_shape = shape("patch_embedding.weight")
        text_shape = shape("text_embedding.0.weight")
        ffn_shape = shape("blocks.0.ffn.0.weight")
        head_shape = shape("head.head.weight")
        block_ids = {
            int(match.group(1))
            for key in key_map
            if (match := re.match(r"blocks\.(\d+)\.", key))
        }
        if not block_ids:
            raise ValueError("No Wan transformer blocks found in checkpoint.")
        num_layers = max(block_ids) + 1
        if len(block_ids) != num_layers:
            raise ValueError("Wan block indices are not contiguous from zero.")

        if len(patch_shape) != 5:
            raise ValueError(f"Unsupported Wan patch embedding shape {patch_shape}")
        patch_size = (1, patch_shape[3], patch_shape[4])
        if patch_size != (1, 2, 2):
            raise ValueError(f"Unsupported Wan patch embedding shape {patch_shape}")
        dim = patch_shape[0]
        if dim % 128:
            raise ValueError(f"Cannot infer Wan attention heads from dim={dim}")
        out_dim = head_shape[0] // (patch_size[0] * patch_size[1] * patch_size[2])

        with init_empty_weights():
            model = WanModel(
                model_type="i2v",
                patch_size=patch_size,
                text_len=512,
                in_dim=patch_shape[1],
                dim=dim,
                ffn_dim=ffn_shape[0],
                freq_dim=256,
                text_dim=text_shape[1],
                out_dim=out_dim,
                num_heads=dim // 128,
                num_layers=num_layers,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
            )
        _replace_linears_with_empty_int8(model)

        lora_plans = {}
        if adapter is not None:
            linear_dimensions = {
                name: (module.in_features, module.out_features)
                for name, module in model.named_modules()
                if isinstance(module, Int8Linear) and name.startswith("blocks.")
            }
            adapter_keys = set(adapter.keys())
            adapter_shapes = {
                key: tuple(adapter.get_slice(key).get_shape())
                for key in adapter_keys
            }
            lora_plans = plan_lora_tensors(
                linear_dimensions,
                adapter_keys,
                adapter_shapes,
                expected_targets=set(linear_dimensions),
            )

        params_loaded = 0
        fp8_scaled_weights = 0
        lora_targets_merged = 0
        quantization_started = time.perf_counter()
        int8_linear_count = count_int8_modules(model)
        for name, module in model.named_modules():
            if isinstance(module, Int8Linear):
                weight_name = f"{name}.weight"
                if weight_name not in key_map:
                    raise ValueError(f"Checkpoint is missing {weight_name}")
                source_weight = handle.get_tensor(key_map[weight_name])
                scale_key = next(
                    (
                        key_map[candidate]
                        for candidate in (f"{name}.scale_weight", f"{name}.weight_scale")
                        if candidate in key_map
                    ),
                    None,
                )
                scale = handle.get_tensor(scale_key) if scale_key else None
                weight = _dequantize_fp8_tensor(source_weight, scale)
                if tuple(weight.shape) != (module.out_features, module.in_features):
                    raise ValueError(
                        f"{weight_name} shape {tuple(weight.shape)} does not match "
                        f"Linear {(module.out_features, module.in_features)}"
                    )
                lora_plan = lora_plans.get(name)
                if lora_plan is not None:
                    weight = merge_lora_weight(
                        weight,
                        adapter.get_tensor(lora_plan.down_key),
                        adapter.get_tensor(lora_plan.up_key),
                        adapter.get_tensor(lora_plan.alpha_key),
                    )
                    lora_targets_merged += 1
                maximum = weight.abs().max()
                quant_scale = maximum / 127.0 if maximum.item() else torch.tensor(1.0)
                quantized = torch.round(weight / quant_scale).clamp(-127, 127).to(torch.int8)
                set_module_tensor_to_device(
                    module, "weight_int8", "cpu", value=quantized
                )
                set_module_tensor_to_device(
                    module, "scale", "cpu", value=quant_scale.float().reshape(())
                )
                if source_weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    fp8_scaled_weights += 1
                if module.bias is not None:
                    bias_name = f"{name}.bias"
                    if bias_name not in key_map:
                        raise ValueError(f"Checkpoint is missing {bias_name}")
                    bias = handle.get_tensor(key_map[bias_name]).to(torch.float16)
                    set_module_tensor_to_device(module, "bias", "cpu", value=bias)
                params_loaded += 1
                del source_weight, weight, quantized, scale
                continue

            # Load parameters outside Linear layers (norms, modulation,
            # convolutions and embeddings), decoding scaled FP8 as necessary.
            for parameter_name, parameter in list(module.named_parameters(recurse=False)):
                full_name = f"{name}.{parameter_name}" if name else parameter_name
                if full_name not in key_map:
                    raise ValueError(f"Checkpoint is missing parameter {full_name}")
                source = handle.get_tensor(key_map[full_name])
                scale_key = next(
                    (
                        key_map[candidate]
                        for candidate in (f"{full_name[:-7]}.scale_weight", f"{full_name[:-7]}.weight_scale")
                        if candidate in key_map
                    ),
                    None,
                ) if full_name.endswith(".weight") else None
                scale = handle.get_tensor(scale_key) if scale_key else None
                value = _dequantize_fp8_tensor(source, scale).to(torch.float16)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"{full_name} shape {tuple(value.shape)} does not match "
                        f"model shape {tuple(parameter.shape)}"
                    )
                set_module_tensor_to_device(module, parameter_name, "cpu", value=value)
                if source.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    fp8_scaled_weights += 1
                params_loaded += 1
                del source, value, scale

        if count_int8_modules(model) != int8_linear_count:
            raise RuntimeError("Some Linear modules were not replaced with Int8Linear")
        if lora_targets_merged != len(lora_plans):
            raise RuntimeError(
                f"Merged {lora_targets_merged} LoRA targets, expected {len(lora_plans)}"
            )

        dense_fp16_bytes = sum(
            parameter.numel() * 2
            for parameter in model.parameters()
        ) + sum(
            module.weight_int8.numel() * 2
            + (
                module.bias.numel() * 2
                if module.bias is not None
                else 0
            )
            for module in model.modules()
            if isinstance(module, Int8Linear)
        )

    model.eval().requires_grad_(False)
    return model, {
        "checkpoint_format": "Comfy Wan FP8-scaled safetensors",
        "checkpoint_file": str(checkpoint_file),
        "model_parameters_loaded": params_loaded,
        "fp8_scaled_weights_decoded": fp8_scaled_weights,
        "int8_modules": int8_linear_count,
        "lora_file": str(lora_file) if lora_file else None,
        "lora_targets_merged": lora_targets_merged,
        "lora_max_rank": max(
            (target.rank for target in lora_plans.values()), default=0
        ),
        "model_storage_bytes_before": dense_fp16_bytes,
        "quantize_seconds": time.perf_counter() - quantization_started,
    }


def dispatch_wan_model_across_gpus(
    model: torch.nn.Module,
    gpu_ids: list[int],
    max_memory_gib: float = 8.0,
    max_memory_gib_by_device: dict[int, float] | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Place whole Wan transformer blocks across multiple CUDA devices.

    The two T4s in a Kaggle session do not pool VRAM automatically. Accelerate
    installs module hooks that move block inputs between the selected GPUs.
    Keep several GiB free on each device for attention and INT8 dequantization
    workspaces.
    """
    if len(gpu_ids) < 2:
        raise ValueError("At least two CUDA devices are required for model sharding.")
    if any(device_id < 0 or device_id >= torch.cuda.device_count() for device_id in gpu_ids):
        raise ValueError(f"Invalid CUDA device list: {gpu_ids}")

    try:
        from accelerate import dispatch_model
    except ImportError as exc:
        raise RuntimeError("Multi-GPU placement needs accelerate.") from exc

    budgets = {}
    for device_id in gpu_ids:
        budget = (max_memory_gib_by_device or {}).get(device_id, max_memory_gib)
        if budget <= 0:
            raise ValueError(f"GPU {device_id} memory budget must be positive, got {budget} GiB")
        budgets[device_id] = int(budget * 2**30)

    # Accelerate's size-based auto mapper fills devices in order. With the
    # asymmetric budgets used on two T4s that can leave the larger-budget
    # device carrying nearly all blocks and no activation headroom. Wan's
    # direct children are safe placement boundaries, so assign them with a
    # normalized-capacity greedy pass instead. This keeps every device below
    # its requested storage ceiling while balancing blocks by budget.
    children = list(model.named_children())
    if not children:
        raise ValueError("Wan model has no child modules to shard.")
    units = []
    for name, child in children:
        if name == "blocks":
            units.extend(
                (f"{name}.{block_name}", block)
                for block_name, block in child.named_children()
            )
        else:
            units.append((name, child))
    loads = {device_id: 0 for device_id in gpu_ids}
    device_map = {}
    for name, child in units:
        size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in list(child.parameters()) + list(child.buffers())
        )
        candidates = sorted(
            gpu_ids,
            key=lambda device_id: (
                loads[device_id] / budgets[device_id],
                loads[device_id],
            ),
        )
        selected = next(
            (device_id for device_id in candidates
             if loads[device_id] + size <= budgets[device_id]),
            None,
        )
        if selected is None:
            raise RuntimeError(
                "Wan model does not fit within the requested per-GPU memory "
                f"budgets without CPU/disk offload: module={name!r}, "
                f"module_bytes={size}, loads={loads}, budgets={budgets}"
            )
        device_map[name] = selected
        loads[selected] += size

    assigned_devices = set(device_map.values())
    allowed_devices = set(gpu_ids)
    if not assigned_devices.issubset(allowed_devices):
        raise RuntimeError(
            "Wan model does not fit within the requested per-GPU memory budget "
            f"without CPU/disk offload: {device_map}"
        )

    logging.info("Dispatching Wan modules over CUDA devices: %s (bytes=%s)", device_map, loads)
    model = dispatch_model(model, device_map=device_map, main_device=gpu_ids[0])
    return model, {str(module): device for module, device in device_map.items()}


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Kaggle, select a GPU accelerator.")

    expert_dir = args.expert_dir.resolve() if args.expert_dir else None
    checkpoint_file = args.checkpoint_file.resolve() if args.checkpoint_file else None
    if expert_dir:
        require_diffusers_directory(expert_dir)
    if checkpoint_file and not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    device = torch.device(f"cuda:{args.device}")
    dtype = dtype_from_name(args.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This GPU does not support BF16. Use --dtype float16 on a Tesla T4."
        )
    if args.two_gpu and torch.cuda.device_count() < 2:
        raise RuntimeError("--two-gpu needs at least two visible CUDA devices.")

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.2f} GiB)")
    print(f"Checkpoint: {checkpoint_file or expert_dir}")
    print(f"Load/compute dtype: {args.dtype}")
    torch.cuda.reset_peak_memory_stats(device)
    if args.two_gpu:
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    report = {
        "expert_dir": str(expert_dir) if expert_dir else None,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file else None,
        "gpu": props.name,
        "gpu_total_bytes": props.total_memory,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
    }

    # Keep the source expert on CPU while replacing layers to avoid an
    # FP16+INT8 GPU double allocation on a 16-GiB T4. For the standalone FP8
    # file, stream and quantize one tensor at a time.
    started = time.perf_counter()
    if checkpoint_file:
        model, source_metadata = load_fp8_scaled_wan_checkpoint(checkpoint_file)
        report.update(source_metadata)
    else:
        from wan.modules.model import WanModel

        model = WanModel.from_pretrained(
            str(expert_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
    report["load_seconds"] = time.perf_counter() - started
    report["linear_modules_before"] = (
        report["int8_modules"]
        if checkpoint_file
        else count_linear_modules(model)
    )
    report.setdefault("model_storage_bytes_before", tensor_storage_bytes(model))
    if checkpoint_file:
        report["linear_modules_after"] = count_linear_modules(model)
        report["int8_modules"] = count_int8_modules(model)
        report["model_storage_bytes_after"] = tensor_storage_bytes(model)
        report["replaced_modules"] = report["int8_modules"]
        report["weight_storage_reduction_percent"] = 100.0 * (
            1.0
            - report["model_storage_bytes_after"]
            / report["model_storage_bytes_before"]
        )
        print(
            f"Streamed checkpoint in {report['load_seconds']:.1f}s "
            f"({report['quantize_seconds']:.1f}s spent decoding/quantizing); "
            f"{report['int8_modules']} INT8 modules; "
            f"{report['model_storage_bytes_after'] / 1024**3:.2f} GiB storage"
        )
    else:
        report["linear_modules_before"] = count_linear_modules(model)
        report["model_storage_bytes_before"] = tensor_storage_bytes(model)
        print(
            f"Loaded one expert in {report['load_seconds']:.1f}s; "
            f"{report['linear_modules_before']} Linear modules; "
            f"{report['model_storage_bytes_before'] / 1024**3:.2f} GiB tensor storage"
        )

    if not checkpoint_file:
        started = time.perf_counter()
        replaced = quantize_linear_modules(model)
        report["quantize_seconds"] = time.perf_counter() - started
        report["replaced_modules"] = len(replaced)
        report["linear_modules_after"] = count_linear_modules(model)
        report["int8_modules"] = count_int8_modules(model)
        report["model_storage_bytes_after"] = tensor_storage_bytes(model)
        report["weight_storage_reduction_percent"] = 100.0 * (
            1.0 - report["model_storage_bytes_after"] / report["model_storage_bytes_before"]
        )
        if not replaced or report["linear_modules_after"] != 0:
            raise RuntimeError(
                "INT8 replacement incomplete: "
                f"replaced={len(replaced)}, remaining Linear="
                f"{report['linear_modules_after']}"
            )
        print(
            f"Replaced {len(replaced)} Linear modules in "
            f"{report['quantize_seconds']:.1f}s; model tensor storage is "
            f"{report['model_storage_bytes_after'] / 1024**3:.2f} GiB "
            f"({report['weight_storage_reduction_percent']:.1f}% reduction)"
        )

    gc.collect()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    try:
        if args.two_gpu:
            gpu_ids = [args.device] + [
                index
                for index in range(torch.cuda.device_count())
                if index != args.device
            ]
            model, device_map = dispatch_wan_model_across_gpus(model, gpu_ids)
            report["device_map"] = device_map
        else:
            model.to(device)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "GPU_LOAD_OOM"
        report["error"] = str(exc)
        report["gpu_allocated_at_failure_bytes"] = torch.cuda.memory_allocated(device)
        report["gpu_reserved_at_failure_bytes"] = torch.cuda.memory_reserved(device)
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        save_report(report, args.report)
        raise
    report["cuda_transfer_seconds"] = time.perf_counter() - started
    report["gpu_allocated_after_load_bytes"] = torch.cuda.memory_allocated(device)
    report["gpu_reserved_after_load_bytes"] = torch.cuda.memory_reserved(device)
    report["gpu_allocated_after_load_by_device"] = {
        str(index): torch.cuda.memory_allocated(index)
        for index in range(torch.cuda.device_count())
    }

    if not args.skip_forward:
        # Minimal I2V-shaped inputs exercise every block and its CUDA attention
        # path while keeping activation/workspace memory small.
        latent_channels = 16
        cond_channels = 20  # four-frame mask plus 16 VAE latent channels
        latent = torch.randn(
            latent_channels, 1, 2, 2, device=device, dtype=dtype
        )
        condition = torch.randn(cond_channels, 1, 2, 2, device=device, dtype=dtype)
        context = [torch.randn(8, model.text_dim, device=device, dtype=dtype)]
        timesteps = torch.tensor([500], device=device, dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = model(
                    [latent],
                    timesteps,
                    context,
                    seq_len=4,
                    y=[condition],
                )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as exc:
            report["status"] = "FORWARD_OOM"
            report["error"] = str(exc)
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
            save_report(report, args.report)
            raise
        report["forward_seconds"] = time.perf_counter() - started
        report["forward_output_shapes"] = [list(value.shape) for value in output]
        report["forward_output_finite"] = all(
            bool(torch.isfinite(value).all().item()) for value in output
        )
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["gpu_peak_allocated_by_device"] = {
            str(index): torch.cuda.max_memory_allocated(index)
            for index in range(torch.cuda.device_count())
        }
        print(
            f"Expert forward PASS in {report['forward_seconds']:.1f}s; "
            f"output={report['forward_output_shapes']}; "
            f"peak allocated={report['gpu_peak_allocated_bytes'] / 1024**3:.2f} GiB"
        )
        if not report["forward_output_finite"]:
            raise RuntimeError("Expert forward produced NaN or infinite values.")

    report["status"] = "PASS"
    save_report(report, args.report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--expert-dir",
        type=Path,
        help="One Diffusers expert folder containing config.json and model weights",
    )
    source_group.add_argument(
        "--checkpoint-file",
        type=Path,
        help="One Comfy Wan FP8-scaled .safetensors expert checkpoint",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--two-gpu",
        action="store_true",
        help="Shard whole Wan blocks across two GPUs, reserving about 6.5 GiB per T4 for workspaces",
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--report", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
