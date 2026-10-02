"""Generate an image-to-video clip with Wan 2.2 I2V on Kaggle.

The script downloads the official T5 and VAE components and the two ComfyUI
FP8-scaled experts from Hugging Face. Expert files are downloaded and
converted to this repository's custom weight-only INT8 layers one at a time.

This is an experimental route: INT8 Linear weights are dequantized on each
forward pass, so generation can be slow. The full 14B experts may also exceed
one T4's available VRAM. This script is not a performance claim.

Use ``--lightning`` to merge the official four-step I2V-A14B LoRA into each
expert during streamed loading and sample with the matching shifted Euler path.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import sys
from pathlib import Path
from types import MethodType

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path("/kaggle/working/wan_models")
HF_I2V_REPO = "Wan-AI/Wan2.2-I2V-A14B"
HF_COMFY_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
HF_LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_ADAPTER_DIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
COMFY_PREFIX = "split_files/diffusion_models/"
EXPERT_FILES = {
    "high_noise_model": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "low_noise_model": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
}


def download(repo_id: str, filename: str, destination_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(destination_dir),
        )
    )
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face download did not produce {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=REPO_ROOT / "examples" / "i2v_input.JPG"
    )
    parser.add_argument(
        "--prompt",
        default="A gentle camera push-in. The subject moves naturally while the scene remains consistent.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working/wan_i2v_preview.mp4")
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--lightning",
        action="store_true",
        help="Merge the official 4-step I2V-A14B Lightning LoRAs into each expert before INT8 conversion, then use shifted Euler sampling",
    )
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=8.0,
        help="Maximum planned DiT storage per GPU; leave headroom for T4 activations",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional local Wan 2.2 I2V model directory with high_noise_model, low_noise_model, T5, and VAE files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this script.")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.device} is unavailable on this Kaggle session.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This staged 14B Kaggle path needs two visible GPUs so the expert can be sharded."
        )
    if not args.image.is_file():
        raise FileNotFoundError(
            f"Input image not found: {args.image}. Upload an image as a Kaggle input and pass --image /kaggle/input/.../your_image.png"
        )
    if args.lightning and args.model_dir:
        raise ValueError("--lightning currently requires the streamed Comfy FP8 checkpoint path; omit --model-dir.")
    if args.lightning and args.steps not in (4, 20):
        raise ValueError("The Lightning adapter is distilled for exactly 4 steps; pass --steps 4 or omit --steps.")
    if args.lightning:
        args.steps = 4
    if args.steps < 2:
        raise ValueError("Use at least 2 denoising steps so both experts can run.")
    if args.frames < 1 or (args.frames - 1) % 4:
        raise ValueError("Wan frame counts must be 4n+1, for example 17 or 49.")

    torch.cuda.set_device(args.device)
    logging.info("Using %s", torch.cuda.get_device_name(args.device))

    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V
    from wan.utils.utils import save_video

    config = copy.deepcopy(WAN_CONFIGS["i2v-A14B"])
    config.param_dtype = torch.float16
    config.t5_dtype = torch.bfloat16
    if args.lightning:
        config.sample_shift = 5.0
        config.sample_steps = 4
        config.sample_guide_scale = (1.0, 1.0)

    # The local-directory path is for a compatible full checkpoint prepared
    # elsewhere. The default path stages Comfy FP8 experts from Hugging Face.
    checkpoint_dir = args.model_dir or MODELS_DIR
    downloaded_shared = []
    if args.model_dir:
        if not args.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
        for component in (config.t5_checkpoint, config.vae_checkpoint):
            if not (args.model_dir / component).is_file():
                raise FileNotFoundError(
                    f"{args.model_dir / component} is missing from --model-dir"
                )
    else:
        downloaded_shared = [
            download(HF_I2V_REPO, config.t5_checkpoint, MODELS_DIR),
            download(HF_I2V_REPO, config.vae_checkpoint, MODELS_DIR),
        ]

    model = None
    try:
        model = WanI2V(
            config,
            str(checkpoint_dir),
            device_id=args.device,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=False,
        )
    finally:
        # Keep downloaded files if initialization failed so a retry does not
        # download the 11+ GiB T5 checkpoint and VAE again. On success, release
        # disk space before the much larger DiT experts are staged.
        if model is not None:
            for component_path in downloaded_shared:
                component_path.unlink(missing_ok=True)
        gc.collect()

    # This script generates one video per process. Release the 11+ GiB CPU T5
    # weights after prompt encoding, before streaming either 14B DiT expert.
    model.release_t5_after_encode = True

    if args.model_dir:
        # The upstream method loads both experts from Diffusers folders and
        # offloads inactive weights to CPU. This supports a full prepared pair.
        pass
    else:
        # The upstream loop calls this hook at every denoising step. Keep only
        # the active expert, stream it, and shard whole transformer blocks over
        # both T4s so neither card needs to hold the entire 14B expert.
        model._staged_expert_loading = True

        def prepare_expert(self, t, boundary, offload_model):
            del offload_model
            expert_name = (
                "high_noise_model" if t.item() >= boundary else "low_noise_model"
            )
            other_name = (
                "low_noise_model"
                if expert_name == "high_noise_model"
                else "high_noise_model"
            )
            expert = getattr(self, expert_name)
            if expert is None:
                other = getattr(self, other_name)
                if other is not None:
                    setattr(self, other_name, None)
                    del other
                    gc.collect()
                    torch.cuda.empty_cache()

                checkpoint = download(
                    HF_COMFY_REPO,
                    COMFY_PREFIX + EXPERT_FILES[expert_name],
                    MODELS_DIR,
                )
                adapter = None
                try:
                    if args.lightning:
                        adapter_filename = f"{expert_name}.safetensors"
                        adapter = download(
                            HF_LIGHTNING_REPO,
                            f"{LIGHTNING_ADAPTER_DIR}/{adapter_filename}",
                            MODELS_DIR / "lightning",
                        )
                    from kaggle_single_expert import load_fp8_scaled_wan_checkpoint
                    from kaggle_single_expert import dispatch_wan_model_across_gpus

                    logging.info(
                        "Converting %s to custom INT8 Linear layers%s",
                        expert_name,
                        " with merged 4-step Lightning LoRA" if adapter else "",
                    )
                    expert, details = load_fp8_scaled_wan_checkpoint(
                        checkpoint, lora_file=adapter
                    )
                    logging.info(
                        "Converted %s FP8 tensors; replaced %s Linear layers; merged %s LoRA targets",
                        details["fp8_scaled_weights_decoded"],
                        details["int8_modules"],
                        details["lora_targets_merged"],
                    )
                    gpu_ids = [self.device.index] + [
                        index
                        for index in range(torch.cuda.device_count())
                        if index != self.device.index
                    ]
                    expert, device_map = dispatch_wan_model_across_gpus(
                        expert, gpu_ids, max_memory_gib=args.max_memory_gib
                    )
                    logging.info("Expert device map: %s", device_map)
                    torch.cuda.synchronize(self.device)
                    setattr(self, expert_name, expert)
                finally:
                    checkpoint.unlink(missing_ok=True)
                    if adapter is not None:
                        adapter.unlink(missing_ok=True)
                    gc.collect()
                logging.info("Loaded %s on %s", expert_name, self.device)
            return getattr(self, expert_name)

        model._prepare_model_for_timestep = MethodType(prepare_expert, model)

        # WanI2V.generate() moves both experts to CPU at the end when
        # offload_model=True. In staged mode one field is None, so use the normal
        # path with offloading disabled and let the cleanup below release CUDA.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as image:
        video = model.generate(
            input_prompt=args.prompt,
            img=image.convert("RGB"),
            max_area=480 * 832,
            frame_num=args.frames,
            sample_solver="euler" if args.lightning else "unipc",
            sampling_steps=args.steps,
            guide_scale=(1.0, 1.0) if args.lightning else (3.5, 3.5),
            seed=42,
            shift=5.0,
            offload_model=bool(args.model_dir),
        )
    save_video(
        tensor=video[None],
        save_file=str(args.output),
        fps=config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Video writer did not produce a non-empty file at {args.output}")
    print(f"Video saved: {args.output}")
    del video, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
