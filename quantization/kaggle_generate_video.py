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
import hashlib
import json
import logging
import math
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
DAEMON_CUDA_OOM_EXIT_CODE = 75


class WanCudaOOMError(RuntimeError):
    """Fatal worker error carrying diagnostics for the parent pipeline."""

    error_type = "cuda_oom"
    exit_code = DAEMON_CUDA_OOM_EXIT_CODE


def cuda_oom_diagnostics(error: BaseException) -> str:
    """Return actionable per-GPU state for a CUDA allocation failure."""
    details = [f"{type(error).__name__}: {error}"]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            details.append(
                "gpu=%d allocated=%.2fGiB reserved=%.2fGiB free=%.2fGiB total=%.2fGiB peak=%.2fGiB"
                % (
                    index,
                    torch.cuda.memory_allocated(index) / 2**30,
                    torch.cuda.memory_reserved(index) / 2**30,
                    free / 2**30,
                    total / 2**30,
                    torch.cuda.max_memory_allocated(index) / 2**30,
                )
            )
    return "CUDA out of memory during Wan2.2 generation; " + "; ".join(details)


def raise_cuda_oom(error: BaseException) -> None:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        raise WanCudaOOMError(cuda_oom_diagnostics(error)) from error
    raise error


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
    parser.add_argument("--negative-prompt", default="")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--max-memory-gib",
        type=float,
        default=8.0,
        help="Maximum planned DiT storage per GPU; leave headroom for T4 activations",
    )
    parser.add_argument(
        "--gpu0-memory-gib", type=float,
        help="Override the planned DiT storage budget for CUDA device 0",
    )
    parser.add_argument(
        "--gpu1-memory-gib", type=float,
        help="Override the planned DiT storage budget for CUDA device 1",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional local Wan 2.2 I2V model directory with high_noise_model, low_noise_model, T5, and VAE files",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=480 * 832,
        help="Maximum image area passed to Wan. Lower this for longer shots when activation VRAM is limiting.",
    )
    parser.add_argument(
        "--memory-telemetry",
        action="store_true",
        help="Log allocated, reserved, free, and total CUDA memory around major stages.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Keep the model loaded and process JSON task objects from stdin.",
    )
    return parser


def log_cuda_memory(label: str) -> None:
    """Log per-device memory without changing allocation behavior."""
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        logging.info(
            "CUDA memory %s gpu=%d allocated=%.2fGiB reserved=%.2fGiB free=%.2fGiB total=%.2fGiB",
            label,
            index,
            torch.cuda.memory_allocated(index) / 2**30,
            torch.cuda.memory_reserved(index) / 2**30,
            free / 2**30,
            total / 2**30,
        )


def normalize_daemon_task(task: dict, *, sample_fps: int = 16) -> dict:
    """Validate and normalize the wrapper-compatible JSON task fields."""
    required = ("input_image", "output_video", "prompt")
    missing = [key for key in required if not str(task.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing daemon task field(s): {', '.join(missing)}")
    normalized = {
        "image": task["input_image"],
        "output": task["output_video"],
        "prompt": task["prompt"],
        "negative_prompt": task.get("negative_prompt", ""),
    }
    if "frames" in task:
        try:
            normalized["frames"] = int(task["frames"])
        except (TypeError, ValueError) as error:
            raise ValueError("frames must be an integer") from error
    elif "clip_duration" in task:
        duration = float(task["clip_duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("clip_duration must be a finite number greater than zero")
        normalized["frames"] = max(17, 4 * math.ceil((duration * sample_fps - 1) / 4) + 1)
    if "seed" in task:
        try:
            normalized["seed"] = int(task["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError("seed must be an integer") from error
    elif task.get("clip_name"):
        normalized["seed"] = int.from_bytes(
            hashlib.sha256(str(task["clip_name"]).encode("utf-8")).digest()[:4],
            "big",
        )
    for key in ("steps", "max_area"):
        if key in task:
            try:
                normalized[key] = int(task[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} must be an integer") from error
    if "frames" in normalized and (
        normalized["frames"] < 1 or (normalized["frames"] - 1) % 4
    ):
        raise ValueError("frames must be 4n+1, for example 17 or 49")
    if "steps" in normalized and normalized["steps"] < 2:
        raise ValueError("steps must be at least 2")
    if "max_area" in normalized and normalized["max_area"] <= 0:
        raise ValueError("max_area must be greater than zero")
    return normalized


class WanVideoWorker:
    """Reusable Wan runtime with staged, one-expert-at-a-time loading."""

    def __init__(self, args: argparse.Namespace, *, daemon: bool = False):
        self.args = args
        self._validate_runtime_args()
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU accelerator before running this script.")
        if args.device < 0 or args.device >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.device} is unavailable on this Kaggle session.")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("This staged 14B Kaggle path needs two visible GPUs so the expert can be sharded.")
        torch.cuda.set_device(args.device)
        logging.info("Using %s", torch.cuda.get_device_name(args.device))
        if args.memory_telemetry:
            log_cuda_memory("startup")

        from wan.configs import WAN_CONFIGS
        from wan.image2video import WanI2V
        config = copy.deepcopy(WAN_CONFIGS["i2v-A14B"])
        config.param_dtype = torch.float16
        config.t5_dtype = torch.bfloat16
        if args.lightning:
            config.sample_shift, config.sample_steps = 5.0, 4
            config.sample_guide_scale = (1.0, 1.0)

        checkpoint_dir = args.model_dir or MODELS_DIR
        shared = []
        if args.model_dir:
            if not args.model_dir.is_dir():
                raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
            for component in (config.t5_checkpoint, config.vae_checkpoint):
                if not (args.model_dir / component).is_file():
                    raise FileNotFoundError(f"{args.model_dir / component} is missing from --model-dir")
        else:
            shared = [download(HF_I2V_REPO, config.t5_checkpoint, MODELS_DIR),
                      download(HF_I2V_REPO, config.vae_checkpoint, MODELS_DIR)]
        model = None
        try:
            model = WanI2V(config, str(checkpoint_dir), device_id=args.device,
                           t5_cpu=True, init_on_cpu=True, convert_model_dtype=False)
        finally:
            if model is not None:
                for path in shared:
                    path.unlink(missing_ok=True)
            gc.collect()
        self.config, self.model = config, model
        model.release_t5_after_encode = not daemon
        model.vae_cpu_offload = not bool(args.model_dir)
        self._t5_checkpoint = checkpoint_dir / config.t5_checkpoint
        self._t5_tokenizer_path = config.t5_tokenizer
        self._t5_released_for_expert_swap = False
        if not args.model_dir:
            model._staged_expert_loading = True
            model._wan_worker = self
            model._prepare_model_for_timestep = MethodType(WanVideoWorker._prepare_expert, model)

    def _release_t5_for_expert_swap(self) -> None:
        """Free CPU T5 weights before converting the opposite DiT expert."""
        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is None or text_encoder.model is None:
            return
        text_encoder.model = None
        gc.collect()
        self._t5_released_for_expert_swap = True
        logging.info("Released CPU T5 weights before expert conversion")

    def _restore_t5_after_expert_swap(self) -> None:
        """Restore T5 after expert conversion so the daemon can serve the next task."""
        if not self._t5_released_for_expert_swap:
            return
        if not self._t5_checkpoint.is_file():
            self._t5_checkpoint = download(
                HF_I2V_REPO, self.config.t5_checkpoint, MODELS_DIR
            )
        from wan.modules.t5 import T5EncoderModel

        tokenizer_path = self._t5_tokenizer_path
        local_tokenizer_path = self._t5_checkpoint.parent / tokenizer_path
        if local_tokenizer_path.is_dir():
            tokenizer_path = str(local_tokenizer_path)
        self.model.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(self._t5_checkpoint),
            tokenizer_path=tokenizer_path,
        )
        self._t5_released_for_expert_swap = False
        logging.info("Restored CPU T5 weights after expert conversion")

    def _validate_runtime_args(self):
        args = self.args
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
        if args.max_area <= 0:
            raise ValueError("--max-area must be greater than zero.")

    def _prepare_expert(self, t, boundary, offload_model):
        del offload_model
        worker = self._wan_worker
        args = worker.args
        expert_name = "high_noise_model" if t.item() >= boundary else "low_noise_model"
        other_name = "low_noise_model" if expert_name == "high_noise_model" else "high_noise_model"
        expert = getattr(self, expert_name)
        if expert is None:
            other = getattr(self, other_name)
            if other is not None:
                worker._release_t5_for_expert_swap()
                setattr(self, other_name, None)
                del other
                gc.collect()
                torch.cuda.empty_cache()
            checkpoint = download(HF_COMFY_REPO, COMFY_PREFIX + EXPERT_FILES[expert_name], MODELS_DIR)
            adapter = None
            try:
                if args.lightning:
                    adapter = download(HF_LIGHTNING_REPO,
                                       f"{LIGHTNING_ADAPTER_DIR}/{expert_name}.safetensors",
                                       MODELS_DIR / "lightning")
                from kaggle_single_expert import load_fp8_scaled_wan_checkpoint, dispatch_wan_model_across_gpus
                logging.info("Converting %s to custom INT8 Linear layers%s", expert_name,
                             " with merged 4-step Lightning LoRA" if adapter else "")
                expert, details = load_fp8_scaled_wan_checkpoint(checkpoint, lora_file=adapter)
                logging.info("Converted %s FP8 tensors; replaced %s Linear layers; merged %s LoRA targets",
                             details["fp8_scaled_weights_decoded"], details["int8_modules"], details["lora_targets_merged"])
                gpu_ids = [self.device.index] + [i for i in range(torch.cuda.device_count())
                                                  if i != self.device.index]
                expert, device_map = dispatch_wan_model_across_gpus(
                    expert, gpu_ids, max_memory_gib=args.max_memory_gib,
                    max_memory_gib_by_device={i: budget for i, budget in
                                              ((0, args.gpu0_memory_gib), (1, args.gpu1_memory_gib))
                                              if budget is not None})
                logging.info("Expert device map: %s", device_map)
                if args.memory_telemetry:
                    log_cuda_memory(f"after-{expert_name}")
                torch.cuda.synchronize(self.device)
                setattr(self, expert_name, expert)
            finally:
                checkpoint.unlink(missing_ok=True)
                if adapter is not None:
                    adapter.unlink(missing_ok=True)
                gc.collect()
            worker._restore_t5_after_expert_swap()
            logging.info("Loaded %s on %s", expert_name, self.device)
        return getattr(self, expert_name)

    def cleanup(self):
        for name in ("high_noise_model", "low_noise_model"):
            expert = getattr(self.model, name, None)
            if expert is not None:
                setattr(self.model, name, None)
                del expert
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, task: dict) -> Path:
        args = copy.copy(self.args)
        normalized = normalize_daemon_task(task, sample_fps=self.config.sample_fps)
        for key, value in normalized.items():
            setattr(args, key, value)
        args.image, args.output = Path(args.image), Path(args.output)
        if not args.image.is_file():
            raise FileNotFoundError(f"Input image not found: {args.image}")
        if args.frames < 1 or (args.frames - 1) % 4:
            raise ValueError("Wan frame counts must be 4n+1, for example 17 or 49.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(args.image) as image:
                video = self.model.generate(input_prompt=args.prompt, n_prompt=args.negative_prompt,
                    img=image.convert("RGB"), max_area=args.max_area, frame_num=args.frames,
                    sample_solver="euler" if args.lightning else "unipc", sampling_steps=args.steps,
                    guide_scale=(1.0, 1.0) if args.lightning else (3.5, 3.5),
                    seed=args.seed, shift=5.0, offload_model=bool(args.model_dir))
        except torch.cuda.OutOfMemoryError as error:
            raise_cuda_oom(error)
        if args.memory_telemetry:
            log_cuda_memory("after-generation")
        from wan.utils.utils import save_video
        save_video(tensor=video[None], save_file=str(args.output), fps=self.config.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        del video
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise RuntimeError(f"Video writer did not produce a non-empty file at {args.output}")
        return args.output


def run_daemon(args: argparse.Namespace) -> int:
    try:
        worker = WanVideoWorker(args, daemon=True)
    except Exception as error:
        logging.exception("Daemon initialization failed")
        response = {
            "status": "error",
            "error": (
                cuda_oom_diagnostics(error)
                if isinstance(error, torch.cuda.OutOfMemoryError)
                else str(error)
            ),
        }
        if isinstance(error, torch.cuda.OutOfMemoryError):
            response.update(error_type="cuda_oom", fatal=True)
        print(json.dumps(response), flush=True)
        sys.stderr.flush()
        return DAEMON_CUDA_OOM_EXIT_CODE if response.get("error_type") else 1
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            task = None
            try:
                task = json.loads(line)
                if not isinstance(task, dict):
                    raise ValueError("Each daemon input line must be a JSON object.")
                request_id = task.get("request_id", task.get("id"))
                action = task.get("action", task.get("op"))
                if action in ("exit", "shutdown", "stop"):
                    response = {"status": "success", "action": "shutdown"}
                    if request_id is not None:
                        response["request_id"] = request_id
                    print(json.dumps(response), flush=True)
                    return 0
                if action is not None:
                    raise ValueError(f"Unsupported daemon action: {action!r}")
                worker.generate(task)
                response = {"status": "success"}
                if request_id is not None:
                    response["request_id"] = request_id
                print(json.dumps(response), flush=True)
            except Exception as error:
                logging.exception("Daemon task failed")
                is_oom = isinstance(error, (WanCudaOOMError, torch.cuda.OutOfMemoryError))
                response = {
                    "status": "error",
                    "error": (
                        str(error) if isinstance(error, WanCudaOOMError)
                        else cuda_oom_diagnostics(error)
                        if is_oom
                        else str(error)
                    ),
                }
                if is_oom:
                    response.update(error_type="cuda_oom", fatal=True)
                if isinstance(task, dict):
                    request_id = task.get("request_id", task.get("id"))
                    if request_id is not None:
                        response["request_id"] = request_id
                print(json.dumps(response), flush=True)
                if is_oom:
                    sys.stderr.write(f"Wan daemon CUDA OOM: {response['error']}\n")
                    sys.stderr.flush()
                    return DAEMON_CUDA_OOM_EXIT_CODE
        return 0
    finally:
        # Keep T5/VAE and the last staged expert resident between tasks. Only
        # process termination or an explicit exit releases CUDA allocations.
        worker.cleanup()
        sys.stderr.flush()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.daemon:
        raise SystemExit(run_daemon(args))
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this script.")
    if not args.image.is_file():
        raise FileNotFoundError(
            f"Input image not found: {args.image}. Upload an image as a Kaggle input and pass --image /kaggle/input/.../your_image.png"
        )
    worker = WanVideoWorker(args)
    try:
        output = worker.generate({
            "input_image": args.image,
            "output_video": args.output,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "frames": args.frames,
            "seed": args.seed,
        })
    finally:
        worker.cleanup()
    print(f"Video saved: {output}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
