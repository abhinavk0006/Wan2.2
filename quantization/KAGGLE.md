# Kaggle Wan image-to-video

The Kaggle generation entry point creates an I2V MP4 rather than only running
an expert smoke test. It downloads the T5 and VAE checkpoints from
`Wan-AI/Wan2.2-I2V-A14B`, then downloads the high-noise and low-noise FP8-scaled
DiT experts from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` one at a time. Each
expert is converted to this repository's experimental custom weight-only INT8
Linear layers. Source checkpoint files are deleted after loading to reduce
working-disk use. A single-T4 check found that stored INT8 weights fit, but the
forward pass ran out of temporary VRAM while expanding a matrix to float32.
The Kaggle runner now has an experimental two-GPU option that places whole
transformer blocks across both T4s and moves activations between them.
Wan's transformer uses the repository's PyTorch scaled-dot-product-attention
fallback when FlashAttention is unavailable, as on this Kaggle setup.

## Kaggle setup

1. Create a Kaggle notebook, enable Internet and a GPU accelerator, and clone
   this repository.
2. Install dependencies without replacing Kaggle's PyTorch/CUDA build:

   ```python
   %cd /kaggle/working/Wan2.2
   %pip install 'diffusers>=0.31,<0.33' 'transformers>=4.49,<=4.51.3' \
       'accelerate>=1.1.1' easydict safetensors ftfy huggingface_hub imageio imageio-ffmpeg
   ```

3. Upload an input image using Kaggle's Add Input control. Your current image
   is `/kaggle/input/datasets/abhinavk0006/testimage/1.jpeg`. Run:

   ```python
   %cd /kaggle/working/Wan2.2
   !python quantization/kaggle_generate_video.py \
       --image /kaggle/input/datasets/abhinavk0006/testimage/1.jpeg \
       --prompt "A gentle camera push-in; the subject moves naturally." \
       --frames 17 \
       --steps 20 \
       --max-memory-gib 8 \
       --memory-telemetry \
       --output /kaggle/working/wan_i2v_preview.mp4
   ```

   The `17` frame preview is a short first attempt. Use `49` frames after a
   successful run for a longer clip. Frame counts must be `4n+1`.

## Four-step Lightning experiment

Use the same setup cell and input image with `--lightning`. This downloads the
official 4-step I2V-A14B adapter for each expert when that expert is first
loaded, merges the rank-64 update into streamed base weights before INT8
conversion, and samples with four shifted Euler steps and guidance scales
`(1, 1)`.

```python
%cd /kaggle/working/Wan2.2
!python quantization/kaggle_generate_video.py \
    --image /kaggle/input/datasets/abhinavk0006/testimage/1.jpeg \
    --prompt "A gentle camera push-in; the subject moves naturally." \
    --lightning \
    --frames 17 \
    --max-memory-gib 8 \
    --output /kaggle/working/wan_i2v_lightning_4step.mp4
```

The first run still needs to download T5, VAE, both base experts, and both
adapters, and quantize each expert when it is first activated. Four steps
reduce denoising work but do not eliminate initial load and conversion costs.
Laptop checks validate adapter structure and merge math only; the Kaggle run
is required to validate actual video generation.

## Activation-memory fallback

The default image area is `480 * 832`. For a longer shot that approaches the
temporary activation limit, try a lower area without changing the model:

```python
!python quantization/kaggle_generate_video.py \
    --image /kaggle/input/datasets/abhinavk0006/testimage/1.jpeg \
    --prompt "A gentle camera push-in; the subject moves naturally." \
    --lightning \
    --frames 33 \
    --gpu0-memory-gib 5 \
    --gpu1-memory-gib 11 \
    --max-area 345600 \
    --memory-telemetry \
    --output /kaggle/working/wan_i2v_33_low_area.mp4
```

`--max-area` is an activation-memory experiment, not a guaranteed output
resolution setting; Wan preserves the input aspect ratio while resizing to the
requested area. Compare output dimensions and visual detail before adopting
it for close-up chemistry shots. Telemetry reports per-GPU allocated,
reserved, free, and total memory after startup, expert dispatch, and
generation.

## Persistent JSON-lines worker

For pipelines that render several clips, opt into the persistent worker. It
loads T5/VAE and the Wan runtime once, then processes tasks sequentially.
High- and low-noise experts are still streamed and released between tasks;
both 14B experts are never intentionally resident at the same time. The
daemon deliberately does **not** run expert cleanup after each task: T5/VAE
and whichever expert was last staged remain resident for reuse. On a typical
run the final low-noise expert remains on the T4s, so the next task can reuse
that state; switching noise phases still evicts it before loading the other
expert. This is the best VRAM-safe tradeoff without keeping both 14B experts
resident. Exit/EOF releases the staged expert and CUDA allocations.

```python
%cd /kaggle/working/Wan2.2
!python quantization/kaggle_generate_video.py --daemon \
    --max-memory-gib 8 --frames 17 --steps 20 <<'TASKS'
{"input_image":"/kaggle/input/datasets/abhinavk0006/testimage/1.jpeg","output_video":"/kaggle/working/clip-1.mp4","prompt":"A gentle camera push-in.","clip_name":"clip-1"}
{"input_image":"/kaggle/input/datasets/abhinavk0006/testimage/1.jpeg","output_video":"/kaggle/working/clip-2.mp4","prompt":"A slow natural movement.","clip_duration":1.0,"seed":2}
{"action":"exit"}
TASKS
```

The first stdout line is `READY`. Each subsequent task produces one JSON
response: `{"status":"success"}` or `{"status":"error","error":"..."}`;
errors do not stop the worker or discard reusable model state. Tasks use the
existing wrapper fields `input_image`, `output_video`, `prompt`,
`negative_prompt`, `clip_duration` (or `frames`), and optional
`clip_name`/`seed`. An optional `request_id` (or `id`) is copied into the
response for pipeline correlation. `{"action":"exit"}`,
`{"action":"shutdown"}`, and `{"op":"shutdown"}` cleanly exit after emitting a
shutdown acknowledgement. Unsupported actions and malformed task fields
produce explicit error responses. Blank lines are ignored and EOF also exits
cleanly. Logs are written to stderr, so stdout remains machine-readable.

A CUDA OOM during a task is the intentional exception: the response includes
`"error_type":"cuda_oom"` and `"fatal":true`, includes the request ID when
provided, and is flushed before the daemon exits with status `75`. The same
typed response and exit status are used for an OOM during daemon startup.
Ordinary task errors remain recoverable and continue serving later tasks.

4. When the command finishes, the MP4 is at
   `/kaggle/working/wan_i2v_preview.mp4`. Download it from the Kaggle output
   pane or copy it into a Kaggle dataset for reuse.

## Limits

Kaggle's two T4 GPUs do not combine into one larger GPU; `--two-gpu` explicitly
shards the transformer blocks and moves activations between cards. Before
generating a video, test one expert across both visible GPUs with:

```python
!python quantization/kaggle_single_expert.py \
    --checkpoint-file /kaggle/working/wan_models/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
    --dtype float16 --device 0 --two-gpu \
    --report /kaggle/working/high_noise_int8_report.json
```

The custom Linear implementation still expands weights during every forward
pass, so two-GPU inference remains experimental and may be slow. The custom
loader accepts the Comfy Wan FP8-scaled key layout with scalar per-tensor
scales; unsupported formats stop with an error. The expert dispatcher assigns
each Wan transformer block with a capacity-normalized balance pass, so an
asymmetric budget such as `--gpu0-memory-gib 5 --gpu1-memory-gib 11` does not
automatically pack all blocks onto GPU 1. Each requested budget is a hard
storage ceiling; keep at least 1--2 GiB of additional headroom for attention
and INT8 dequantization workspaces.

For the streamed Kaggle path, the VAE is explicitly moved to CPU after image
encoding, while the active expert is denoising, and restored to the selected
GPU only after the expert is released for decoding. This is enabled only for
the streamed path; local `--model-dir` runs retain the original VAE residency
behavior. If a CUDA allocation fails, the daemon returns an error containing
allocated, reserved, free, total, and peak memory for every visible GPU;
check that diagnostic before changing model or resolution settings.

The separate `kaggle_single_expert.py` command remains available for loading
and smoke-testing one expert. It does not generate a video.
