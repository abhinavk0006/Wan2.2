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
scales; unsupported formats stop with an error.

The separate `kaggle_single_expert.py` command remains available for loading
and smoke-testing one expert. It does not generate a video.
