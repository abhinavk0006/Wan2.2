# Kaggle Wan image-to-video

The Kaggle generation entry point creates an I2V MP4 rather than only running
an expert smoke test. It downloads the T5 and VAE checkpoints from
`Wan-AI/Wan2.2-I2V-A14B`, then downloads the high-noise and low-noise FP8-scaled
DiT experts from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` one at a time. Each
expert is converted to this repository's experimental custom weight-only INT8
Linear layers. Source checkpoint files are deleted after loading to reduce
working-disk use. This full 14B path is memory- and time-intensive on a T4, and
the custom Linear implementation dequantizes its weights on each forward pass.

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
       --output /kaggle/working/wan_i2v_preview.mp4
   ```

   The `17` frame preview is a short first attempt. Use `49` frames after a
   successful run for a longer clip. Frame counts must be `4n+1`.

4. When the command finishes, the MP4 is at
   `/kaggle/working/wan_i2v_preview.mp4`. Download it from the Kaggle output
   pane or copy it into a Kaggle dataset for reuse.

## Limits

Kaggle's two T4 GPUs do not combine into one larger GPU here; the script uses
one selected GPU (`--device 0` by default). If CUDA out-of-memory occurs, try
`--device 1` in a fresh run. The 480p 14B experts may still exceed available
VRAM after conversion. The custom loader accepts the Comfy Wan FP8-scaled key
layout with scalar per-tensor scales; unsupported formats stop with an error.

The separate `kaggle_single_expert.py` command remains available for loading
and smoke-testing one expert. It does not generate a video.
