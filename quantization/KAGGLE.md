# Kaggle single-expert INT8 validation

This first GPU gate loads one Wan I2V expert, replaces its `nn.Linear` layers
with the custom weight-only INT8 modules, transfers the compressed expert to
CUDA, and runs a minimal I2V-shaped forward pass. Run high-noise and low-noise
experts in separate Kaggle sessions so only one A14B expert is resident at a
time.

## Checkpoint formats

The runner supports either a Diffusers WanModel expert directory or the
standalone Comfy Wan FP8-scaled `.safetensors` file. For the FP8 checkpoint it
infers the Wan architecture from tensor shapes, decodes each FP8 tensor with
its scalar `scale_weight`, and immediately converts each Linear matrix to the
custom INT8 representation. It streams tensors one at a time, so Kaggle does
not need a full FP16 expert copy in CPU RAM. Unsupported key layouts or
non-scalar scales fail with an explicit error instead of being guessed.

## Kaggle setup

1. Create a Kaggle notebook with a GPU accelerator (the T4 uses FP16).
2. Add this repository's updated code and the high-noise FP8-scaled checkpoint
   dataset as notebook inputs. The handoff paths are expected to look like
   `/kaggle/input/wan2-2-i2v-high-noise-14b-fp8-scaled/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`.
   Add the low-noise dataset for the second run, after restarting the session.
3. In a setup cell, install only the Python dependencies not already present;
   keep Kaggle's installed PyTorch/CUDA build:

   ```python
   %pip install 'diffusers>=0.31,<0.33' 'transformers>=4.49,<=4.51.3' \
       'accelerate>=1.1.1' easydict safetensors ftfy
   ```

   Flash Attention is optional here: Wan's attention wrapper falls back to
   PyTorch SDPA when Flash Attention is absent. Do not reinstall PyTorch.

4. Run the HIGH-noise expert first (adjust the Kaggle input path if its dataset
   slug differs):

   ```python
   %cd /kaggle/input/wan22-code/Wan2.2
   !python quantization/kaggle_single_expert.py \
       --checkpoint-file /kaggle/input/wan2-2-i2v-high-noise-14b-fp8-scaled/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
       --dtype float16 \
       --report /kaggle/working/high_noise_int8_report.json
   ```

5. Save/download the JSON report, restart the Kaggle session to clear memory,
   then run the LOW-noise expert by changing the checkpoint path to
   `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` and the report name.

The runner records stream/dequantization/quantization time, number of replaced layers,
model tensor-storage reduction, transfer time, CUDA peak allocated/reserved
memory, forward time, output shape, and finite-output status. It does not claim
that this weight-only prototype is faster: `Int8Linear` dequantizes each weight
matrix before `F.linear`.

## What this gate does not establish

Passing both expert runs establishes that each expert can load, be quantized,
fit on the selected GPU for a minimal forward, and produce finite output. It
does not establish full 832x480x49 I2V generation or visual quality. Those
remain the next gate after both experts pass. Compare against existing
quantized checkpoints if their supported runtime is easier to run or gives a
better memory/quality/speed tradeoff.
