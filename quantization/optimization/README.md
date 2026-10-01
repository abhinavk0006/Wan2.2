# Wan optimization infrastructure (experimental)

This package is deliberately separate from `wan/image2video.py` and from the existing Kaggle generation path. It provides composable configuration and CPU-safe tests; it does not enable new inference modes end to end.

## Components

- `config.py`: precision-mode selection, module-prefix mixed-precision policy, sampling settings, and explicit 4-step distillation metadata. Four steps require an artifact identifier; specifying four steps alone is rejected.
- `quantization.py`: backend resolution. The existing `Int8Linear` path is reused without changing its implementation. FP8 and INT4 can be described and tested as requested modes, but backend execution raises `NotImplementedError` until implementations are added. FP16/BF16 are native precision selections, with actual device capability checks left to the eventual model loader.
- `experts.py`: model-independent lazy expert pool. `get()` keeps separately loaded experts independent; `activate()` lazily swaps and releases the previous expert by default.
- `lora.py`: validates Wan/Comfy-style rank-factorized LoRA keys, checks their target dimensions, and offers a layer-streamed safetensors merge into ordinary `nn.Linear` layers. This is designed to run before an eventual quantization pass; it is not wired into the Wan loader.
- `euler.py`: builds the shifted flow-matching Euler schedule and applies an explicit Euler update, independently of the video loop.
- `instrumentation.py`: CPU wall-clock measurements and optional CUDA allocated/reserved/peak memory measurements.
- `sensitivity.py`: explicit layer or block forward-output error comparison; it does not launch sweeps automatically.
- `DistillationConfig`: records the LightX2V four-step recipe values (Euler, shift 5, guide scale 1 for both experts). These values do not install a sampler into the generation loop.

## Smoke tests

From the repository root:

```bash
python -m pytest quantization/optimization -q
```

These tests use small CPU modules and fake experts. They cover the Euler schedule/update, LoRA mapping/merge math, and optional safetensors file streaming. No A14B weights, GPU, video pipeline, or full-video sensitivity run is involved. Existing INT8 tests remain in place.

The Lightning artifact headers were also checked separately on a laptop using HTTP byte ranges: both experts report 1,200 tensors, with 400 rank-64 down/up pairs and scalar alphas across the 40 blocks. Sampled shapes match the I2V A14B width (5120) and FFN width (13824); one real pair from each expert produced finite CPU outputs. This validates the artifact header and isolated low-rank math, not a real model merge or generated-video quality.

## Not yet validated

No FP8 or INT4 inference backend is implemented. The streamed LoRA merge helper has not yet been exercised against full Lightning files and is not connected to the custom FP8-to-INT8 checkpoint loader. The Euler schedule/update utility is tested separately but is not connected to `WanI2V`. Mixed-precision policies are declarative and not yet applied to model layers. The expert manager is independently tested but is not connected to the current WanI2V lifecycle. GPU memory/timing fields require a later GPU run.
