# Bounded runtime optimizations

The baseline remains four-step Lightning, per-tensor INT8, balanced GPU placement
and 33 generated frames. None of the changes establishes a measured speedup or
scientific quality improvement until a target-hardware comparison is completed.

## Expert reuse

`--cpu-expert-cache-gib 14 --host-reserve-gib 3` retains only the high-noise expert
on CPU after it leaves the GPUs. The low-noise expert remains disposable. This
avoids keeping two approximately 13-GiB CPU experts beside conversion workspaces.
Before conversion or disk restore, cached RAM is evicted if needed. A cache miss
falls back to streamed conversion. Set the budget to 0 to disable CPU reuse.

CPU retention removes Accelerate dispatch hooks before moving tensors. A reused
expert is dispatched again for its next use. The cache key includes source and
LoRA content hashes, quantization, FP16 exemptions and loader/INT8/LoRA code.
Check logs for `cpu_hit`, `disk_hit`, `miss`, retention skips and evictions.

`--expert-cache-dir PATH --expert-cache-max-gib 32` enables optional converted
safetensors shards, a manifest and tensor checksums. No pickled module objects
are loaded. Cache writes are atomic directory renames and bounded in size;
low disk space skips the cache. Two experts need approximately 27 GiB, excluding
downloaded source files. Kaggle working storage may be smaller: prefer a mounted
source dataset and sufficient writable storage. CPU caching can also be evicted
on smaller host RAM configurations, reducing or eliminating expected savings.

Checksums are integrity checks, not signatures against a malicious local user.
Cache identity is not a claim that a numerical representation preserves video
quality. Keep Lightning/base and different precision configurations separate.

## Stream conversion directly to GPUs

`--stream-expert-to-gpu` plans the same whole-module placement on a meta model,
then sends each converted tensor directly to its assigned GPU. CPU quantization
and LoRA merging are unchanged. This avoids building a full second CPU expert
beside the retained high expert. Conversion still needs working memory and GPU
storage; it does not remove attention activation costs or guarantee longer shots.
The final dispatch validates actual module sizes against the precomputed map.

## Prompt reuse

`--prompt-cache-dir PATH` stores small tensor-only positive/negative contexts.
Identity includes the T5 checkpoint hash, tokenizer identifier, encoder source,
dtype, text length and exact prompt contents. Lightning omits the unused negative
encoding. Changed prompts miss the cache. Across fresh sessions, persist or mount
the directory to obtain hits. Normal worker initialization still loads T5; a full
cache hit skips prompt encoding and releases those weights. It does not currently
eliminate T5 startup loading. The same explicit tokenizer version must be used
when comparing runtimes.

## VAE output streaming

`--vae-stream-to-cpu` copies each completed temporal decoder chunk to CPU and
concatenates completed frames there. Causal feature caches stay on the decoder
device. Decoder weights, precision, latent scaling and frame order are unchanged.
Even without this flag the decoder now concatenates chunks once, rather than
repeatedly growing a GPU tensor. This targets decode/output memory, not attention
activation memory; longer denoising can still OOM. Measure host usage too.

## Precision and placement experiments

`--quantization per_channel` uses an output-row scale instead of one scale per
matrix. `--keep-fp16 'blocks.0.self_attn.q'` retains matching Linear weights in
FP16, merging the same adapter before use. Repeat patterns as needed. Unmatched
patterns fail. These options can change numerical output and increase VRAM;
default settings remain unchanged. This path still uses floating-point F.linear,
not a fused INT8 kernel.

`--placement contiguous` fills devices in sequence at whole-module boundaries.
It may reduce inter-device boundaries relative to balanced placement, but changes
activation headroom and may fail where balanced placement fits. It is opt-in.
Never equate storage-only budgets with a guarantee that activations fit.

## Calibration tooling

To collect limited real layer inputs, pass `--capture-linear-inputs PATH
--capture-layer 'blocks.0.self_attn.q' --capture-max-tokens 32`.
Capture is off by default; it introduces synchronization and I/O, so do not include
capture runs in ordinary speed comparisons. Captures retain model activations;
keep them with the experiment provenance and use a new folder for each setting.

```bash
python quantization/precision_calibration.py --captures CAPTURES/high_noise_model \
  --checkpoint-file ORIGINAL_HIGH_FP8.safetensors \
  --lora-file HIGH_LIGHTNING.safetensors --expert high_noise_model \
  --quantization per_channel --relative-error-limit 0.01 --output precision.json
```

This compares local Linear output errors on captured inputs, without loading a
full expert. It reports candidate FP16 exemptions; inspect and pass them with
`--keep-fp16` for the next experiment. It is not an automatic policy or a complete
sensitivity sweep across prompts, seeds, timesteps and both experts. The threshold
is configurable, not a validated optimal sensitivity weight. Full videos and
scientific evaluation are required before selecting a final precision policy.

## Target-hardware comparison

Compare cache disabled versus bounded CPU reuse with the same prompts, seed,
area, frame count, weight source and precision. Report startup separately, then
each clip, each cache state and full wall time. Compare streaming on/off separately;
keep per-channel precision and contiguous placement as additional ablations.
Do not add expert-load times to denoising totals: loads are nested within them.
