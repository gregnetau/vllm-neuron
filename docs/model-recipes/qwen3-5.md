# Qwen3.5 / Qwen3.8 (Dense) Model Recipe

<!-- meta: description: Model recipe for serving the dense Qwen3.5 / Qwen3.8 hybrid
(Gated DeltaNet + full attention) checkpoints with vLLM on Neuron, including supported
checkpoints, feature support, FP8 calibration, performance and known limitations. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.5, Qwen3.8, Qwen3.8-27B, Gated DeltaNet, hybrid,
linear attention, FP8, model recipe, model card, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-10-01 -->
<!-- Content type: model-card -->

## Introduction

The dense Qwen3.5 / Qwen3.8 checkpoints (`Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`) are hybrid decoders: 48 of the 64 layers of the 27B model are Gated
DeltaNet (GDN) linear attention with a fixed-size recurrent state per request, and every
fourth layer is full attention (24 query / 4 KV heads, head_dim 256, output gate, partial
rotary). This recipe serves the text tower; the vision tower and the MTP head are not loaded.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Trn2 (trn2.3xlarge, TP=4) | BF16, static FP8 MLP |

> `Qwen/Qwen3.8-27B-FP8` uses 128x128 block weight scales with dynamic activation
> quantization, which Trn2 does not support. Serve the BF16 checkpoint and let the plugin
> quantize at load time (see [FP8](#fp8)).

## Features

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Image / video | ❌ |
| **Quantization** | BF16 | ✅ |
| | Static FP8 (E4M3, per tensor), MLP | ✅ |
| | Static FP8, attention and GDN projections | ❌ |
| | FP8 KV cache | ❌ |
| **Parallelism** | Tensor parallelism (TP=4, one KV head per rank) | ✅ |
| | Pipeline / context parallelism | ❌ |
| **Performance** | Fused NKI GDN decode kernel (batch 1) | ✅ |
| | Segmented prefill (`max_model_len` > prefill bucket) | ✅ |
| | Prefix caching | ❌ |
| | Multi-token prediction (MTP) | ❌ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (unit tests) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3.8-27B
- ❌ Not supported yet

## Serving

On a `trn2.3xlarge` (one Trainium2 chip, 4 logical NeuronCores with 24 GB HBM each):

```bash
MODEL=<path-to-checkpoint>/Qwen3.8-27B examples/vllm_neuron/models/qwen3_5/serve.sh
```

The script sets the required options:

- `--tensor-parallel-size 4`: every head count divides by 4 and each rank holds one KV head.
- `--gpu-memory-utilization 0.65` and `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15`: HBM is
  24 GB per logical core and the 27B weights take 13.5 GB per rank. A larger KV budget is
  filled with recurrent-state pages and the graph is rejected (`NCC_EVRF009`).
- `--no-enable-prefix-caching`: GDN layers have no reusable prefix state.
- `--limit-mm-per-prompt '{"image": 0, "video": 0}'`: text-only serving.

An offline example is in `examples/vllm_neuron/models/qwen3_5/run.py`.

### FP8

The plugin quantizes the MLP weights per tensor at load time (`scale = amax / 240`, values
kept within the Trn2 E4M3 range) and uses static activation scales from an offline
calibration file:

```bash
python examples/vllm_neuron/models/qwen3_5/calibrate_fp8.py \
    --model <path-to-checkpoint>/Qwen3.8-27B --output fp8_act_amax.json
MODEL=<path-to-checkpoint>/Qwen3.8-27B FP8_SCALES=$PWD/fp8_act_amax.json \
    examples/vllm_neuron/models/qwen3_5/serve.sh
```

Calibration runs the Hugging Face model on CPU (about 2 s per prompt for 27B; 124 GB host
RAM is sufficient). `neuron_config.modules_to_not_convert` keeps selected MLPs in BF16,
for example `["layers.59.mlp"]`.

## Performance

`trn2.3xlarge`, TP=4, batch 1, greedy, measured with
`examples/vllm_neuron/models/qwen3_5/benchmark_serve.py` (mean of 3 rounds after a warmup).

| Configuration | Prompt / output tokens | TTFT | TPOT |
|---|---|---|---|
| BF16, `max_model_len` 2048, prefill bucket 1024 | 1024 / 128 | 350 ms | 41.6 ms |

With `max_model_len` 128 (one attention block per request, no decode-time block clearing),
a 5-token prompt with 40 output tokens takes 1.15 s end to end in BF16 and 0.93 s with FP8
MLPs (about 28 ms and 23 ms per token including prefill).

Cold compilation takes about 4 minutes for the `max_model_len` 2048 configuration and is
cached afterwards.

## Accuracy

Per-module outputs of the Neuron model (BF16, all 64 layers and the final norm) match the
Hugging Face reference with cosine similarity >= 0.9999. Static FP8 MLP outputs match the
BF16 reference with cosine similarity >= 0.998 per layer. A task-level evaluation is in
progress (see [Roadmap](#roadmap)).

## Known limitations

- **Batch size 1 for the fused GDN decode kernel.** Larger decode batches fall back to the
  PyTorch GDN path.
- **Decode-time KV block clearing when `max_model_len` exceeds one attention block.** KV
  blocks are shared with float32 recurrent-state pages, so a reused block is cleared before
  its first read. When all of a request's blocks are allocated at prefill this happens at
  prefill only; otherwise decode clears on block boundaries, which adds about 14 ms per
  token at `max_model_len` 2048.
- **Full-attention prefill falls back to PyTorch** when segmented prefill is active
  (`head_dim` 256 exceeds the segmented attention kernel's limit of 128).
- **`on_device_sampling_config.all_greedy` returns invalid token ids** with this model;
  use the default on-device sampling configuration.
- **`logprobs` requests fail.** The sampler output does not include top log-probabilities.

## Related work

[vllm-project/vllm-neuron#54](https://github.com/vllm-project/vllm-neuron/pull/54) adds
Qwen3.5 dense support (2B and 27B) on the same release branch, derived from the Qwen3.5
port in [qingzwang/vllm-neuron](https://github.com/qingzwang/vllm-neuron). It established
the hybrid KV-cache-group plumbing for this plugin, the per-core HBM sizing for 27B
(`--gpu-memory-utilization 0.65`), the PyTorch fallback for head_dim 256 under segmented
prefill, and the latency methodology this recipe follows. This implementation was developed
in parallel on the same base and is measured against #54 as the reference:

| Same instance and client; 27B BF16, TP=4, `max_model_len` 2048, 1024 / 128 tokens | #54 | This implementation |
|---|---|---|
| TTFT | 277 ms | 350 ms |
| TPOT | 48.6 ms | 41.6 ms |
| Cold compile | 37 min | 4 min |

The decode and compile differences come from the GDN state-page layout and DMA pattern, the
fused NKI decode kernel, and compiling at the default optimization level with modular flow
enabled. #54 has the faster prefill. The intent is to converge with #54 rather than to
propose a competing implementation.

## Roadmap

Before this work is proposed upstream:

1. **FP8**: static FP8 for the attention and GDN projections, and FP8 KV cache.
2. **MTP**: the checkpoint's multi-token-prediction head for speculative decoding.
3. **End-to-end benchmark suite**: accuracy and serving benchmarks against the Hugging Face
   reference and #54, including [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark)
   for code generation, plus latency and throughput sweeps over context length and batch size.

## Tutorials

- [Hybrid recurrent state for Gated DeltaNet models (design)](../design/hybrid-state-gdn.md)
- [Onboarding models](../model-dev/onboarding-models.md)
