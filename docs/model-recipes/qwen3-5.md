# Qwen3.5 / Qwen3.8 (Dense) Model Recipe

<!-- meta: description: Model recipe for serving the dense Qwen3.5 / Qwen3.8 hybrid
(Gated DeltaNet + full attention) checkpoints with vLLM on Neuron, including supported
checkpoints, feature support, FP8 calibration, performance and known limitations. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.5, Qwen3.8, Qwen3.8-27B, Gated DeltaNet, hybrid,
linear attention, FP8, model recipe, model card, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-10-03 -->
<!-- Content type: model-card -->

## Introduction

The dense Qwen3.5 / Qwen3.8 checkpoints (`Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`) are hybrid decoders: 48 of the 64 layers of the 27B model are Gated
DeltaNet (GDN) linear attention with a fixed-size recurrent state per request, and every
fourth layer is full attention (24 query / 4 KV heads, head_dim 256, output gate, partial
rotary). This recipe serves the text tower: the vision tower is not loaded, and the
multi-token-prediction (MTP) head is loaded as a speculative-decoding draft when enabled.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Trn2 (trn2.3xlarge, TP=4) | BF16, static FP8 |

> `Qwen/Qwen3.8-27B-FP8` uses 128x128 block weight scales with dynamic activation
> quantization, which Trn2 does not support. Serve the BF16 checkpoint and let the plugin
> quantize at load time (see [FP8](#fp8)).

## Features

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Image / video | ❌ |
| **Quantization** | BF16 | ✅ |
| | Static FP8 (E4M3, per tensor): MLP, GDN and attention projections | ✅ |
| | FP8 KV cache (unit KV scales) | ✅ |
| **Parallelism** | Tensor parallelism (TP=4, one KV head per rank) | ✅ |
| | Decode batches > 1 (`max_num_seqs` > 1), with and without MTP | ✅ |
| | Pipeline / context parallelism | ❌ |
| **Performance** | Fused NKI GDN decode kernel (BF16: one token; FP8: whole sub-layer, up to 32 tokens) | ✅ |
| | NKI FP8 decode kernels: MLP (norm + gate/up + down), projections | ✅ |
| | Segmented prefill (`max_model_len` > prefill bucket) | ✅ |
| | Long context (`max_model_len` 32768; about 600k tokens of KV in BF16, 1.2M in FP8) | ✅ |
| | `max_model_len` above 32768 | ❌ (prefill, see [Known limitations](#known-limitations)) |
| | Prefix caching (1024-token blocks, Mamba `align` mode) | ✅ |
| | Multi-token prediction (MTP) speculative decoding, 1-2 draft tokens | ✅ |
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
- `--max-num-seqs 1` (`MAX_NUM_SEQS`) and decode batch buckets `SEQS_BUCKETS` (default
  `MAX_NUM_SEQS`; each bucket is a compiled decode graph, for example `SEQS_BUCKETS="1, 8"`).
- `--gpu-memory-utilization 0.65` (`GPU_MEM_UTIL`) and
  `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15` (`KV_CAP_FRACTION`): a small KV/state pool
  that fits every configuration, including BF16 weights. For long contexts see
  [Context length and KV capacity](#context-length-and-kv-capacity).
- `--max-model-len 2048` (`MAX_MODEL_LEN`) and `--max-num-batched-tokens 1024`
  (`PREFILL_BUCKET`, the prefill chunk).
- `--no-enable-prefix-caching` unless `PREFIX_CACHING=1` (see below).
- `--limit-mm-per-prompt '{"image": 0, "video": 0}'`: text-only serving.
- Asynchronous scheduling (vLLM's default); `ASYNC_SCHEDULING=0` disables it.

An offline example is in `examples/vllm_neuron/models/qwen3_5/run.py`.

### Context length and KV capacity

HBM is 24 GB per logical core (96 GB on the chip). The weights take 13.5 GB per rank in BF16
and 7.0 GiB in FP8 (8.5 GiB with the MTP draft). The plugin sizes the KV pool per core as the
smaller of the free HBM and `KV_CAP_FRACTION * GPU_MEM_UTIL * 24 GB`. With the defaults (0.15
and 0.65) that is 2.3 GB per core, so `neuron-top` shows about half the chip in use.

Per rank, a token of context takes 16 KB of attention KV in BF16 (16 layers, one KV head of
256, K and V) and 8 KB in FP8. Without prefix caching the recurrent state lives in a separate
pool per GDN layer, outside the KV cache: one 1 MB page per layer for each of a request's
`1 + k` state blocks (`k` MTP draft tokens) and each of the `max_num_seqs` batch slots, about
1.2 GB per rank for 8 sequences with `k = 2`. Keeping the state out of the shared KV buffers
matters for the compiler: neuronx-cc rejects a graph whose input and output tensors exceed
24 GB (`NCC_EVRF009`), and a buffer bound both as KV and as state is counted twice.

FP8 weights, MTP with 2 draft tokens, `max_model_len` 32768, `MAX_NUM_SEQS=8`:

| KV cache | `GPU_MEM_UTIL` | `KV_CAP_FRACTION` | KV pool (blocks x tokens) | vLLM's reported capacity |
|---|---|---|---|---|
| BF16 | 0.95 | 0.48 | about 590 x 1024 | 469,941 tokens (14.3 requests of 32k) |
| FP8 (`KV_CACHE_DTYPE=fp8`) | 0.95 | 0.48 | about 630 x 2048 | 770,703 tokens (23.5 requests of 32k) |

```bash
MODEL=<path-to-checkpoint>/Qwen3.8-27B FP8_SCALES=$PWD/fp8_act_amax.json MTP_TOKENS=2 \
    KV_CACHE_DTYPE=fp8 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=8 SEQS_BUCKETS="1, 8" \
    GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 examples/vllm_neuron/models/qwen3_5/serve.sh
```

vLLM still allocates the state groups' blocks from the shared pool (unused there), `3 (1 + k)`
blocks per request, which is why its reported capacity is below the pool. Prefix caching keeps
the state in the shared pool (state checkpoints are cached blocks), where the compiler's double
count limits the pool to about 360k tokens (`KV_CAP_FRACTION=0.35`, `GPU_MEM_UTIL=0.9`).
`VLLM_NEURON_SHARED_STATE_POOL=1` selects the shared pool without prefix caching.

When parallel compilation fails after the pool size or KV dtype changes, the compile cache
(`~/.cache/neuron_libtorch/neuron/compile_cache`) may hold graphs captured for the earlier
configuration, which parallel compilation picks up again. Remove the directories whose
`log-neuron-cc.txt` reports the error.

### MTP speculative decoding

`MTP_TOKENS=1` or `2` serves with vLLM's `mtp` speculative method: the checkpoint's
multi-token-prediction head (one full-attention decoder layer on top of the backbone's last
hidden state) drafts tokens, and the target verifies them in one decode pass.

```bash
MODEL=<path-to-checkpoint>/Qwen3.8-27B FP8_SCALES=$PWD/fp8_act_amax.json MTP_TOKENS=2 \
    examples/vllm_neuron/models/qwen3_5/serve.sh
```

Each GDN layer keeps the state after every verified token in its own state block (vLLM's
speculative state blocks for Mamba layers), and the next step starts from the block of the
last accepted token; see [the design note](../design/hybrid-state-gdn.md#speculative-decoding-mtp).

### Prefix caching

`PREFIX_CACHING=1` (`--enable-prefix-caching`) caches prompt prefixes in 1024-token blocks: the
attention KV blocks and a checkpoint of every GDN layer's state at each block boundary (vLLM's
Mamba cache mode `align`; see [the design note](../design/hybrid-state-gdn.md#prefix-caching)).
Prefills are scheduled in 1024-token chunks, so `PREFILL_BUCKET` must be at least 1024. A
request reuses the longest cached prefix that ends on a block boundary, below the prompt
length.

| FP8, `max_model_len` 4096 | Prompt tokens (cached) | TTFT |
|---|---|---|
| First request | 2,500 (0) | 1,242 ms |
| Same prompt again | 2,500 (2,048) | 430 ms |
| First request | 3,500 (0) | 1,607 ms |
| Same prompt again | 3,500 (3,072) | 420 ms |

Greedy outputs are identical with and without prefix caching. Each cached block holds a GDN
state page per GDN layer as well as the attention KV (per rank and 1024 tokens: 48 MB of
state, 16 MB of KV), so fewer prefixes fit in the KV pool than with an attention-only model.

### FP8

The plugin quantizes weights per tensor at load time (`scale = amax / 240`, values kept within
the Trn2 E4M3 range) and uses static activation scales from an offline calibration file:

| Module | FP8 | BF16 |
|---|---|---|
| MLP | gate, up, down | |
| Gated DeltaNet | `in_proj_qkv`, `in_proj_z`, `out_proj` | `in_proj_a`, `in_proj_b`, conv |
| Full attention | q / k / v (one scale per part), `o_proj` | output gate |

The output gate stays BF16: `sigmoid` turns small absolute errors on large negative gate
logits into large relative errors, and per-tensor FP8 on the gate raises the attention output
error from about 1% to about 10%.

```bash
python examples/vllm_neuron/models/qwen3_5/calibrate_fp8.py \
    --model <path-to-checkpoint>/Qwen3.8-27B --output fp8_act_amax.json
MODEL=<path-to-checkpoint>/Qwen3.8-27B FP8_SCALES=$PWD/fp8_act_amax.json \
    examples/vllm_neuron/models/qwen3_5/serve.sh
```

`KV_CACHE_DTYPE=fp8` (`--kv-cache-dtype fp8`) stores K and V in FP8 with unit scales (K is
post-QK-norm and bounded; values are clamped to the E4M3 range on write), which doubles the
tokens per KV byte (see [Context length and KV capacity](#context-length-and-kv-capacity)).

Calibration runs the Hugging Face model on CPU (about 2 s per prompt for 27B; 124 GB host
RAM is sufficient). `neuron_config.modules_to_not_convert` keeps selected modules in BF16,
for example `["layers.30.linear_attn", "layers.59.mlp"]`. For prefills above 128 tokens the
FP8 MLP runs as two kernel calls over halves of the intermediate dimension, since the full
width exceeds the kernel's limit for this model.

Decode (up to 32 tokens) uses the model's own NKI kernels (`fp8_kernels.py`,
`gdn_kernels.py`), which stream FP8 weights through the Tensor Engine in double-row mode with
the activation quantized in the kernel:

| Kernel | Fuses |
|---|---|
| `fp8_mlp_decode` | post-attention RMSNorm, gate / up, SiLU-mul, down |
| `gdn_decode_fp8` | input RMSNorm, `in_proj` (q / k / v / z, b / a), conv, delta rule, state update, gated norm, `out_proj` |
| `fp8_matvec` | attention `o_proj` |

Each kernel splits its work over the two logical-core programs by output columns or by heads,
so no core barrier is needed. `examples/vllm_neuron/models/qwen3_5/check_fp8_kernels.py`
compares them against a PyTorch FP8 emulation on device. Decode attention uses a vendored copy
of the library `attention_block_tkg` (`attention_block_tkg.py`) that returns the new K token
as rows; the library writes it one element per DMA descriptor at `head_dim` 256.

### Compilation

`OPTLEVEL` (`--optimization-level`, default 1) and `HLO2TENSORIZER_OPTIONS` (`""` compiles the
whole graph instead of modular flow) are exposed by `serve.sh`. Measured in BF16 at
`max_model_len` 2048:

| Compile settings | TTFT | TPOT | Cold compile |
|---|---|---|---|
| `-O1`, modular flow (default) | 368 ms | 27.2 ms | 4 min |
| `-O3`, modular flow | 355 ms | 26.7 ms | 40 min |
| `-O3`, whole graph | 353 ms | 26.7 ms | 39 min |

## Performance

`trn2.3xlarge`, TP=4, batch 1, greedy, measured with
`examples/vllm_neuron/models/qwen3_5/benchmark_serve.py` (mean of 5 rounds after a warmup).

| Configuration | Prompt / output tokens | TTFT | TPOT |
|---|---|---|---|
| BF16, `max_model_len` 2048, prefill bucket 1024 | 1024 / 128 | 369 ms | 26.6 ms |
| FP8, `max_model_len` 2048, prefill bucket 1024 | 1024 / 128 | 395 ms | 17.7 ms |
| FP8 + MTP, 1 draft token, same | 1024 / 128 | 398 ms | 14.2 ms |
| FP8 + MTP, 2 draft tokens, same | 1024 / 128 | 401 ms | 10.5 ms |
| FP8 + MTP, 2 draft tokens, asynchronous scheduling (default), same | 1024 / 128 | 396 ms | 8.4 ms |

FP8 decode per token on device: 48 GDN layers at 0.099 ms, 64 MLPs at 0.123 ms, 16 attention
layers at 0.140 ms, 1.0 ms for the BF16 `lm_head` and 1.7 ms for the 132 TP all-reduces. The
FP8 kernels stream weights at about 600 GB/s per logical core, close to the rate the Tensor
Engine consumes FP8 weights for a single token; the attention layer and the per-layer
collectives are the remaining overheads.


With MTP the draft acceptance rate is about 86% for the first draft token and 62% for the
second (mean 1.85 and 2.4 tokens per step). With synchronous scheduling a step with 2 draft
tokens takes about 33 ms: 20.5 ms for the target's verify pass (3 tokens), 4.4 ms for the
draft and about 7 ms on the host; asynchronous scheduling overlaps the host work.

At `max_model_len` 32768 (FP8, MTP with 2 draft tokens, asynchronous scheduling, batch 1):

| Prompt tokens | TTFT | TPOT |
|---|---|---|
| 1,000 | 0.82 s | 9.1 ms |
| 8,000 | 6.5 s | 9.1 ms |
| 16,000 | 13 s | 9.1 ms |
| 30,000 | 24.5 s | 9.1 ms |

Decode time does not depend on context length. Prefill does: it costs about 0.8 s per
1024-token chunk at any position, compared with 0.4 s at `max_model_len` 2048. Each chunk's
full-attention layers (the PyTorch fallback, see [Known limitations](#known-limitations))
gather and score the whole `max_model_len` of KV, a `[6, 1024, 32768]` float32 score tensor
per rank and layer (about 24 ms per layer). A prefill attention kernel that reads only the
context in use would bound this by the prompt length (see [Roadmap](#roadmap)).

Several concurrent requests at `max_model_len` 32768 (FP8, MTP with 2 draft tokens,
`MAX_NUM_SEQS=8`, `SEQS_BUCKETS="1, 8"`; 1,000-token prompts, greedy, measured with a streaming
client; TTFT includes waiting for the other requests' prefills):

| KV cache | Concurrent requests | Output tokens | Aggregate tokens/s | TPOT per request |
|---|---|---|---|---|
| BF16 | 1 | 256 | 73 | 10.5 ms |
| BF16 | 4 | 256 | 108 | 22-33 ms |
| BF16 | 8 | 256 | 159 | 23-46 ms |
| BF16 | 8 | 1,024 | 183 | 24-26 ms when all requests decode |
| FP8 | 8 | 512 | 216 | 23-38 ms |

Prefills run one request at a time between decode steps, so a new request pauses the others'
decoding for its prefill (about 0.8 s per 1,000 tokens); MTP acceptance at batch 8 is about
2.7 tokens per step.

Cold compilation of the `max_model_len` 2048 configuration takes about 4 minutes in BF16 and
12 minutes in FP8, and is cached afterwards.

## Accuracy

Per-module outputs of the Neuron model (BF16, all 64 layers and the final norm) match the
Hugging Face reference with cosine similarity >= 0.9999. Static FP8 MLP outputs match the
BF16 reference with cosine similarity >= 0.998 per layer. Against the Hugging Face reference
on real activations, FP8 layer outputs have relative errors of 1-7% (GDN), 1-6% (MLP) and
1-2.5% (full attention). With all FP8 modules, greedy 64-token continuations of 12 prompts
match BF16 exactly for 2-3 prompts, with a mean identical prefix of 29-31 tokens; divergent
continuations remain coherent. The FP8 decode kernels quantize the same tensors with the same
scales as the library kernels they replace and leave this agreement unchanged. At
`max_model_len` 32768 (FP8, batch 1), greedy continuations of the same 12 prompts are
identical to those at `max_model_len` 2048 for 8 prompts; changing the graph shapes changes
the reduction order. For the task-level evaluation see [Aider Polyglot](#aider-polyglot).

## Aider Polyglot

[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) (225 Exercism exercises in
C++, Go, Java, JavaScript, Python and Rust, two tries each) runs the Aider benchmark harness
in Docker against the OpenAI-compatible server. Use the 32k configuration from
[Context length and KV capacity](#context-length-and-kv-capacity): Aider prompts with the
exercise files, test output from the first try and the model's thinking can exceed 8k
tokens.

```bash
git clone https://github.com/Aider-AI/aider && cd aider
mkdir -p tmp.benchmarks
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
./benchmark/docker_build.sh
cp <vllm-neuron>/examples/vllm_neuron/models/qwen3_5/aider-model-settings.yml .
docker run --rm --memory=12g --memory-swap=12g --add-host=host.docker.internal:host-gateway \
    -v $PWD:/aider -v $PWD/tmp.benchmarks/.:/benchmarks \
    -e OPENAI_API_KEY=none -e OPENAI_API_BASE=http://host.docker.internal:8000/v1 \
    -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks aider-benchmark \
    bash -c "pip install -q -e .[dev] && ./benchmark/benchmark.py qwen3.8-27b-neuron \
        --model openai/qwen3.8-27b --read-model-settings aider-model-settings.yml \
        --edit-format diff --threads 2 --exercises-dir polyglot-benchmark"
```

`aider-model-settings.yml` selects thinking mode with Qwen's recommended sampling
(temperature 0.6, top-p 0.95, top-k 20) and strips the thinking, which the server returns
inline up to `</think>`. It sets no `max_tokens`: vLLM rejects requests whose prompt plus
`max_tokens` exceeds `max_model_len`, and by default it allows the rest of the context.
With `MAX_NUM_SEQS` above 1, `--threads` up to that number runs exercises concurrently; with
one sequence, `--threads 2` still overlaps one exercise's tests with the other's request. Pass `--num-tests N` for a subset.

A 3-exercise smoke run took about 200 s per exercise, with 16k completion tokens per exercise
(thinking included), all responses well formed and no exhausted context windows. At that rate
the full benchmark takes about 13 hours.

## Known limitations

- **Prefill cost grows with `max_model_len`**, not with the prompt length (see
  [Performance](#performance)), and `max_model_len` well above 32768 does not fit: the
  PyTorch attention fallback materializes a `[6, 1024, max_model_len]` float32 score tensor
  per layer (6.4 GB at 256k). Long contexts need a flash-attention prefill kernel for
  `head_dim` 256 over the paged KV cache.
- **BF16 decode at batch > 1** uses the PyTorch GDN path (the fused BF16 GDN kernel is
  single-request); use FP8 for batched serving.
- **vLLM allocates state blocks in the shared KV pool** that the dedicated state pool does
  not use: `3 (1 + k)` blocks per request (see
  [Context length and KV capacity](#context-length-and-kv-capacity)).
- **Full-attention prefill falls back to PyTorch** when segmented prefill is active
  (`head_dim` 256 exceeds the segmented attention kernel's limit of 128).
- **`on_device_sampling_config.all_greedy` returns invalid token ids** with this model;
  use the default on-device sampling configuration.
- **`logprobs` requests fail.** The sampler output does not include top log-probabilities.
- **MTP supports at most 2 draft tokens** and is not combined with prefix caching. With 3
  draft tokens the verify pass produced invalid indices; this is not resolved.

## Related work

[vllm-project/vllm-neuron#54](https://github.com/vllm-project/vllm-neuron/pull/54) adds
Qwen3.5 dense support (2B and 27B) on the same release branch, derived from the Qwen3.5
port in [qingzwang/vllm-neuron](https://github.com/qingzwang/vllm-neuron). It established
the hybrid KV-cache-group plumbing for this plugin, the per-core HBM sizing for 27B
(`--gpu-memory-utilization 0.65`), the PyTorch fallback for head_dim 256 under segmented
prefill, and the latency methodology this recipe follows. This implementation was developed
in parallel on the same base and is measured against #54 as the reference:

| Same instance and client; 27B, TP=4, `max_model_len` 2048, 1024 / 128 tokens | #54 (BF16) | This implementation (BF16) | This implementation (FP8) |
|---|---|---|---|
| TTFT | 277 ms | 369 ms | 395 ms |
| TPOT | 48.6 ms | 26.6 ms | 17.7 ms |
| Cold compile | 37 min | 4 min | 12 min |

The decode and compile differences come from the GDN state-page layout and DMA pattern, the
fused NKI decode kernel, clearing recycled KV blocks in the runner rather than in the decode
graph, and compiling at the default optimization level with modular flow enabled. #54 has the
faster prefill: its prefill graph executes in 267 ms against 357 ms here, with more matmul
instructions, and the difference persists with this implementation compiled the same way
(`-O3`, whole graph). Both implementations replaced the power-series intra-chunk inverse with
blocked inversion after finding it unstable on device. The intent is to converge with #54
rather than to propose a competing implementation.

## Roadmap

Before this work is proposed upstream:

1. **Long-context prefill**: a flash-attention prefill kernel for `head_dim` 256 over the
   paged KV cache that reads only the context in use, for `max_model_len` up to 256k.
2. **KV pool**: stop allocating unused state blocks in the shared pool; dedicated state pool
   with prefix caching.
3. **Aider Polyglot** and serving benchmarks at the target context length and concurrency.
4. **FP8**: calibrated KV-cache scales, and accuracy-driven selection of layers kept in BF16.
5. **MTP**: prefix caching together with MTP, more than 2 draft tokens, and FP8 for the
   draft's `lm_head`.
6. **End-to-end benchmark suite**: accuracy and serving benchmarks against the Hugging Face
   reference and #54, plus latency and throughput sweeps over context length and batch size.

## Tutorials

- [Hybrid recurrent state for Gated DeltaNet models (design)](../design/hybrid-state-gdn.md)
- [Onboarding models](../model-dev/onboarding-models.md)
