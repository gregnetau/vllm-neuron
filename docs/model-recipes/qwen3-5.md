# Qwen3.5 / Qwen3.8 (Dense) Model Recipe

<!-- meta: description: Model recipe for serving the dense Qwen3.5 / Qwen3.8 hybrid
(Gated DeltaNet + full attention) checkpoints with vLLM on Neuron, including supported
checkpoints, feature support, recommended configurations, long context up to 262k tokens,
FP8, performance and known limitations. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.5, Qwen3.8, Qwen3.8-27B, Gated DeltaNet, hybrid,
linear attention, FP8, long context, MTP, model recipe, model card, LLM serving, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-10-28 -->
<!-- Content type: model-card -->

## Introduction

The dense Qwen3.5 / Qwen3.8 checkpoints (`Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`) are hybrid decoders: 48 of the 64 layers of the 27B model are Gated
DeltaNet (GDN) linear attention with a fixed-size recurrent state per request, and every
fourth layer is full attention (24 query / 4 KV heads, head_dim 256, output gate, partial
rotary). This recipe serves the text tower: the vision tower is not loaded, and the
multi-token-prediction (MTP) head is loaded as a speculative-decoding draft when enabled.

Tested with vLLM Neuron plugin 0.29.0.1.2.0 (vLLM 0.29.0) and Neuron SDK 2.33 on trn2.3xlarge.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization | Context |
|-------|-------------|----------|--------------|---------|
| Qwen3.8-27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Trn2 (trn2.3xlarge, TP=4) | BF16, static FP8 | up to 262,144 (native) |

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
| | Packed FP8 KV layout (`fp8_packed_kv`) | ❌ (decode kernel, `head_dim` 256) |
| **Parallelism** | Tensor parallelism (TP=4, one KV head per rank) | ✅ |
| | Decode batches > 1 (`max_num_seqs` > 1), with and without MTP | ✅ |
| | Pipeline / context parallelism | ❌ |
| **Context** | Native `max_model_len` 262144 (KV pool about 640k tokens in BF16, 1.26M in FP8) | ✅ |
| | Segmented (chunked) prefill: NKI attention kernel at `head_dim` 256, BF16 and FP8 KV | ✅ |
| | Several prefill buckets; each step picks its KV segment size by the cached length (`PREFILL_BUCKETS`) | ✅ |
| | Decode context-length buckets (`DECODE_CTX_BUCKETS`) | ✅ |
| | Decode steps between the chunks of a long prefill (`DECODE_INTERLEAVE`) | ✅ |
| | Prefix caching (Mamba `align` mode) | ✅ (not re-validated with the segmented prefill kernel) |
| **Performance** | Fused NKI GDN decode kernel (BF16: one token; FP8: whole sub-layer, up to 32 tokens) | ✅ |
| | NKI FP8 decode kernels: MLP (norm + gate/up + down), projections | ✅ |
| | Multi-token prediction (MTP) speculative decoding, 1-2 draft tokens | ✅ |
| | Asynchronous scheduling | ✅ |
| **Compilation** | torch.compile, XLA route (`serve.sh` default) | ✅ |
| | torch.compile, native route (`BACKEND=neuron_native`) | ✅ (slower, see [Compilation](#compilation)) |
| | CPU mode (unit tests) | ✅ |
| **Runner** | Model Runner V1 | ✅ |
| | Model Runner V2 | ❌ (hybrid models and speculative decoding route to V1) |

**Status legend:**

- ✅ Supported: integrated and tested for Qwen3.8-27B
- ❌ Not supported yet

## Quick start

On a `trn2.3xlarge` (one Trainium2 chip, 4 logical NeuronCores with 24 GB HBM each), with
the vLLM Neuron virtual environment on `PATH` (neuronx-cc must be found):

```bash
# 1. Static FP8 activation scales (CPU, a few minutes; once per checkpoint)
python examples/vllm_neuron/models/qwen3_5/calibrate_fp8.py \
    --model <path-to-checkpoint>/Qwen3.8-27B --output <path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json

# 2. Serve (recommended long-context configuration, see below)
MODEL=<path-to-checkpoint>/Qwen3.8-27B FP8_SCALES=<path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json \
    MTP_TOKENS=2 KV_CACHE_DTYPE=fp8 BLOCK_SIZE=128 MAX_MODEL_LEN=262144 MAX_NUM_SEQS=4 \
    SEQS_BUCKETS="1, 4" DECODE_CTX_BUCKETS="16384, 65536, 131072" DECODE_INTERLEAVE=1 \
    GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 examples/vllm_neuron/models/qwen3_5/serve.sh
```

### Recommended configurations

All with FP8 weights (`FP8_SCALES`), MTP with 2 draft tokens (`MTP_TOKENS=2`), FP8 KV
(`KV_CACHE_DTYPE=fp8`), `BLOCK_SIZE=128`, `GPU_MEM_UTIL=0.95`, `KV_CAP_FRACTION=0.48`:

| Use | `MAX_MODEL_LEN` | `MAX_NUM_SEQS` / `SEQS_BUCKETS` | `DECODE_CTX_BUCKETS` | Other | Cold compile |
|---|---|---|---|---|---|
| Native context, mixed prompt lengths | 262144 | 4 / `"1, 4"` | `"16384, 65536, 131072"` | `DECODE_INTERLEAVE=1` | about 30 min |
| Native context, mostly 100k+ prompts | 262144 | 4 / `"1, 4"` | `"16384, 65536, 131072"` | `DECODE_INTERLEAVE=1 PREFILL_BUCKETS="1024, 4096"` | about 65 min |
| Agentic coding, throughput (Aider) | 65536 | 8 / `"1, 8"` | `"16384"` | `DECODE_INTERLEAVE=1` | about 30 min |
| Lowest latency, short prompts | 2048 | 1 / `"1"` | | | about 12 min |

The 4096-token prefill bucket cuts the TTFT of a 240k-token prompt from 226 s to 174 s but
raises it for prompts below about 100k tokens (29k: 13.3 s to 16.6 s); see
[Long-context prefill](#long-context-prefill). Compiled graphs are cached in
`~/.cache/neuron_libtorch/neuron/compile_cache`; a restart with the same configuration takes
about 10 minutes (weight loading and warmup). Changing the KV pool size (`GPU_MEM_UTIL`,
`KV_CAP_FRACTION`, `KV_CACHE_DTYPE`, `BLOCK_SIZE`, `MAX_NUM_SEQS`) recompiles every graph.

## Serving

`examples/vllm_neuron/models/qwen3_5/serve.sh` starts `vllm serve` with the options this
model needs, controlled by environment variables:

| Variable | Option | Default | Notes |
|---|---|---|---|
| `MODEL` | model path | required | BF16 checkpoint directory |
| `SERVED_MODEL_NAME` | `--served-model-name` | checkpoint directory name | |
| | `--tensor-parallel-size 4` | | every head count divides by 4; one KV head per rank |
| `MAX_MODEL_LEN` | `--max-model-len` | 2048 | up to 262144 |
| `PREFILL_BUCKET` | `--max-num-batched-tokens` | 1024 | prefill chunk; segmented prefill above it |
| `PREFILL_BUCKETS` | `num_batched_tokens_buckets`, `kv_segment_size_buckets` | `PREFILL_BUCKET` | e.g. `"1024, 4096"`: short prompts pad to the smallest, long prompts are chunked by the largest; the KV segment size is picked per step from the same list |
| `MAX_NUM_SEQS` | `--max-num-seqs` | 1 | |
| `SEQS_BUCKETS` | `num_seqs_buckets` | `MAX_NUM_SEQS` | decode batch buckets, one compiled decode graph each |
| `DECODE_CTX_BUCKETS` | `decode_context_length_buckets` | none | below `MAX_MODEL_LEN` (the implicit last bucket); decode reads only the bucket's KV |
| `DECODE_INTERLEAVE` | `decode_steps_between_prefill_chunks` | 0 | decode steps for the other requests after each prefill chunk |
| `BLOCK_SIZE` | `--block-size` | vLLM's (32) | attention kernel block; 128 for FP8 KV prefill |
| `GPU_MEM_UTIL` | `--gpu-memory-utilization` | 0.65 | |
| `KV_CAP_FRACTION` | `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | 0.15 | KV/state pool cap, see [KV capacity](#context-length-and-kv-capacity) |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `auto` (BF16) | `fp8` doubles the tokens per byte |
| `FP8_SCALES` | `quantization: fp8` | none | static FP8 weights with these activation scales |
| `MTP_TOKENS` | `--speculative-config` (`mtp`) | none | 1 or 2 draft tokens |
| `PREFIX_CACHING` | `--enable-prefix-caching` | 0 | see [Prefix caching](#prefix-caching) |
| `ALL_GREEDY` | `on_device_sampling_config.all_greedy` | 0 | 1: greedy-only on-device sampling (argmax; requests' temperature / top-k / top-p are ignored) |
| `ASYNC_SCHEDULING` | `--no-async-scheduling` when 0 | 1 | |
| `BACKEND` | `VLLM_NEURON_BACKEND` | `vllm_neuron` (XLA route) | `neuron_native` selects the native torch.compile route |
| `OPTLEVEL`, `HLO2TENSORIZER_OPTIONS` | compiler options | 2, modular flow | see [Compilation](#compilation) |
| `PROFILE_DIR`, `PROFILE_DELAY`, `PROFILE_STEPS` | `--profiler-config`, `neuron_profiler` | none | NRT device profiles between `POST /start_profile` and `/stop_profile` |
| `NEURON_LIBTORCH_COMPILATION_TIMEOUT` | | 3600 | seconds a rank waits for another rank's compilation |

The script also passes `--limit-mm-per-prompt '{"image": 0, "video": 0}'` (text only) and
sets `NEURON_SKIP_EFA_AFFINITY=1` (trn2.3xlarge has no EFA). An offline example is in
`examples/vllm_neuron/models/qwen3_5/run.py`.

### Context length and KV capacity

HBM is 24 GB per logical core (96 GB on the chip). The weights take 13.5 GB per rank in BF16
and 7.0 GiB in FP8 (8.5 GiB with the MTP draft). The plugin sizes the KV pool per core as the
smaller of the free HBM and `KV_CAP_FRACTION * GPU_MEM_UTIL * 24 GB`. With the defaults (0.15
and 0.65) that is 2.3 GB per core, which fits every configuration, including BF16 weights.

Per rank, a token of context takes 16 KB of attention KV in BF16 (16 layers, one KV head of
256, K and V) and 8 KB in FP8. Without prefix caching the recurrent state lives in a separate
pool per GDN layer, outside the KV cache: one 1 MB page per layer for each of a request's
`1 + k` state blocks (`k` MTP draft tokens) and each of the `max_num_seqs` batch slots, about
1.2 GB per rank for 8 sequences with `k = 2`. Keeping the state out of the shared KV buffers
matters for the compiler: neuronx-cc rejects a graph whose input and output tensors exceed
24 GB (`NCC_EVRF009`), and a buffer bound both as KV and as state is counted twice.

FP8 weights, MTP with 2 draft tokens, `GPU_MEM_UTIL=0.95`, `KV_CAP_FRACTION=0.48`:

| KV cache | `max_model_len` | `max_num_seqs` | vLLM's reported KV capacity |
|---|---|---|---|
| BF16 | 32768 | 8 | 469,941 tokens (14.3 requests of 32k) |
| FP8 | 32768 | 8 | 770,703 tokens (23.5 requests of 32k) |
| BF16 | 262144 | 1 | 640,027 tokens (2.4 requests of 262k) |
| FP8 | 262144 | 4 | 1,263,772 tokens (4.8 requests of 262k) |

The 32k and BF16 rows were measured before the placeholder state page described next (the
262k FP8 row was 1,190,172 tokens before it). vLLM still allocates `1 + k` blocks per request
for each of the 3 state groups from the shared pool, unused with the dedicated state pool. The
runner reports those groups with a placeholder page of one 256-token attention block (the
real page is kept for the dedicated pool), so each such block costs 256 tokens of KV instead of
an attention block enlarged to the state page (2048 tokens with FP8 KV). Prefix caching keeps
the state in the shared pool (state checkpoints are cached blocks), where the compiler's double
count limits the pool to about 360k tokens (`KV_CAP_FRACTION=0.35`, `GPU_MEM_UTIL=0.9`). With
FP8 weights and FP8 KV the graphs compile with prefix caching only at `KV_CAP_FRACTION` up to
about 0.22 (`NCC_EVRF009` above it).
`VLLM_NEURON_SHARED_STATE_POOL=1` selects the shared pool without prefix caching.

When parallel compilation fails after the pool size or KV dtype changes, the compile cache may
hold graphs captured for the earlier configuration, which parallel compilation picks up again.
Remove the directories whose `log-neuron-cc.txt` reports the error.

### Long-context prefill

Prompts longer than `PREFILL_BUCKET` are prefilled in chunks (segmented prefill). The
full-attention layers of each chunk run the nkilib segmented attention kernel
(`NF.segmented_attention`), which loops over the cached context actually in use: a chunk's cost
depends on its position, not on `max_model_len`. The kernel tiles `head_dim` 256 over two
128-partition tiles. With one KV head per rank the plugin views the KV blocks as the kernel's
preferred block without a copy: 64 tokens for a BF16 cache (DMA-transpose K load), 128 for an
FP8 cache. An FP8 K cache uses the vendored `attention_segmented_cte.py`, which loads 1-byte K
per block with a PE transpose; the library reads FP8 K only in the packed layout, which the
decode kernel cannot read at `head_dim` 256.
`examples/vllm_neuron/models/qwen3_5/check_segmented_attention.py` compares the kernel with a
float32 reference on device (cosine similarity >= 0.99996 for every prefix up to 261k).

Per layer and rank, one 1024-token chunk after a cached prefix (31k row on plugin 0.29, the
others on 0.24):

| Cached prefix | BF16 KV | FP8 KV (block 128) | PyTorch attention (previous implementation) |
|---|---|---|---|
| 0 | 0.9 ms | 0.9 ms | 15 ms at `max_model_len` 32k, 108 ms at 128k, out of memory at 256k |
| 31k | 5.1 ms | 5.8 ms | same as above (whole `max_model_len`) |
| 130k | 23.6 ms | | |
| 261k | 46.9 ms | | |

A 1024-token chunk also costs about 0.4 s in the dense layers at any position, so long prompts
spend about half their prefill in attention (quadratic in the prompt) and half in the dense
layers (linear).

Larger segments make the attention cheaper per token. Per 1024 query tokens at a 258k
prefix, on plugin 0.24 (where the segment equalled the chunk):

| Chunk (`PREFILL_BUCKETS` largest) | BF16 KV | FP8 KV |
|---|---|---|
| 1024 | 45.9 ms | 73.5 ms |
| 2048 | 27.0 ms | 40.6 ms |
| 4096 | 17.6 ms | 24.4 ms |

The dense layers, however, take about 25-30% more time per token in the 2048- and 4096-token
prefill graphs than in the 1024-token graph (not the MLP kernel or the GDN recurrence, which
scale linearly; under investigation). End to end on plugin 0.24 (262k recommended
configuration, passkey retrieval correct in every case):

| `PREFILL_BUCKETS` | 29k TTFT | 115k TTFT | 240k TTFT | Longest decode stall during a 28k prefill (`DECODE_INTERLEAVE=1`) |
|---|---|---|---|---|
| 1024 | 13.3 s | 75 s | 226 s | 0.55 s |
| 1024, 2048 | 17.1 s | 80 s | 205 s | 1.29 s |
| 1024, 4096 | 16.6 s | 72.5 s | 174 s | 2.39 s |

Short prompts are padded to the smallest bucket and are not affected (1,024 tokens: TTFT
0.39 s with any of them). The 4096-token graph takes over 20 minutes to compile.

### MTP speculative decoding

`MTP_TOKENS=1` or `2` serves with vLLM's `mtp` speculative method: the checkpoint's
multi-token-prediction head (one full-attention decoder layer on top of the backbone's last
hidden state) drafts tokens, and the target verifies them in one decode pass.

Each GDN layer keeps the state after every verified token in its own state block (vLLM's
speculative state blocks for Mamba layers), and the next step starts from the block of the
last accepted token; see [the design note](../design/hybrid-state-gdn.md#speculative-decoding-mtp).
The draft acceptance rate is about 86% for the first draft token and 62% for the second (mean
1.85 and 2.4 tokens per step at batch 1; about 2.7 at batch 8).

### Prefix caching

`PREFIX_CACHING=1` (`--enable-prefix-caching`) caches prompt prefixes in 1024-token blocks: the
attention KV blocks and a checkpoint of every GDN layer's state at each block boundary (vLLM's
Mamba cache mode `align`; see [the design note](../design/hybrid-state-gdn.md#prefix-caching)).
Prefills are scheduled in 1024-token chunks, so `PREFILL_BUCKET` must be at least 1024. A
request reuses the longest cached prefix that ends on a block boundary, below the prompt
length. V1 rejects prefix caching together with MTP (`MTP_TOKENS`). With FP8 weights and FP8
KV, V1 compiles prefix caching only at `KV_CAP_FRACTION` up to about 0.22: the shared pool
buffers are bound both as KV and as state, and neuronx-cc counts them twice toward its 24 GB
graph I/O limit (`NCC_EVRF009`).

Measured with the previous (PyTorch) prefill attention:

| FP8, `max_model_len` 4096 | Prompt tokens (cached) | TTFT |
|---|---|---|
| First request | 2,500 (0) | 1,242 ms |
| Same prompt again | 2,500 (2,048) | 430 ms |
| First request | 3,500 (0) | 1,607 ms |
| Same prompt again | 3,500 (3,072) | 420 ms |

Greedy outputs were identical with and without prefix caching. Each cached block holds a GDN
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

`KV_CACHE_DTYPE=fp8` (`--kv-cache-dtype fp8`) stores K and V in FP8 with unit scales (K is
post-QK-norm and bounded; values are clamped to the E4M3 range on write), which doubles the
tokens per KV byte. Use `BLOCK_SIZE=128` with it: the FP8 prefill attention loads K per block
and is 2.4x faster with 128-token blocks than with the default 32.

Calibration (`calibrate_fp8.py`) runs the Hugging Face model on CPU over 20 prompts (a few
minutes for 27B; 124 GB host RAM is sufficient). `neuron_config.modules_to_not_convert` keeps
selected modules in BF16, for example `["layers.30.linear_attn", "layers.59.mlp"]`. For
prefills above 128 tokens the FP8 MLP runs as two kernel calls over halves of the intermediate
dimension, since the full width exceeds the kernel's limit for this model.

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

vLLM Neuron 0.29 compiles through one of two torch.compile routes. `serve.sh` selects the XLA
route (`VLLM_NEURON_BACKEND=vllm_neuron`); `BACKEND=neuron_native` selects the native route,
the plugin's default. FP8 weights, FP8 KV, MTP 2, `max_model_len` 32768, `-O1`:

| Route | TTFT, 1,024 tokens | TPOT | TTFT, 27k tokens |
|---|---|---|---|
| XLA | 385 ms | 9.1 ms | 11.1 s |
| Native | 535 ms | 10.0 ms | 15.1 s |

On the native route the model compiles with `--model-type=generic`
(`NATIVE_COMPILER_MODEL_TYPE`). The route's default, `transformer`, partitions the prefill
graph per layer and made it about 2.5x slower. The remaining gap in the prefill graph (510 ms
against 361 ms of device time for 1,024 tokens) is time the two physical cores of a logical
core spend waiting for each other at synchronizations; FLOPs and memory traffic are equal.

`serve.sh` compiles at `-O2` (`OPTLEVEL=2`). On plugin 0.29 that decodes faster than `-O1`
with the same TTFT (262k configuration: TPOT 8.2 ms against 8.6 ms at 1k, 10.4 ms against
11.7 ms at 115k) and takes about 30 minutes to compile cold, against 28.

`OPTLEVEL` (`--optimization-level`) and `HLO2TENSORIZER_OPTIONS` (`""` compiles the
whole graph instead of modular flow) are exposed by `serve.sh`. Measured on plugin 0.24 in BF16 at
`max_model_len` 2048:

| Compile settings | TTFT | TPOT | Cold compile |
|---|---|---|---|
| `-O1`, modular flow (default) | 368 ms | 27.2 ms | 4 min |
| `-O3`, modular flow | 355 ms | 26.7 ms | 40 min |
| `-O3`, whole graph | 353 ms | 26.7 ms | 39 min |

Every (decode batch bucket, decode context bucket) pair is a compiled graph for the target and
for the MTP draft, so cold compilation grows with the number of buckets: about 4 minutes in
BF16 and 12 minutes in FP8 at `max_model_len` 2048, and about 28 minutes for the 262k
configuration above.

## Performance

`trn2.3xlarge`, TP=4, greedy, plugin 0.29 unless marked 0.24. Single-stream numbers with
`examples/vllm_neuron/models/qwen3_5/benchmark_serve.py`, concurrent numbers with
`benchmark_concurrency.py`, retrieval with `check_long_context.py` (passkey at several depths
of a filler document, chat API, thinking disabled).

### Short prompts (batch 1, 1024 prompt / 128 output tokens)

Rows other than the recommended configuration were measured on plugin 0.24.

| Configuration | TTFT | TPOT |
|---|---|---|
| BF16, `max_model_len` 2048 | 369 ms | 26.6 ms |
| FP8, `max_model_len` 2048 | 395 ms | 17.7 ms |
| FP8 + MTP, 1 draft token, same | 398 ms | 14.2 ms |
| FP8 + MTP, 2 draft tokens, same | 401 ms | 10.5 ms |
| FP8 + MTP, 2 draft tokens, asynchronous scheduling (default), same | 396 ms | 8.4 ms |
| FP8 + MTP 2, FP8 KV, `max_model_len` 262144 (recommended configuration) | 387 ms | 8.2 ms |
| FP8 + MTP 2, BF16 KV, `max_model_len` 262144, no decode buckets | 396 ms | 15.3 ms |

Without decode context buckets every decode step reads the whole `max_model_len` of KV.

FP8 decode per token on device (0.24): 48 GDN layers at 0.099 ms, 64 MLPs at 0.123 ms, 16 attention
layers at 0.140 ms, 1.0 ms for the BF16 `lm_head` and 1.7 ms for the 132 TP all-reduces. The
FP8 kernels stream weights at about 600 GB/s per logical core, close to the rate the Tensor
Engine consumes FP8 weights for a single token; the attention layer and the per-layer
collectives are the remaining overheads. With synchronous scheduling an MTP step with 2 draft
tokens takes about 33 ms: 20.5 ms for the target's verify pass (3 tokens), 4.4 ms for the
draft and about 7 ms on the host; asynchronous scheduling overlaps the host work.

### Long context (recommended 262k configuration, batch 1)

| Prompt tokens | TTFT | TPOT | Passkey retrieval |
|---|---|---|---|
| 1,024 | 0.39 s | 8.2 ms | |
| 27,896 | 11.5 s | | correct at depths 0.25, 0.75 |
| 58,616 | 26.0 s | | correct at depth 0.5 |
| 110,456 | 55 s | 10.4 ms (at 115,000 tokens, 131k decode bucket) | correct at depths 0.25, 0.75 |
| 230,456 | 148 s | | correct at depth 0.5 |

On plugin 0.24 the same configuration took 13.3 s at 29k, 75 s at 115k and 226 s at 240k (174 s
with `PREFILL_BUCKETS="1024, 4096"`); TPOT at 115k was 15.0 ms without the 131k decode bucket.
On 0.24 with a BF16 KV cache (640k-token pool, block 32, no decode buckets) TTFT was 11.8 s at 29k,
31 s at 61k, 76 s at 123k and 199 s at 240k, with retrieval correct at the same depths. With
the previous PyTorch prefill attention, `max_model_len` was limited to about 32k and a 30k
prompt took 24.5 s.

### Concurrent requests

Recommended 262k configuration (`MAX_NUM_SEQS=4`), 1,000-token prompts, 512 output tokens,
all requests started together (TTFT includes waiting for the other requests' prefills, which
run one at a time):

| Concurrent requests | Aggregate output tokens/s | TPOT per request | TTFT |
|---|---|---|---|
| 1 | 112 | 8.1 ms | 0.41 s |
| 4 | 260 | 12.2-14.5 ms | 0.49-1.64 s |

On plugin 0.24: four concurrent 19k-token passkey requests all retrieve correctly (TTFT 8.3, 16.5, 24.7 and
32.9 s: one prefill after another). With `DECODE_INTERLEAVE=1` the other requests keep
decoding while a long prompt prefills: one request decoding while a 28k-token prompt arrives
stalls for at most 0.55 s between tokens (1024-token chunks) instead of the whole prefill
(about 12 s), and the long prompt's TTFT grows by about 53 ms per chunk (12.2 s to 13.7 s).

Throughput configuration on 0.24 (`MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=8`): 8 concurrent requests,
1,000-token prompts, 512 output tokens: 318 output tokens/s aggregate, TPOT 18.8-24.1 ms.

## Accuracy

`test/integration/model/qwen3_5/test_model.py` serves the model through vLLM (BF16, TP=4) and
compares the raw logits of 16 greedy tokens for 4 prompts against the Hugging Face text model
(an FP32 baseline and a teacher-forced BF16 pass), at batch 1 and 4. It passes with no argmax
divergence:

```bash
QWEN3_5_MODEL_PATH=<path-to-checkpoint>/Qwen3.8-27B \
    pytest test/integration/model/qwen3_5/test_model.py -v -s
```

Re-validated on 2026-10-30 at this branch (trn2.3xlarge, Neuron SDK 2.33): Qwen3.8-27B (TP=4) and
Qwen3.5-2B (`QWEN3_5_TP=2`) both pass at batch 1 and 4.

The Hugging Face reference runs on CPU and needs about 110 GB of host memory for its FP32 pass;
it is cached on disk after the first run. Its top-K bounds are wider than the template's BF16
defaults for one ill-conditioned token, where Hugging Face's own BF16 model has a larger error
than Neuron (see the comment in the test).

Per-module outputs of the Neuron model (BF16, all 64 layers and the final norm) match the
Hugging Face reference with cosine similarity >= 0.9999. Static FP8 MLP outputs match the
BF16 reference with cosine similarity >= 0.998 per layer. Against the Hugging Face reference
on real activations, FP8 layer outputs have relative errors of 1-7% (GDN), 1-6% (MLP) and
1-2.5% (full attention). With all FP8 modules, greedy 64-token continuations of 12 prompts
match BF16 exactly for 2-3 prompts, with a mean identical prefix of 29-31 tokens; divergent
continuations remain coherent. The FP8 decode kernels quantize the same tensors with the same
scales as the library kernels they replace and leave this agreement unchanged. The segmented
prefill kernel matches a float32 reference with cosine similarity >= 0.99996 (BF16 and FP8 KV),
and long-context retrieval is correct up to 240k tokens (see [Performance](#performance)). For
the task-level evaluation see [Aider Polyglot](#aider-polyglot).

### GSM8K

GSM8K (test split, chat API, thinking disabled) on Model Runner V1 with FP8 weights, FP8 KV
and MTP 2 in the throughput configuration (`MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=8`):

| Run | Questions | Accuracy |
|---|---|---|
| Concurrency 1 | first 500 | 96.6% |
| Concurrency 8, same questions | first 500 | 96.4% (3 questions flip between the runs) |
| Concurrency 8 | all 1,319 | 95.9% |

Concurrent batching is within noise of single-stream serving: on the same 500 questions the
two runs differ in 3 answers.

FP8 graphs differ in numerics, so greedy outputs are not a strict reference for each other:
at high-entropy positions (two candidate tokens with nearly equal logits) the FP8 prefill,
decode and MTP verify graphs can each pick a different token for the same context, and all of
them can be within the error of FP8. When a greedy parity test (for example concurrent
against sequential requests) diverges from an FP8 reference, teacher-force both continuations
through the Hugging Face BF16 model on CPU at the divergent position before calling it
corruption: a divergence where BF16 also rates both tokens as near-equal is FP8 noise, a
token BF16 gives near-zero probability is a bug.

## Aider Polyglot

[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) (225 Exercism exercises in
C++, Go, Java, JavaScript, Python and Rust, two tries each) runs the Aider benchmark harness
in Docker against the OpenAI-compatible server. Serve the throughput configuration (Aider is
decode bound: thinking mode produces about 16k completion tokens per exercise):

```bash
MODEL=<path-to-checkpoint>/Qwen3.8-27B SERVED_MODEL_NAME=qwen3.8-27b \
    FP8_SCALES=<path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json MTP_TOKENS=2 KV_CACHE_DTYPE=fp8 \
    BLOCK_SIZE=128 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 SEQS_BUCKETS="1, 8" DECODE_CTX_BUCKETS="16384" \
    DECODE_INTERLEAVE=1 GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 examples/vllm_neuron/models/qwen3_5/serve.sh
```

Then set up Aider once and start the run, which goes to the background and logs to a file:

```bash
git clone https://github.com/Aider-AI/aider && cd aider
mkdir -p tmp.benchmarks
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
./benchmark/docker_build.sh
cd <vllm-neuron>
AIDER_DIR=<path>/aider LOG=$PWD/aider_polyglot.log \
    examples/vllm_neuron/models/qwen3_5/run_aider_polyglot.sh
tail -f aider_polyglot.log
```

`run_aider_polyglot.sh` copies `aider-model-settings.yml` into the Aider checkout and runs
`benchmark.py` with the diff edit format, 8 threads (one per sequence) and 2 tries; `CONT=1`
resumes a stopped run, `NUM_TESTS=N` runs a subset. The settings select thinking mode with
Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20), raise the request timeout
to 3600 s, and strip the thinking,
which the server returns inline up to `</think>`. They set no `max_tokens`: vLLM rejects
requests whose prompt plus `max_tokens` exceeds `max_model_len`, and by default it allows the
rest of the context.

### Results

| Run | Pass rate, 2 tries | First try | Well-formed responses | Exhausted context windows | Client timeouts |
|---|---|---|---|---|---|
| 2026-10-19 (3600 s request timeout) | **71.1%** | 33.8% | 82.7% (46 malformed) | 7 | 1 |
| 2026-10-14 to 2026-10-16 (default timeout) | **73.8%** | 30.2% | 97.3% (6 malformed) | 3 | 196 |

Both runs used plugin 0.24.0.1.1.0 and the throughput configuration above. With the longer
timeout the client no longer abandoned long thinking responses. The second run had more
malformed edit blocks; the 2.7-point difference in the 2-try score is about one standard error at 225 exercises.

First run in detail, 2026-10-14 to 2026-10-16 (Aider 0.86.3.dev, commit `5dc9490`; the throughput
configuration above; diff edit format, 2 tries, 8 threads):

| | |
|---|---|
| Pass rate, 2 tries | **73.8%** (166 / 225) |
| Pass rate, first try | 30.2% (68) |
| Well-formed responses | 97.3% (6 malformed) |
| Exhausted context windows | 3 |
| Prompt / completion tokens | 3.19M / 3.22M |
| Wall time | about 47.5 hours |

| Language | Exercises | Pass, 2 tries | Pass, first try | Timed-out requests |
|---|---|---|---|---|
| JavaScript | 49 | 87.8% | 24.5% | 18 |
| Python | 34 | 82.4% | 26.5% | 33 |
| Rust | 30 | 73.3% | 46.7% | 44 |
| Go | 39 | 71.8% | 38.5% | 47 |
| C++ | 26 | 69.2% | 15.4% | 15 |
| Java | 47 | 57.4% | 29.8% | 48 |

The per-language first-try split is tallied from the per-exercise result files. The client
timed out 196 requests (`litellm.Timeout`; 205 error outputs): thinking responses of tens of
thousands of tokens at batch 8 outlast the default request timeout, and Aider retries them
from the start. That wasted generation accounts for most of the wall time, and timed-out
requests may also have cost passes, so the score is a lower bound for this configuration.
`aider-model-settings.yml` now sets a 3600 s request timeout (`extra_params.timeout`) for
the rerun.

On the server side the run was clean: one server process for the whole run, no tracebacks,
recompiles or engine errors; 801 of 804 requests returned HTTP 200, and the 3 HTTP 400s match
the 3 prompts over `max_model_len`. Mean logged generation throughput was 159 tokens/s (about
270 tokens/s with 8 requests decoding); MTP accepted 2.65-2.69 tokens per step.

## Known limitations

- **Long-prompt prefill is compute bound**: about 0.4 s per 1024-token chunk in the dense
  layers plus attention that grows with the position (46 ms per layer for a 1024-token chunk
  at a 258k prefix on 0.24, about 20% of the Tensor Engine peak), so a 230k prompt takes about 2.5
  minutes. Larger prefill buckets cut the attention cost but slow the dense layers (see
  [Long-context prefill](#long-context-prefill)).
- **Prefills run one request at a time.** With `DECODE_INTERLEAVE` the other requests decode
  between chunks, at one decode step per chunk (or per setting); without it they pause for the
  whole prefill.
- **FP8 KV prefill attention** loads K per 128-token block (PE transpose), about 1.6x the BF16
  kernel time at long prefixes.
- **Decode above the largest decode context bucket** reads the whole `max_model_len` of KV;
  each bucket adds compiled decode graphs.
- **BF16 decode at batch > 1** uses the PyTorch GDN path (the fused BF16 GDN kernel is
  single-request); use FP8 for batched serving.
- **vLLM allocates state blocks in the shared KV pool** that the dedicated state pool does
  not use: `3 (1 + k)` blocks per request, each now one 256-token attention block.
- **Packed FP8 KV (`fp8_packed_kv`) is not supported**: the library decode kernel reads the
  packed layout only at `head_dim` <= 128.
- **Prefix caching** is not combined with MTP and has not been re-validated since the
  segmented prefill kernel replaced the PyTorch prefill attention.
- **The native compile route is slower than the XLA route** (see [Compilation](#compilation));
  `serve.sh` uses the XLA route.
- **Model Runner V2 is not supported.** Hybrid models and speculative decoding run on V1.
- **MTP with synchronous scheduling (`ASYNC_SCHEDULING=0`) can corrupt outputs near
  `max_model_len`** on V1: requests that run into `max_model_len` next to other requests may
  emit wrong tokens in their last steps. This predates the fixes in this recipe and is not
  fixed; keep asynchronous scheduling (the default) or use Model Runner V2.
- **With MTP and asynchronous scheduling, hybrid requests near `max_model_len` end early** on
  V1: the scheduler caps a request's `max_tokens` so that every step stays a full verify step,
  so a request finishes with `finish_reason` `length` up to `4 * (1 + k)` tokens (12 for
  `MTP_TOKENS=2`) before `max_model_len`.
- **Prefix caching on V1** with FP8 weights and FP8 KV compiles only at `KV_CAP_FRACTION` up to
  about 0.22 (`NCC_EVRF009`: the shared pool buffers bound as KV and as state count twice
  toward the 24 GB graph I/O limit), and V1 rejects prefix caching together with MTP.
- **Per-request `seed`** is accepted but not reproducible: on-device sampling does not use the
  request's generator.
- **MTP supports at most 2 draft tokens**. With 3 draft tokens the verify pass produced
  invalid indices; this is not resolved.
- **Device profiles of prefill graphs are incomplete**: the runtime drops profiler
  notifications for graphs of this size, so per-layer prefill breakdowns are not available yet.

## Related work

[vllm-project/vllm-neuron#54](https://github.com/vllm-project/vllm-neuron/pull/54) adds
Qwen3.5 dense support (2B and 27B) on the same release branch, derived from the Qwen3.5
port in [qingzwang/vllm-neuron](https://github.com/qingzwang/vllm-neuron). It established
the hybrid KV-cache-group plumbing for this plugin, the per-core HBM sizing for 27B
(`--gpu-memory-utilization 0.65`), the PyTorch fallback for head_dim 256 under segmented
prefill (replaced here by the segmented attention kernel), and the latency methodology this
recipe follows. This implementation was developed in parallel on the same base and is
measured against #54 as the reference:

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

Done:

- Long-context prefill: segmented attention kernel at `head_dim` 256 for BF16 and FP8 KV,
  serving the native 262144 context (passkey retrieval correct to 240k), with several prefill
  buckets and matching segment sizes.
- Decode context-length buckets for hybrid models (the recurrent-state block tables keep their
  `1 + k` blocks).
- Decode steps between the chunks of a long prefill.
- Placeholder state page for the dedicated GDN state pool (+6% KV at 262k).
- Dedicated GDN state pool, batched decode with MTP, FP8 KV cache, prefix caching (`align`).
- Aider Polyglot: 73.8% and 71.1% with 2 tries (two runs).
- Port to vLLM Neuron 0.29 (vLLM 0.29, Neuron SDK 2.33) on Model Runner V1, both compile
  routes.
- `logprobs` (raw logits, full vocabulary) and `all_greedy` on-device sampling, with a
  logit-validation test against Hugging Face (`test/integration/model/qwen3_5/test_model.py`).
- `logprobs` with MTP and with asynchronous scheduling.
- Stress testing of concurrent, aborted and near-`max_model_len` requests with MTP and
  asynchronous scheduling, with the fixes it found; GSM8K 95.9% (FP8, MTP 2, concurrency 8).

Next:

1. **Prefill throughput**: the dense layers (0.4 s per 1024-token chunk; #54's prefill graph
   is 25% faster) and their slowdown in larger prefill graphs; the attention kernel's
   efficiency at long prefixes (about 20% of peak; loading K/V once per query head is not the
   bottleneck); a DMA-transpose load for FP8 K; the native route's core-synchronization waits.
2. **Aider Polyglot** at long context.
3. **KV pool**: no state blocks in the shared pool at all; dedicated state pool with prefix
   caching; re-validate prefix caching with the segmented prefill kernel.
4. **FP8**: calibrated KV-cache scales, and accuracy-driven selection of layers kept in BF16.
5. **MTP**: prefix caching together with MTP, more than 2 draft tokens, and FP8 for the
   draft's `lm_head`.
6. **Model Runner V2** together with MTP: speculative decoding and hybrid state on V2.
7. **End-to-end benchmark suite**: accuracy and serving benchmarks against the Hugging Face
   reference and #54, plus latency and throughput sweeps over context length and batch size.

## Tutorials

- [Hybrid recurrent state for Gated DeltaNet models (design)](../design/hybrid-state-gdn.md)
- [Onboarding models](../model-dev/onboarding-models.md)
