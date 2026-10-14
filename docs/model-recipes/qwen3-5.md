# Qwen3.5 / Qwen3.8 (Dense) Model Recipe

<!-- meta: description: Model recipe for serving the dense Qwen3.5 / Qwen3.8 hybrid
(Gated DeltaNet + full attention) checkpoints with vLLM on Neuron, including supported
checkpoints, feature support, recommended configurations, long context up to 262k tokens,
FP8, performance and known limitations. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.5, Qwen3.8, Qwen3.8-27B, Gated DeltaNet, hybrid,
linear attention, FP8, long context, MTP, model recipe, model card, LLM serving, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-10-14 -->
<!-- Content type: model-card -->

## Introduction

The dense Qwen3.5 / Qwen3.8 checkpoints (`Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`) are hybrid decoders: 48 of the 64 layers of the 27B model are Gated
DeltaNet (GDN) linear attention with a fixed-size recurrent state per request, and every
fourth layer is full attention (24 query / 4 KV heads, head_dim 256, output gate, partial
rotary). This recipe serves the text tower: the vision tower is not loaded, and the
multi-token-prediction (MTP) head is loaded as a speculative-decoding draft when enabled.

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
| **Context** | Native `max_model_len` 262144 (KV pool about 640k tokens in BF16, 1.19M in FP8) | ✅ |
| | Segmented (chunked) prefill: NKI attention kernel at `head_dim` 256, BF16 and FP8 KV | ✅ |
| | Decode context-length buckets (`DECODE_CTX_BUCKETS`) | ✅ |
| | Prefix caching (Mamba `align` mode) | ✅ (not re-validated with the segmented prefill kernel) |
| **Performance** | Fused NKI GDN decode kernel (BF16: one token; FP8: whole sub-layer, up to 32 tokens) | ✅ |
| | NKI FP8 decode kernels: MLP (norm + gate/up + down), projections | ✅ |
| | Multi-token prediction (MTP) speculative decoding, 1-2 draft tokens | ✅ |
| | Asynchronous scheduling | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (unit tests) | ✅ |

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
    SEQS_BUCKETS="1, 4" DECODE_CTX_BUCKETS="16384, 65536" GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 \
    examples/vllm_neuron/models/qwen3_5/serve.sh
```

### Recommended configurations

All with FP8 weights (`FP8_SCALES`), MTP with 2 draft tokens (`MTP_TOKENS=2`), FP8 KV
(`KV_CACHE_DTYPE=fp8`), `BLOCK_SIZE=128`, `GPU_MEM_UTIL=0.95`, `KV_CAP_FRACTION=0.48`:

| Use | `MAX_MODEL_LEN` | `MAX_NUM_SEQS` / `SEQS_BUCKETS` | `DECODE_CTX_BUCKETS` | Cold compile |
|---|---|---|---|---|
| Long documents, native context | 262144 | 4 / `"1, 4"` | `"16384, 65536"` | about 28 min |
| Agentic coding, throughput (Aider) | 65536 | 8 / `"1, 8"` | `"16384"` | |
| Lowest latency, short prompts | 2048 | 1 / `"1"` | | about 12 min |

Compiled graphs are cached in `~/.cache/neuron_libtorch/neuron/compile_cache`; a restart with
the same configuration takes about 10 minutes (weight loading and warmup).

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
| `MAX_NUM_SEQS` | `--max-num-seqs` | 1 | |
| `SEQS_BUCKETS` | `num_seqs_buckets` | `MAX_NUM_SEQS` | decode batch buckets, one compiled decode graph each |
| `DECODE_CTX_BUCKETS` | `decode_context_length_buckets` | none | below `MAX_MODEL_LEN` (the implicit last bucket); decode reads only the bucket's KV |
| `BLOCK_SIZE` | `--block-size` | vLLM's (32) | attention kernel block; 128 for FP8 KV prefill |
| `GPU_MEM_UTIL` | `--gpu-memory-utilization` | 0.65 | |
| `KV_CAP_FRACTION` | `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | 0.15 | KV/state pool cap, see [KV capacity](#context-length-and-kv-capacity) |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `auto` (BF16) | `fp8` doubles the tokens per byte |
| `FP8_SCALES` | `quantization: fp8` | none | static FP8 weights with these activation scales |
| `MTP_TOKENS` | `--speculative-config` (`mtp`) | none | 1 or 2 draft tokens |
| `PREFIX_CACHING` | `--enable-prefix-caching` | 0 | see [Prefix caching](#prefix-caching) |
| `ASYNC_SCHEDULING` | `--no-async-scheduling` when 0 | 1 | |
| `OPTLEVEL`, `HLO2TENSORIZER_OPTIONS` | compiler options | 1, modular flow | see [Compilation](#compilation) |

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
| FP8 | 262144 | 4 | 1,190,172 tokens (4.5 requests of 262k) |

vLLM still allocates the state groups' blocks from the shared pool (unused there), `3 (1 + k)`
blocks per request, which is why its reported capacity is below the pool at short
`max_model_len`. Prefix caching keeps the state in the shared pool (state checkpoints are
cached blocks), where the compiler's double count limits the pool to about 360k tokens
(`KV_CAP_FRACTION=0.35`, `GPU_MEM_UTIL=0.9`). `VLLM_NEURON_SHARED_STATE_POOL=1` selects the
shared pool without prefix caching.

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

Per layer and rank, one 1024-token chunk after a cached prefix:

| Cached prefix | BF16 KV | FP8 KV (block 128) | PyTorch attention (previous implementation) |
|---|---|---|---|
| 0 | 0.9 ms | 0.9 ms | 15 ms at `max_model_len` 32k, 108 ms at 128k, out of memory at 256k |
| 31k | 6.8 ms | 10.3 ms | same as above (whole `max_model_len`) |
| 130k | 23.6 ms | | |
| 261k | 46.9 ms | | |

A prefill chunk also costs about 0.4 s in the dense layers at any position, so long prompts
spend about half their prefill in attention (quadratic in the prompt) and half in the dense
layers (linear).

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
length. Prefix caching is not combined with MTP.

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

`OPTLEVEL` (`--optimization-level`, default 1) and `HLO2TENSORIZER_OPTIONS` (`""` compiles the
whole graph instead of modular flow) are exposed by `serve.sh`. Measured in BF16 at
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

`trn2.3xlarge`, TP=4, greedy. Single-stream numbers with
`examples/vllm_neuron/models/qwen3_5/benchmark_serve.py`, concurrent numbers with
`benchmark_concurrency.py`, retrieval with `check_long_context.py` (passkey at several depths
of a filler document, chat API, thinking disabled).

### Short prompts (batch 1, 1024 prompt / 128 output tokens)

| Configuration | TTFT | TPOT |
|---|---|---|
| BF16, `max_model_len` 2048 | 369 ms | 26.6 ms |
| FP8, `max_model_len` 2048 | 395 ms | 17.7 ms |
| FP8 + MTP, 1 draft token, same | 398 ms | 14.2 ms |
| FP8 + MTP, 2 draft tokens, same | 401 ms | 10.5 ms |
| FP8 + MTP, 2 draft tokens, asynchronous scheduling (default), same | 396 ms | 8.4 ms |
| FP8 + MTP 2, FP8 KV, `max_model_len` 262144 (recommended configuration) | 392 ms | 8.9 ms |
| FP8 + MTP 2, BF16 KV, `max_model_len` 262144, no decode buckets | 396 ms | 15.3 ms |

Without decode context buckets every decode step reads the whole `max_model_len` of KV.

FP8 decode per token on device: 48 GDN layers at 0.099 ms, 64 MLPs at 0.123 ms, 16 attention
layers at 0.140 ms, 1.0 ms for the BF16 `lm_head` and 1.7 ms for the 132 TP all-reduces. The
FP8 kernels stream weights at about 600 GB/s per logical core, close to the rate the Tensor
Engine consumes FP8 weights for a single token; the attention layer and the per-layer
collectives are the remaining overheads. With synchronous scheduling an MTP step with 2 draft
tokens takes about 33 ms: 20.5 ms for the target's verify pass (3 tokens), 4.4 ms for the
draft and about 7 ms on the host; asynchronous scheduling overlaps the host work.

### Long context (recommended 262k configuration, batch 1)

| Prompt tokens | TTFT | TPOT | Passkey retrieval |
|---|---|---|---|
| 1,024 | 0.39 s | 8.9 ms | |
| 28,856 | 13.3 s | 10.0 ms | correct at depths 0.25, 0.75 |
| 115,256 | 75 s | 15.0 ms (262k decode bucket) | correct at depths 0.25, 0.75 |
| 240,056 | 226 s | | correct at depth 0.5 |

With a BF16 KV cache (640k-token pool, block 32, no decode buckets) TTFT is 11.8 s at 29k,
31 s at 61k, 76 s at 123k and 199 s at 240k, with retrieval correct at the same depths. With
the previous PyTorch prefill attention, `max_model_len` was limited to about 32k and a 30k
prompt took 24.5 s.

### Concurrent requests

Recommended 262k configuration (`MAX_NUM_SEQS=4`), 1,000-token prompts, 512 output tokens,
all requests started together (TTFT includes waiting for the other requests' prefills, which
run one at a time):

| Concurrent requests | Aggregate output tokens/s | TPOT per request | TTFT |
|---|---|---|---|
| 1 | 104 | 8.8 ms | 0.41 s |
| 4 | 249 | 12.8-15.1 ms | 0.47-1.68 s |

Four concurrent 19k-token passkey requests all retrieve correctly (TTFT 8.3, 16.5, 24.7 and
32.9 s: one prefill after another).

## Accuracy

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

## Aider Polyglot

[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) (225 Exercism exercises in
C++, Go, Java, JavaScript, Python and Rust, two tries each) runs the Aider benchmark harness
in Docker against the OpenAI-compatible server. Serve the throughput configuration (Aider is
decode bound: thinking mode produces about 16k completion tokens per exercise):

```bash
MODEL=<path-to-checkpoint>/Qwen3.8-27B SERVED_MODEL_NAME=qwen3.8-27b \
    FP8_SCALES=<path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json MTP_TOKENS=2 KV_CACHE_DTYPE=fp8 \
    BLOCK_SIZE=128 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 SEQS_BUCKETS="1, 8" DECODE_CTX_BUCKETS="16384" \
    GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 examples/vllm_neuron/models/qwen3_5/serve.sh
```

Then set up Aider once and start the run, which goes to the background and logs to a file:

```bash
git clone https://github.com/Aider-AI/aider && cd aider
mkdir -p tmp.benchmarks
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
./benchmark/docker_build.sh
cd <vllm-neuron>
AIDER_DIR=<path>/aider LOG=/root/runs/aider_polyglot.log \
    examples/vllm_neuron/models/qwen3_5/run_aider_polyglot.sh
tail -f /root/runs/aider_polyglot.log
```

`run_aider_polyglot.sh` copies `aider-model-settings.yml` into the Aider checkout and runs
`benchmark.py` with the diff edit format, 8 threads (one per sequence) and 2 tries; `CONT=1`
resumes a stopped run, `NUM_TESTS=N` runs a subset. The settings select thinking mode with
Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20) and strip the thinking,
which the server returns inline up to `</think>`. They set no `max_tokens`: vLLM rejects
requests whose prompt plus `max_tokens` exceeds `max_model_len`, and by default it allows the
rest of the context.

Results: in progress (see the run log). A 3-exercise smoke run at `max_model_len` 32768 with
one sequence took about 200 s per exercise, with 16k completion tokens per exercise (thinking
included), all responses well formed and no exhausted context windows.

## Known limitations

- **Long-prompt prefill is compute bound**: about 0.4 s per 1024-token chunk in the dense
  layers plus attention that grows with the position (47 ms per layer for the last chunk of a
  256k prompt, about 20% of the Tensor Engine peak), so a 240k prompt takes 3-4 minutes.
- **Prefills run one request at a time** and pause the other requests' decoding for their
  duration.
- **FP8 KV prefill attention** loads K per 128-token block (PE transpose), about 1.5x the
  BF16 kernel time at long prefixes.
- **Decode between the largest configured decode bucket and `max_model_len`** reads the whole
  `max_model_len` of KV (15 ms per token at 115k in the 262k configuration); add a bucket
  (for example 131072) at the cost of more compiled graphs.
- **BF16 decode at batch > 1** uses the PyTorch GDN path (the fused BF16 GDN kernel is
  single-request); use FP8 for batched serving.
- **vLLM allocates state blocks in the shared KV pool** that the dedicated state pool does
  not use: `3 (1 + k)` blocks per request.
- **Packed FP8 KV (`fp8_packed_kv`) is not supported**: the library decode kernel reads the
  packed layout only at `head_dim` <= 128.
- **Prefix caching** is not combined with MTP and has not been re-validated since the
  segmented prefill kernel replaced the PyTorch prefill attention.
- **`on_device_sampling_config.all_greedy` returns invalid token ids** with this model;
  use the default on-device sampling configuration.
- **`logprobs` requests fail.** The sampler output does not include top log-probabilities.
- **MTP supports at most 2 draft tokens**. With 3 draft tokens the verify pass produced
  invalid indices; this is not resolved.

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
  serving the native 262144 context (passkey retrieval correct to 240k).
- Decode context-length buckets for hybrid models (the recurrent-state block tables keep their
  `1 + k` blocks).
- Dedicated GDN state pool, batched decode with MTP, FP8 KV cache, prefix caching (`align`).

Before this work is proposed upstream:

1. **Prefill throughput**: the dense layers (0.4 s per 1024-token chunk; #54's prefill graph
   is 25% faster), the attention kernel at long prefixes (one K/V load for the 6 query heads of
   a KV head), and a DMA-transpose load for FP8 K.
2. **Scheduling**: interleave prefill chunks with decode steps so a long prefill does not pause
   the other requests.
3. **Aider Polyglot**: the full 225-exercise result (in progress), then at long context.
4. **KV pool**: stop allocating unused state blocks in the shared pool; dedicated state pool
   with prefix caching; re-validate prefix caching with the segmented prefill kernel.
5. **FP8**: calibrated KV-cache scales, and accuracy-driven selection of layers kept in BF16.
6. **MTP**: prefix caching together with MTP, more than 2 draft tokens, and FP8 for the
   draft's `lm_head`.
7. **API gaps**: `logprobs`, `all_greedy` on-device sampling.
8. **End-to-end benchmark suite**: accuracy and serving benchmarks against the Hugging Face
   reference and #54, plus latency and throughput sweeps over context length and batch size.

## Tutorials

- [Hybrid recurrent state for Gated DeltaNet models (design)](../design/hybrid-state-gdn.md)
- [Onboarding models](../model-dev/onboarding-models.md)
