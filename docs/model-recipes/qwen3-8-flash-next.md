# Qwen3.8-Flash-Next Model Recipe

<!-- meta: description: Model recipe for serving Qwen3.8-Flash-Next (qwen4_exp: Gated DeltaNet +
Qwen Sparse Attention hybrid MoE with gated residual and n-gram embeddings) with vLLM on Neuron,
including feature support, configuration, data-parallel serving, performance, accuracy and
known limitations. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3.8-Flash-Next, qwen4_exp, Qwen Sparse Attention, QSA,
Gated DeltaNet, MoE, n-gram embedding, hyper-connections, FP8, model recipe, Trn2, Trainium -->
<!-- meta: date_updated: 2026-10-27 -->
<!-- Content type: model-card -->

## Introduction

Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, `model_type=qwen4_exp`) is a 125B-parameter
MoE (6B active) with a 51B-parameter n-gram embedding and a 4B MTP head. Its 48 layers follow
`12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))`. What is new compared
with the dense Qwen3.5 / Qwen3.8 hybrid ([recipe](qwen3-5.md)):

- **Qwen Sparse Attention (QSA).** A 4-head x 128 indexer scores mean-pooled 4-token key blocks.
  Each query attends to its top 512 blocks (2,048 tokens) plus the open tail block. Up to 2,051
  tokens of context, every block is selected and QSA is exactly dense causal attention.
- **Gated residual (hyper-connections).** The residual stream is 4 x 2,560 wide. Every
  sub-layer reads a gated mean of the normalised streams and writes back to each stream with its
  own gate.
- **N-gram embedding (PLE) on layer 2.** 16 hashed bigram/trigram heads index a 320M x 160 table
  (51B parameters, 102 GB in BF16).
- **Sparse MoE on every layer.** 512 experts, top-10 plus a sigmoid-gated shared expert,
  expert intermediate size 640.

This recipe serves the text tower; the vision tower is not loaded.

Tested with vLLM Neuron plugin 0.29.0.1.2.0 (vLLM 0.29.0) and Neuron SDK 2.33 on trn2.48xlarge.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization | Context |
|-------|-------------|----------|--------------|---------|
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Trn2 (trn2.48xlarge, TP=8 per replica) | FP8 routed experts | up to 65,536 tested (262,144 native) |

> `Qwen/Qwen3.8-Flash-Next-FP8` uses 128x128 block weight scales with dynamic activation
> quantization, which this port does not load. Serve the BF16 checkpoint: the plugin quantizes
> the routed experts at load time (see [FP8](#fp8)).

## Features

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| | Image / video | ❌ |
| **Quantization** | FP8 routed experts (per-channel weights, BF16 activations) | ✅ |
| | Full BF16 (48 layers) | ❌ does not fit 8 NeuronCores |
| | FP8 KV cache | ⚠️ works, 1.67x KV tokens; fails logit validation (see [FP8](#fp8)) |
| **Attention** | QSA sparse attention (exact selection) | ✅ |
| **Parallelism** | TP=8 (KV heads replicated, experts split across the TP ranks) | ✅ |
| | Data parallel (independent replicas behind a proxy) | ✅ |
| **Scheduling** | Async scheduling | ✅ (host-token join, see [Host inputs](../design/host-inputs.md)) |
| | Chunked (segmented) prefill | ✅ |
| | KV overcommit with preemption (`VLLM_NEURON_MAX_KV_CONCURRENT`) | ✅ |
| | Prefix caching | ❌ |
| **Speculative decoding** | MTP (1-2 draft tokens) | ✅ (see [MTP](#mtp-speculative-decoding)) |

## Quick start

```bash
# One replica on NeuronCores 0-7, port 8000 (64k context)
MODEL=<path>/Qwen3.8-Flash-Next CORES=0-7 PORT=8000 \
    examples/vllm_neuron/models/qwen4_exp/serve.sh

# Offline generation (truncate the model with --num-layers for quick bring-up)
python examples/vllm_neuron/models/qwen4_exp/run.py --model <path>/Qwen3.8-Flash-Next --fp8
```

The first start compiles about 30 graphs (prefill buckets, dense-prefill variants, and decode
batch x context buckets). This takes roughly 20-30 minutes on a trn2.48xlarge; later starts hit the
compile cache.

## Serving

### Layout

A replica runs TP=8 on 8 logical NeuronCores (2 Trainium2 chips at LNC=2):

| Component | Per rank |
|---|---|
| Attention | 3 of 24 query heads; one replicated KV head (rank r holds head r // 4) |
| GDN | 2 of 16 key heads, 6 of 48 value heads |
| MoE | 64 of 512 experts at full intermediate size (expert parallel inside the TP group); router and shared expert replicated |
| Gated residual, PLE projections | replicated |
| N-gram table | host RAM only |

The n-gram ids and table rows are computed on the host every step (`HasHostInputs`). The table
is memory-mapped from the checkpoint shards, so all replicas share one copy in the page cache.
Hashing and gathering 8,192 tokens takes about 21 ms with a warm cache.

> The table shards live in files 5-37 of the checkpoint. Warm the page cache before serving, for
> example `cat <checkpoint>/model-000{05..37}-of-00131.safetensors > /dev/null`.
> Cold random reads from EBS are too slow for serving.

### Data-parallel replicas

A trn2.48xlarge fits 8 replicas. `run_replicas.sh` starts them on consecutive core ranges and
ports behind `dp_proxy.py`. The proxy sends each request to the healthy replica with the fewest
requests in flight.

```bash
MODEL=<path>/Qwen3.8-Flash-Next REPLICAS=8 MAX_MODEL_LEN=65536 PREFILL_BUCKETS="1024, 4096" \
    DECODE_CTX_BUCKETS="4096, 16384, 32768" GPU_MEM_UTIL=0.85 DECODE_INTERLEAVE=1 \
    examples/vllm_neuron/models/qwen4_exp/run_replicas.sh
curl -s localhost:8000/proxy/stats                   # health and load per replica
STOP=1 examples/vllm_neuron/models/qwen4_exp/run_replicas.sh
```

### Memory and concurrency

- **Per-core memory.** With FP8 experts the weights take 16.5 GiB of each 24 GiB core. At
  `gpu_memory_utilization=0.85` the KV cache gets about 2.9 GiB, plus a 0.7 GiB side cache for
  the QSA indexer keys: about 256k tokens.
- **Concurrency cap.** The scheduler caps concurrency at KV tokens // max_model_len (3 requests
  at 65,536). `VLLM_NEURON_MAX_KV_CONCURRENT` overrides the cap; requests that outgrow the KV
  cache are then preempted and recomputed.
  - **Throughput.** At 65,536 with 3-11k-token prompts and 1.5-4.5k-token outputs, a cap of 8
    gives 72.7 generated tok/s per replica, against 42.1 at the default cap of 3.
  - **Resumed requests** continue exactly as a fresh request with the same tokens would, until
    the decode batch composition changes. Greedy outputs depend on batch composition, as with
    any change of concurrency.
- **Prefill buckets.** Prompts whose cached context plus chunk fits 2,048 tokens run the dense
  prefill kernels (compiled as separate "dense" variants). Longer contexts use the sparse QSA
  prefill. Prompts of 1,025-2,048 tokens pad to the 4,096 bucket and take the sparse path; add a
  2,048 bucket to avoid that.

### FP8

`quantization="fp8"` (set by `serve.sh`) stores the routed experts in float8_e4m3fn with one
scale per output channel, quantized at load from the BF16 checkpoint. No calibration is needed.
Attention, GDN, the shared expert, the router, the gated residual and the PLE projections stay
BF16.

- Decode uses a sparse MoE kernel that loads only the local experts some token selected.
- **FP8 KV cache** (`--kv-cache-dtype fp8`). It holds 1.67x the tokens (427k per replica at
  65,536, default cap 6), and the QSA indexer side cache stays BF16. It uses unit scales and
  fails logit validation: the maximum L2 error is up to 2.0x HF BF16's, against 1.31x with BF16
  KV. Greedy output on the 3,532-token prompt is still 8/8. BF16 KV is the default.
- Prefill dequantizes the experts to BF16 inside the graph before the blockwise kernel.

### MTP speculative decoding

`--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'` (or `run.py --mtp 2`)
loads the checkpoint's MTP head as the draft. Async scheduling turns off automatically
with speculative decoding for this model.

- **Accepted length.** With greedy sampling the mean accepted length is 2.25-2.46 with 2
  draft tokens, and 1.71-1.82 with 1, the same on the 3,532-token (QSA-sparse) prompt.
- **Outputs.** Greedy outputs match plain decode up to the first position where the two best
  logits are within bf16 rounding, as plain decode itself does between batch sizes. The verify
  step matches Hugging Face exactly in an FP32 CPU test.
- **Logprobs** are returned for every emitted token, speculated ones included, with and
  without async scheduling.

## Performance

Decode, one replica, FP8 experts, greedy:

| Configuration | ms/step |
|---|---|
| B=1, short context | ~40 (host-bound) |
| B=3, short context | ~61 (49 tok/s) |
| B=3, 20-40k context | ~107 (28 tok/s) |
| B=1, 14.5k / 29k / 58k context | 40-45 / 43-48 / 48-51 |
| B=3, 14.5k / 29k / 58k context | 73 / 82 / 106 |

With MTP (short context, ms per output token per request):

| Draft tokens | B=1 | B=4 |
|---|---|---|
| none | 47 | 55 |
| 1 | 36 | 47 |
| 2 | 27 | 44 |

- **Aggregate.** 8 replicas with 24 concurrent short requests: 364 tok/s.
- **Prefill.** About 1.1k tokens/s at long context: 3.6-3.9 s per 4,096-token chunk, 57.7k
  tokens in 50.6 s.
- **Long-context decode.** QSA decode still reads the full context and masks it, so it is the
  main cost at long context.

## Accuracy

Logit validation against Hugging Face Transformers (FP32 reference,
`test/integration/model/qwen4_exp/test_model.py`):

- **Short prompts.** The full 48-layer FP8 model passes: every prompt's L2 error stays within
  1.5x of HF's own BF16 CPU run, with a per-token ratio of about 1.0x on average.
- **3,532-token prompt (QSA sparse, 65,536-token serving configuration).** Greedy 8/8 against
  HF.
- **Batch 4.** The 48-layer FP8 model also passes at batch 4. The truncated 4- and 8-layer
  models pass in BF16 and FP8 at batch 1, and at batch 4 except 4-layer FP8, which is 1% over
  the threshold.
- **Long-context retrieval** (full model, serving configuration, thinking off): passkey and
  natural-text needles at 15k-63k tokens and depths 0-99%, 34 of 35 correct at B=1 and B=3. The
  one miss was cut off by max_tokens and is correct with a longer limit. The device's block selection
  overlaps HF BF16's by 99.85%, against 99.56% between HF BF16 and FP32.

## Aider Polyglot

225 exercises with `run_aider_polyglot.sh`: diff edit format, 2 tries, thinking mode at the
template's default `reasoning_effort=xhigh`, model-card sampling (temperature 1.0, top_p 0.95,
top_k 20), Aider 0.86.3.dev (5dc9490). Served by 8 replicas (64k context, 3 concurrent requests
each) with 24 threads.

| | |
|---|---|
| Pass rate, 1 try / 2 tries | 48.9% / **86.2%** (194 of 225) |
| Well-formed responses | 99.6% |
| Error outputs | 24 (15 Aider-side 3,600 s request timeouts on very long thinking responses) |
| Exhausted context windows | 6 |
| Tokens | 3.0M prompt, 5.47M completion (about 24k per exercise) |
| Wall time | 12.5 h |

## Known limitations

- **Long-context decode cost.** QSA decode computes dense-cost attention with a selection mask;
  a compact gather of the selected 2,051 tokens is the next speedup.
- **Concurrency cap.** Concurrency is capped by the worst-case KV reservation (see
  [Memory and concurrency](#memory-and-concurrency)).
- **Not supported:** prefix caching (the recurrent state uses dedicated pools), and BF16 for the
  full model (needs more than 8 NeuronCores per replica).
- **MTP** supports at most 2 draft tokens and no prefix caching, and was tested up to 8,192
  tokens of context and 4 concurrent requests. Its verify step uses the GDN decode kernel for up
  to 16 requests; larger batches use a slower torch path.
- **Prefill is sequential.** Long prompts are prefilled one after another: three concurrent
  15k-token prompts take about 43 s, against about 14 s for one.
- **FP8 KV cache** uses unit scales and reduces accuracy (see [FP8](#fp8)).
- **Text only.** The vision tower is not loaded.
