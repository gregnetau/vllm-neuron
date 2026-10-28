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
| **Attention** | QSA sparse attention (exact selection; decode reads only the selected tokens) | ✅ |
| **Parallelism** | TP=8 (KV heads replicated, experts split across the TP ranks) | ✅ |
| | Data parallel (independent replicas behind a proxy) | ✅ |
| **Scheduling** | Async scheduling | ✅ (host-token join, see [Host inputs](../design/host-inputs.md)) |
| | Chunked (segmented) prefill | ✅ |
| | KV overcommit with preemption (`VLLM_NEURON_MAX_KV_CONCURRENT`) | ✅ |
| | Prefix caching | ❌ |
| **Speculative decoding** | MTP (1-2 draft tokens) | ⚠️ faster at batch 1 only (batch 2 with an extra batch-2 graph), slower above (see [MTP](#mtp-speculative-decoding)) |

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
  `gpu_memory_utilization=0.85` the KV cache gets about 2.8 GiB, plus 0.9 GiB of side caches for
  the QSA indexer's raw and compressed keys: about 244k tokens.
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

`MTP_TOKENS=2 serve.sh`, `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`
or `run.py --mtp 2` loads the checkpoint's MTP head as the draft. Async scheduling turns off
automatically with speculative decoding for this model.

MTP is not part of the recommended 8-request configuration. In the 65,536-token configuration
it speeds up a single request, but is slower than plain decode as soon as two or more requests
decode together in the default batch buckets, and it shrinks the KV cache (see [Performance](#performance)). Use it for
single-request latency: `MTP_TOKENS=2` with `serve.sh`'s default `GPU_MEM_UTIL` of 0.82.

- **Batch 1.** With 2 draft tokens, 1.4-1.8x per token up to 60k context (1.2-1.6x with
  model-card sampling); with 1 draft token, 1.1-1.35x.
- **Two or more requests.** Batches of 2-8 run the batch-8 graphs (`SEQS_BUCKETS="1, 8"`).
  There, a verify step costs 2.3-3.2x a plain decode step: for a batch of 3 at 20/30/40k
  context, 242 ms with 1 draft token and 332 ms with 2, against 105 ms; for 8 at 8k, 182 ms
  with 2, against 63 ms. That outweighs the 1.8-2.5 tokens accepted per step. Turning async
  scheduling off costs only 0-13% of a plain step. With an extra batch-2 graph
  (`SEQS_BUCKETS="1, 2, 8"`), 1 draft token is 1.2-1.25x faster than plain decode at batch 2;
  with 2 draft tokens those graphs run out of device memory. An Aider-like load (8 requests at a time,
  3-11k-token prompts, 1.5-4.5k-token outputs, model-card sampling) generates 65.8 tok/s per
  replica with 2 draft tokens, against 79.6 without MTP (both at `gpu_memory_utilization` 0.82).
- **Memory.** The draft head (about 1 GiB per core) and its state pools reduce the KV cache at
  65,536 context:

  | Draft tokens | gpu_memory_utilization | KV tokens |
  |---|---|---|
  | none | 0.85 | 256,094 |
  | 1 | 0.85 | 164,840 |
  | 2 | 0.82 (`serve.sh` default) | 109,061 |

  With 2 draft tokens, 0.85 runs out of device memory while loading the graphs, and 0.83 at
  the first long prefill. With less KV, fewer long requests run at once and the rest wait or
  are preempted: of eight 20-34k-token prompts, 5 run at a time with 1 draft token and 3 with 2.
- **Accepted length.** With greedy sampling the mean accepted length is 2.0-2.6 with 2
  draft tokens, and 1.71-1.86 with 1, also at 15-60k context. Model-card sampling (temperature
  1.0, top_p 0.95, top_k 20) accepts 2.24-2.5 and 1.73-1.81.
- **Outputs.** Greedy outputs match plain decode up to the first position where the two best
  logits are within bf16 rounding, as plain decode itself does between batch sizes. The verify
  step matches Hugging Face exactly in an FP32 CPU test. At 65,536 context with up to 8
  concurrent 1k-60k-token prompts, 800-4,000-token outputs and preemption, every first
  divergence from plain decode was at a top-2 logprob gap of at most 0.5 (mostly 0 or 0.125).
- **Long context.** Natural-text needles at 15k-60k tokens (depths 10-90%, three at a time) are
  all retrieved with 1 and 2 draft tokens, and passkeys at 14k-58k tokens with 2.
- **Logprobs** are returned for every emitted token, speculated ones included, with and
  without async scheduling.

## Performance

Decode, one replica, FP8 experts, greedy:

| Configuration | ms/step |
|---|---|
| B=1, short context | ~38 (host-bound) |
| B=3, short context | ~48 (62 tok/s) |
| B=3, 15-60k context | 48-52 (58-62 tok/s) |
| B=1, 14.5k / 29k / 58k context | 36 / 36 / 36-38 |
| B=3, 15k / 30k / 60k context | 48-50 / 49-50 / 50-52 |
| B=8, 1k / 15k / 25k context (`VLLM_NEURON_MAX_KV_CONCURRENT=8`) | 45.5 / 45.8 / 46.9 (~175 tok/s) |

Long-context decode costs about the same as short-context decode: the attention layers read
only the selected tokens. Short contexts gain too, as decode runs in the 4,096-token context
bucket, which takes the QSA path (more than 2,048 slots). Before that change (masked
full-context decode), B=3 took 57-61 ms at 1-15k and 75 / 102-105 ms at 30k / 60k, and B=1
took 46-49 ms at 58k.

With MTP, ms per output token per request (aggregate generated tok/s in parentheses).
Short context, 8,192-token configuration with batch buckets 1 and 4:

| Draft tokens | B=1 | B=4 |
|---|---|---|
| none | 47 | 55 |
| 1 | 36 | 47 |
| 2 | 27 | 44 |

65,536-token configuration (`serve.sh` with `VLLM_NEURON_MAX_KV_CONCURRENT=8`, decode context
buckets 4,096 / 16,384 / 32,768, `DECODE_INTERLEAVE=1`); greedy unless noted:

| Batch, context | No MTP | 1 draft token | 2 draft tokens |
|---|---|---|---|
| B=1, 1k | 41 | 33 | 30 |
| B=1, 15k | 40 | | 26 |
| B=1, 30k | 52 | 38 | 29 |
| B=1, 30k, model-card sampling | 48 | 42 | 30 |
| B=1, 60k | 50 | 46 | 33 |
| B=1, 60k, model-card sampling | 46 | | 39 |
| B=2, 30k | 75 | | 88 |
| B=2, 30k, `SEQS_BUCKETS="1, 2, 8"` | 59 | 50 | out of memory |
| B=2, 8k, `SEQS_BUCKETS="1, 2, 8"` | 49 | 39 | out of memory |
| B=3, 20/30/40k | 105 (28.6) | 131 (23.0) | 130 (22.9) |
| B=3, 20/30/40k, model-card sampling | 105 (28.6) | 140 (21.5) | 150 (20.2) |
| B=8, 8k | 63 (128) | | 72 (111) |
| B=8, 12k | 69 (116) | 70 (115) | KV holds 6 |
| B=8, 12k, model-card sampling | 72 (111) | 76 (105) | KV holds 6 |
| B=8, 20-34k | 106 (76) | KV holds 5: 30-35 tok/s | KV holds 3: 21-24 tok/s |

- **Aggregate.** 8 replicas with 24 concurrent short requests: 364 tok/s.
- **Prefill.** About 1.4k tokens/s at long context: 2.9 s per 4,096-token chunk at 30k, 57.7k
  tokens in 41.2 s.
- **Long-context decode.** Decode attends only to the selected tokens: the decode kernel reads
  a compact context of the 512 selected blocks plus the open tail (2,304 slots) through a block
  table over the KV cache, whatever the context length. The indexer scores cached compressed
  keys and takes its top 512 with an NKI top-k kernel. MTP verify steps still read the full
  context with a selection mask.

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
- **Compact decode** (attending only to the selected tokens) against the earlier masked
  full-context decode: the 3,532-token prompt stays greedy 8/8 against HF; passkeys at 14k-58k
  tokens 9/9; at 15k-60k the answer tokens' logprobs agree within 5e-4, and greedy outputs
  diverge only at later near-ties, as between batch sizes.

## Aider Polyglot

225 exercises with `run_aider_polyglot.sh`:
- diff edit format, 2 tries;
- thinking mode at the template's default `reasoning_effort=xhigh`, model-card sampling
  (temperature 1.0, top_p 0.95, top_k 20);
- Aider 0.86.3.dev (5dc9490);
- served by 8 replicas at 64k context.

| | Run 1 (2026-10-22) | Run 2 (2026-10-27) |
|---|---|---|
| Serving | 3 concurrent requests per replica (default cap), 24 threads | 8 per replica (`VLLM_NEURON_MAX_KV_CONCURRENT=8`), 64 threads |
| Pass rate, 1 try / 2 tries | 48.9% / 86.2% (194 of 225) | 47.1% / **87.1%** (196 of 225) |
| Well-formed responses | 99.6% | 100% |
| Error outputs | 24 (15 Aider-side 3,600 s request timeouts) | 19 (13 timeouts) |
| Exhausted context windows | 6 | 6 |
| Tokens (prompt / completion) | 3.0M / 5.47M | 2.8M / 5.67M |
| Wall time, average generation rate | 12.5 h, ~120 tok/s | 8.8 h, ~180 tok/s |

- **Errors.** The timeouts are very long thinking responses, which Aider abandons after its
  3,600 s request timeout.
- **Run 2's code.** It used the final branch (MTP not enabled). No preemption occurred at 8
  requests per replica.

## Known limitations

- **MTP verify steps at long context** read the full context with a QSA selection mask (only
  plain decode uses the compact context).
- **Concurrency cap.** Concurrency is capped by the worst-case KV reservation (see
  [Memory and concurrency](#memory-and-concurrency)).
- **Not supported:** prefix caching (the recurrent state uses dedicated pools), and BF16 for the
  full model (needs more than 8 NeuronCores per replica).
- **MTP** is faster than plain decode only at batch 1 in the 65,536-token configuration, and it
  reduces the KV cache by 36% (1 draft token) or 57% (2 draft tokens, which also need
  `gpu_memory_utilization` 0.82). It supports at most 2 draft tokens, no prefix caching and no
  async scheduling. Its verify step uses the GDN decode kernel for up to 16 requests; larger
  batches use a slower torch path.
- **Prefill is sequential.** Long prompts are prefilled one after another: three concurrent
  15k-token prompts take about 43 s, against about 14 s for one.
- **FP8 KV cache** uses unit scales and reduces accuracy (see [FP8](#fp8)).
- **Text only.** The vision tower is not loaded.
