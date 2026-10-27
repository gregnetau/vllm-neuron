# Qwen3.8-Flash-Next: development notes

<!-- meta: description: Design decisions, hardware and compiler constraints, measurements,
validation method and open work from porting Qwen3.8-Flash-Next (qwen4_exp) to vLLM Neuron on
Trn2. -->
<!-- meta: date_updated: 2026-10-26 -->

These notes record how the Qwen3.8-Flash-Next port was built and validated: the decisions, the
Trn2 and compiler constraints found on the way, the measurements behind them, and the open work.
For deployment, see the [model recipe](../model-recipes/qwen3-8-flash-next.md).

## References

- **Hugging Face Transformers 5.17**, `models/qwen4_exp/modular_qwen4_exp.py`: the reference
  implementation, and the oracle for the CPU unit tests. Transformers 5.18 renames the attention
  layer type to `indexed_attention`. The config accepts `full_attention` (checkpoint),
  `qwen_sparse_attention` and `indexed_attention`.
- **vLLM upstream**, `vllm/models/qwen4_exp/` (CUDA and ROCm): a reference for the MTP head,
  QSA metadata and n-gram hashing.
- **The Qwen3.8-Flash-Next technical report**: QSA training, the gated residual, n-gram
  placement and host prefetching.

## Architecture as implemented

| Component | Implementation |
|---|---|
| Gated DeltaNet (36 layers) | The Qwen3.5 GDN with a sigmoid output gate (`output_gate_type`), same head dims as Qwen3.8-27B |
| Attention (12 layers) | Qwen3.5 gated attention, 24 Q / 2 KV heads, head_dim 256, partial rotary 64, plus the QSA indexer (4 x 128 query heads, 1 x 128 key) |
| QSA | Mean-pooled 4-token key blocks, then RMSNorm, then RoPE at the block start. Score = sum over heads of relu(q.k). Top 512 blocks plus the open tail block. A query's own complete block is not force-selected. |
| Gated residual | 4 streams x 2,560. Read: grouped (1+w) RMSNorm, low-rank (320) sigmoid gate, mean over streams. Write: per-stream 2*sigmoid gate. No final norm; the last read feeds the LM head. |
| PLE (layer 2) | 16 hashed heads (8 bigram, 8 trigram), 160 dims each, gated per stream, plus a depthwise causal conv (kernel 4, dilation 3) |
| MoE | 512 experts, softmax top-10 renormalised, sigmoid-gated shared expert, intermediate 640 |
| MTP | One QSA + MoE layer on the pre-mixer 4-stream residual: `x = fc_hidden(norm(h)) + fc_embedding(norm(e))` per stream; its own final mixer |

## Design decisions

### Parallel layout: TP=8, experts split inside the TP group

- **Why 8 cores per replica.** The 48-layer model needs 8 logical cores: FP8 experts are about
  15 GB per rank at TP=8, and BF16 experts (31 GB per rank) do not fit a 24 GiB core.
- **Why not more.** 24 query heads rule out TP=16, and the GDN decode kernel needs an even
  number of key heads per rank (TP ≤ 8).
- **KV heads.** The 2 KV heads are replicated, one per rank, so the Qwen3.5 "one KV head per
  rank" kernels and KV views apply unchanged.
- **Experts.** Each rank holds 64 experts at full intermediate size. Tokens are replicated
  anyway, so each rank computes its experts' share, and the reduce-scatter (prefill) or
  all-reduce (decode) sums it. Sharding the intermediate instead would give 80 columns per
  rank, below the blockwise kernel minimum of 128.
- **Throughput.** It comes from data parallelism: up to 8 independent replicas per
  trn2.48xlarge behind a least-loaded proxy.

### FP8: per-channel routed experts only

The published FP8 checkpoint uses 128 x 128 block scales with dynamic activations, which the Trn2
kernels used here do not take. Instead, the BF16 experts are quantized at load with one scale per
output channel (`QuantizationType.ROW`: FP8 weights, BF16 activations), so no calibration is
needed. Everything else stays BF16.

### N-gram embedding on the host

The 51B-parameter table (102 GB) and its int64 wrapping hash stay on the host:

- **Graphs downcast integers.** neuronx-cc compiles with `--implicit-integer-downcast=all`, so
  64-bit wraparound hashing in a graph is not reliable.
- **No room on device.** The table would cost 12.8 GB per rank at TP=8.

The model implements `HasHostInputs`:

- every step it hashes the current and previous two tokens and gathers 16 rows of 160 from the
  memory-mapped checkpoint shards;
- the graph receives a [T, 2560] tensor.

Measured with a warm page cache: about 21 ms for 8,192 tokens and under 1 ms for a decode batch.
Cold random reads from EBS are IOPS-bound and unusable, so warm the page cache before serving.

**Getting the previous tokens right.** The rule is HF's EOS reset: tokens at or before the most
recent EOS become EOS. The previous token comes from an earlier row of the same request when the
batch has one; otherwise from `token_ids_cpu`. This covers prefill chunks and spec-verify rows
whose drafts are not committed yet.

### Async scheduling with a host-token join

Host inputs need the true sampled tokens, so async scheduling first had to be turned off, and
the synchronous step then spent about half its time on the host.

The join keeps async scheduling:

- step N+1 is prepared while step N runs;
- right before the forward, the runner applies step N's sampled tokens on the host;
- it builds only the host inputs from them and uploads them with input_ids.

Each rank applies every step's tokens on its own main thread, because upstream only applies them
on the output rank. See [Host inputs](../design/host-inputs.md).

Gain: B=8 decode from 154 to 145 ms/step at the time, because decode was mostly device-bound.

### PLE conv state

The dilated conv keeps the last 9 inputs x 10,240 channels per request, float32. It is a separate
recurrent-state layer (`layers.1.ple`) in the dedicated state pool. Pages are viewed as rows of
128 floats; see [Trn2 and compiler constraints](#trn2-and-compiler-constraints).

### QSA

- **Dense threshold.** Up to 2,051 tokens of context every block is selected, so attention is
  plain dense causal attention. The model checks this statically per graph.
- **Indexer keys.** Raw indexer keys (128 per token, BF16) live in a per-layer side cache shaped
  like the V cache and addressed with the attention layer's block table.
  - **Why not a KV group.** A key-only vLLM KV group shares the KV pool through a second view.
    The graph-I/O check (NCC_EVRF009) counts the pool twice, and the full model then failed to
    compile, 128 MB over the limit.
  - **Budget.** The worker shrinks the KV budget by the side-cache ratio (0.25 with BF16 KV).
  - **Compressed keys.** They are recomputed from the raw keys every step.
- **Top-k without sort.**
  - It bisects on the fp32 bit pattern (31 count passes), then breaks ties by lowest block
    index. That is exactly min(512, visible) blocks.
  - ReLU scores are often exactly 0, so ties are common. HF's CPU `topk` breaks ties
    arbitrarily, so the tests compare against HF with a stable top-k patched in.
- **Prefill.**
  - Short contexts use dense-prefill graph variants with block tables trimmed to 2,048 tokens,
    which keep the segmented/flash kernels.
  - Longer ones use a masked torch attention in query blocks, with transient memory under about
    0.5 GB per layer at a 4,096-token chunk and 64k context.
- **Decode.** The selection mask is ANDed into the fused decode kernel's pre-generated mask
  (`_restrict_decode_mask`). Correct, but it still costs the full context.

### Kernels

- **MoE sparse decode (`moe_tkg_sparse`).** A dynamic loop over only the local experts that some
  token selected. Selective `moe_tkg` loaded an expert per slot (T x 10 slots, ~87% non-local).
  Per layer: 0.25 → 0.067 ms at T=1, 1.07 → 0.27 ms at T=8.
- **GDN decode for several requests (`own_conv_rows`).** In the original kernel, program 0
  wrote every conv-state row while program 1 read them unsynchronised, which was safe only for
  one request. With `own_conv_rows` each program writes only the rows of the channels it reads.
  At B=8 GDN decode went from 1.35 ms per layer (the torch fallback) to 0.29 ms.

## Trn2 and compiler constraints

- **`index_copy_` of whole [half] rows.** On flat state pages it fails with NCC_ITEN406 (too many
  partition dimensions). View pages as [pages, rows, 128] outside the graph, as the GDN state
  does.
- **Scalar constants.** Create them with `device=`. A CPU scalar inside the traced graph fails
  with "Expected XLA tensor".
- **NCC_EVRF009** counts each graph input. Two views of the same pool count twice.
- **NKI DMA on trn2:**
  - DMAs that scatter 2- or 4-byte elements into SBUF returned wrong data for more than one
    token, and the NKI CPU simulator does not reproduce this. Move contiguous runs and transpose
    on the PE.
  - Matmul accumulation groups placed as slices of one PSUM tile gave wrong sums; use one bank
    per group.
  - A stride-0 broadcast over a middle dimension gave wrong results in vector ops.
  - A DMA `scalar_offset` into a flattened [E, ...] tensor read wrong data at E=64; use
    `.select(0, idx)` views.
- **Compile caches.**
  - The libtorch compile cache key follows the traced code, not input shapes. After changes
    that only alter shapes or binding, stale traces are reused, so clear the cache or set
    `NEURON_LIBTORCH_CACHE_ROOT`.
  - The NKI cache keys on the kernel's source, so pass options as kernel arguments, not module
    globals.
- **Core selection.** `NEURON_VISIBLE_DEVICES` takes logical core ids, and multi-process vLLM
  rejects `NEURON_RT_VISIBLE_CORES`. Set `NEURON_SKIP_EFA_AFFINITY=1` on instances without EFA.
- **Teardown.** A killed vLLM run can leave orphan workers holding cores; check `neuron-ls`.

## Measurements

Device time of one B=8 decode step, full model, from NRT device profiles of a 16-layer
truncation scaled to 48 layers (approximate):

| Sub-layer | Before GDN kernel | After |
|---|---|---|
| MoE (48 layers) | 54.8 ms | 52.2 ms (before the sparse MoE kernel, about −38 ms) |
| GDN (36 layers) | 48.7 ms | 10.6 ms |
| Gated residual reads (96) | 5.7 ms | 4.5 ms |
| Attention (12 layers) | 4.1 ms | 2.0 ms |
| PLE | 1.2 ms | 0.85 ms |
| LM head + sampling | 0.5 ms | 0.6 ms |
| Total device | ≈119 ms | ≈74 ms |

Other measurements:

- **Remaining host overhead:**
  - each step dispatches a graph with about 1,177 inputs, so the per-input runtime work is
    16-33 ms;
  - Dynamo guard checks are about 9 ms; `VLLM_NEURON_SKIP_GUARD_EVAL=1` removes them.
- **QSA decode.** About 0.5 ms extra per attention layer per step (mask build), and dense-cost
  attention over the context.
- **Serving.** B=3 decode is about 61 ms/step at short context and about 107 ms/step at
  20-40k context.

## Validation method

- **CPU unit tests.** Every component is checked against the Transformers implementation at a
  tiny config that keeps the real structure: dense and QSA-sparse model logits for padded,
  chunked and decode steps; the n-gram hash and EOS rule bit-exact; QSA selection; MoE; MTP
  mappings; PLE state.
- **Device logit validation.** `test/integration/model/qwen4_exp/test_model.py` against HF FP32
  goldens, plus a BF16 pass teacher-forced along the FP32 greedy sequence.
  - **Memory.** The full FP32 reference needs about 650 GB of host RAM; the 102 GB n-gram
    table is loaded even for truncated models.
  - **Truncated models.** These produce flat, gibberish distributions, so judge them
    teacher-forced or up to the first divergence.
  - **Sparse vs dense at 3.5k tokens.** The two differ by no more than BF16 noise
    (KL 1.7e-2 vs 3.3e-2). To test the selection itself, compare the indexer's choices
    directly: device vs HF overlap is 99.85%.

## Aider Polyglot run (2026-10-22)

Setup:

- 8 replicas x TP=8, FP8 experts, max_model_len 65,536, 3 concurrent requests per replica (the
  worst-case KV cap), 24 Aider threads;
- thinking at the template default `reasoning_effort=xhigh`, sampling temperature 1.0,
  top_p 0.95, top_k 20;
- Aider 5dc9490, diff edit format, 2 tries.

| Metric | Value |
|---|---|
| test_cases | 225 |
| pass_rate_1 / pass_rate_2 | 48.9 / 86.2 (110 / 194) |
| percent_cases_well_formed | 99.6 |
| error_outputs | 24 (15 Aider-side 3,600 s request timeouts) |
| num_malformed_responses | 1 |
| user_asks | 125 |
| exhausted_context_windows | 6 |
| test_timeouts | 0 |
| prompt / completion tokens | 3,009,653 / 5,474,817 |
| seconds_per_case | 4,040 |
| wall time | 12.5 h |

The run averaged about 120 tok/s, where 24 short requests reach 364 tok/s. Reasons:

- each replica decoded at batch 3;
- long-context decode drops to about 28 tok/s per replica;
- the long tail: the last 13 exercises took 2.5 h with replicas mostly idle;
- no prefix caching, so every turn re-prefills its context.

## Device validation (2026-10-26)

Full 48-layer FP8 model on trn2.48xlarge unless noted.

- **Long context.**
  - Retrieval in the 65,536-token serving configuration: passkey and natural-text needles at
    15k-63k tokens and depths 0-99%, 34 of 35 correct at B=1 and B=3. The one miss was
    truncated by max_tokens.
  - Prefill is about 1.1k tokens/s (57.7k tokens in 50.6 s).
  - B=1 decode is 40-51 ms/token up to 58k context.
- **Logit validation.**
  - The full model passes at batch 1 and batch 4.
  - The 4- and 8-layer truncations pass in BF16 and FP8, except 4-layer FP8 at batch 4 (1% over
    the threshold).
  - The GDN decode kernel matches the torch path on device at B=1/8/16 with padded rows.
- **MTP.**
  - With 2 draft tokens: 27 ms/token at B=1 (1.72x) and 44 at B=4 (1.26x). Mean accepted
    length is 2.25-2.46 greedy.
  - Getting there needed draft warm-up for the dense-prefill variants and the GDN decode kernel
    in the verify step: the torch fallback made MTP slower than plain decode.
  - Greedy outputs match plain decode up to bf16 near-ties. The verify step matches HF exactly
    on CPU (FP32).
- **Preemption.**
  - Overcommit first livelocked. A prefill step hides the decodes, so only the prefill request
    could be preempted, and it never fit again. Resumed requests were also misclassified as
    decode once their recompute passed the prompt. Both are fixed.
  - With 5 forced preemptions the outputs are deterministic across async, async-off and sync
    scheduling. Resumed requests continue exactly as fresh requests would.
  - At 65,536 context, a cap of 8 gives 1.73x the generated tokens per second of the default
    cap of 3.
- **FP8 KV.** 1.67x KV tokens, but fails logit validation (unit scales).
- **Not covered:**
  - MTP beyond 8k context and 4 concurrent requests.

### On the 2026-10-27 qwen3_5-upstream base

The work was replayed onto the updated Qwen3.5 branch. That brings in logprobs for speculative
and async-scheduled steps, seeded sampling on Neuron, int64 slot mappings, and dedicated
state-pool pages kept inside the pool. Re-validated on device:
- the 48-layer FP8 logit test passes with the same numbers as before;
- through the async server (65,536 context) and the MTP server, logprobs come back for every
  generated token, with the greedy token equal to the top-1;
- seeded requests are reproducible;
- a passkey at 40,842 prompt tokens is retrieved.

## Open work

- **QSA decode compact gather.** Gather only the ≤ 2,051 selected K/V rows per query instead of
  masking the full context. This is the main long-context decode cost.
- **Serving defaults.**
  - Raise per-replica concurrency (`VLLM_NEURON_MAX_KV_CONCURRENT=8`, validated above) and use
    MTP for Aider-style workloads; both are validated but were not used in the 2026-10-22 run.
  - FP8 KV needs calibrated scales before it can be a default.
- **FP8 prefill MoE without in-graph dequantization.** The blockwise kernel's per-channel FP8
  path fails with NCC_IGCA108 because scale buffers are allocated inside its dynamic loop.
- **Prefix caching.** Recurrent state in the shared pool, with align-mode checkpoints for the GDN
  and PLE state.
- **Smaller costs:**
  - batch the GDN decode kernel's per-request instruction chains (~5 ms/step at B=8);
  - fix the gated-residual group RMSNorm layout (~2 ms/step at B=1);
  - the PLE state write;
  - fuse the top-10 router (nkilib's router stops at top-8).
- **Fewer graph inputs.** Per-step runtime overhead scales with the number of graph inputs
  (~1,177 here).
