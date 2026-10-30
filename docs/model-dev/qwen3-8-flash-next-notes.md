# Qwen3.8-Flash-Next: development notes

<!-- meta: description: Design decisions, hardware and compiler constraints, measurements,
validation method and open work from porting Qwen3.8-Flash-Next (qwen4_exp) to vLLM Neuron on
Trn2. -->
<!-- meta: date_updated: 2026-10-28 -->

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
  - **Budget.** The worker shrinks the KV budget by the side-cache ratio (0.3125 with BF16 KV:
    raw plus compressed keys).
  - **Compressed keys.** A second side cache holds one compressed key per 4-token block
    (128 BF16 values, a quarter of the raw keys' size), same block ids. Each step every token
    recomputes the entry of its own block from the cached raw keys (`write_compressed_keys`),
    so a block's entry is final once its last token is written, and scoring reads S/4 cached
    keys. Recomputing all of them from the S raw keys every step cost about 3 ms per layer at
    64k context and B=8 (the gather plus the FP32 mean, norm and RoPE).
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
- **Decode: compact context.** With one token per request, decode attends only to the
  selected tokens (`compact_decode_selection`):
  - **Top-k as an index list.** The NKI rotational top-k kernel (`vllm_neuron.functional.topk`)
    returns the 512 block indices; entries with a negative score are invisible blocks. Ties may
    resolve to other blocks than the bisection's lowest-index rule; both are an exact top-k.
  - **No K/V copy.** The fused decode kernel reads the K/V caches viewed in pages of one QSA
    block, `[num_blocks * 64, 4, 256]`, through a block table of the selected pages, then the
    open tail block, then the null page as padding: 2,304 slots instead of the bucket. Its mask
    allows the selected tokens before the current position and the active slot if the current
    token's block is selected or open (`_decode_context` hook of the Qwen3.5 attention).
  - **Why not compaction of the bisection mask.** Building the index list from the bool mask
    without sort (0/1 matmul prefix counts and a two-level search) cost 0.6 ms per layer at
    64k and B=8, on top of 2.7 ms for the bisection itself; the NKI top-k takes 0.4 ms.
  - **MTP verify rows** (several tokens per request) keep the masked full-context kernel.

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
- **QSA decode, per attention layer.** One rank (3 Q heads, 1 KV head), indexer + top-k +
  attention, device time from chains of 2-8 layers (`write_raw_keys` to the decode kernel's
  output):

  | Context bucket, batch | Masked full context (before) | Compact context | Kernel over the whole bucket, no indexer |
  |---|---|---|---|
  | 4k, B=1 / B=8 | 0.78 / 2.5-2.7 ms | 0.43 / 0.53-0.93 ms | 0.08 / 0.17 ms |
  | 16k, B=1 / B=8 | 1.4 / 3.3-3.7 ms | 0.53 / 1.0-1.15 ms | 0.11 / 0.36 ms |
  | 32k, B=1 / B=8 | 0.9-1.2 / 2.8 ms | 0.34-0.61 / 0.71 ms | 0.10 / 0.61 ms |
  | 64k, B=1 / B=8 | 1.1 / 4.5-4.7 ms | 0.47 / 0.77-0.79 ms | 0.31 / 1.17 ms |

  Pieces at 64k and B=8, each timed alone (they overlap in the full graph, so they do not
  add up): recomputing the compressed keys 3.2 ms, the bisection top-k 2.7 ms, the kernel over
  the whole bucket 1.2 ms; the NKI top-k 0.4 ms, scoring the cached compressed keys 0.2 ms.
- **Serving.** B=3 decode is about 48 ms/step at short context and 48-52 ms/step at 15-60k
  context; before the compact decode it was 57-61 ms/step at 1-15k and 102-105 at 60k.

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

### Run 2 (2026-10-27)

Same configuration on the final branch, but with 8 concurrent requests per replica
(`VLLM_NEURON_MAX_KV_CONCURRENT=8`) and 64 Aider threads:

| Metric | Value |
|---|---|
| pass_rate_1 / pass_rate_2 | 47.1 / 87.1 (106 / 196) |
| percent_cases_well_formed | 100.0 |
| error_outputs | 19 (13 Aider-side 3,600 s request timeouts) |
| user_asks | 136 |
| exhausted_context_windows | 6 |
| prompt / completion tokens | 2,807,446 / 5,674,140 |
| wall time | 8.8 h (about 180 tok/s average; about 900 tok/s at the start) |

- **No preemption occurred.** The pass rate matches run 1 within run-to-run variance.
- **The long tail remains.** The last few exercises run near batch 1 for up to an hour each.

### Run 3 on Model Runner V2 (2026-10-28)

Same configuration as run 2 on Model Runner V2 (`RUNNER=v2`, the `qwen4_exp-mrv2` branch), with
7 replicas (one TP=8 range was kept for validation) and 56 Aider threads:

| Metric | Value |
|---|---|
| pass_rate_1 / pass_rate_2 | 50.2 / 86.7 (113 / 195) |
| percent_cases_well_formed | 100.0 |
| error_outputs | 9 (2 Aider-side 3,600 s request timeouts, 7 exhausted context windows) |
| user_asks | 132 |
| prompt / completion tokens | 2,629,036 / 5,931,550 |
| wall time | 5.5 h (about 300 tok/s average) |

- **Same quality as V1.** pass_rate_2 is within run-to-run variance of runs 1 and 2.
- **Shorter, but not from faster steps.** About 52 tok/s per replica averaged over the run,
  against about 22 in run 2, while V2 and V1 decode at the same speed on a fixed workload
  (see [Model runner](#model-runner)). Run 2 lost 13 requests to Aider's 3,600 s
  timeout (run 3: 2), and the long tail of the last exercises varies from run to run.
- **22 preemptions** under `VLLM_NEURON_MAX_KV_CONCURRENT=8`, no server errors.

### Run 4 on Model Runner V2 with prefix caching (2026-10-29)

Run 3's configuration with `PREFIX_CACHING=1` (512-token blocks, 188k KV tokens per replica
against 200k):

| Metric | Value |
|---|---|
| pass_rate_1 / pass_rate_2 | 48.9 / 84.9 (110 / 191) |
| percent_cases_well_formed | 99.1 |
| error_outputs | 21 (19 exhausted context windows, no Aider-side timeouts) |
| user_asks | 135 |
| prompt / completion tokens | 3,245,110 / 6,288,167 |
| wall time | 4.7 h (about 370 tok/s average) |

- **Pass rate within run-to-run variance** (191 against 194-196).
- **Exhausted context windows are the long thinking responses.** Without `max_tokens` a
  response that reaches `max_model_len` ends with `finish_reason=length`, which Aider counts as
  an exhausted context window (the one inspected: about 22k output tokens). Slower runs hit
  Aider's 3,600 s timeout first: timeouts plus exhausted windows are 21, 19, 9 and 19 in runs
  1-4.
- **Hits equal misses.** Checked after the run on its replicas, greedy at batch 1 on 10 Aider
  transcripts of 3k-39k tokens: a repeated request (full hit) and a follow-up with a new tail
  (hit on the shared blocks) against the same request uncached. Every divergence within 300
  tokens is at an exact top-2 tie (gap 0.000; the BF16 logit step there is 0.125, and two
  uncached runs on two replicas differ by one step). Preemption with hits on the preempted
  request's own blocks matches HF on CPU (`test_preemption_with_prefix_caching_matches_hf`).
- **175 preemptions** (hit rate 10-18% per replica), no server errors.

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

### On the 2026-10-27 qwen3_5-upstream base

The work was replayed onto the updated Qwen3.5 branch. That brings in logprobs for speculative
and async-scheduled steps, seeded sampling on Neuron, int64 slot mappings, and dedicated
state-pool pages kept inside the pool. Re-validated on device:
- the 48-layer FP8 logit test passes with the same numbers as before;
- through the async server (65,536 context) and the MTP server, logprobs come back for every
  generated token, with the greedy token equal to the top-1;
- seeded requests are reproducible;
- a passkey at 40,842 prompt tokens is retrieved.

### MTP at the 64k serving configuration (2026-10-28)

The measurements in this section predate the compact QSA decode: their plain-decode baseline is the
masked full-context decode. Since then plain decode is about 2x faster at 30-60k context with
3 requests, while MTP verify rows keep the masked path, so MTP's position relative to plain
decode at long context is worse than measured here. A compact context for verify rows is in
the open work.

`serve.sh` with MAX_MODEL_LEN=65536, PREFILL_BUCKETS="1024, 4096", DECODE_CTX_BUCKETS="4096,
16384, 32768", DECODE_INTERLEAVE=1, VLLM_NEURON_MAX_KV_CONCURRENT=8, max_num_seqs 8, against
the same configuration without MTP (async scheduling on, GPU_MEM_UTIL 0.85).

- **Memory.**
  - The KV cache drops from 256,094 tokens to 164,840 with 1 draft token (0.85). The draft
    weights take about 1 GiB per core and the recurrent-state pools grow from 0.16 to 0.45 GiB.
  - With 2 draft tokens, 0.85 fails during warm-up (HBM full while loading the batch-8,
    32,768-context graphs) and 0.83 fails at the first long prefill (`nrt_tensor_allocate`
    status 4). 0.82 runs every test, with 109,061 KV tokens.
- **Graphs.** Warm-up covers the batch {1, 8} x context {4,096, 16,384, 32,768, 65,536} verify
  graphs at 1 + k tokens per request, the plain decode graphs and the draft graphs. No graph
  compiled while serving B=1-8 at 1k-60k context.
- **Correctness.**
  - Nine greedy requests (1k-60k prompts, 800-4,000 generated tokens, staggered starts, up to 8
    running at once, fewer with MTP's smaller KV cache) with top-2 logprobs, with 1 and 2 draft
    tokens and without MTP. No errors.
  - Every first divergence from plain decode is at a top-2 logprob gap of 0-0.25, except one at
    0.5: the first token of a 15k prompt with 2 draft tokens, an exact tie on the MTP server.
    The greedy token always equals the top-1 logprob.
  - With 2 draft tokens, the first-token logprobs of prompts over 1,024 tokens differ from
    plain decode by up to 0.25, reproducibly and also for a single request (1 draft token: up
    to 0.04; plain decode at gpu_memory_utilization 0.82 and 0.85: identical). The cause is not
    known. On the 3,532-token HF golden prompt both agree with HF: greedy 8 of 8, and a maximum
    top-20 logprob error against HF FP32 of 0.62-1.03 per token with 2 draft tokens, 0.56-1.78
    without MTP and 0.58-0.96 for HF BF16.
  - Preemption: five 30k prompts with 3,500 tokens each (1 draft token) and three 33k prompts
    with 4,000 tokens each (2 draft tokens) each preempt one request. All complete, the outputs
    stay coherent, and they diverge from plain decode only at near-ties (top-2 gap at most
    0.375).
  - Retrieval: the needle in each of the nine prompts (9 of 9 with 1 and with 2 draft tokens),
    natural-text needles at 15k/30k/60k and three depths, sent three at a time (9 of 9 with 1
    and with 2), and passkeys at 14k-58k (9 of 9 with 2).
- **Speed.** See the recipe's performance table. MTP is faster only at batch 1: 1.4-1.8x with
  2 draft tokens (1.2-1.6x with model-card sampling), 1.1-1.35x with 1.
  - A batch of 2 already runs the batch-8 graphs. A verify step there costs 2.3-3.2x a plain
    step (B=3 at 20/30/40k: 105 ms plain, 242 ms with 1 draft token, 332 ms with 2; B=8 at 8k:
    63 ms plain, 182 ms with 2). At batch 1 the verify step costs 1.4-1.7x a plain step.
  - Plain decode with async scheduling off: 50 ms (B=1, 30k), 111 ms (B=3, 20-40k), 70 ms
    (B=8, 8k), against 52, 105 and 63 with it on.
  - End to end, the nine-request mix takes 677 s without MTP, 1,031 s with 1 draft token and
    1,214 s with 2; the five 30k prompts take 489 s and 724 s (1 draft token), the three 33k
    prompts 570 s and 867 s (2 draft tokens).
  - With a batch-2 graph (`SEQS_BUCKETS="1, 2, 8"`), plain decode at B=2 takes 59 ms/token at
    30k and 49 at 8k, against 75 at 30k in the batch-8 graph. 1 draft token (0.85, same KV
    cache) then gives 50 and 39 ms/token, 1.2-1.25x. With 2 draft tokens the extra graphs run
    out of device memory at 0.82 while serving two 30k requests.
  - An Aider-like load (16 requests, 8 at a time, 3-11k-token prompts, 1.5-4.5k-token outputs,
    model-card sampling, thinking on) generates 65.8 tok/s with 2 draft tokens against 79.6
    without MTP (gpu_memory_utilization 0.82 for both).

## Model runner

V1 (`neuron_model_runner.py`) is the default. Model Runner V2 (`neuron_model_runner_v2.py`)
serves the model with `VLLM_USE_V2_MODEL_RUNNER=1` (`serve.sh` `RUNNER=v2`), with async
scheduling also with MTP, decode context-length buckets, preemption, and prefix caching
(`PREFIX_CACHING=1`, V2 only). It builds on V2's support for recurrent-state models and MTP
(the Qwen3.5 work, the generic commits of the `qwen3_5-mrv2` branch).

What V2 needed for this model:
- **Host inputs.** V2 builds `input_ids` and positions in the graph, so the runner keeps a host
  copy of every slot's tokens and computed count, advanced as the graphs commit them, and
  joins on the earlier steps' sampled tokens (and, for verify rows, the device draft table)
  before it builds the n-gram rows. See [Host inputs](../design/host-inputs.md#model-runner-v2).
- **Dense prefill variants, QSA side caches.** As on V1 (`prefill_ctx_variants`,
  `kv_side_cache_ratio`). With MTP the drafter runs inside the prefill graph, and neuronx-cc
  2.28 fails the dense prefill graph with the MTP head in it at `-O1` (`NCC_INLA001` in
  `lower_act`); it compiles at `-O2`, which the model's `compiler_args_for` asks for that one
  graph (`prefill_dense_spec`).
- **PLE state, MTP head.** Nothing new: the PLE conv state is one more recurrent-state layer, and
  `Qwen4ExpMTP` has V2's fused draft forward (`mtp_hidden_size` is not needed: the hidden
  state stays in the graph).
- **Preemption.** Nothing new: V2 readmits a preempted request with its whole history.
- **Prefix caching.** Two fixes for state layers of different sizes (36 GDN layers, one PLE
  layer): one state page for every layer in align mode (vLLM needs one block size), and one
  placeholder shape in the specs (vLLM groups layers by spec equality, and otherwise made 49
  one-layer groups, so every attention layer's KV view, and the QSA side caches sized from it,
  spanned the whole pool). Blocks are 512 tokens with BF16 KV (FP8 KV: twice that).

Found on device:
- **KV-block clearing overflowed the runtime queue.** V2 keeps 32-token attention blocks, so a
  45k-token prompt allocated about 1,400 blocks in one step, cleared 8 per graph call; the
  runtime refused the next submission (`Execution Queue Full`). The clearer now also uses
  256-block calls and waits every 32 calls.
- **KV-block clearing slowed every step.** Clearing new blocks before the forward, every few
  decode steps with 32-token blocks, made the runtime switch NEFFs twice per clear (an 8-layer
  runtime trace: 414 executions and 354 model switches where V1 had 309 and 126; device time of
  the decode graph itself within 2.5% of V1's). Not clearing is wrong: stale K/V in reused
  blocks changed the logit test's second prompt (K5 0.0201 to 0.061-0.069). Without prefix
  caching V2 now clears a request's blocks when it is removed (one call per finished or
  preempted request), which restores the numbers exactly and V1's step time.
- **Compile cache.** `parallel_compile` compiles every captured HLO without a NEFF in the
  cache directory, also one left by an earlier failed run; delete such entries.
- **`vllm serve` and the checkout.** The `vllm` launcher runs the venv's Python; a checkout
  added with a `.pth` in another venv is not on its path. Put it on `PYTHONPATH`.

Validation (CPU): `test/unit/model/qwen4_exp/test_v2_runner.py` runs the tiny model (sparse
QSA, real n-gram rows from the HF table) through the V2 runner and schedulers against HF:
plain and MTP (oracle drafts with partial acceptance, and the real head), sync and async,
chunked prefills, prefix caching (hits on GDN/PLE checkpoints and cached QSA keys), FP8 KV
with and without prefix caching, preemption with prefix caching (readmission with a hit on
the request's own blocks). `test/unit/model/qwen3_5/test_v2_host_inputs.py` checks the
host-built rows against the graph's and the host copy against the device state.

Validation (trn2.48xlarge, 48-layer FP8, 2026-10-28):
- **Logit test** (`test/integration/model/qwen4_exp/test_model.py`): passes on V2 and V1 at
  batch 1 and batch 4 (2026-10-29/30), no divergent token, on the native and the XLA route
  (`VLLM_NEURON_BACKEND=vllm_neuron`, as `serve.sh`). On the XLA route V2 compiles its
  2048-token prefill graph only at `-O2` (NCC_INLA001 at `-O1`: more than 8 activation tables
  in one instruction; repro `ncc_inla001_qwen4exp_v2_prefill2048.tgz`), so the test asks for
  `-O2` on V2 (`QWEN4_EXP_OPTLEVEL`); the serving buckets (1024, 4096) compile at `-O1`. Its references now hold the FP8
  experts the model serves (`_round_experts_to_fp8`, the loader's own per-channel
  quantization, bit-identical to the served weights). Against the BF16-expert references, V2,
  V1 and the `qwen4_exp-upstream` branch (before any V2 work) all failed 2 of 4 prompts with the
  same numbers to four decimals (prompt 1 diverging at token 4 on a 0.375 near-tie): the test
  measured the FP8 quantization against bounds meant for BF16 compute. The 4-layer BF16
  truncation passes (three-way).
- **Serving** (64k configuration, `RUNNER=v2`): needles at 3k-45k tokens 6 of 6, three at a
  time; 24 mixed requests 8 at a time without errors. Startup with a warm compile cache about
  19 minutes.
- **Speed against V1** (same commit, same knobs, one freshly started replica each, greedy,
  `ignore_eos`): equal. Inter-token time 35.5 ms against 35.2 at B=1 (8k prompt), 55.5 ms
  against 55.6 at B=8 (4k prompts), 116 ms against 115 at B=4 (30k prompts, decode interleaved
  with the others' prefill chunks); time to first token equal.
  - Before KV blocks were cleared at free time, V2 steps were about 8% slower (37.3 / 59.9 /
    125 ms) and the first token of a B=8 batch took longer: see "KV-block clearing" above.
  - Where a B=8 step goes on the host: input preparation 1.2 ms (overlapped with the previous
    step), the join's wait for the previous step 52 ms, host inputs 0.9 ms and the forward call
    3.2 ms (2.5 with `VLLM_NEURON_SKIP_GUARD_EVAL=1`, which gains 0.7 ms per step).
- **Aider Polyglot** (runs 3 and 4 above): 86.7% after 2 tries in 5.5 h on 7 replicas; with
  prefix caching 84.9% in 4.7 h.
- **Preemption** under `VLLM_NEURON_MAX_KV_CONCURRENT=8`: 22 preemptions during the Aider run,
  no errors.
- **Prefix caching:** greedy outputs equal the server without it for 4 questions on a shared
  12k-token prefix, sequential and concurrent, and for a turn continuing an answer; hit rate
  84%; time to first token 9.8 s cold, 1.1 s cached; 16 mixed requests 8 at a time without
  errors.
- **MTP** (2 draft tokens, async scheduling): greedy outputs equal plain decode up to exact
  BF16 ties (top-2 gap 0.000); 84% of drafts accepted (2.68 tokens per step, V1: 2.25-2.46);
  B=1 27-34 ms/token (V1: 27). KV cache 107k tokens at `GPU_MEM_UTIL=0.82` (V1: 109k).
  Same benchmark as the speed comparison (greedy, `ignore_eos`, tokens counted from `usage`),
  MTP against plain V2: inter-token 19.2 ms against 35.5 at B=1 (1.85x), 67.1 against 55.5 at
  B=8 (21% slower), 111 against 116 at B=4 with 30k prompts; time to first token 6.1 / 13.8 /
  70.4 s against 5.7 / 12.9 / 57.5 (the prefill graphs carry the drafter), so the 30k batch
  takes longer overall (146 s against 117).
- **FP8 KV cache**: the logit test (FP8-expert references) passes at batch 4 and fails 2 of 4
  prompts at batch 1, without a divergent token (prompt 4: all-K error 0.136 against the 0.09
  bound, BF16 KV 0.091). Unit scales lose nothing to calibration: in the HF model (5 prompts of
  2k-4k tokens: code, prose, Chinese), per-layer K amax is at most 15.8 and V amax at most 34.8,
  inside the +-240 FP8 range, and the FP8 round trip errs by 2.65% with unit scales and with
  per-layer amax/240 scales alike (the e4m3 mantissa); finer scales (per head or token) would be
  needed to do better. Prefix caching, FP8 KV and MTP together (1024-token blocks, 146k KV tokens): misses
  and hits give the same greedy output, sequential and concurrent, equal to the BF16 server's
  answers; needles at 12k-45k 3 of 3; 24 mixed requests 8 at a time without errors.
  Two-hour soak of that server (2026-10-29): 18 rounds of 16 sampled requests (prompts of
  500-24k words, 8 at a time), each followed by the hit/miss consistency check: all
  consistent, no errors, 12 preemptions, prefix-cache hit rate 35%.

Qwen3.5 regression (the generic V1/V2 commits, 2026-10-30):
- **Logit test** (`test/integration/model/qwen3_5/test_model.py`) passes on V1 and V2 at batch
  1 and 4, no divergent token: Qwen3.8-27B (TP=4) on the native and the XLA route, Qwen3.5-2B
  (`QWEN3_5_TP=2`) on the native route. V1 gives the `qwen4_exp-upstream` branch's numbers to
  four decimals on both routes.
- **V1 MTP was broken by `ddfcba8`** (preemption under `VLLM_NEURON_MAX_KV_CONCURRENT`, on
  `qwen4_exp-upstream`): Qwen3.8-27B with MTP 2 and the recipe's `serve.sh` dropped tokens
  ("OrderedDictUCache:"); `qwen3_5-upstream` and the commit before were correct (bisected on
  device). Its partial-prefill check (`num_computed + num_scheduled < num_tokens`, to keep a
  resumed request's recompute from emitting tokens) also flagged verify steps under async
  scheduling, whose token count already holds the in-flight steps' accepted tokens, and
  discarded their tokens. Fixed: a step with scheduled drafts is never a partial prefill. MTP
  then equals plain decode up to BF16 ties (gap 0.000 and 0.125), 24 ms/token at batch 1. V2
  was not affected.
- **Compile routes.** `VLLM_NEURON_BACKEND` defaults to the native route; `serve.sh` (both
  models) selects the XLA route (`vllm_neuron`), and the logit tests take whichever the
  environment sets. MTP on the native route hits an out-of-bounds access on device during warmup
  or the first request, for Qwen3.8-27B on every branch tried (this one, `qwen4_exp-upstream`,
  `qwen3_5-upstream`, `qwen3_5-mrv2`) and for Qwen3.5-2B on this branch and
  `qwen4_exp-upstream` (`qwen3_5-mrv2` serves 2B); not investigated further. Qwen3.5-2B on the
  native route also fails to compile at batch 8 and with a 4096-token prefill bucket
  (NCC_ITRF901 / NCC_IGCA044).

## Open work

- **QSA verify rows.** MTP verify steps still mask the full context; a compact context of the
  union of the rows' selections would apply there too.
- **QSA top-k in prefill.** Prefill still selects with the bisection top-k (about 45 count
  passes per query block).
- **Serving defaults.**
  - Per-replica concurrency of 8 (used in Aider run 2) as the recommended Aider configuration.
  - MTP stays off: at the 64k configuration it helps only at batch 1 and shrinks the KV cache
    by 36-57%.
    Next steps: profile the batch-8 verify graph at long context, and turn drafting off above
    batch 2 (vLLM's `num_speculative_tokens_per_batch_size`) with a batch-2 graph, which makes
    1 draft token 1.2x faster at batch 2.
  - FP8 KV stays opt-in: its error is the e4m3 rounding, which per-layer scales do not reduce
    (see above); per-head or per-token scales would.
- **FP8 prefill MoE without in-graph dequantization.** The blockwise kernel's per-channel FP8
  path fails with NCC_IGCA108 because scale buffers are allocated inside its dynamic loop.
- **Prefix caching on V1.** V2 serves it (align-mode checkpoints of the GDN and PLE state).
- **Dense prefill with MTP on V2.** The compiler error above at `-O1` (worked around with
  `-O2` for that graph; repro: the captured HLO and command).
- **Device profiles of the full model.** The runtime cannot allocate its profiler buffers next
  to the 48-layer model and KV cache (HBM full); profile truncations, as above.
- **Smaller costs:**
  - batch the GDN decode kernel's per-request instruction chains (~5 ms/step at B=8);
  - fix the gated-residual group RMSNorm layout (~2 ms/step at B=1);
  - the PLE state write;
  - fuse the top-10 router (nkilib's router stops at top-8).
- **Fewer graph inputs.** Per-step runtime overhead scales with the number of graph inputs
  (~1,177 here).
