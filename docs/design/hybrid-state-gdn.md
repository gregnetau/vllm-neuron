# Hybrid Recurrent State for Gated DeltaNet Models

<!-- meta: description: Design of per-request recurrent state (Gated DeltaNet) alongside paged
KV cache in the vLLM Neuron plugin, as used by Qwen3.5 / Qwen3.8 dense models. -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-27 -->

## Overview

Qwen3.5 / Qwen3.8 models (dense, and the sparse MoE Qwen3.5-35B-A3B, whose FFNs are the only
difference) interleave Gated DeltaNet (GDN) linear-attention layers with full-attention layers
(48 + 16 on 27B, 30 + 10 on 35B-A3B). A GDN layer carries a fixed-size state per request
instead of a KV cache:

| State | Shape per rank (27B, TP=4) | Dtype |
|---|---|---|
| Conv state | `[K-1, conv_dim/tp]` = `[3, 2560]` | float32 |
| Recurrent state | `[Hv/tp, Dk, Dv]` = `[12, 128, 128]` | float32 |

This document covers how that state is allocated through vLLM's hybrid KV cache manager,
how it is laid out in device memory, and how the model reads and writes it on Neuron.

## KV cache groups

The model reports GDN layers through `KVSpec.state_layers` (`StateLayerSpec`: shapes and
dtypes per layer, see `model/kv_cache.py`). The runner emits one `MambaSpec` per GDN layer
with `mamba_cache_mode="none"` (see [Prefix caching](#prefix-caching) for `align`), so vLLM
allocates one state page per request per GDN layer group. A request's page is the first block id of the group's block table; no extra
scheduler input is needed.

vLLM unifies page sizes across groups by enlarging the attention block size. The runner pads
the state page so the enlarged attention block is a whole number of 256-token blocks, and
the attention metadata splits each manager block back into kernel-sized blocks
(`--block-size`, 32 by default; 128 is recommended with an FP8 KV cache, see
[Full-attention prefill](#full-attention-prefill)). With one KV head per rank, a manager block
is a pure reshape of its kernel blocks. Checkpoints with fewer KV heads than ranks (Qwen3.5-35B-A3B:
2 KV heads at TP=4) replicate each KV head over `tp / num_key_value_heads` ranks, so they also
keep one KV head per rank (`Qwen3_5Config.kv_head_shard`). The runner logs the result, for example
`KV cache block_size resolved: cache_config=32, per_group=[1024, 2048, 2048, 2048]`.

## Dedicated state pool

Without prefix caching, the runner allocates the state in its own pool per GDN layer instead of
using the shared raw KV buffers (`NeuronModelRunner._state_pool_pages`): the null page 0 plus
`1 + num_speculative_blocks` pages for each of the `max_num_seqs` batch slots. The GDN groups'
block tables are translated on the host (`_state_pool_table`): a request takes a slot on its
first step and keeps it until it finishes or is preempted, including steps where the
scheduler leaves it out of the persistent batch (during another request's prefill). The pool
is taken out of the KV budget (`NeuronWorker.determine_available_memory`).

In the shared layout every raw buffer is a graph input twice, as attention K/V views and as
state views, and neuronx-cc's graph I/O check (`NCC_EVRF009`, 24 GB per logical core) counts
both; that capped the KV pool at about half the free HBM. vLLM still allocates the state
groups' blocks (`1 + k` per request and group) from the shared pool; they stay unused. So that
they cost little, the runner reports those groups with a placeholder page of one 256-token
attention block (`_build_state_cache_specs`) and keeps the real page for the dedicated pool
(`_state_page`). That also stops vLLM from enlarging the attention block to the state page
(2048 tokens with FP8 KV; `per_group=[256, ...]` instead). Prefix caching (state checkpoints
are cached blocks) keeps the shared layout and the real page, as does
`VLLM_NEURON_SHARED_STATE_POOL=1`.

## Model Runner V2

V1 is the default for these models. With `VLLM_USE_V2_MODEL_RUNNER=1`, Model Runner V2
serves them, with or without MTP speculative decoding, with or without prefix caching (see
[Prefix caching on Model Runner V2](#prefix-caching-on-model-runner-v2)), but not with the
shared state pool. That configuration, and those without segmented prefill, stay on V1.
Both runners build the
specs, pools and KV-block clearing with `vllm_neuron/vllm/worker/hybrid_state.py`. V2
differs from V1 in two ways:

- Without prefix caching the placeholder page is one attention page, not a 256-token block,
  so vLLM does not enlarge the attention block. The attention group keeps `--block-size`,
  and a state group's block spans `max_model_len`. No manager block is split into kernel
  blocks.
- V2 keeps a request in the same batch slot from admission until it finishes or is
  preempted. A freed slot is quarantined while an in-flight graph still touches it. So the
  slot indexes the pool directly: page `1 + slot` (`build_state_pool_rows`), with no
  translation table.

Every prefill uses the first KV segment size, so V2 requires all prefill buckets to be
multiples of 128 for these models.

With MTP, V2 gives each slot `1 + k` pool pages (`1 + slot * (1 + k) + j`) and runs every
decode step as a verify step whose GDN metadata carries `state_in`. Unlike V1 it does not
pass `state_in` from the host or from the previous step's rejection output: the verify
graph reads it from a slot-indexed device tensor of accepted counts that the previous
graph of the same slot committed (`num_accepted - 1`), so a request held back for any
number of steps resumes from its own last accepted block. See
[Speculative decoding on Model Runner V2](speculation/mrv2-spec-decode.md).

## Prefix caching

This section describes Model Runner V1 (no MTP); for V2, with and without MTP, see
[Prefix caching on Model Runner V2](#prefix-caching-on-model-runner-v2).

With `--enable-prefix-caching` the platform selects vLLM's Mamba cache mode `align`, and the
runner emits each `MambaSpec` with a block of as many tokens as the enlarged attention block
(1024 for Qwen3.8-27B at TP=4), so the per-group block sizes are equal
(`per_group=[1024, 1024, 1024, 1024]`) and a state block lines up with an attention prefix
block. vLLM then keeps a request's running state in the state block of its last scheduled
token, schedules prefills in block-aligned chunks, and caches each block whose state ends on
the block boundary; earlier non-cached state blocks are released.

The runner implements the parts of align mode that vLLM's GPU runner does in
`mamba_utils.preprocess_mamba`:

- Before each step, when a request's running state moves to a new block (a prefill chunk or
  a decode step crossing a boundary, or a new request resuming from a cached prefix), the
  previous block's page is copied to the new one for every GDN layer, in one small compiled
  graph over `[pages, rows, 128]` views (`_preprocess_mamba_align`).
- The GDN metadata carries only the running state's block (`[B, 1]`), at warmup and at run
  time, and the attention group's slot mapping (state blocks before the running one can be
  the null block, which would mark their tokens as padding).
- KV-block clearing skips the blocks of a new or resumed request below `num_computed_tokens`,
  which are prefix-cache hits.

A prefill with a cached prefix reads its initial GDN state from the copied page (as for any
chunk after the first) and attends to the cached KV blocks. Prefix-cache granularity is the
1024-token block, and the prefill chunk (`max_num_batched_tokens`) must be at least one block.

## Prefix caching on Model Runner V2

V2 keeps the running state in its dedicated per-slot pool also with prefix caching, and
uses vLLM's state blocks only for the block-boundary checkpoints that vLLM caches
(`hybrid_state.StateCheckpointer`, set up in `NeuronModelRunnerV2.initialize_kv_cache`).

### Why the checkpoints are vLLM blocks

In align mode vLLM decides which state blocks are cached, keeps them in its block pool
next to the attention blocks and evicts them with the pool's LRU. Once it admits a request
with a hit, the scheduler counts the prefix as computed, so the state block of the hit must
hold the state: nothing recomputes it. A checkpoint store of the runner's own would need
either a page per vLLM block id (the KV pool's size a second time) or its own eviction,
which vLLM does not know about (a hit on an evicted checkpoint silently gives wrong output).
So the checkpoints live in the state blocks vLLM allocates, with the real page size, as in
vLLM's GPU runner and in V1.

What V2 does not share with V1 is the running state. The compiled forward graphs read and
write the GDN state only in the dedicated pool, as without prefix caching, and bind the
shared KV buffers only as attention KV. In V1's shared layout every raw buffer is a forward
graph input twice (K/V and state views), and neuronx-cc counts it twice against the 24 GB
graph I/O limit (`NCC_EVRF009`; Qwen3.8-27B FP8 with `KV_CAP_FRACTION=0.48` needs 28 GB).
V2's forward graphs count the pool once. Only two small copy graphs bind the buffers as
state, once per buffer (layers on one buffer share the view objects,
`hybrid_state.shared_state_views`), and without the weights.

Costs: vLLM enlarges the attention block to the state page (1024 tokens for Qwen3.8-27B at
TP=4 with BF16 KV and `--block-size 32`, 2048 with FP8 KV and `BLOCK_SIZE=128`), which is
the prefix-cache granularity, and allocates `2 + k` state blocks per request and state group
(used only as checkpoint targets) plus one per retained checkpoint. The dedicated pool is
taken out of the KV budget as without prefix caching.

### Blocks

The runner requires one block size for every KV group (the enlarged attention block,
`_prefix_cache_geometry`), so vLLM hashes at that block and never makes partial hits. Each
attention manager block `b` is viewed and addressed as kernel blocks
`b * r .. b * r + r - 1` of `--block-size` tokens, `r = block / --block-size`:

- the KV views are `(2, num_blocks * r, heads, --block-size, head_dim)`, the same bytes of
  each K / V half that the state's paired halves use, for any number of KV heads;
- `NeuronRequestState` stores the attention groups' kernel block ids
  (`configure_block_sizes(kernel_ratios=...)`), so slot mappings, kernel block tables,
  decode context-length trims and the metadata's `block_size` are in kernel blocks, as
  without prefix caching;
- KV-block clearing zeroes all `r` kernel blocks of each new manager block.

The state groups' rows keep vLLM's manager block ids (the checkpoint blocks; in align mode
the columns below the running block are the null block), and their slot mapping is the
attention group's (the models mark valid tokens by slot > 0). The GDN block tables of the
graphs are the slot's pool pages (`[B, 1 + k]`, `-1` for padded rows), as without prefix
caching.

### Save and restore

- **Save**, after each step's forward (`_save_prefix_checkpoints`). A step that reaches
  block boundary `B` leaves the state after exactly `B` tokens in a pool page of its slot:
  page 0 after a prefill chunk or a plain decode step; page `j = B - n - 1` after a verify
  step that started at `n` computed tokens and accepted at least `j + 1` tokens (the verify
  step writes the state after token `j` to page `j`). The copy graph
  (`save_state_checkpoints`) takes a candidate boundary and the destination (the state
  groups' blocks at column `B / block - 1`) from the host, and copies only if the counts the
  step committed on the device satisfy `n < B <= n + accepted`. The host candidates
  (`checkpoint_boundaries`) are a chunk's or plain step's end if it is a boundary, and for a
  verify step the boundaries in `[n_host - slack + 1, n_host + 1 + drafts]`: the scheduler's
  count `n_host` is exact with sync scheduling and exceeds the device count by at most the
  in-flight step's rejected drafts with async scheduling (`slack = k`). A block larger than
  `2k + 1` tokens has at most one boundary there. This is vLLM's GPU align postprocess (the
  state at the accepted boundary to the block ending there), without a device-to-host read.
- **Restore**, before the forward (`_restore_prefix_checkpoints`). A request admitted with
  `num_computed_tokens = h > 0` (a multiple of the block) gets the state groups' blocks at
  column `h / block - 1` copied to page 0 of its slot. Its prefill reads that as the initial
  state (`cached_seq_len > 0`), and its attention reads the cached KV blocks (the segmented
  prefill kernel takes a prior length that is a multiple of the kernel block). A hit one
  token before the end of the prompt runs that token in the decode graph, from the same
  page. With MTP the slot's accepted count is reset to 1 at admission, as for any request.
- Both graphs process one row per call (`STATE_CKPT_ROWS`, one compile each, compiled in
  `warm_up_hybrid_graphs`), use 1-D `index_select` / `index_copy_` with int64 indices over
  `[pages, rows, 128]` views made at initialization, and read-modify-write each page, so a
  row that does not copy (not reached, padding, warmup) writes the null block or page onto
  itself.

Ordering: vLLM hashes a block when it schedules the step that computes it, lets other
requests hit it from the next step on (`cached_blocks_this_step`), and frees state blocks
only below the processed (not optimistic) count. The save is submitted right after the
forward of the step that reached the boundary, and NRT runs submissions in order, so every
later restore, KV-block clearing and forward that touches the block runs after it.
`new_block_ids_to_zero` leaves prefix hits out.

vLLM retains a checkpoint at every block boundary with MTP (its default for hybrid models
with an EAGLE-style drafter) or with `--prefix-cache-retention-interval` a multiple of the
block; otherwise only at a prompt's last block boundary and at shared-prefix junctions.
With MTP it frees a boundary block that a verify step passed (accepted tokens beyond the
boundary) before caching it, so the boundaries that get cached are those a step ends on.
The runner checkpoints every boundary a step reaches (one copy per block of a request);
the blocks vLLM does not retain are freed and reused.

### Configuration

- `VLLM_USE_V2_MODEL_RUNNER=1 PREFIX_CACHING=1`, with or without `MTP_TOKENS` (V1 rejects
  prefix caching with MTP). The platform selects Mamba cache mode `align`.
- Segmented prefill (V2 requires it for prefix caching), prefill buckets multiples of 128.
  The scheduler ends prefill chunks on block boundaries, and stops a chunk at the prompt's
  last full block and at shared-prefix junctions; with a block larger than the chunk
  (`PREFILL_BUCKET=1024` with 2048-token blocks) every other chunk ends on a boundary.
  Prompts shorter than one block plus one token never hit.
- With MTP the block must exceed `2k + 1` tokens (true for the served models), and vLLM drops
  the last matched block of a hit (EAGLE-style drafter), so hits are one block shorter.

### Tests

`test/unit/model/qwen3_5/test_v2_prefix_caching_e2e.py` runs vLLM's scheduler and KV cache
manager (prefix caching, the engine core's block hasher), the V2 runner and the Qwen3.5
module (dense and sparse MoE, NKI stubbed), with 16-token blocks split into 8-token kernel
blocks, against HF: concurrent sharers, a request continuing a finished one's output
(decode checkpoints, blocks reused from the cache), hits of one and of several blocks, a hit
followed by chunked prefill across boundaries, a hit one token before the end of a prompt,
evictions with a small pool and decode steps between prefill chunks, vLLM's default
retention, sync and async scheduling, without MTP and with MTP (oracle drafts with partial
acceptance, and the real MTP head, whose drafts after a hit equal a from-scratch run). Every
state block vLLM cached was written by a save that reached its boundary; the checkpoints
saved from a verify step's draft column and every restored page equal HF's GDN state after
as many tokens; zeroing never touches a hit block; decode steps with several rows, padding
and rows out of slot order carry each row's pool pages and `-1` for padded rows. With an FP8
KV cache (64-token blocks, 8 kernel blocks each) and batched decode the outputs equal the run
without prefix caching. `test/unit/worker/test_v2_prefix_caching.py` covers the copy
functions, the candidates, the block split and the runner's specs and layout.

### Device validation

Not run on a device yet. On a trn2 instance with no other process on the devices:

```bash
# Qwen3.5-2B, TP=2, BF16 KV: without and with MTP
VLLM_USE_V2_MODEL_RUNNER=1 PREFIX_CACHING=1 MODEL=<ckpt>/Qwen3.5-2B TP=2 \
    MAX_MODEL_LEN=8192 PREFILL_BUCKET=1024 MAX_NUM_SEQS=4 SEQS_BUCKETS="1, 4" \
    examples/vllm_neuron/models/qwen3_5/serve.sh
VLLM_USE_V2_MODEL_RUNNER=1 PREFIX_CACHING=1 MTP_TOKENS=2 MODEL=<ckpt>/Qwen3.5-2B TP=2 \
    MAX_MODEL_LEN=8192 PREFILL_BUCKET=1024 MAX_NUM_SEQS=4 SEQS_BUCKETS="1, 4" \
    examples/vllm_neuron/models/qwen3_5/serve.sh
# Qwen3.8-27B, TP=4, static FP8 + FP8 KV (2048-token blocks), with the KV budget that V1's
# shared layout cannot compile (NCC_EVRF009); then the same with MTP_TOKENS=2
VLLM_USE_V2_MODEL_RUNNER=1 PREFIX_CACHING=1 MODEL=<ckpt>/Qwen3.8-27B TP=4 \
    FP8_SCALES=<path>/fp8_act_amax.json KV_CACHE_DTYPE=fp8 BLOCK_SIZE=128 \
    MAX_MODEL_LEN=16384 PREFILL_BUCKET=2048 MAX_NUM_SEQS=4 SEQS_BUCKETS="1, 4" \
    GPU_MEM_UTIL=0.95 KV_CAP_FRACTION=0.48 examples/vllm_neuron/models/qwen3_5/serve.sh
```

Log lines: `Prefix caching for ... uses Mamba cache mode 'align'`; `V2 prefix caching for
recurrent state (Mamba align): 2048-token prefix-cache blocks (attention manager blocks
split into 16 kernel blocks of 128 tokens)` (27B FP8 KV; 1024 and 32 with BF16 KV);
`Recurrent-state pools: ... attention block sizes [128]` (kernel blocks); no
`NCC_EVRF009`; vLLM's `Prefix cache hit rate` above 0 once prompts repeat. Graphs: the
prefill graphs `prefill_s{P}_kv{S}`, the decode graphs `decode_b{B}` (`decode_b{B}_s3` with
MTP), the KV-block clearing graph and the two checkpoint copy graphs
(`save_state_checkpoints`, `restore_state_checkpoints`), all compiled at startup; nothing
compiles while serving. Checks: greedy outputs equal those without prefix caching for a
shared multi-block prefix sent sequentially and concurrently (batch 4, also with fewer
than 4 requests so decode rows are padded), for a prompt continuing a previous answer, and
with MTP; TTFT drops for a repeated multi-block prompt.

## Page layout

Each raw KV buffer is shared by one layer of every group. The attention view of a buffer is
`(2, num_blocks, ...)` for K and V, so a state page is stored as two contiguous float32
halves aligned with that layout (`state_cache.paired_half_views`). The model binds each
half as `[pages, rows, 128]` outside the compiled graph (`Qwen3_5GatedDeltaNet.bind_state`).

A page is `[ssm | conv | pad]` in rows of 128 floats:

| Rows | Contents |
|---|---|
| `h*128 + dk` | recurrent state of head `h`, row `dk`, columns `dv` |
| `Hv*128 + j*NCB + f` | conv state tap `j`, channels `f*128 .. f*128+127` |

With 1024-row halves no head straddles the two halves, which lets one DMA move all of a
program's heads.

## Reads and writes

- **Prefill** (one request per step, padded to the bucket): the GDN layer reads its page,
  runs the causal conv and the chunked delta rule (`gdn_ops.chunk_gated_delta_rule`,
  chunk 64), and writes the final state back. Padding tokens are identity updates
  (`g = beta = 0`); a token is valid iff its slot is not the null block.
- **Decode**: `gdn_kernels.gdn_decode` runs the conv step, q/k L2 norm, gating, the gated
  delta rule, the gated RMSNorm and the in-place page update in one NKI kernel. Heads are
  split across the two programs of a logical core; page DMAs use a runtime page offset with
  hardware descriptor generation. With static FP8, `gdn_kernels.gdn_decode_fp8` runs the
  whole sub-layer instead (input RMSNorm, projections, the step above, `out_proj`): each
  program owns half the value heads and the key heads they read, and updates only its own
  state and conv rows, so it needs no core barrier and supports up to 32 requests. With BF16
  (and FP8 where the fused kernel does not apply), steps of more than one token (B > 1, or an
  MTP verify step) run `gdn_kernels.gdn_decode_batched`: the same head split and recurrence
  after the PyTorch input projections, for up to 32 tokens, returning the normed output for
  `out_proj`. `VLLM_NEURON_GDN_BATCHED_DECODE` selects it: `0` off (PyTorch path), `1`
  (default) for steps of more than one token, `2` also for single-token steps instead of
  `gdn_decode`. Other cases use the PyTorch ops in `gdn_ops`.
- State writes are in place (`index_copy_` on the bound views, or the kernel's aliased
  outputs). The runner's block tables use `-1` for unused entries; the model maps them to
  the null block before indexing.

## Speculative decoding (MTP)

With vLLM's `mtp` speculative method and `k` draft tokens, each `MambaSpec` gets
`num_speculative_blocks = k`, so a request holds `1 + k` state blocks per GDN layer group
(`block_table_tensor` is `[B, 1 + k]`). A verify step runs `1 + k` tokens per request through
the GDN layers in order (`gdn_kernels.gdn_decode_fp8` and `gdn_decode_batched` keep the state
in SBUF between them)
and writes the state after token `j` to block `j`. The next step reads the state from the
block of the last accepted token, passed as `state_in` in the GDN metadata. Prefills and
single-token steps read `state_in` (0 after a prefill) and write block 0.

With synchronous scheduling the runner sets `state_in` from the previous step's sampled
tokens. With asynchronous scheduling the target graph derives it from the previous step's
rejection-sampler output, in the same on-device correction that adjusts positions and slot
mappings for rejected drafts (`functional.spec_decode_correction`); the "no previous step"
placeholder rows carry one valid token and no drafts, so they resume from block 0. Selecting
the accepted state inside the verify graph instead (copying it to block 0 after rejection
sampling) was tried and rejected: in-place copies on the bound state buffers after the
kernels that write them gave wrong results and cost about 13 ms per step.

## Decode batches

Decode buckets above one request pad the batch with rows whose block-table entries are `-1`.
Padded rows write their K/V to slot 0 of the null block, their GDN state writes are skipped
(out-of-bounds page index; `gdn_decode_batched` reads the null page for them, so their output
stays finite), and their hidden state is zeroed after every layer, so the null
block (read, masked, for unused table entries) stays finite. Padded sampling rows get neutral
parameters. Two device issues were specific to batches:

- The library `attention_block_tkg` with `head_dim` > 128 wrote the transposed Q heads
  head-major while RoPE and attention read them batch-major, mixing the heads of different
  requests; the vendored copy writes them batch-major.
- `index_select` of the final hidden states by the sampling positions returned NaN rows when
  the same hidden states were also a graph output (the MTP draft input). Decode samples every
  row in order, so the gather is skipped there.

With `decode_context_length_buckets`, the runner trims each attention group's block table to
the bucket's block count. The GDN groups' tables (`1 + k` state blocks, block size as large as
`max_model_len`) are not trimmed: they do not grow with the context, and warmup compiles them
whole. Model Runner V2 does the same (with or without MTP; see
[MRv2 speculative decoding](speculation/mrv2-spec-decode.md#decode-context-length-buckets)).

## Full-attention prefill

Under segmented prefill (prompt longer than `max_num_batched_tokens`) the 16 full-attention
layers attend each chunk to the cached prefix and to the chunk itself, already written to the
cache. They run the nkilib `attention_segmented_cte` kernel through `NF.segmented_attention`,
which iterates over the prefix with dynamic loops, so a chunk's cost depends on the cached
length and not on `max_model_len` (the previous PyTorch path gathered and scored the whole
block table, a `[6, 1024, max_model_len]` float32 score tensor per layer). Kernel constraints
and how the plugin meets them:

| Kernel constraint | Handling |
|---|---|
| `head_dim` <= 512, tiled over 128 partitions | 256 is supported; the wrapper's former 128 limit was stale |
| K tile of 512 tokens: block size divides 512 | blocks viewed as sub-blocks (below) |
| DMA-transpose K load: 2-byte dtype, blocks per segment a multiple of 16 | BF16 cache viewed as 64-token blocks (16 per 1024-token segment) |
| Per-block K load (otherwise): block size <= 128 | FP8 cache viewed as 128-token blocks |
| FP8 K only pre-transposed or packed | vendored `attention_segmented_cte.py`: 1-byte K takes the per-block load, widened to bf16 for the PE transpose |
| `prior_tokens` a multiple of the block size | prefills are isolated and chunked by `max_num_batched_tokens`; cached prefixes are block aligned |
| Query length a multiple of 128 | the Q chunk is padded to a prefill bucket (`num_batched_tokens_buckets`), independent of the KV segment; each step picks its segment from `kv_segment_size_buckets` by the cached length (`pick_kv_segment_size`) |

With one KV head per rank a block `[1, B, Dh]` is contiguous, so the cache viewed as
`[nb * r, 1, B / r, Dh]` is a view and block `b` becomes sub-blocks `b*r .. b*r+r-1`
(`_split_kv_blocks`, opted into with `segmented_attention(sub_block_kv=True)`); unused
table entries (-1) become the null block's sub-blocks. The packed FP8 K layout that the library kernel reads directly is not used: the decode kernel
reads it only at `head_dim` <= 128.

## Scheduling

Prefill and decode are separate graphs and a step is one or the other; prefills run one
request at a time. With `neuron_config.decode_steps_between_prefill_chunks` = N the scheduler
runs N decode-only steps for the other running requests after each chunk of an ongoing
prefill (`NeuronScheduler.schedule`, step 3.0), instead of holding them back until the last
chunk. Nothing in the runner assumes a prefill's chunks are consecutive: requests absent from a
step leave the persistent batch and are re-added, and each keeps its dedicated state-pool slot.

## Recycled blocks

Block ids are global across groups, so a KV block can previously have held float32 state.
Read as bf16, those bytes include NaN and Inf, and the decode attention kernel reads whole
blocks (a masked `0 * NaN` is still NaN). The runner clears attention blocks when the
scheduler allocates them: vLLM lists them for models with Mamba layers
(`SchedulerOutput.new_block_ids_to_zero`, which leaves out prefix-cache hits and blocks an
async KV load will write), and each step that has any runs one small compiled graph that
zeroes them for every attention layer in place (`_setup_kv_block_clearing`). The
model then skips its own clearing (`set_runner_clears_kv_blocks`); without the runner hook it
clears blocks on their first write, at prefill and at block boundaries during decode. Writing
the cache ahead of `attention_decode` inside the decode graph makes the kernel input a copy of
the whole cache, which is why clearing moved out of the graph.

Unused block-table entries are passed to the decode kernel as the null block rather than
`-1`: the kernel skips out-of-bounds reads but keeps the previous SBUF contents.

## Compiler constraints

The model avoids patterns that lower incorrectly or slowly on Neuron:

| Pattern | Replacement |
|---|---|
| `torch.split` with a size list | `torch.tensor_split` or slicing |
| Column slices of a fused projection at unaligned offsets | separate projections |
| Grouped `conv1d` | sum of K shifted multiplies |
| `index_copy_` through an in-graph `.view()` | views created at bind time |
| A page moved as one `[B, half]` row | `[B, rows, 128]` gathers and scatters |
| `solve_triangular` | matmul-only unit-lower inverse |
| Writing the KV cache ahead of `attention_decode` | clear only when required |
| Gathering the whole block table for prefill attention | segmented attention kernel over the cached prefix |

## Static FP8

With `neuron_config.quantization="fp8"`, weights are quantized per tensor at load
(`scale = amax(W) / 240` over the full tensor, so every rank uses the same scale) and
activations use static scales from an offline calibration file
(`neuron_config.fp8_activation_scales_path`).

| Module | Prefill | Decode (<= 32 tokens) |
|---|---|---|
| MLP | `NF.mlp` STATIC, post-attention RMSNorm fused (as in the Llama static-FP8 path) | `fp8_kernels.fp8_mlp_decode` |
| GDN `in_proj_qkv`, `in_proj_z` | `NF.qkv_proj` STATIC as a generic projection (one scale for all parts) | `gdn_kernels.gdn_decode_fp8` |
| GDN `out_proj` | `NF.o_proj` STATIC | `gdn_kernels.gdn_decode_fp8` |
| Attention q / k / v | `NF.qkv_proj` STATIC; one scale per part | `NF.attention_decode` STATIC QKV |
| Attention `o_proj` | `NF.o_proj` STATIC on `[B, Nh*Dh/128, 128, S]` (head_dim 256 exceeds the kernel's 128) | `fp8_kernels.fp8_matvec` |

The decode kernels keep the activation as the stationary matmul operand (one column per token,
zero-padded to 32) and stream the weights as the moving operand in FP8 double-row mode, which
consumes 256 contraction rows per Tensor Engine cycle; for a single token that rate, about
300 GB/s of FP8 weight per physical core, is the bound rather than HBM. Weight DMAs cover
whole rows of each PSUM pass and are chunked along the contraction so matmuls start early;
weights a later stage needs (GDN `in_proj_z`, `out_proj`) are loaded while the recurrence
runs.

The attention output gate stays BF16. `NF.mlp` falls back to PyTorch, without quantization,
where its kernel does not apply (CTE with intermediate size per rank above 4096 and hidden
size below 7168). The FP8 MLP then splits the intermediate dimension into two kernel calls
(the MLP is separable along it) and uses BF16 on dequantized weights only if that does not
fit either.
