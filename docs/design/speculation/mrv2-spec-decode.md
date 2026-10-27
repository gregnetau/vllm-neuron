# Speculative Decoding on Model Runner V2

<!-- meta: description: Design of speculative decoding (MTP first, EAGLE3 next) on the Neuron
Model Runner V2: fixed-layout verify graph, in-graph rejection and multi-token commit, and a
device-resident per-slot accepted count. -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-27 -->

## Overview

Model Runner V2 (`vllm_neuron/vllm/worker/neuron_model_runner_v2.py`, "V2") keeps every piece
of request progress that a step changes on the device, indexed by a stable request slot, and
commits it inside the compiled graph (`NeuronForwardAdapter`, `neuron_forward_adapter.py`).
Speculative decoding fits that model directly: a verify step changes a request's progress by
the number of tokens the rejection sampler accepts, which only the device knows, so the
device commits it, and every later input that depends on it (positions, slot mappings, the
recurrent-state block to resume from, the next input tokens) is derived from the committed
state inside the next graph.

This removes the host-side machinery that V1 (`neuron_model_runner.py`) needs for
asynchronous speculative decoding: `_get_prev_spec_tensors_for_on_device_correction`,
`_remap_prev_sampled_by_req_id`, `_record_last_spec_rows` / `_fill_from_last_spec_rows`,
`_dummy_prev_sampled`, the spec / non-spec transition cache
(`_async_spec_transition_full_draft_cache`) and the host `_mamba_state_in` map. Each of the
recent V1 fixes covers a case that this design handles by construction:

| V1 fix | Failure | V2 design |
|---|---|---|
| 1327be7 | A held-back request's positions were corrected twice | Positions come from the device count, which is never corrected; nothing is corrected |
| f83411a | A request was corrected by the wrong step's drafts | Each slot's accepted count is committed by the step that verified it and read by the slot's next step, whichever step that is |
| d9c33bd | A request absent from the previous step resumed from the wrong state block | `state_in` is read from the slot's own committed accepted count |
| 932f443 | A step with mixed draft counts misaligned the `[B, 1 + k]` verify layout | The layout is always `[B, 1 + k]`; a row with fewer drafts is padded inside its own row |
| d88c405 | The rejection sampler dropped the bonus token of a row with fewer drafts | The fixed-layout sampler masks a row's positions past its draft count as accepted |

The design follows vLLM's GPU MRv2 (`vllm/v1/worker/gpu/`): positions come from the
device `num_computed_tokens` (`prepare_pos_seq_lens`), draft token values come from a
device per-request table (`req_states.draft_tokens`, read by
`combine_sampled_and_draft_tokens`) while the scheduler supplies only draft counts, and a
hybrid model resumes from the state block given by a per-slot accepted count
(`model_states/mamba_hybrid.py`, `num_accepted_tokens_gpu`).

Scope: MTP (`speculative_config.method == "mtp"`, Qwen3.5 / Qwen3.8 dense and the sparse
MoE Qwen3.5-35B-A3B, the `Qwen3_5MTP` / `Qwen3_5MoeMTP` head) first. The draft-model contract is the one EAGLE3 and the MTP proposer of
upstream PR #78 share (`forward(...) -> (tokens [B, 1 + k], drafts [B, k], logits)`), so
EAGLE3 is a target-output parser and a hidden-state width away (see [EAGLE3](#eagle3)).

## Configuration and policy gating

Speculative decoding stays on V1 unless both hold:

- `VLLM_USE_V2_MODEL_RUNNER=1`, and
- the method and the target architecture are in `V2_SPEC_DECODE_METHODS`
  (`vllm/worker/v2_spec_decode.py`; initially `mtp` for `Qwen3_5ForCausalLM` /
  `Qwen3_5ForConditionalGeneration`, which also covers Qwen3.8 dense, and the MoE
  `Qwen3_5MoeForCausalLM` / `Qwen3_5MoeForConditionalGeneration`, the same Neuron class
  with the sparse MoE FFN; see [Sparse MoE targets](#sparse-moe-targets)).

`get_neuron_model_runner_v2_policy` keeps `default_enabled=False` for every speculative
configuration (V1 stays the default until device parity runs pass), and
`_spec_decode_unsupported` records as unsupported (so a forced V2 fails validation) any
speculative configuration with: a method or architecture outside the allowlist, KV transfer
(DI; the bootstrap slot reads its first token from history, and the last-token recompute of
a transferred prompt is scheduled with spec width), sliding-window groups (trimmed tables are
cut from the host count, which is an upper bound), expert parallelism with data parallelism
(the spec / non-spec NEFF vote is not synchronized), debug logits, and a
`rejection_sample_method` other than `standard` / `synthetic`. Structured outputs, tensor
capture and decode context-length buckets are unsupported on V2 for every model. Prefix
caching of recurrent-state models works with MTP on V2 (V1 does not support MTP with prefix
caching): a verify step's state after each token is in its own pool page, and the checkpoint
for a block boundary it reaches is copied from the page of the token on the boundary, using
the accepted count committed on the device (see
[Prefix caching on Model Runner V2](../hybrid-state-gdn.md#prefix-caching-on-model-runner-v2)).

The runner repeats the checks in `NeuronModelRunnerV2._validate_spec_decode_config`, and
requires every prefill bucket to exceed `1 + k` (models tell prefill from decode by
`max_query_len > decode_token_threshold = 1 + k`).

## Graphs

With speculative decoding V2 compiles one graph family per phase:

| Phase | Tokens | Rows sampled | Drafter |
|---|---|---|---|
| Prefill (per prefill bucket `P`, segment size) | `P` (one request) | 1 | runs over the chunk, writes `k` drafts for the slot |
| Verify (per decode bucket `B`) | `B * (1 + k)` | `B * (1 + k)` | runs over the verify tokens, writes `k` drafts per slot |

There is **no single-token decode graph** when speculative decoding is on. A decode step
always uses the verify graph; a row whose request has fewer than `k` scheduled drafts (near
`max_model_len`, or after the scheduler stopped drafting for every request) uses
`1 + n_r` of its `1 + k` positions and the rest are padding. The cost is `1 + k` tokens of
work for the few steps where a request cannot draft, against a whole graph family less to
compile and no spec / non-spec transitions (V1's transition cache and its
`_transition_bonus_tensor` exist only because a step can switch graph family). Recurrent-state
models benefit twice: every decode step reads `state_in`, so the state after a verify step
(left in block `num_accepted - 1`) is always found; a plain decode graph would read block 0.

`k` is fixed (`num_speculative_tokens`); dynamic speculation lengths are not supported (the
scheduler's `dynamic_sd_lookup` would change `num_spec_tokens_to_schedule`; the runner rejects
a row with more than `k` drafts).

### Verify layout

Row `r` of a verify step holds request `r` (scheduler order) at stable slot `s_r` and its
`1 + n_r` real tokens `[x_r, d_r1, ..., d_rn]` in columns `0 .. n_r`, padded to `1 + k`:

```text
            col 0        col 1 .. n_r      col n_r+1 .. k
row r   [ next_token[s_r] | draft[s_r, :n_r] | PAD ... ]
row pad [ PAD ...                                     ]   (rows num_active .. B-1)
```

- `next_token[s]` is the non-aliased next-token carry (`_next_token_by_slot`), the last token
  the slot's previous step committed. `draft[s, j]` is the device draft table, written by the
  slot's previous step's drafter. The scheduler's draft token ids are placeholders (`-1`)
  and only their count `n_r` is used (`num_draft_by_row`, a `[B]` int32 host operand; 0 for
  padded rows).
- Positions are `num_computed_tokens[s_r] + j`, read from the device count in the graph,
  clamped to `max_model_len - 1`. The host never computes verify positions; its
  `scheduler_num_computed_tokens` is an upper bound under async scheduling (see
  [Async scheduling](#async-scheduling)).
- Slot mappings of the attention groups are computed in the graph from those positions and
  the group's kernel block rows (`block_table[r, pos // block_size] * block_size +
  pos % block_size`). Padding positions and padded rows get slot 0 of the null block, V2's
  write sink (`NULL_BLOCK_ID`). Host block rows are built from the scheduler's allocation,
  which covers the optimistic count plus the lookahead (`vllm_config.num_lookahead_tokens`),
  so every real position is inside a valid block; a `-1` entry also maps to the null slot.
- `max_query_len = decode_token_threshold = 1 + k` in every group's metadata, so models treat
  the step as decode with `1 + k` tokens per request (V1 uses the same threshold).
- `sampling_positions = arange(B * (1 + k))`: every position is sampled.

## Adapter ABI

`NeuronForwardAdapter` (`neuron_forward_adapter.py`) gains an optional speculator submodule
and three operands. Without speculative decoding none of them is passed and every existing
graph signature is unchanged.

| Operand | Shape | Kind | Meaning |
|---|---|---|---|
| `draft_tokens` | `[max_num_reqs, k]` int32 | device state, aliased bundle member | drafts proposed for each slot by its last step |
| `num_accepted` | `[max_num_reqs]` int32 | device state, aliased bundle member | tokens committed by each slot's last step (1 after a prefill) |
| `num_draft_by_row` | `[B]` int32 | host operand, verify only | scheduled draft count per row |

### Prologue (verify)

1. `input_ids [B, 1 + k]` from `next_token_by_slot` and `draft_tokens` gathered by
   `row_to_slot` (padded rows clamp to slot 0 and are masked to `PAD_TOKEN`).
2. `positions [B * (1 + k)]` from `num_computed_tokens[row_to_slot]`.
3. Slot mappings of non-state groups from positions and block rows; for recurrent-state
   groups `state_in = clamp(num_accepted[row_to_slot] - 1, 0)` is added to the metadata
   (the key that selects `Qwen3_5GatedDeltaNet._decode_spec` / the fused kernel's
   `slots_in`).
4. Sampling parameters gathered per token (`row_to_slot` expanded over the row).

### Target, rejection, drafter, commit

1. The unchanged model runs by keyword. With `set_mtp_hidden_output(True)` it returns
   `(sampled [B * (1 + k)], hidden [T, H], gathered_logits)`.
2. **Rejection sampling in the graph**: `nn.rejection_sampler.rejection_sample_fixed(targets
   [B, 1 + k], drafts [B, k], num_drafts [B])` returns `[B, 1 + k]` with `-1` past the
   accepted prefix (the same acceptance rule as V1's `rejection_sampler`, sharing its
   `synthetic` mode; positions past a row's own drafts count as accepted so the bonus token
   is kept, d88c405). The on-device sampler draws each target token from the target
   distribution, so accept-if-equal keeps the output distribution exact at any temperature,
   as in V1. `valid_count [B]` (1 .. 1 + n_r) is the accepted count.
3. **Drafter in the same graph**: the draft model's contract is the EAGLE3 / MTP fused draft
   forward (`input_ids, positions, initial_target_hidden_states, attn_metadata,
   sampling_positions, rank, raw_sampled_token_ids`). Inputs: the verify tokens shifted left
   by one inside each row; positions and slot mappings from the prologue; the target hidden
   states; `sampling_positions = r * (1 + k) + k`; `raw_sampled_token_ids` = the rejection
   output. The draft model's `extract_accepted_tokens` moves each row's sampling position back
   to `r * (1 + k) + valid_count - 1` (the last accepted token, independent of `n_r`) and
   patches the next token there, then runs its `k - 1` recurrent steps. It returns
   `drafts [B, k]`. On prefill the shifted ids are gathered from the token store at
   `position + 1` (the true next prompt token for a non-final chunk, fixing V1's use of the
   discarded sample there) and the last position is patched with the sample on the final chunk.
4. **Multi-token commit** (`_compute_next_device_state`): for each active slot, with `W = 1 + k`
   columns and `c = valid_count` (prefill and single-token rows: `W = c = 1`):
   - `num_computed_tokens += c` on verify rows (`query_len` on prefill rows, as today).
   - The `c` accepted tokens are appended to `token_ids` at `total_len .. total_len + c - 1`
     (one guarded `index_put_` per column, capacity-checked against `max_model_len`), and
     `total_len` grows by the number stored.
   - `last_sampled_tokens` and the next-token carry take the last accepted token.
   - `draft_tokens[s] = drafts[row]`, `num_accepted[s] = c` (1 on prefill rows).
   The bundle returned to `apply_committed_state` grows from four to six tensors.
5. Outputs: `sampled [B, 1 + k]` with `-1` padding (host trims each row to its valid prefix),
   the bundle, the next-token carry, and gathered logits when enabled.

### Why the drafter runs in the target graph

V1 compiles the draft model as its own NEFF and passes the target hidden states through HBM
as a graph output. In V2 the drafter is a submodule of the adapter, so:

- the draft table is device state committed by the same graph that verified the drafts (no
  cross-graph future, no host stash like V1's `_futures_drafts_only`);
- the target hidden states are not a graph output, which avoids the class of failures where a
  tensor that is both a graph output and an `index_select` / lm_head input lowered wrongly
  (NaN rows in batch > 1 verify steps; see the Qwen3.5 decode note in
  [Hybrid recurrent state](../hybrid-state-gdn.md) and upstream PR #78);
- one NEFF dispatch per step instead of two.

The cost is a larger graph per bucket and one combined compile. A separate drafter NEFF
remains possible behind the same speculator interface (the speculator would become its own
compiled module fed from a hidden-state output), if a draft model ever needs compiler options
the target cannot share.

## Device-resident accepted count

`NeuronRequestState` (`neuron_request_state.py`) owns two new slot-indexed device tensors,
`draft_tokens [max_num_reqs, k]` and `num_accepted [max_num_reqs]`. Admission resets the
slot's row (`draft_tokens = 0`, `num_accepted = 1`); the prefill graph overwrites both. A
request keeps its slot while it is held back during another request's prefill
(`decode_steps_between_prefill_chunks` interleave, or plain scheduling), so its next step reads
its own last commit regardless of how many steps it sat out. Slot recycle is guarded by V2's
quarantine ledger, unchanged.

Everything the next step needs is derived from the device state in the graph:

| Input | V1 (async) | V2 |
|---|---|---|
| Positions | host optimistic positions corrected in the graph by the previous step's rejections (`correct_spec_decode_positions_and_slot_mapping`) | `num_computed_tokens[slot] + j` |
| Slot mappings | recomputed from corrected positions | computed from positions |
| `state_in` | `valid_count - 1` of the previous step's rejection row, remapped on the host by request id, or the host map in sync mode | `num_accepted[slot] - 1` |
| Input tokens | previous step's `[B, 1 + k]` future (bonus first), remapped on composition change | `next_token[slot]`, `draft_tokens[slot]` |

## Hybrid recurrent state (Qwen3.5 / Qwen3.8)

- Specs: `hybrid_state.build_state_cache_specs(..., num_spec_blocks=k)` gives every state
  `MambaSpec` `num_speculative_blocks = k`; vLLM then allocates `1 + k` state blocks per
  request and group (`MambaSpec.max_num_blocks_per_req`).
- Pool: `state_pool_pages = 1 + max_num_reqs * (1 + k)`; slot `s` owns pages
  `1 + s * (1 + k) + j` (`build_state_pool_rows` with `pages_per_slot = 1 + k`). State
  groups' block rows are widened to `1 + k` columns (the attention-sized width of a state
  group with `block_size >= max_model_len` is 1).
- Verify: `block_table_tensor [B, 1 + k]` (padded rows `-1`), `state_in [B]` from
  `num_accepted`. The GDN decode reads block `state_in` and writes the state after token `j`
  to block `j` (`_decode_spec`, `gdn_decode_fp8` with `slots_in` / `slots_out`). Padding
  positions write their own slot's blocks past the accepted ones, which no later step reads.
- Prefill reads and writes block 0 and the commit sets `num_accepted = 1`, so the first
  verify step resumes from block 0. A preempted request is re-admitted with a full prefill
  (V2 scheduler contract), which also starts from block 0.
- The approach rejected in V1 (copying the accepted state to block 0 inside the verify graph)
  is not needed: the next step reads the accepted block in place.

## Async scheduling

With async scheduling (batch queue depth 2, `NeuronAsyncScheduler`) step `N + 1` is scheduled
before step `N`'s output reaches the scheduler. `AsyncScheduler._update_after_schedule`
assumes every draft of step `N` is accepted: it advances `num_computed_tokens` by `1 + n`,
adds `-1` placeholders as the next drafts, and allocates blocks for the optimistic length plus
the lookahead. When step `N`'s output arrives, `Scheduler.update_from_output` rolls
`num_computed_tokens` and `num_output_placeholders` back by the rejected count
(`len(scheduled_spec_decode_tokens) - (len(generated) - 1)`).

V2 consumes only the parts of a scheduler output that the rollback cannot invalidate:

- the set of requests, their row order and slots, and their draft counts `n_r` (graph shape
  and padding);
- block ids (an upper bound covers every real position);
- prefill windows (prefills are never speculative, so the host count is exact there).

The device count is exact at all times because each graph commits its own accepted count
before the next graph reads it (NRT executes submissions in order, and the bundle tensors chain
the graphs). So step `N + 1` uses step `N`'s true progress without a device-to-host read, no
matter whether the scheduler has processed step `N`'s output yet.

`take_draft_token_ids` returns `-1` placeholders for the last step's requests (GPU MRv2's
`DraftTokensHandler` without structured outputs): with sync scheduling the engine feeds them
to `update_draft_token_ids` so the next step schedules `k` drafts; under async scheduling the
engine does not call it. Draft values never leave the device.

The new-request join (`_drain_async_output_for_new_request_join`) and the quarantine ledger are
unchanged: a verify graph touches exactly the slots of its rows.

## Near max_model_len

- The scheduler limits a step to `max_model_len - 1 - num_computed` tokens, so a request near
  the end gets `n_r < k` drafts (sync) or none (async: `NeuronAsyncScheduler` stops drafting
  for every request while any request is near the end, 932f443). Both are rows with fewer
  drafts in the fixed layout.
- The scheduler-side rule of 932f443 does not exclude every mixed step: once the request near
  the end has finished, a request whose final prefill chunk ran while the others were held back
  gets placeholders, while the held-back requests keep the empty draft lists of their last
  step, so the next step mixes rows with `k` and with 0 drafts (seen in the CPU test with
  `NeuronAsyncScheduler`, `decode_steps_between_prefill_chunks=0`). V1's packed layout would
  misalign there; the fixed layout does not depend on the rule.
- Padding positions are clamped to `max_model_len - 1` and write the null slot; the drafter's
  recurrent positions are clamped the same way (`mtp._slot_mapping` maps `-1` table entries to
  slot 0).
- The commit appends at most `max_model_len - total_len` tokens.
- `max_tokens`: the scheduler truncates the output list at the stop condition; extra accepted
  tokens on the device are never read (the request finishes and its slot is freed).

## Output, logprobs, metrics

- `sampled [B, 1 + k]` is materialized once (sync: in `sample_tokens`; async: in
  `_AsyncNeuronModelRunnerOutput.get_output`) and trimmed per row to its valid prefix.
  Prefill rows keep the `[token]` / `[]` rule.
- Logprobs: the gathered logits are `[B * (1 + k), V]`; request `r`'s `c_r` tokens use rows
  `r * (1 + k) + j`, `j < c_r` (the token at column `j` is sampled from the target logits at
  row `j`). The runner returns one `LogprobsLists` row per generated token with
  `cu_num_generated_tokens`, the layout vLLM slices with `slice_request`.
- Acceptance metrics come from the scheduler (`SpecDecodingStats` from the output lengths).
  The runner logs the NEFF bucket as `decode_b{B}_s{1 + k}`.

## Decode context-length buckets

With `decode_context_length_buckets` (models with recurrent state only; the policy keeps
other models on V1 for it), every decode step trims the attention groups' kernel block
tables to the smallest bucket `S` of `decode_context_length_buckets + [max_model_len]` with
`S >= max(num_computed) + headroom`, `headroom = max(1, 2k)`
(`neuron_input_build.select_decode_ctx_bucket`): the verify step writes and attends
`1 + k` positions from the device computed count, and the MTP head's `k - 1` recurrent draft
steps go one position past the last accepted token (at most `num_computed + k`) each, so the
step touches positions up to `num_computed + 2k - 1`. The count is the scheduler's, an upper
bound of the device count under async scheduling, so the in-graph positions stay inside the
trimmed table by construction.

- Trim: the first `ceil(S / block_size)` columns of each row (`build_kernel_block_rows(...,
  num_columns=...)`, a fresh row-major array); the `max_model_len` bucket keeps the whole
  aligned row (the graph without buckets). Prefill and the recurrent-state groups (`1 + k`
  pool pages) are never trimmed. All attention groups, the MTP head's layer included, get the
  same width, so the head's attention and its draft-step slot mapping (`mtp._slot_mapping`)
  read the trimmed table too.
- Slot mappings: the host slot mapping resolves through the untrimmed rows; the verify
  graph's slot mapping (`_build_verify_inputs`) resolves through the trimmed table. When the
  table is trimmed (`width * block_size < max_model_len`, a trace-time constant), a position
  past it writes the null slot instead of the table's last block, and so does
  `mtp._slot_mapping`; neither happens with a correctly chosen bucket (the CPU tests assert it
  inside the graph).
- Graphs: one verify graph `decode_b{B}_s{1 + k}` per `(num_seqs bucket, ctx bucket)`, the
  compile metric labelled `decode_b{B}_s{1 + k}_ctx{S}`; the runtime NEFF label stays
  `decode_b{B}_s{1 + k}`.
- Not with expert parallelism across DP ranks (the bucket is not synchronized across the
  group).

## Warmup and compile

- Prefill: every `(num_batched_tokens_bucket, kv_segment_size)` as today, with the drafter in
  the graph (synthetic slot 0).
- Decode: every `num_seqs_bucket` `B` (times every decode context bucket, with
  `decode_context_length_buckets`) as a verify graph with `n_r = k` for all rows
  (`_build_decode_synthetic_inputs`), the same assembly path as serving.
- The worker's V1 drafter hooks stay off: V2 keeps `self.drafter = None` (the drafter lives in
  the adapter), so `NeuronWorker` builds no separate draft trace jobs; the draft model's weights
  are counted through `NeuronModelRunnerV2.draft_model`.
- The no-recompile guard applies unchanged; the only per-step shape inputs are the buckets.

## EAGLE3

EAGLE3 reuses everything above except two target-side pieces: the target is configured with
`set_aux_hidden_state_layers` and returns the concatenated auxiliary hidden states instead of
the final hidden state, and the draft model class is `Eagle3<arch>` with
`target_hidden_states` of width `3 * hidden_size`. The speculator class
(`V2DraftSpeculator`) takes a target-output parser per method; the draft models already
implement the fused draft forward. Llama's decode `index_select` of sampling positions must be
checked against the in-graph hidden-state use. Sliding-window targets (GPT-OSS) additionally
need in-graph window trimming, since the host count is an upper bound.

## Files

| File | Change |
|---|---|
| `vllm/model_runner_v2_policy.py` | `_spec_decode_unsupported`: allowlist gate for speculative configurations |
| `vllm/worker/v2_spec_decode.py` | `V2_SPEC_DECODE_METHODS`, `V2DraftSpeculator`, `load_draft_model` |
| `vllm/worker/neuron_model_runner_v2.py` | `_validate_spec_decode_config`, `_load_draft_model`, `_build_speculator`, KV specs incl. draft layers and spec state blocks (`_kv_spec_layers`), verify classification in `_prepare_model_input`, verify assembly in `_assemble_model_input`, `state_in` placeholders and spec operands in `_build_forward_adapter_kwargs`, six-tensor `apply_committed_state`, `_trim_sampled` / `_sampled_rows`, `_build_verify_logprobs`, `take_draft_token_ids` |
| `vllm/worker/neuron_forward_adapter.py` | `_forward_speculative`, `_build_verify_inputs`, `_shifted_prefill_draft_ids`, `_compute_next_device_state_multi` (non-speculative graphs keep `_compute_next_device_state`) |
| `vllm/worker/neuron_request_state.py` | `draft_tokens`, `num_accepted`, `configure_block_sizes(min_widths=...)` |
| `vllm/worker/neuron_input_build.py` | `build_verify_layout`; `NeuronModelInput.num_draft_by_row`, `spec_width` |
| `nn/rejection_sampler.py` | `rejection_sample_fixed` (shares `_accept_prefix` with the packed sampler) |
| `vllm/worker/neuron_worker.py` | count `draft_model` weights in the KV budget |

## Sparse MoE targets

Qwen3.5-35B-A3B (`qwen3_5_moe`) is the Qwen3.5 class with `Qwen3_5SparseMoeBlock`
(`model/qwen3_5/moe.py`) in every decoder layer, the MTP layer's included, so the verify graph,
the state handling and the drafter are the dense model's. What differs is the FFN's dispatch on
the token count of a step, which a verify step raises to `B * (1 + k)`:

| Verify bucket (k = 2) | Tokens | Routed experts |
|---|---|---|
| `B` = 1, 4, 8 | 3, 12, 24 | fused `moe_block_tkg`, selected experts' weights only |
| `B` = 16, 32 | 48, 96 | fused `moe_block_tkg`, every local expert (`T * top_k >= E`) |
| `B` = 64 | 192 | blockwise `moe_cte` (more than `MAX_TKG_TOKENS` = 128) |

The decode path has no token mask: padding tokens of a row and padded rows go through the
router and the experts like real tokens, and their outputs are discarded (padded rows are also
zeroed after every layer). That is exact for the real tokens because every token selects
exactly `top_k` experts, also with a non-finite hidden state (the router scatters its top-k
values), so the blockwise mapping's static block count (`T * top_k` slots) never overflows, and
both kernels compute each token's output from its own row only. Prefill masks padding (slot
`<= 0`) out of the expert blocks as on V1. The MTP head's experts load from per-expert
checkpoint tensors into the same device layout, sharded the same way (TP on the intermediate
dimension, or `--enable-expert-parallel`), and the head runs the same dispatch on its own
token counts (`B * (1 + k)`, then `B` per recurrent draft step).

## Tests (CPU)

- `test/unit/nn/test_rejection_sampler.py`: the fixed layout against the packed sampler,
  mixed draft counts, synthetic rates.
- `test/unit/worker/test_v2_spec_adapter.py`: the adapter with a toy target and draft:
  per-row acceptance and commit, inputs from device state with an optimistic host count,
  `state_in`, a held-back slot, capacity near `max_model_len`, prefill drafting.
- `test/unit/model/qwen3_5/test_v2_spec_decode_e2e.py`: the real Neuron schedulers (sync,
  and async with the engine core's batch queue of depth 2), the V2 runner and the Qwen3.5
  module (NKI stubbed) against the HF model: requests arriving together, a chunked prefill
  arriving while others decode (held back or interleaved), partial acceptance (oracle
  draft), trimmed drafts and drafting stopped near `max_model_len`, all drafts rejected,
  logprobs, the pool pages and `state_in`, and the real MTP head, whose in-graph drafts
  equal a from-scratch run of the head over each committed sequence. Every case runs for
  a dense and a sparse MoE checkpoint (MoE ops on their PyTorch fallbacks). With `SPEC_DYNAMO=1`
  the adapter runs through `torch.compile(backend="eager", fullgraph=True)`; the run must
  compile one graph per bucket and nothing per step (`TORCH_LOGS=recompiles`).
  `test_decode_ctx_buckets_match_hf` adds decode context buckets `[128, 256]` (max_model_len
  320): requests cross both boundaries and reach the `max_model_len` graph mid-generation,
  verify steps straddle each boundary, batches mix short and long rows, with the oracle and
  the real MTP head; the positions of the verify slot mapping and of the head's draft steps
  are checked against the trimmed table inside the graph.
- `test/unit/model/qwen3_5/test_v2_runner_inputs.py::test_v2_decode_ctx_buckets_match_hf_across_bucket_boundaries`:
  the same without speculative decoding, logits against HF at every step.
- `test/unit/worker/test_v2_decode_ctx_buckets.py`: bucket selection, the head trim, the
  verify slot mapping through a trimmed table, and the worker's decode jobs per
  `(B, ctx)`.
- `test/unit/worker/test_v2_policy_recurrent_state.py`: the policy matrix, also with vLLM's
  real `ModelConfig` and MTP `SpeculativeConfig` of the Qwen3.5-35B-A3B checkpoint config.
- `test/unit/model/qwen3_5/test_moe.py::test_moe_verify_bucket_padded_rows_do_not_leak`:
  non-finite padded verify rows through both decode dispatch paths of the MoE block.

## Not done yet

- Device validation (below). Nothing here has run on a Neuron device.
- EAGLE3 (see [EAGLE3](#eagle3)), DFlash, and dense targets other than Qwen3.5.
- KV transfer (DI) with speculative decoding, sliding-window targets, decode context-length
  buckets on V2 for models without recurrent state.
- A separate drafter NEFF (only needed if a draft model must compile with other options).

## Device validation

Run on a trn2 instance with no other process on the devices. V2 needs
`VLLM_USE_V2_MODEL_RUNNER=1`; `DECODE_CTX_BUCKETS` and `PREFIX_CACHING=1` are optional (for
prefix caching see
[its device validation](../hybrid-state-gdn.md#prefix-caching-on-model-runner-v2)):

```bash
# Qwen3.5-2B, TP=2, MTP with 2 draft tokens, 4 sequences, async scheduling (default)
VLLM_USE_V2_MODEL_RUNNER=1 MODEL=<ckpt>/Qwen3.5-2B TP=2 MTP_TOKENS=2 MAX_NUM_SEQS=4 \
    SEQS_BUCKETS="1, 4" examples/vllm_neuron/models/qwen3_5/serve.sh
# Qwen3.8-27B, TP=4 (add FP8_SCALES=... KV_CACHE_DTYPE=fp8 BLOCK_SIZE=128 for FP8)
VLLM_USE_V2_MODEL_RUNNER=1 MODEL=<ckpt>/Qwen3.8-27B TP=4 MTP_TOKENS=2 MAX_NUM_SEQS=4 \
    SEQS_BUCKETS="1, 4" examples/vllm_neuron/models/qwen3_5/serve.sh
# Qwen3.5-35B-A3B (sparse MoE, BF16; about 17.1 GiB of weights per rank with the draft)
VLLM_USE_V2_MODEL_RUNNER=1 MODEL=<ckpt>/Qwen3.5-35B-A3B TP=4 MTP_TOKENS=2 MAX_NUM_SEQS=4 \
    SEQS_BUCKETS="1, 4" GPU_MEM_UTIL=0.9 examples/vllm_neuron/models/qwen3_5/serve.sh
# Repeat each with ASYNC_SCHEDULING=0, and with DECODE_INTERLEAVE=1.
# Qwen3.8-27B FP8, 64k context with a 16k decode context bucket, 8 sequences: verify graphs
# decode_b1_s3 and decode_b8_s3, each at ctx 16384 (128 blocks of 128) and 65536 (512)
VLLM_USE_V2_MODEL_RUNNER=1 MODEL=<ckpt>/Qwen3.8-27B TP=4 FP8_SCALES=<path>/fp8_act_amax.json \
    KV_CACHE_DTYPE=fp8 BLOCK_SIZE=128 MTP_TOKENS=2 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=8 \
    SEQS_BUCKETS="1, 8" DECODE_CTX_BUCKETS="16384" examples/vllm_neuron/models/qwen3_5/serve.sh
```

With `DECODE_CTX_BUCKETS`, also check `V2 decode context-length buckets [16384] +
max_model_len 65536: ...` at startup, the compile metric labels `decode_b{B}_s3_ctx{S}` for
every `(B, S)`, no recompile once serving (the serve guard raises), and identical greedy
output with and without `DECODE_CTX_BUCKETS` for requests that cross 16384 while decoding.

Log lines to check:

- `Using the standalone V2 Neuron model runner.` and
  `V2 speculative decoding: method=mtp k=2 draft=...; verify graphs decode_b{B}_s3 for
  num_seqs_buckets=[1, 4], drafter in the target graph`.
- `Recurrent-state pools: 48 layers in KV groups [...], N pages per layer (3 per slot)`
  (Qwen3.8-27B; the GDN layer count for Qwen3.5-2B), with attention block sizes equal to
  `--block-size`.
- Compile metrics / NEFF buckets `prefill_s1024_kv1024` and `decode_b1_s3`, `decode_b4_s3`
  only (no `decode_b{B}` without `_s3`).
- No `VLLM_USE_V2_MODEL_RUNNER=1 forces Neuron Model Runner V2 with unsupported features`
  warning.
- vLLM's `SpecDecoding metrics: ... Mean acceptance length: ...` comparable to V1 MTP on the
  same prompts.

Checks: greedy (`temperature=0`) completions equal V1 MTP and V2 without
`MTP_TOKENS`, token for token, at concurrency 1 and 4 with prompts of different lengths
arriving together, a long prompt (several chunks) arriving while others decode, and requests
that run to `max_model_len` (for example `MAX_MODEL_LEN=2048` with `max_tokens` past it);
`logprobs=1` returns one entry per generated token. Signs of a layout or state bug are the
ones 932f443 describes: tokens with logprob far below the top token in the last tokens of
requests near the end, or divergence after the first verify step with partial acceptance.
