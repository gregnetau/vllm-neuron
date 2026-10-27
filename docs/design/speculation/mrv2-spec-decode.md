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

Scope: MTP (`speculative_config.method == "mtp"`, Qwen3.5 / Qwen3.8 dense, the
`Qwen3_5MTP` head) first. The draft-model contract is the one EAGLE3 and the MTP proposer of
upstream PR #78 share (`forward(...) -> (tokens [B, 1 + k], drafts [B, k], logits)`), so
EAGLE3 is a target-output parser and a hidden-state width away (see [EAGLE3](#eagle3)).

## Configuration and policy gating

Speculative decoding stays on V1 unless both hold:

- `VLLM_USE_V2_MODEL_RUNNER=1`, and
- the method and the target model class are in `V2_SPEC_DECODE_METHODS` / the method's
  architecture allowlist in `model_runner_v2_policy.py` (initially `mtp` for
  `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration`).

`get_neuron_model_runner_v2_policy` keeps `default_enabled=False` for every speculative
configuration (V1 stays the default until device parity runs pass) and records as unsupported
(so a forced V2 warns, and the runner refuses at construction) any speculative configuration
with: a method or architecture outside the allowlist, KV transfer (DI; the bootstrap slot
reads its first token from history), structured outputs (draft validation needs host drafts),
sliding-window groups (trimmed tables are cut from the host count, which is an upper bound),
expert parallelism with data parallelism (the spec / non-spec NEFF vote is not synchronized),
debug logits and tensor capture, and `rejection_sample_method` other than `standard` /
`synthetic`. Prefix caching remains unsupported for recurrent-state models on V2 (and V1 does
not support MTP with prefix caching either).

The runner repeats the checks in `NeuronModelRunnerV2._validate_spec_decode_config` because
`VLLM_USE_V2_MODEL_RUNNER=1` forces V2 with only a warning.

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

## Warmup and compile

- Prefill: every `(num_batched_tokens_bucket, kv_segment_size)` as today, with the drafter in
  the graph (synthetic slot 0).
- Decode: every `num_seqs_bucket` `B` as a verify graph with `n_r = k` for all rows
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
| `vllm/model_runner_v2_policy.py` | allowlist gate for speculative configurations |
| `vllm/worker/neuron_model_runner_v2.py` | spec config, draft model load, KV specs incl. draft layers and spec state blocks, verify input assembly, bundle, output trimming, logprobs, `take_draft_token_ids` |
| `vllm/worker/neuron_forward_adapter.py` | verify prologue, rejection, drafter call, multi-token commit |
| `vllm/worker/neuron_request_state.py` | `draft_tokens`, `num_accepted`, per-group block-row widths |
| `vllm/worker/neuron_input_build.py` | `build_verify_layout` (host verify operands) |
| `vllm/worker/v2_spec_decode.py` | speculator registry, draft model loading, `V2DraftSpeculator` |
| `nn/rejection_sampler.py` | `rejection_sample_fixed` |
| `vllm/worker/neuron_worker.py` | count `draft_model` weights in the KV budget |

## Device validation

See the runner's log lines `V2 speculative decoding: method=... k=...` (construction),
`Recurrent-state pools: ... (k+1 per slot)` (KV init), the compiled buckets
`decode_b{B}_s{1+k}`, and vLLM's `SpecDecoding metrics` (acceptance). Greedy outputs must
match V1 MTP and non-speculative V2 token for token.
