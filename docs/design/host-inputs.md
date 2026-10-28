# Host Inputs: Per-Token Tensors Built on the Host

<!-- meta: description: The HasHostInputs model capability. The V1 and V2 runners build extra
per-token tensors on the host every step and trace every compiled graph with them. -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-28 -->

## Overview

Some models need a per-token input that is cheap to compute on the host and expensive or
impossible on the device. The motivating case is Qwen3.8-Flash-Next: its hashed n-gram embedding
(51B parameters, 102 GB in BF16) stays in host RAM. Each step, the host hashes each token's
n-gram history in int64 and gathers the rows. The graph receives a `[T, 2560]` BF16 tensor aligned
with the padded `input_ids` rows.

The forward runs as static-shape compiled graphs. So the extra kwarg must appear, with the same
shape and dtype, in every graph the runner traces or dispatches. Otherwise the graph mismatches or
recompiles.

## The capability

`vllm_neuron.model.interfaces.HasHostInputs` is a runtime-checkable protocol with two instance
methods:

```python
def host_input_specs(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]
    # kwarg name -> (per-token trailing shape, dtype), e.g. {"ple_embeds": ((2560,), torch.bfloat16)}

def build_host_inputs(self, input_ids: np.ndarray, req_rows: np.ndarray, positions: np.ndarray,
                      token_ids_cpu: np.ndarray) -> dict[str, torch.Tensor]
    # returns CPU tensors [T, *shape], one per spec name
```

| Argument | Shape / dtype | Meaning |
|---|---|---|
| `input_ids` | `[T]` int64 | Padded batch rows, exactly as the graph sees them |
| `req_rows` | `[T]` int64 | `input_batch` row of each token; `-1` on padding rows |
| `positions` | `[T]` int64 | Absolute position of each token in its request; `0` on padding rows |
| `token_ids_cpu` | `[max_num_reqs, max_model_len]` | `input_batch.token_ids_cpu` |

A request's rows are contiguous and in position order. Some rows hold tokens that are not yet
committed to `token_ids_cpu`: spec-decode drafts and jump-forward tokens. In synchronous
spec decode, vLLM writes the drafts into `token_ids_cpu` past `num_tokens_no_spec`. They are
placeholders, though, and get trimmed after rejection. So a model must resolve a token's
predecessors from earlier rows of the same request first, and use `token_ids_cpu` only for
positions before the first row of that request in this step.

Two different checks read the capability:

- The platform checks the class registered in `registry.get_models()` with `issubclass`, before
  the model is built. That class must declare `HasHostInputs`. A factory can inherit it as
  stubs.
- The runner checks the module that `from_configs` returns, with `isinstance`, and calls its
  methods. It looks through `torch.compile`'s `_orig_mod`. The `@async_speculative_decoding`
  decorator patches `forward` on the class, so the module needs no other unwrapping.

## Runner integration (V1)

All paths are gated on `NeuronModelRunner._host_input_specs()`. It resolves once, after the
model loads, and returns None without the capability. A model without it gets no extra
kwargs.

| Path | Where | What |
|---|---|---|
| Regular step (prefill, decode, spec verify, mixed DI batch) | `_prepare_model_input_impl`, after `_pad_to_compiled_shapes` | `host_input_req_rows(actual_num_tokens, padding_map, T)`: per request, its real rows, then its prefill padding, then trailing decode/DP padding. Positions are the padded `cache_positions`. |
| Jump-forward step | `_prepare_jf_model_input` | `req_rows = where(real_grid, request, -1)` over the `[B, 1 + jf_max]` grid |
| Dispatch | `_execute_model_forward` | Moves the CPU tensors to the device and adds them to `model_kwargs`. It raises if the model has the capability and no input was built (an uncovered input path). Under async scheduling it builds them here instead, at the host-token join (below). |
| Prefill warmup / graph capture / parallel trace | `_build_prefill_synthetic_inputs` | zeros `[bucket, *shape]` on the requested device (meta for the trace) |
| Decode warmup / capture / trace, spec verify and non-spec, DP dummy batch | `_build_decode_synthetic_inputs` | zeros `[B * tokens_per_req, *shape]` |
| JF dummy batch | `_build_jf_dummy_inputs` | zeros `[B * (1 + jf_max), *shape]` |
| JF warmup | inline `jf_kwargs` in `warmup_decode` | zeros `[B * (1 + jf_max), *shape]` |

`build_step_host_inputs` checks the returned names, shapes and dtypes against the specs. A wrong
result fails on the host, so it never reaches the graph guards.

The pure pieces are module-level functions in `neuron_model_runner.py`:
`host_input_req_rows`, `host_input_zero_kwargs` and `build_step_host_inputs`. They have unit
tests in `test/unit/worker/test_host_inputs.py`.

## Async scheduling: the host-token join

Only the host inputs and the decode `input_ids` depend on the tokens that the previous step
samples. Everything else (attention metadata, sampling params, padding, slot mappings) depends
only on the scheduler output. So under async scheduling the V1 runner prepares step N+1 while
step N runs on the device, as for any model, and joins on step N's tokens only right before the
forward (`NeuronModelRunner._host_input_join`):

1. `execute_model` runs `_update_states` and `_prepare_model_input` as usual. The decode rows of
   the CPU-built `input_ids` may be stale. `_set_step_host_inputs` only records the padded
   `(req_rows, positions)` layout.
2. The swap (`_maybe_swap_async_input_ids`) and the in-graph reorder are off for these models:
   the host needs the true tokens anyway.
3. `_execute_model_forward` uploads every other input, then joins:
   - It materializes every queued async output, oldest first (`_drain_host_join_outputs`). This
     waits for step N on the device and copies its sampled tokens back.
   - It sets each real row of `input_ids` to `token_ids_cpu[req_row, position]`.
   - It calls `build_host_inputs` on those true tokens, then uploads `input_ids` and the host
     inputs and dispatches the graph.

The device bubble between two steps is then the token copy, `build_host_inputs` and the
upload, plus the dispatch. The scheduler, `_update_states` and the input preparation overlap
the previous forward.

### Why every rank materializes every step

Upstream materializes an async output only on the output rank, on its async-output thread
(`multiproc_executor.py`: `if output_rank is None or self.rank == output_rank`). The other TP
ranks never write the sampled tokens into their `InputBatch`, so their `token_ids_cpu` history
and their write cursor (`num_tokens_no_spec`) lag. Each rank builds the host inputs itself, from
its own `token_ids_cpu`, so with host inputs every rank queues each step's
`AsyncNeuronModelRunnerOutput` (`_track_host_join_output`) and applies it on its main thread at
the next join, in step order. `get_output` is idempotent under a per-object lock, so the
output rank's async-output thread and the join never apply a step twice. The write cursors
then stay exact, and the join checks it: a real row whose position is not below
`num_tokens_no_spec` raises instead of reading a stale token.

### Batch composition changes

The queued outputs are applied by request id into the current InputBatch row, after
`_update_states` has condensed the batch, so new and finished requests need nothing.
`_host_join_needs_early_drain` drains the queue before `_update_states` (which then blocks
before the step's preparation) only when the step:

- resumes a request, since it is re-seeded from its `output_token_ids`;
- takes a live request out of the batch (unscheduled), since that request would lose its
  pending token;
- reuses a request id (abort and resubmit), since the new request would receive the old one's
  token.

A zero-token step that keeps the batch needs no drain. Preempted requests resume through
recompute: the scheduler sends their full token ids, and the prefill rows carry them.

### When async scheduling stays off

`NeuronPlatform._configure_host_input_models` turns async scheduling off for a registered class
that implements `HasHostInputs` in two cases:

- speculative decoding is configured on V1: V1's join covers non-speculative steps only, and
  V1's async spec decode builds its drafts and positions on the device (V2's join covers
  verify steps, see below);
- `VLLM_NEURON_HOST_INPUTS_ASYNC=0`.

As a backstop the V1 runner raises in `_host_input_specs()` for async scheduling with
speculative decoding. With `async_scheduling=False` (explicit, or turned off by the hook) the
synchronous path builds the host inputs in `_prepare_model_input`.

## Model Runner V2

V2 (`neuron_model_runner_v2.py`) builds `input_ids` and positions inside the graph from
device state (`NeuronForwardAdapter`), so the host never sees the tokens of a decode or verify
step. To build host inputs it keeps its own copy of the history:

- **Host token history.** `_host_token_ids [max_num_reqs, max_model_len]` with
  `_host_num_computed` and `_host_total_len` per stable slot, seeded at admission from
  `prefill_token_ids` (the device's token store gets the same row) and advanced by
  `_apply_host_tokens` from each step's committed tokens, exactly as the graph commits them
  (`_compute_next_device_state(_multi)`: a prefill chunk advances by its length and appends
  the sampled token on the final chunk; a decode or verify row advances by its accepted
  tokens and appends them; the history stops at `max_model_len`).
- **Join.** `execute_model` prepares the step as usual, then `_host_input_join` materializes
  every queued async output, oldest first, on every rank (upstream does so only on the output
  rank), and applies its tokens. Steps that free or admit slots (finished, preempted or new
  requests) join before `update_states`, since the history is kept by slot. With sync
  scheduling `sample_tokens` applies the tokens directly.
- **Layout.** `_with_host_inputs` rebuilds the graph's token layout on the host
  (`host_inputs.build_v2_host_input_layout`): the prefill chunk from the history; a decode
  row's token at its computed count; a `[B, 1 + k]` verify row's first token from the
  history and the rest from the slot's drafts, read back from the device draft table (the
  join completed the graph that wrote it). Positions come from the host computed counts,
  which are exact after the join, while the scheduler's are an upper bound under async
  speculative decoding.
- **Operand.** The result goes to the model through the adapter's `host_inputs` operand.
  Every synthetic input (warmup, capture, trace jobs, DP dummies) carries zeros of the graph's
  width (`_assemble_model_input`).

So V2 keeps async scheduling also with MTP. Preempted requests are readmitted with their whole
history and prefix-cache hits with their whole prompt, which the host copy takes as is. Tests:
`test/unit/model/qwen3_5/test_v2_host_inputs.py` (echo host inputs checked against the graph's
input ids and positions and the host copy against the device state, sync and async, chunked
prefills, MTP, preemption) and `test/unit/model/qwen4_exp/test_v2_runner.py` (Qwen3.8-Flash-Next
against HF, also with prefix caching).

## Not covered

- Draft models: the drafter (EAGLE/MTP/DFlash proposers) builds its own forward kwargs, and those
  kwargs do not include host inputs. The Qwen3.8-Flash-Next MTP layer has no n-gram embedding,
  so it does not need them.
- Host-input cost is on the step's critical path in every mode and on both runners: under async
  scheduling it is the device bubble. Make `build_host_inputs` fast, for example with vectorized
  hashing and a multi-threaded gather.
- Async speculative decoding (MTP) with host inputs on V1: the join would have to resolve the
  accepted drafts on the host before building the inputs of the verify rows (V2 does).
