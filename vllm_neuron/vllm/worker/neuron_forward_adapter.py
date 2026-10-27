# SPDX-License-Identifier: Apache-2.0
"""Adapt standalone Neuron runner inputs to the compiled model forward pass.

Use one runner-owned host-to-device interface to build ``input_ids`` in the
graph, select per-request sampling parameters, call the unchanged model by
keyword, and commit device-authoritative state in the graph. Return sampled
tokens with the aliased state bundle and ordinary next-token carry.

Specialize prefill and decode as separate compiled graphs according to whether
``linear_idx`` is present. Allow index values to change between steps while
keeping their shapes fixed. Compile the adapter on a Neuron device and execute
it eagerly during CPU simulation.

Build one explicit attention-metadata dictionary per KV-cache group and alias
that dictionary across all layer names in the group.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_neuron.functional.combine_sampled_tokens import (
    build_decode_input_ids_from_store,
    scatter_sampled_into_slot_store,
)

# In-vocab pad id for masked-out input rows/tokens. 0 is a valid embedding index (never out of range);
# padded rows are excluded from sampling host-side and from attention by the block-table -1 sentinel, so
# the pad embedding never pollutes a real output. NEVER use a negative id (embedding is not defined there).
PAD_TOKEN = 0
# KV write sink for padded tokens: slot 0 of the reserved null block (NULL_BLOCK_ID 0),
# the runner's host padding value for slot mappings.
NULL_SLOT = 0
# Rejected or padding position of a verify step's rejection output.
PLACEHOLDER = -1


# Forked from the V1 model runner's _reinterpret_uint32_as_int32 helper.
def _reinterpret_uint32_as_int32(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Reinterpret uint32 argmax bits as int32 without a device cast.

    Return other dtype combinations unchanged. Token IDs fit in int32, so
    viewing the same bits is exact and avoids materializing a value cast.

    Args:
        tensor: Tensor containing sampled token IDs.
        target_dtype: Integer dtype required by the token store.

    Returns:
        The bit-reinterpreted tensor, or the original tensor when no
        reinterpretation is required.
    """
    if tensor.dtype == torch.uint32 and target_dtype == torch.int32:
        return tensor.view(torch.int32)
    return tensor


def build_attention_metadata(
    kv_cache_groups,
    block_tables: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    max_query_len: int,
    decode_token_threshold: int = 1,
    cached_seq_len: torch.Tensor | None = None,
    kv_segment_size: int = 0,
    slot_mappings: Sequence[torch.Tensor] | None = None,
    swa_kv_pos_offsets: Sequence[torch.Tensor | None] | None = None,
    always_cached_seq_len: bool = False,
    block_sizes: Sequence[int] | None = None,
) -> dict[str, dict]:
    """Build per-layer attention metadata for dense model execution.

    Create one metadata dictionary per KV-cache group and share it by
    reference across every layer in that group. Accept all values explicitly
    so the same builder serves real execution and synthetic warmup. Consume
    block rows that already contain the ``-1`` kernel sentinel.

    Use ``max_query_len`` and ``decode_token_threshold`` to distinguish
    prefill from decode. Include ``cached_seq_len`` and ``kv_segment_size``
    only for segmented prefill, and ``cached_seq_len`` on every prefill when
    ``always_cached_seq_len`` is set.

    Args:
        kv_cache_groups (Sequence[KVCacheGroupSpec]): Cache groups with layer
            names and block sizes.
        block_tables: Per-group ``[rows, width]`` int32 kernel block tables.
        slot_mapping: Padded ``[tokens]`` int64 slot mapping, used for every
            group when ``slot_mappings`` is not given.
        max_query_len: Padded prefill token count or one for decode.
        decode_token_threshold: Query length above which execution is prefill.
        cached_seq_len: Optional ``[[length]]`` int32 cached-prefix tensor.
        kv_segment_size: Positive segment size for segmented prefill, or zero.
        slot_mappings: Optional per-group slot mappings. Required whenever
            groups own distinct block tables (e.g. a sliding-window group next
            to a full-attention group): KV writes must resolve slots through
            their own group's table.
        swa_kv_pos_offsets: Optional per-group ``[rows]`` int32 KV position
            offsets. A non-None entry marks a sliding-window group whose block
            table is window-trimmed; the model subtracts the offset from
            position IDs so the causal mask is computed in the trimmed frame.
        always_cached_seq_len: Include ``cached_seq_len`` even without a KV
            segment size. Recurrent-state models read it on every prefill
            (zero means "start from zero state").
        block_sizes: Optional per-group block size of ``block_tables`` (the kernel
            block of a group whose KV-manager blocks are split; default: the
            group spec's block size).

    Returns:
        A mapping from layer name to its group's shared metadata dictionary.
    """
    metadata: dict[str, dict] = {}
    for group_idx, group in enumerate(kv_cache_groups):
        block_table = block_tables[group_idx]
        group_slot_mapping = (
            slot_mappings[group_idx] if slot_mappings is not None else slot_mapping
        )
        per_layer = {
            "block_table_tensor": block_table,
            "slot_mapping": group_slot_mapping,
            "max_query_len": max_query_len,
            "block_size": (
                block_sizes[group_idx]
                if block_sizes is not None
                else group.kv_cache_spec.block_size
            ),
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": decode_token_threshold,
        }
        if swa_kv_pos_offsets is not None:
            offset = swa_kv_pos_offsets[group_idx]
            if offset is not None:
                per_layer["swa_kv_pos_offset"] = offset
        if kv_segment_size or always_cached_seq_len:
            per_layer["cached_seq_len"] = cached_seq_len
        if kv_segment_size:
            per_layer["kv_segment_size"] = kv_segment_size
        for layer_name in group.layer_names:
            metadata[layer_name] = (
                per_layer  # one dict aliased across the group's layers
            )
    return metadata


def _compute_next_device_state(
    token_ids: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    total_len: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
    slot_active_row: torch.Tensor,
    query_len_by_slot: torch.Tensor,
    sampled_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the next device-authoritative state bundle.

    Advance counts for every active slot. Commit sampled tokens to history only
    for decode steps and final prefill chunks whose token store has capacity.
    The scheduler normally stops full sequences; the capacity guard prevents a
    late dispatch from overwriting the final stored token. Update token history
    in place with one indexed cell per stable slot; produce the smaller state
    vectors out of place.

    Args:
        token_ids: ``[max_num_reqs, max_model_len]`` token history.
        num_computed_tokens: ``[max_num_reqs]`` computed-token counts.
        total_len: ``[max_num_reqs]`` stored token lengths.
        last_sampled_tokens: ``[max_num_reqs, 1]`` sampled-token store.
        prefill_len: ``[max_num_reqs]`` prefill completion lengths.
        slot_active_row: ``[max_num_reqs]`` output row per active slot.
        query_len_by_slot: ``[max_num_reqs]`` scheduled token count by slot.
        sampled_tokens: Sampled tokens in output-row order.

    Returns:
        Updated token history, computed counts, total lengths, and sampled
        token store.
    """
    max_model_len = token_ids.shape[1]

    active = slot_active_row >= 0
    delta = torch.where(
        active,
        query_len_by_slot.to(num_computed_tokens.dtype),
        torch.zeros_like(num_computed_tokens),
    )
    new_num_computed = num_computed_tokens + delta
    already_decode = num_computed_tokens >= prefill_len
    in_bounds = total_len < max_model_len
    commit = active & (already_decode | (new_num_computed == prefill_len)) & in_bounds

    inactive_row = torch.full_like(slot_active_row, -1)
    commit_slot_row = torch.where(commit, slot_active_row, inactive_row)
    new_last_sampled = scatter_sampled_into_slot_store(
        last_sampled_tokens, commit_slot_row, sampled_tokens
    )

    row_idx = torch.clamp(slot_active_row.to(torch.int64), min=0)
    token_by_slot = sampled_tokens.reshape(-1).to(token_ids.dtype)[row_idx]

    state_rows = torch.arange(
        token_ids.shape[0], dtype=torch.int64, device=token_ids.device
    )
    append_col = torch.clamp(total_len.to(torch.int64), max=max_model_len - 1)
    old_token_at_append = token_ids[state_rows, append_col]
    token_to_store = torch.where(commit, token_by_slot, old_token_at_append)
    token_ids.index_put_((state_rows, append_col), token_to_store)
    new_token_ids = token_ids
    new_total_len = total_len + commit.to(total_len.dtype)

    return new_token_ids, new_num_computed, new_total_len, new_last_sampled


def _compute_next_device_state_multi(
    token_ids: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    total_len: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
    slot_active_row: torch.Tensor,
    query_len_by_slot: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_committed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the next state bundle of a step that commits several tokens per row.

    Same contract as :func:`_compute_next_device_state` (and the same result for
    one committed token per row), for speculative decoding: a verify step commits
    the accepted prefix of each row of its ``[rows, 1 + k]`` rejection output, and
    the computed count advances by the committed count instead of by the scheduled
    query length (which includes the rejected drafts). A prefill step of a
    speculative configuration passes ``num_committed`` ones and advances by its
    query length. Token history gets one guarded indexed write per column.

    Args:
        token_ids: ``[max_num_reqs, max_model_len]`` token history.
        num_computed_tokens: ``[max_num_reqs]`` computed-token counts.
        total_len: ``[max_num_reqs]`` stored token lengths.
        last_sampled_tokens: ``[max_num_reqs, 1]`` sampled-token store.
        prefill_len: ``[max_num_reqs]`` prefill completion lengths.
        slot_active_row: ``[max_num_reqs]`` output row per active slot.
        query_len_by_slot: ``[max_num_reqs]`` scheduled token count by slot.
        sampled_tokens: Sampled tokens in output-row order, ``[rows]`` or
            ``[rows, width]`` (the committed prefix of each row first).
        num_committed: ``[rows]`` committed-token count per row.

    Returns:
        Updated token history, computed counts, total lengths, and sampled
        token store.
    """
    max_model_len = token_ids.shape[1]
    num_rows = sampled_tokens.shape[0]
    sampled_2d = sampled_tokens.reshape(num_rows, -1)
    width = sampled_2d.shape[1]

    active = slot_active_row >= 0
    row_idx = torch.clamp(slot_active_row.to(torch.int64), min=0)
    committed_by_slot = num_committed.reshape(-1).to(num_computed_tokens.dtype)[row_idx]
    # A prefill row advances by its chunk; a verify row by its accepted tokens.
    delta = torch.where(
        num_computed_tokens >= prefill_len,
        committed_by_slot,
        query_len_by_slot.to(num_computed_tokens.dtype),
    )
    delta = torch.where(active, delta, torch.zeros_like(num_computed_tokens))
    new_num_computed = num_computed_tokens + delta
    already_decode = num_computed_tokens >= prefill_len
    in_bounds = total_len < max_model_len
    commit = active & (already_decode | (new_num_computed == prefill_len)) & in_bounds

    # The last committed token of each row feeds the next step.
    last_col = torch.clamp(committed_by_slot.to(torch.int64) - 1, min=0, max=width - 1)
    # One 1-D gather per column, and a select for the last committed token: neuronx-cc
    # fails 2-D gathers here (NCC_ISMP902, simplifier "tuple index out of range").
    token_by_slot = [
        sampled_2d[:, col].reshape(-1).to(token_ids.dtype)[row_idx]
        for col in range(width)
    ]  # width x [max_num_reqs]
    last_by_slot = torch.zeros_like(token_by_slot[0])
    for col in range(width):
        last_by_slot = torch.where(last_col == col, token_by_slot[col], last_by_slot)
    last_by_slot = last_by_slot.view(-1, 1)
    new_last_sampled = torch.where(
        commit.view(-1, 1),
        last_by_slot.to(last_sampled_tokens.dtype),
        last_sampled_tokens,
    )

    state_rows = torch.arange(
        token_ids.shape[0], dtype=torch.int64, device=token_ids.device
    )
    stored = torch.zeros_like(total_len)
    for col in range(width):
        dest = total_len.to(torch.int64) + col
        store = commit & (committed_by_slot > col) & (dest < max_model_len)
        append_col = torch.clamp(dest, max=max_model_len - 1)
        old_token_at_append = token_ids[state_rows, append_col]
        token_to_store = torch.where(store, token_by_slot[col], old_token_at_append)
        token_ids.index_put_((state_rows, append_col), token_to_store)
        stored = stored + store.to(total_len.dtype)
    new_token_ids = token_ids
    new_total_len = total_len + stored

    return new_token_ids, new_num_computed, new_total_len, new_last_sampled


def _build_verify_inputs(
    *,
    attn_metadata: dict,
    row_to_slot: torch.Tensor,
    input_valid_mask: torch.Tensor,
    num_draft_by_row: torch.Tensor,
    decode_token_by_slot: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    max_model_len: int,
    token_dtype: torch.dtype,
):
    """Build a verify step's tokens, positions and metadata from device state.

    Row ``r`` holds the slot's next token and its first ``num_draft_by_row[r]``
    drafts in a ``1 + k`` wide row, then padding. Positions start at the slot's
    device computed count, which is exact even when the scheduler's count is
    optimistic (async scheduling); slot mappings of attention groups follow from
    the positions, through the group's kernel block table (head-trimmed to the
    step's decode context bucket, which covers every position of the step: the
    bucket is chosen from the scheduler's counts, never below the device counts).
    Recurrent-state groups (metadata with ``state_in``) resume from
    block ``num_accepted - 1``. Padding tokens and padded rows get the pad token
    and slot 0 of the null block.

    Positions stay consecutive inside a row (padding tokens continue the row,
    clamped to ``max_model_len - 1``; padded rows count from 0), as the decode
    attention mask expects.

    Returns:
        ``(input_ids [B, 1 + k], drafts [B, k], positions [B * (1 + k)] int32,
        attn_metadata)``.
    """
    num_rows = row_to_slot.shape[0]
    k = draft_tokens.shape[1]
    width = 1 + k
    slot_idx = torch.clamp(row_to_slot.to(torch.int64), min=0)
    real_row = input_valid_mask.reshape(num_rows, 1)
    cols = torch.arange(width, dtype=torch.int64, device=row_to_slot.device).view(
        1, width
    )
    token_valid = real_row & (cols <= num_draft_by_row.to(torch.int64).view(-1, 1))

    first = decode_token_by_slot.reshape(-1).to(token_dtype)[slot_idx].view(-1, 1)
    drafts = draft_tokens.to(token_dtype)[slot_idx]  # [B, k]
    input_ids = torch.cat([first, drafts], dim=1)
    input_ids = torch.where(
        token_valid, input_ids, torch.full_like(input_ids, PAD_TOKEN)
    )
    drafts = input_ids[:, 1:]

    base = num_computed_tokens.to(torch.int64)[slot_idx].view(-1, 1)
    base = torch.where(real_row, base, torch.zeros_like(base))
    positions_2d = torch.clamp(base + cols, max=max_model_len - 1)

    state_in = torch.clamp(num_accepted.to(torch.int32)[slot_idx] - 1, min=0)
    state_in = torch.where(
        input_valid_mask.reshape(-1), state_in, torch.zeros_like(state_in)
    )

    new_metadata: dict[str, dict] = {}
    # One rebuilt dict per KV group (the runner tags each group's dict with its
    # constant kv_group index); untagged metadata is rebuilt per layer.
    by_group: dict[int, dict] = {}
    for layer_name, meta in attn_metadata.items():
        group = meta.get("kv_group")
        new_meta = by_group.get(group) if group is not None else None
        if new_meta is None:
            new_meta = dict(meta)
            if "state_in" in meta:
                new_meta["state_in"] = state_in.to(meta["state_in"].dtype)
            else:
                block_table = meta["block_table_tensor"]
                block_size = meta["block_size"]
                table_blocks = block_table.shape[1]
                position_blocks = positions_2d // block_size
                blocks = torch.clamp(position_blocks, max=table_blocks - 1)
                block_ids = torch.gather(block_table.to(torch.int64), 1, blocks)
                slots = block_ids * block_size + positions_2d % block_size
                null_slot = torch.full_like(slots, NULL_SLOT)
                write = token_valid & (block_ids >= 0)
                if table_blocks * block_size < max_model_len:
                    # A decode context-length bucket trimmed the table. The runner
                    # picks a bucket that covers every position of the step, so this
                    # never drops a real token; it keeps a violation from writing
                    # into the table's last block.
                    write = write & (position_blocks < table_blocks)
                slots = torch.where(write, slots, null_slot)
                new_meta["slot_mapping"] = slots.reshape(-1)
            if group is not None:
                by_group[group] = new_meta
        new_metadata[layer_name] = new_meta
    return (
        input_ids,
        drafts,
        positions_2d.reshape(-1).to(torch.int32),
        new_metadata,
    )


def _shifted_prefill_draft_ids(
    token_store: torch.Tensor,
    linear_idx: torch.Tensor,
    input_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather each prefill token's successor from the token store.

    A draft head reads token ``i + 1`` with the target state of token ``i``. Inside
    a chunk that is the next prompt token, also for the chunk's last token when
    the chunk is not the final one (the store holds the whole prompt). The final
    chunk's last token has no successor in the store; the caller patches it with
    the sampled token.
    """
    max_model_len = token_store.shape[1]
    in_row = linear_idx % max_model_len
    next_idx = torch.where(in_row < max_model_len - 1, linear_idx + 1, linear_idx)
    shifted = torch.index_select(token_store.reshape(-1), 0, next_idx)
    return torch.where(input_valid_mask, shifted, torch.full_like(shifted, PAD_TOKEN))


class NeuronForwardAdapter(torch.nn.Module):
    """Wrap an unchanged model with the runner's compiled forward contract.

    Use an ``nn.Module`` so ``torch.compile`` traces the explicit tensor ABI in
    :meth:`forward`. Retain the wrapped model as an unmodified submodule.

    With a ``speculator`` (speculative decoding) the adapter also runs, in the same
    graph, the rejection sampler of a verify step and the draft model, and commits
    each slot's accepted tokens, drafts and accepted count (see
    ``docs/design/speculation/mrv2-spec-decode.md``).
    """

    def __init__(
        self, model, emit_gathered_logits: bool = False, speculator=None
    ) -> None:
        super().__init__()
        self.model = model
        self.emit_gathered_logits = emit_gathered_logits
        self.speculator = speculator

    def forward(
        self,
        positions: torch.Tensor,
        attn_metadata,
        sampling_positions: torch.Tensor,
        row_to_slot: torch.Tensor,
        input_valid_mask: torch.Tensor,
        slot_active_row: torch.Tensor,
        query_len_by_slot: torch.Tensor,
        prefill_len: torch.Tensor,
        token_store: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        total_len: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        sampling_params: torch.Tensor,
        rank: torch.Tensor,
        next_token_by_slot: torch.Tensor,
        linear_idx: torch.Tensor | None = None,
        draft_tokens: torch.Tensor | None = None,
        num_accepted: torch.Tensor | None = None,
        num_draft_by_row: torch.Tensor | None = None,
        host_inputs: dict[str, torch.Tensor] | None = None,
    ):
        """Execute one model step and commit state inside the graph.

        * Builds prefill inputs from `linear_idx`.
        * Builds initial decode inputs from token history.
        * Builds subsequent decode inputs from the previous non-aliased output.
        * Selects sampling parameters by stable slot.
        * Runs the model.
        * Updates token history in place.

        Return sampled tokens, four updated state tensors, and a non-aliased
        ``[max_num_reqs, 1]`` clone of the sampled-token store. Every
        tensor-parallel rank passes that clone to the next decode step to keep
        collective ordering stable. In debug mode, append gathered logits
        without moving the existing output positions.

        Args:
            positions: Padded token positions for prefill or decode.
            attn_metadata (dict[str, dict]): Per-layer attention metadata.
            sampling_positions: Hidden-state rows to sample.
            row_to_slot: Mapping from sampled output rows to stable slots.
                ``-1`` marks a padded row.
            input_valid_mask: Validity mask for input tokens or rows.
            slot_active_row: Mapping from stable slots to output rows.
            query_len_by_slot: Scheduled token count for each stable slot.
            prefill_len: Prefill completion length for each stable slot.
            token_store: Persistent token-history tensor.
            num_computed_tokens: Device-authoritative computed-token counts.
            total_len: Device-authoritative token-history lengths.
            last_sampled_tokens: Persistent sampled-token store.
            sampling_params: Per-slot sampling parameters.
            rank: Tensor-parallel rank tensor passed to the model.
            next_token_by_slot: Non-aliased sampled-token output from the
                previous step.
            linear_idx: Optional flat prefill gather index. Omit for decode.
            draft_tokens: Speculative decoding only: ``[max_num_reqs, k]`` drafts
                each slot's last step proposed (device state).
            num_accepted: Speculative decoding only: ``[max_num_reqs]`` tokens
                each slot's last step committed (device state).
            num_draft_by_row: Verify steps only: ``[rows]`` scheduled draft
                count per row.
            host_inputs: Models with host inputs only (``interfaces.HasHostInputs``):
                per-token tensors built on the host, aligned with the step's
                ``input_ids`` rows, passed to the model by keyword.

        Returns:
            Sampled tokens, updated state tensors, and the non-aliased
            next-token output. Append gathered logits in debug mode. With
            speculative decoding the sampled tokens of a verify step are the
            ``[rows, 1 + k]`` rejection output (``-1`` past the accepted
            prefix) and the state bundle also holds the new draft table and
            accepted counts.

        Raises:
            AssertionError: If the decode carry dtype does not match the token
                history dtype.
            RuntimeError: If debug mode does not receive tensor logits.
        """
        if self.speculator is not None:
            return self._forward_speculative(
                positions=positions,
                attn_metadata=attn_metadata,
                sampling_positions=sampling_positions,
                row_to_slot=row_to_slot,
                input_valid_mask=input_valid_mask,
                slot_active_row=slot_active_row,
                query_len_by_slot=query_len_by_slot,
                prefill_len=prefill_len,
                token_store=token_store,
                num_computed_tokens=num_computed_tokens,
                total_len=total_len,
                last_sampled_tokens=last_sampled_tokens,
                sampling_params=sampling_params,
                rank=rank,
                next_token_by_slot=next_token_by_slot,
                linear_idx=linear_idx,
                draft_tokens=draft_tokens,
                num_accepted=num_accepted,
                num_draft_by_row=num_draft_by_row,
                host_inputs=host_inputs,
            )

        # --- prologue: build input_ids on device (never read on host) ---
        if linear_idx is not None:
            # PREFILL: gather each scored token from its request's stable-slot row in the flattened store.
            gathered = torch.index_select(token_store.reshape(-1), 0, linear_idx)
            input_ids = torch.where(
                input_valid_mask, gathered, torch.full_like(gathered, PAD_TOKEN)
            )
        else:
            # DECODE: a DI bootstrap slot must consume its first uncached
            # history token. Other slots consume the non-aliased output from
            # the prior step. Select the history token by its computed count.
            assert next_token_by_slot.dtype == token_store.dtype, (
                "decode carry must match token_store dtype before bootstrap "
                f"selection; got {next_token_by_slot.dtype} and {token_store.dtype}"
            )
            bootstrap_slot = (slot_active_row >= 0) & (
                num_computed_tokens < prefill_len
            )
            bootstrap_col = torch.clamp(
                num_computed_tokens.to(torch.int64),
                max=token_store.shape[1] - 1,
            )
            bootstrap_token_by_slot = torch.gather(
                token_store,
                dim=1,
                index=bootstrap_col.unsqueeze(1),
            )
            decode_token_by_slot = torch.where(
                bootstrap_slot.unsqueeze(1),
                bootstrap_token_by_slot,
                next_token_by_slot,
            )
            placeholder = torch.full_like(
                row_to_slot, PAD_TOKEN, dtype=token_store.dtype
            )
            input_ids = build_decode_input_ids_from_store(
                placeholder, row_to_slot, decode_token_by_slot
            )
            input_ids = torch.where(
                input_valid_mask, input_ids, torch.full_like(input_ids, PAD_TOKEN)
            )

        # Padded rows use -1 for the decode-input preserve sentinel. Clamp only
        # for this gather; their selected parameters are masked and ignored.
        sampling_row_to_slot = torch.clamp(row_to_slot, min=0)
        selected_sampling_params = torch.index_select(
            sampling_params, 0, sampling_row_to_slot
        )

        # --- call the UNCHANGED model by keyword (args[0..1] are inputs_embeds/is_token_ids) ---
        result = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
            sampling_params=selected_sampling_params,
            logit_mask=None,
            rank=rank,
            **(host_inputs or {}),
        )
        sampled_raw = result[0]
        sampled = _reinterpret_uint32_as_int32(sampled_raw, token_store.dtype)

        # --- epilogue: commit this step's device-authoritative state in-graph (aliased outputs) ---
        bundle = _compute_next_device_state(
            token_ids=token_store,
            num_computed_tokens=num_computed_tokens,
            total_len=total_len,
            last_sampled_tokens=last_sampled_tokens,
            prefill_len=prefill_len,
            slot_active_row=slot_active_row,
            query_len_by_slot=query_len_by_slot,
            sampled_tokens=sampled,
        )
        # Return a clone so the next step depends on a non-aliased graph output.
        # Every TP rank consumes it before the next decode, which preserves
        # sampling collective order.
        next_token_out = bundle[3].clone()
        if self.emit_gathered_logits:
            gathered_logits = result[1]
            if not torch.is_tensor(gathered_logits):
                raise RuntimeError(
                    "V2 gathered logits require the model to return a tensor"
                )
            return sampled, bundle, next_token_out, gathered_logits
        return sampled, bundle, next_token_out

    @staticmethod
    def _decode_token_by_slot(
        token_store,
        num_computed_tokens,
        prefill_len,
        slot_active_row,
        next_token_by_slot,
    ):
        """Return the ``[max_num_reqs, 1]`` token each slot's decode row consumes."""
        assert next_token_by_slot.dtype == token_store.dtype, (
            "decode carry must match token_store dtype before bootstrap "
            f"selection; got {next_token_by_slot.dtype} and {token_store.dtype}"
        )
        bootstrap_slot = (slot_active_row >= 0) & (num_computed_tokens < prefill_len)
        bootstrap_col = torch.clamp(
            num_computed_tokens.to(torch.int64), max=token_store.shape[1] - 1
        )
        bootstrap_token_by_slot = torch.gather(
            token_store, dim=1, index=bootstrap_col.unsqueeze(1)
        )
        return torch.where(
            bootstrap_slot.unsqueeze(1), bootstrap_token_by_slot, next_token_by_slot
        )

    def _forward_speculative(
        self,
        *,
        positions,
        attn_metadata,
        sampling_positions,
        row_to_slot,
        input_valid_mask,
        slot_active_row,
        query_len_by_slot,
        prefill_len,
        token_store,
        num_computed_tokens,
        total_len,
        last_sampled_tokens,
        sampling_params,
        rank,
        next_token_by_slot,
        linear_idx,
        draft_tokens,
        num_accepted,
        num_draft_by_row,
        host_inputs=None,
    ):
        """Run one prefill or verify step with the draft model in the same graph.

        Prefill: the target runs the chunk, then the draft head runs over the same
        tokens (each with its successor) and proposes ``k`` drafts for the slot.
        Verify: tokens, positions, slot mappings and ``state_in`` come from device
        state (:func:`_build_verify_inputs`); the target samples every position,
        the fixed-layout rejection sampler accepts a prefix of each row, and the
        draft head proposes the next drafts from the last accepted position. The
        commit appends the accepted tokens and records the drafts and the accepted
        count per slot.
        """
        assert draft_tokens is not None and num_accepted is not None, (
            "speculative decoding needs the draft table and the accepted counts"
        )
        speculator = self.speculator
        is_prefill = linear_idx is not None
        num_rows = row_to_slot.shape[0]
        k = draft_tokens.shape[1]
        sampling_row_to_slot = torch.clamp(row_to_slot, min=0)

        if is_prefill:
            gathered = torch.index_select(token_store.reshape(-1), 0, linear_idx)
            input_ids = torch.where(
                input_valid_mask, gathered, torch.full_like(gathered, PAD_TOKEN)
            )
            token_sampling_slots = sampling_row_to_slot
        else:
            assert num_draft_by_row is not None, "a verify step needs draft counts"
            decode_token_by_slot = self._decode_token_by_slot(
                token_store,
                num_computed_tokens,
                prefill_len,
                slot_active_row,
                next_token_by_slot,
            )
            input_ids_2d, drafts_in, positions, attn_metadata = _build_verify_inputs(
                attn_metadata=attn_metadata,
                row_to_slot=row_to_slot,
                input_valid_mask=input_valid_mask,
                num_draft_by_row=num_draft_by_row,
                decode_token_by_slot=decode_token_by_slot,
                draft_tokens=draft_tokens,
                num_computed_tokens=num_computed_tokens,
                num_accepted=num_accepted,
                max_model_len=token_store.shape[1],
                token_dtype=token_store.dtype,
            )
            input_ids = input_ids_2d.reshape(-1)
            # Every token of a row samples with the row's parameters.
            token_sampling_slots = (
                sampling_row_to_slot.view(num_rows, 1)
                .expand(num_rows, 1 + k)
                .reshape(-1)
            )
        selected_sampling_params = torch.index_select(
            sampling_params, 0, token_sampling_slots
        )

        result = self.model(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
            sampling_params=selected_sampling_params,
            logit_mask=None,
            rank=rank,
            **(host_inputs or {}),
        )
        sampled = _reinterpret_uint32_as_int32(result[0], token_store.dtype)
        target_hidden = speculator.target_hidden_states(result)

        if is_prefill:
            # Draft over the chunk: token i + 1 with the target state of token i.
            # The final chunk's last token is followed by the token just sampled.
            draft_ids = _shifted_prefill_draft_ids(
                token_store, linear_idx, input_valid_mask
            )
            slot0 = sampling_row_to_slot[:1]
            final_chunk = (
                num_computed_tokens[slot0] + query_len_by_slot[slot0]
            ) >= prefill_len[slot0]
            next_after_chunk = torch.index_select(draft_ids, 0, sampling_positions)
            next_token = torch.where(
                final_chunk,
                sampled.reshape(-1)[:1].to(draft_ids.dtype),
                next_after_chunk,
            )
            raw_sampled = next_token.view(1, 1)
            draft_sampling_positions = sampling_positions
            sampled_out = sampled
            num_committed = torch.ones(
                num_rows, dtype=torch.int32, device=sampled.device
            )
        else:
            from vllm_neuron.nn.rejection_sampler import rejection_sample_fixed

            targets = sampled.reshape(num_rows, 1 + k)
            sampled_out = rejection_sample_fixed(
                targets,
                drafts_in,
                num_draft_by_row,
                speculator.synthetic_conditional_rates,
            )
            num_committed = (sampled_out != PLACEHOLDER).sum(dim=1).to(torch.int32)
            # The draft head reads token j + 1 with the target state of token j, so
            # shift each row left by one. It patches the row's next token at its
            # last accepted position, read from the rejection output.
            draft_ids = torch.cat(
                [
                    input_ids_2d[:, 1:],
                    torch.full_like(input_ids_2d[:, :1], PAD_TOKEN),
                ],
                dim=1,
            ).reshape(-1)
            raw_sampled = sampled_out
            draft_sampling_positions = (
                torch.arange(num_rows, dtype=torch.int64, device=sampled.device)
                * (1 + k)
                + k
            )

        new_drafts = speculator(
            input_ids=draft_ids,
            positions=positions,
            target_hidden_states=target_hidden,
            attn_metadata=attn_metadata,
            sampling_positions=draft_sampling_positions,
            raw_sampled_token_ids=raw_sampled,
            rank=rank,
        )

        bundle = _compute_next_device_state_multi(
            token_ids=token_store,
            num_computed_tokens=num_computed_tokens,
            total_len=total_len,
            last_sampled_tokens=last_sampled_tokens,
            prefill_len=prefill_len,
            slot_active_row=slot_active_row,
            query_len_by_slot=query_len_by_slot,
            sampled_tokens=sampled_out,
            num_committed=num_committed,
        )
        active = slot_active_row >= 0
        row_idx = torch.clamp(slot_active_row.to(torch.int64), min=0)
        new_draft_tokens = torch.where(
            active.view(-1, 1),
            new_drafts.reshape(num_rows, k).to(draft_tokens.dtype)[row_idx],
            draft_tokens,
        )
        new_num_accepted = torch.where(
            active, num_committed.to(num_accepted.dtype)[row_idx], num_accepted
        )
        bundle = (*bundle, new_draft_tokens, new_num_accepted)
        next_token_out = bundle[3].clone()
        if self.emit_gathered_logits:
            gathered_logits = speculator.gathered_logits(result)
            if not torch.is_tensor(gathered_logits):
                raise RuntimeError(
                    "V2 gathered logits require the model to return a tensor"
                )
            return sampled_out, bundle, next_token_out, gathered_logits
        return sampled_out, bundle, next_token_out
