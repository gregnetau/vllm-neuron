# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 forward adapter with speculative decoding, eager on CPU.

The target is a toy whose sample at every position is the successor of the input token
(token + 1), so a draft is accepted exactly when it continues the run. The draft is a toy
that records its inputs and proposes the successors of the last accepted token. The adapter,
the fixed-layout rejection sampler and the multi-token commit are the real ones.
"""

import pytest
import torch

pytest.importorskip("vllm")

from vllm_neuron.vllm.worker.neuron_forward_adapter import (
    NeuronForwardAdapter,
    _compute_next_device_state,
    _compute_next_device_state_multi,
)

MAX_REQS, MAX_LEN, K, BLOCK, VOCAB = 4, 32, 2, 4, 1000


class _SuccessorTarget(torch.nn.Module):
    """Samples token + 1 at every position; returns a fake hidden state per token."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, positions, attn_metadata, sampling_positions, **kw):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "positions": positions.clone(),
                "md": attn_metadata,
            }
        )
        sampled = (input_ids[sampling_positions] + 1).to(torch.int32)
        hidden = torch.stack([input_ids.float(), positions.float()], dim=1)
        return sampled, hidden, hidden.clone()  # gathered logits stand-in


class _SuccessorDraft(torch.nn.Module):
    """Proposes next + 1 .. next + k after the last accepted token; records its inputs."""

    num_speculative_tokens = K
    synthetic_conditional_rates = None

    def __init__(self):
        super().__init__()
        self.calls = []

    @staticmethod
    def target_hidden_states(result):
        return result[1]

    @staticmethod
    def gathered_logits(result):
        return result[2]

    def forward(self, *, input_ids, raw_sampled_token_ids, sampling_positions, **kw):
        self.calls.append(
            dict(
                input_ids=input_ids.clone(),
                raw=raw_sampled_token_ids.clone(),
                sampling_positions=sampling_positions.clone(),
                **kw,
            )
        )
        valid = raw_sampled_token_ids >= 0
        last = raw_sampled_token_ids.gather(
            1, (valid.sum(1) - 1).clamp(min=0).view(-1, 1)
        )
        return last + torch.arange(1, K + 1, dtype=torch.int32).view(1, K)


def _state():
    return {
        "token_store": torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32),
        "num_computed_tokens": torch.zeros(MAX_REQS, dtype=torch.int32),
        "total_len": torch.zeros(MAX_REQS, dtype=torch.int32),
        "last_sampled_tokens": torch.zeros(MAX_REQS, 1, dtype=torch.int32),
        "sampling_params": torch.zeros(MAX_REQS, 3),
        "next_token_by_slot": torch.zeros(MAX_REQS, 1, dtype=torch.int32),
        "draft_tokens": torch.zeros(MAX_REQS, K, dtype=torch.int32),
        "num_accepted": torch.ones(MAX_REQS, dtype=torch.int32),
    }


def _metadata(block_table, *, state_rows=None, query_len):
    """One attention group (layers.0) and, optionally, one state group (layers.1)."""
    attn = {
        "block_table_tensor": block_table,
        "slot_mapping": torch.full((block_table.shape[0] * query_len,), 123),
        "max_query_len": query_len,
        "block_size": BLOCK,
        "max_blocks_per_seq": block_table.shape[1],
        "decode_token_threshold": 1 + K,
        "kv_group": 0,
    }
    md = {"layers.0.self_attn": attn, "layers.2.self_attn": attn}
    if state_rows is not None:
        md["layers.1.linear_attn"] = dict(
            attn,
            block_table_tensor=state_rows,
            state_in=torch.zeros(state_rows.shape[0], dtype=torch.int32),
            kv_group=1,
        )
    return md


def _verify(
    adapter, state, *, slots, num_drafts, block_table, state_rows=None, bucket=4
):
    B, S = bucket, 1 + K
    row_to_slot = torch.zeros(B, dtype=torch.int64)
    row_to_slot[: len(slots)] = torch.tensor(slots)
    valid = torch.zeros(B, dtype=torch.bool)
    valid[: len(slots)] = True
    slot_active_row = torch.full((MAX_REQS,), -1, dtype=torch.int32)
    query_len_by_slot = torch.zeros(MAX_REQS, dtype=torch.int32)
    for row, (slot, n) in enumerate(zip(slots, num_drafts)):
        slot_active_row[slot] = row
        query_len_by_slot[slot] = 1 + n
    drafts = torch.zeros(B, dtype=torch.int32)
    drafts[: len(slots)] = torch.tensor(num_drafts, dtype=torch.int32)
    out = adapter(
        # Host positions are ignored by the verify graph (device counts rule).
        positions=torch.full((B * S,), 999, dtype=torch.int32),
        attn_metadata=_metadata(block_table, state_rows=state_rows, query_len=S),
        sampling_positions=torch.arange(B * S),
        row_to_slot=row_to_slot,
        input_valid_mask=valid,
        slot_active_row=slot_active_row,
        query_len_by_slot=query_len_by_slot,
        prefill_len=torch.full((MAX_REQS,), 3, dtype=torch.int32),
        rank=torch.zeros(1, dtype=torch.int32),
        num_draft_by_row=drafts,
        **state,
    )
    sampled, bundle, next_token, _ = out
    (
        state["token_store"],
        state["num_computed_tokens"],
        state["total_len"],
        state["last_sampled_tokens"],
        state["draft_tokens"],
        state["num_accepted"],
    ) = bundle
    state["next_token_by_slot"] = next_token
    return sampled


def _seed(state, slot, tokens, num_computed, drafts):
    state["token_store"][slot, : len(tokens)] = torch.tensor(tokens)
    state["total_len"][slot] = len(tokens)
    state["num_computed_tokens"][slot] = num_computed
    state["last_sampled_tokens"][slot] = tokens[-1]
    state["next_token_by_slot"][slot] = tokens[-1]
    state["draft_tokens"][slot] = torch.tensor(drafts)


def _adapter():
    return NeuronForwardAdapter(
        _SuccessorTarget(), emit_gathered_logits=True, speculator=_SuccessorDraft()
    )


def test_verify_accepts_prefix_and_commits_per_row():
    adapter, state = _adapter(), _state()
    # slot 2: next 10, drafts 11 12 (all accepted, bonus 13); slot 0: next 20, drafts
    # 21 99 (second rejected -> 21 22); slot 3: one draft 31 accepted -> 31 32.
    _seed(state, 2, [1, 2, 3, 10], 3, [11, 12])
    _seed(state, 0, [1, 2, 3, 20], 3, [21, 99])
    _seed(state, 3, [1, 2, 3, 30], 3, [31, 77])
    table = torch.tensor([[1, 2, -1, -1], [3, 4, -1, -1], [5, 6, -1, -1], [-1] * 4])
    sampled = _verify(
        adapter, state, slots=[2, 0, 3], num_drafts=[2, 2, 1], block_table=table
    )
    assert sampled.tolist() == [[11, 12, 13], [21, 22, -1], [31, 32, -1], [1, -1, -1]]

    assert state["num_computed_tokens"].tolist() == [5, 0, 6, 5]
    assert state["total_len"].tolist() == [6, 0, 7, 6]
    assert state["token_store"][2, :7].tolist() == [1, 2, 3, 10, 11, 12, 13]
    assert state["token_store"][0, :6].tolist() == [1, 2, 3, 20, 21, 22]
    assert state["token_store"][3, :6].tolist() == [1, 2, 3, 30, 31, 32]
    assert state["last_sampled_tokens"].view(-1).tolist() == [22, 0, 13, 32]
    assert state["next_token_by_slot"].view(-1).tolist() == [22, 0, 13, 32]
    assert state["num_accepted"].tolist() == [2, 1, 3, 2]
    # Drafts follow each row's last accepted token; the idle slot keeps its own.
    assert state["draft_tokens"].tolist() == [[23, 24], [0, 0], [14, 15], [33, 34]]


def test_verify_inputs_come_from_device_state():
    adapter, state = _adapter(), _state()
    _seed(state, 1, [5, 6, 7, 8, 9], 4, [10, 11])
    _seed(state, 3, [1, 2], 1, [42, 43])
    table = torch.tensor([[7, 8, -1, -1], [9, -1, -1, -1], [-1] * 4, [-1] * 4])
    states = torch.tensor([[5, 6, 7], [11, 12, 13], [-1] * 3, [-1] * 3])
    state["num_accepted"][1] = 3
    _verify(
        adapter,
        state,
        slots=[1, 3],
        num_drafts=[2, 0],
        block_table=table,
        state_rows=states,
    )
    call = adapter.model.calls[-1]
    S = 1 + K
    # Row 0: next 9 + drafts; row 1: next 2 + padding; padded rows: pad.
    assert call["input_ids"].view(4, S).tolist() == [
        [9, 10, 11],
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert call["positions"].view(4, S).tolist() == [
        [4, 5, 6],
        [1, 2, 3],
        [0, 1, 2],
        [0, 1, 2],
    ]
    attn = call["md"]["layers.0.self_attn"]
    assert call["md"]["layers.2.self_attn"] is attn  # one dict per group, aliased
    # Block 7 holds positions 0-3, block 8 positions 4-7; padding -> null slot 0.
    assert attn["slot_mapping"].view(4, S).tolist() == [
        [8 * 4 + 0, 8 * 4 + 1, 8 * 4 + 2],
        [9 * 4 + 1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    state_md = call["md"]["layers.1.linear_attn"]
    assert state_md["state_in"].tolist() == [2, 0, 0, 0]
    assert "slot_mapping" in state_md and state_md["slot_mapping"][0] == 123

    draft = adapter.speculator.calls[-1]
    # Shifted rows; the draft head moves the sampling position back itself.
    assert draft["input_ids"].view(4, S).tolist() == [
        [10, 11, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert draft["sampling_positions"].tolist() == [2, 5, 8, 11]
    assert torch.equal(draft["positions"], call["positions"])


def test_verify_near_max_model_len_clamps_and_guards_capacity():
    adapter, state = _adapter(), _state()
    tokens = list(range(1, MAX_LEN))  # 31 tokens stored, one free column
    _seed(state, 0, tokens, MAX_LEN - 2, [MAX_LEN, MAX_LEN + 1])
    table = torch.arange(8).view(1, 8).expand(4, 8).contiguous()
    sampled = _verify(adapter, state, slots=[0], num_drafts=[1], block_table=table)
    assert sampled[0].tolist() == [MAX_LEN, MAX_LEN + 1, -1]
    # Positions clamp at max_model_len - 1; only one token fits the store.
    assert adapter.model.calls[-1]["positions"][:3].tolist() == [30, 31, 31]
    assert state["total_len"][0] == MAX_LEN
    assert state["token_store"][0, -1] == MAX_LEN


def test_held_back_slot_keeps_drafts_and_accepted_count():
    adapter, state = _adapter(), _state()
    _seed(state, 0, [1, 2, 3, 10], 3, [11, 12])
    _seed(state, 1, [1, 2, 3, 20], 3, [21, 22])
    table = torch.tensor([[1, 2, -1, -1], [3, 4, -1, -1], [-1] * 4, [-1] * 4])
    _verify(adapter, state, slots=[0, 1], num_drafts=[2, 2], block_table=table)
    held = (state["draft_tokens"][1].tolist(), int(state["num_accepted"][1]))
    # Slot 1 sits out two steps; then it runs alone in row 0.
    for _ in range(2):
        _verify(adapter, state, slots=[0], num_drafts=[2], block_table=table)
    assert (state["draft_tokens"][1].tolist(), int(state["num_accepted"][1])) == held
    one_row = torch.tensor([[3, 4, -1, -1]] + [[-1] * 4] * 3)
    sampled = _verify(adapter, state, slots=[1], num_drafts=[2], block_table=one_row)
    assert sampled[0].tolist() == [24, 25, 26]
    assert adapter.model.calls[-1]["positions"][:3].tolist() == [6, 7, 8]


def test_prefill_drafts_from_shifted_store_tokens():
    adapter, state = _adapter(), _state()
    prompt = [5, 6, 7, 8, 9, 10]
    state["token_store"][1, : len(prompt)] = torch.tensor(prompt)
    state["total_len"][1] = len(prompt)

    def prefill(start, n, bucket=4):
        linear_idx = torch.zeros(bucket, dtype=torch.int64)
        linear_idx[:n] = 1 * MAX_LEN + torch.arange(start, start + n)
        valid = torch.zeros(bucket, dtype=torch.bool)
        valid[:n] = True
        slot_active_row = torch.full((MAX_REQS,), -1, dtype=torch.int32)
        slot_active_row[1] = 0
        query_len = torch.zeros(MAX_REQS, dtype=torch.int32)
        query_len[1] = n
        positions = torch.zeros(bucket, dtype=torch.int32)
        positions[:n] = torch.arange(start, start + n)
        sampled, bundle, _next_token, _ = adapter(
            positions=positions,
            attn_metadata=_metadata(torch.tensor([[1, 2]]), query_len=bucket),
            sampling_positions=torch.tensor([n - 1]),
            row_to_slot=torch.tensor([1]),
            input_valid_mask=valid,
            slot_active_row=slot_active_row,
            query_len_by_slot=query_len,
            prefill_len=torch.full((MAX_REQS,), len(prompt), dtype=torch.int32),
            rank=torch.zeros(1, dtype=torch.int32),
            linear_idx=linear_idx,
            **state,
        )
        (
            state["token_store"],
            state["num_computed_tokens"],
            state["total_len"],
            state["last_sampled_tokens"],
            state["draft_tokens"],
            state["num_accepted"],
        ) = bundle
        return sampled

    state["num_accepted"][1] = 3  # stale value of a previous occupant
    prefill(0, 4)  # non-final chunk: the next prompt token follows its last token
    draft = adapter.speculator.calls[-1]
    assert draft["input_ids"].tolist() == [6, 7, 8, 9]
    assert draft["raw"].tolist() == [[9]]
    assert state["total_len"][1] == 6 and state["num_computed_tokens"][1] == 4

    sampled = prefill(4, 2)  # final chunk: the sampled token follows it
    draft = adapter.speculator.calls[-1]
    assert sampled.tolist() == [11]
    assert draft["input_ids"].tolist()[:2] == [10, 0]
    assert draft["raw"].tolist() == [[11]]
    assert state["token_store"][1, :7].tolist() == prompt + [11]
    assert state["draft_tokens"][1].tolist() == [12, 13]
    assert state["num_accepted"][1] == 1


def test_multi_commit_matches_single_commit_for_one_token_rows():
    torch.manual_seed(0)
    args = {
        "num_computed_tokens": torch.tensor([3, 0, 5, 2], dtype=torch.int32),
        "total_len": torch.tensor([4, 0, 6, 2], dtype=torch.int32),
        "last_sampled_tokens": torch.zeros(MAX_REQS, 1, dtype=torch.int32),
        "prefill_len": torch.tensor([3, 4, 5, 6], dtype=torch.int32),
        "slot_active_row": torch.tensor([1, -1, 0, 2], dtype=torch.int32),
        "query_len_by_slot": torch.tensor([1, 0, 1, 4], dtype=torch.int32),
        "sampled_tokens": torch.tensor([7, 8, 9], dtype=torch.int32),
    }
    single = _compute_next_device_state(
        token_ids=torch.randint(0, 9, (MAX_REQS, MAX_LEN), dtype=torch.int32), **args
    )
    torch.manual_seed(0)
    multi = _compute_next_device_state_multi(
        token_ids=torch.randint(0, 9, (MAX_REQS, MAX_LEN), dtype=torch.int32),
        num_committed=torch.ones(3, dtype=torch.int32),
        **args,
    )
    for a, b in zip(single, multi):
        assert torch.equal(a, b)


def test_request_state_spec_rows_reset_on_admission_and_widened_groups():
    from vllm_neuron.vllm.worker.neuron_request_state import NeuronRequestState

    rs = NeuronRequestState(
        max_num_reqs=2,
        max_model_len=64,
        vocab_size=VOCAB,
        device=torch.device("cpu"),
        block_sizes=[16],
        num_speculative_tokens=K,
    )
    assert rs.draft_tokens.shape == (2, K) and rs.num_accepted.tolist() == [1, 1]
    rs.draft_tokens = torch.full((2, K), 9, dtype=torch.int32)
    rs.num_accepted = torch.full((2,), 3, dtype=torch.int32)
    slot = rs.add_request("a", 3, [1, 2, 3], 0, 4, ([1],))
    assert rs.draft_tokens[slot].tolist() == [0, 0] and rs.num_accepted[slot] == 1
    assert rs.draft_tokens[1 - slot].tolist() == [9, 9]
    rs.remove_request("a")

    # A state group whose block spans max_model_len holds 1 + k blocks.
    rs.configure_block_sizes([16, 64], min_widths=[0, 1 + K])
    assert rs.block_table(0).shape[1] == 8 and rs.block_table(1).shape[1] == 1 + K
    assert (
        NeuronRequestState(
            max_num_reqs=2,
            max_model_len=64,
            vocab_size=VOCAB,
            device=torch.device("cpu"),
            block_sizes=[16],
        ).draft_tokens
        is None
    )
