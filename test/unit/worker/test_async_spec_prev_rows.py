# SPDX-License-Identifier: Apache-2.0
"""Async spec decode: a request's correction row survives steps it is not part of (CPU).

The on-device correction of async spec decode reads, per request, the previous spec step's
rejection-sampler row (valid count -> rejected drafts, state_in). A request held back for a
step (e.g. during another request's prefill chunk) must still be corrected by the row of
its own last spec step, not by the no-op dummy.
"""

import types

import pytest
import torch

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")


def _runner(req_ids, live=None):
    runner = object.__new__(runner_mod.NeuronModelRunner)
    runner._last_spec_rows = {}
    runner.requests = {r: object() for r in (live if live is not None else req_ids)}
    runner.input_batch = types.SimpleNamespace(
        req_ids=list(req_ids), num_reqs=len(req_ids)
    )
    return runner


def _spec_out(rows):
    return torch.tensor(rows, dtype=torch.int32)  # [bs, k + 1], -1 = rejected


def test_held_back_request_keeps_its_row():
    r = _runner(["a", "b"])
    step_t = _spec_out([[5, -1, -1], [7, 8, 9]])  # a rejected 2 drafts, b accepted all
    r._record_last_spec_rows(step_t, ["a", "b"])
    # Next step is a prefill chunk of "c" (non-spec output, a and b held back).
    r._record_last_spec_rows(torch.tensor([42]), ["c"])
    assert set(r._last_spec_rows) == {"a", "b"}
    # The following decode step sees a non-spec previous future: every row starts as dummy.
    dummy = runner_mod.NeuronModelRunner._dummy_prev_sampled(2, 2, "cpu")
    filled, dummies, counts = r._fill_from_last_spec_rows(dummy, {0, 1}, None)
    assert dummies == set()
    assert filled.tolist() == [[5, -1, -1], [7, 8, 9]]
    assert counts == {0: 0, 1: 2}


def test_rows_from_previous_future_are_not_refilled():
    r = _runner(["a", "b"])
    prev = _spec_out([[1, 2, -1], [3, -1, -1]])
    r._record_last_spec_rows(prev, ["a", "b"])
    # Row 1 is a dummy for another reason (e.g. padded); its record is the previous
    # future itself, which the remap already used: leave it alone.
    _, dummies, counts = r._fill_from_last_spec_rows(prev.clone(), {1}, prev)
    assert dummies == {1}
    assert counts == {}


def test_new_request_and_padding_stay_dummy():
    r = _runner(["a", "new"])
    r._record_last_spec_rows(_spec_out([[1, 2, 3]]), ["a"])
    r._record_last_spec_rows(torch.tensor([0]), ["new"])  # its prefill
    dummy = runner_mod.NeuronModelRunner._dummy_prev_sampled(4, 2, "cpu")
    filled, dummies, _ = r._fill_from_last_spec_rows(dummy, {0, 1, 2, 3}, None)
    assert dummies == {1, 2, 3}  # new request + 2 padded rows
    assert filled[0].tolist() == [1, 2, 3]
    assert filled[1].tolist() == [0, -1, -1]


def test_finished_requests_are_forgotten():
    r = _runner(["a", "b"], live=["a"])
    r._record_last_spec_rows(_spec_out([[1, -1, -1], [2, -1, -1]]), ["a", "b"])
    assert set(r._last_spec_rows) == {"a"}


def test_held_back_request_positions_are_not_corrected_twice():
    """The scheduler has already rolled back a filled row's rejected drafts (its step is
    at least two steps old), so the correction keeps the positions and only resumes the
    recurrent state from the last accepted token's block."""
    from vllm_neuron.functional.spec_decode_correction import (
        correct_spec_decode_positions_and_slot_mapping,
    )

    r = _runner(["a"])
    r._record_last_spec_rows(_spec_out([[5, -1, -1]]), ["a"])  # 2 drafts rejected
    r._record_last_spec_rows(torch.tensor([42]), ["c"])  # c's prefill; a held back
    r.speculative_config = types.SimpleNamespace(num_speculative_tokens=2)
    r.async_execution_buffer = {"futures_sampled_token_ids": torch.tensor([42])}
    r.device = "cpu"
    prev_sampled, prev_num_draft = r._get_prev_spec_tensors_for_on_device_correction(
        None, num_reqs_padded=1, device="cpu"
    )
    positions = torch.tensor([100, 101, 102], dtype=torch.int32)  # already rolled back
    meta = {
        "gdn": {
            "block_table_tensor": torch.tensor([[7, 8, 9]], dtype=torch.int32),
            "block_size": 1024,
            "state_in": torch.zeros(1, dtype=torch.int32),
        }
    }
    out_pos, out_meta = correct_spec_decode_positions_and_slot_mapping(
        positions,
        meta,
        prev_sampled,
        prev_num_draft,
        torch.zeros(3, dtype=torch.int64),
        vocab_size=1000,
    )
    assert out_pos.tolist() == [100, 101, 102]
    assert out_meta["gdn"]["state_in"].tolist() == [0]


def test_request_without_drafts_in_a_spec_step_is_not_rolled_back():
    """Near max_model_len the scheduler stops drafting for some requests, so a spec step
    mixes rows with and without drafts. A row without drafts rejected nothing."""
    r = _runner(["a", "b"])
    prev = _spec_out([[5, 6, 7], [9, -1, -1]])  # a: 2 drafts accepted; b: no drafts
    r._record_last_spec_rows(prev, ["a", "b"], {"a": [1, 2], "b": []})
    r.speculative_config = types.SimpleNamespace(num_speculative_tokens=2)
    r.async_execution_buffer = {
        "futures_sampled_token_ids": prev,
        "prev_req_ids_ordered": ["a", "b"],
    }
    r.device = "cpu"
    _, prev_num_draft = r._get_prev_spec_tensors_for_on_device_correction(
        None, num_reqs_padded=2, device="cpu"
    )
    assert prev_num_draft.tolist() == [2, 0]
