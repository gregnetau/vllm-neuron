# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 decode context-length buckets: bucket selection, block-table trimming
and the verify graph's slot mapping through a trimmed table (CPU).

End to end coverage (real scheduler, KV manager, V2 runner and Qwen3.5 module against the
HF model) is in test/unit/model/qwen3_5/test_v2_runner_inputs.py and
test_v2_spec_decode_e2e.py.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm_neuron.vllm.worker.neuron_forward_adapter import (
    NULL_SLOT,
    _build_verify_inputs,
)
from vllm_neuron.vllm.worker.neuron_input_build import (
    PAD_SLOT_ID,
    build_kernel_block_rows,
    decode_ctx_headroom,
    decode_ctx_num_blocks,
    select_decode_ctx_bucket,
)
from vllm_neuron.vllm.worker.neuron_request_state import NeuronRequestState


def test_headroom_covers_verify_and_mtp_draft_positions():
    # Plain decode: one token. Verify: 1 + k tokens, then k - 1 MTP draft steps past the
    # last accepted token (at most num_computed + k): positions up to num_computed + 2k - 1.
    assert decode_ctx_headroom(0) == 1
    assert decode_ctx_headroom(1) == 2
    assert decode_ctx_headroom(2) == 4
    assert decode_ctx_headroom(3) == 6


@pytest.mark.parametrize(
    "max_computed, headroom, expected",
    [
        (0, 1, 128),
        (127, 1, 128),  # positions 0 .. 127
        (128, 1, 256),
        (124, 4, 128),  # MTP k=2: positions up to 127
        (125, 4, 256),  # a verify step straddling 128 needs the next bucket
        (252, 4, 256),
        (253, 4, 320),  # past the last bucket: the max_model_len graph
        (318, 4, 320),  # near max_model_len (needed > max_model_len)
    ],
)
def test_select_smallest_covering_bucket(max_computed, headroom, expected):
    assert select_decode_ctx_bucket(max_computed, headroom, [128, 256], 320) == expected


def test_trimmed_width_is_ceil_of_bucket_blocks_capped_by_row():
    assert decode_ctx_num_blocks(128, 16, 24) == 8
    assert decode_ctx_num_blocks(16384, 128, 512) == 128
    assert decode_ctx_num_blocks(256, 96, 40) == 3  # not a multiple of the block size
    assert decode_ctx_num_blocks(4096, 16, 24) == 24


def _state(widths_blocks):
    rs = NeuronRequestState(
        max_num_reqs=4,
        max_model_len=320,
        vocab_size=128,
        device=torch.device("cpu"),
        block_sizes=[16],
    )
    rs.configure_block_sizes([16])
    table = rs.block_table(0)
    for slot, n in enumerate(widths_blocks):
        rs._append_block_row(
            slot, 0, list(range(100 * (slot + 1), 100 * (slot + 1) + n))
        )
    return rs, table


def test_kernel_rows_head_trim_keeps_prefix_and_sentinels():
    rs, table = _state([3, 12, 0])
    assert table.shape[1] == 24  # ceil(320 / 16) = 20, aligned to 128 / 16
    full = build_kernel_block_rows(rs, 0, np.array([1, 0], dtype=np.int32), 4)
    trimmed = build_kernel_block_rows(
        rs, 0, np.array([1, 0], dtype=np.int32), 4, num_columns=8
    )
    assert full.shape == (4, 24) and trimmed.shape == (4, 8)
    assert trimmed.dtype == torch.int32 and trimmed.is_contiguous()
    # Head trim: the first columns of each row, as in the full table.
    assert torch.equal(trimmed, full[:, :8])
    # A row holding more blocks than the trim keeps only the first 8 ...
    assert trimmed[0].tolist() == list(range(200, 208))
    # ... a shorter row keeps its sentinel tail, padded rows are all sentinel.
    assert trimmed[1].tolist() == [100, 101, 102] + [PAD_SLOT_ID] * 5
    assert (trimmed[2:] == PAD_SLOT_ID).all()
    with pytest.raises(AssertionError, match="trimmed width"):
        build_kernel_block_rows(rs, 0, np.array([0], dtype=np.int32), 1, num_columns=25)


def _verify(table, num_computed, block_size, max_model_len, k=2):
    rows = table.shape[0]
    meta = {"block_table_tensor": table, "block_size": block_size, "kv_group": 0}
    state_meta = {
        "block_table_tensor": torch.ones(rows, 1 + k, dtype=torch.int32),
        "block_size": max_model_len,
        "kv_group": 1,
        "state_in": torch.zeros(rows, dtype=torch.int32),
    }
    _, _, positions, md = _build_verify_inputs(
        attn_metadata={"attn": meta, "state": state_meta},
        row_to_slot=torch.arange(rows),
        input_valid_mask=torch.ones(rows, dtype=torch.bool),
        num_draft_by_row=torch.full((rows,), k, dtype=torch.int32),
        decode_token_by_slot=torch.zeros(rows, 1, dtype=torch.int32),
        draft_tokens=torch.zeros(rows, k, dtype=torch.int32),
        num_computed_tokens=torch.tensor(num_computed, dtype=torch.int32),
        num_accepted=torch.ones(rows, dtype=torch.int32),
        max_model_len=max_model_len,
        token_dtype=torch.int32,
    )
    return positions.view(rows, 1 + k), md


def test_verify_slot_mapping_through_a_trimmed_table():
    block, mml = 4, 32
    # Two rows trimmed to a ctx bucket of 12 tokens (3 blocks of a 8-block row).
    table = torch.tensor([[7, 3, 9], [5, 6, -1]], dtype=torch.int32)
    positions, md = _verify(table, [8, 3], block, mml)
    assert positions.tolist() == [[8, 9, 10], [3, 4, 5]]
    slots = md["attn"]["slot_mapping"].view(2, 3).tolist()
    assert slots == [
        [9 * 4 + 0, 9 * 4 + 1, 9 * 4 + 2],
        [5 * 4 + 3, 6 * 4 + 0, 6 * 4 + 1],
    ]
    # The state group's metadata is not rebuilt from positions.
    assert "slot_mapping" not in md["state"]
    assert md["state"]["block_table_tensor"].shape == (2, 3)


def test_verify_position_past_a_trimmed_table_writes_the_null_block():
    # The runner never selects a bucket that leaves a position past the table; if it
    # did, the write must not land in the table's last block.
    block, mml = 4, 32
    table = torch.tensor([[7, 3, 9]], dtype=torch.int32)
    _, md = _verify(table, [10], block, mml)  # positions 10, 11, 12; table ends at 12
    assert md["attn"]["slot_mapping"].tolist() == [9 * 4 + 2, 9 * 4 + 3, NULL_SLOT]


@pytest.mark.parametrize(
    "buckets, targets",
    [
        (None, [(1, 65536), (8, 65536)]),
        ([16384], [(1, 16384), (1, 65536), (8, 16384), (8, 65536)]),
    ],
)
def test_worker_builds_one_v2_decode_job_per_row_and_ctx_bucket(buckets, targets):
    from types import SimpleNamespace

    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    built = []
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        max_model_len=65536,
        neuron_config=SimpleNamespace(
            num_seqs_buckets=[1, 8], decode_context_length_buckets=buckets
        ),
        _build_decode_synthetic_inputs=lambda rows, device, ctx_bucket: (
            built.append((rows, ctx_bucket)) or (rows, ctx_bucket)
        ),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        _build_trace_job=lambda model_input, **kw: model_input,
    )
    assert NeuronWorker._decode_compile_targets(worker) == targets
    jobs = NeuronWorker._build_decode_trace_jobs(
        worker, targets, native_compile_only=True
    )
    # Largest first, and the context bucket reaches the runner's synthetic input.
    assert jobs == sorted(targets, reverse=True) == built


def test_verify_untrimmed_table_is_unchanged():
    # Without a trim (table covers max_model_len), positions clamp to max_model_len - 1
    # and the guard is not traced.
    block, mml = 4, 12
    table = torch.tensor([[7, 3, 9]], dtype=torch.int32)
    _, md = _verify(table, [10], block, mml)
    assert md["attn"]["slot_mapping"].tolist() == [9 * 4 + 2, 9 * 4 + 3, 9 * 4 + 3]
