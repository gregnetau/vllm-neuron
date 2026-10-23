# SPDX-License-Identifier: Apache-2.0
"""Per-step device-tensor reuse in the V1 runner (CPU): sampling params, block-table clones,
bucketed block tables and dedicated state-pool tables.

The runner is built without ``__init__`` (``object.__new__``) and given only the attributes
these paths read. Each cached result must equal what the uncached code builds, and must be
rebuilt as soon as an input changes.
"""

from types import SimpleNamespace

import pytest
import torch

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")
NeuronModelRunner = runner_mod.NeuronModelRunner


def make_runner():
    runner = object.__new__(NeuronModelRunner)
    runner.device = torch.device("cpu")
    runner.speculative_config = None
    return runner


# --- sampling params -------------------------------------------------------------------------


def metadata(top_k=None, top_p=None, temperature=None):
    return SimpleNamespace(top_k=top_k, top_p=top_p, temperature=temperature)


def attn(rows):
    return {"layer": {"block_table_tensor": torch.zeros(rows, 4, dtype=torch.int32)}}


def test_sampling_params_reused_while_metadata_unchanged():
    runner = make_runner()
    meta = metadata(
        top_k=torch.tensor([5, 7], dtype=torch.int32),
        temperature=torch.tensor([0.5, 1.0]),
    )
    runner.input_batch = SimpleNamespace(sampling_metadata=meta, num_reqs=2)
    first = runner._sampling_params_device_tensor(attn(4), None)
    expected = torch.tensor(
        [[5.0, 1.0, 0.5], [7.0, 1.0, 1.0], [-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    assert torch.equal(first, expected)
    assert runner._sampling_params_device_tensor(attn(4), None) is first

    # A new metadata object (refresh_metadata after a batch change) rebuilds.
    runner.input_batch.sampling_metadata = metadata()
    second = runner._sampling_params_device_tensor(attn(4), None)
    assert second is not first
    assert torch.equal(second, torch.tensor([[-1.0, 1.0, 0.0]] * 4)[:4])

    # So does a different bucket or request count.
    third = runner._sampling_params_device_tensor(attn(8), None)
    assert third.shape == (8, 3)
    runner.input_batch.num_reqs = 1
    fourth = runner._sampling_params_device_tensor(attn(8), None)
    assert fourth is not third and fourth.shape == (8, 3)


def test_sampling_params_without_attention_metadata():
    runner = make_runner()
    runner.input_batch = SimpleNamespace(sampling_metadata=metadata(), num_reqs=3)
    out = runner._sampling_params_device_tensor(None, None)
    assert out.shape == (3, 3)


# --- block-table clones ----------------------------------------------------------------------


def test_full_block_table_clone_follows_the_source_tensor():
    runner = make_runner()
    gpu = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    clone = runner._full_block_table_clone(0, gpu, gpu[:2])
    assert torch.equal(clone, gpu[:2]) and clone.data_ptr() != gpu.data_ptr()
    assert runner._full_block_table_clone(0, gpu, gpu[:2]) is clone
    # Another group gets its own tensor (distinct graph inputs).
    other = runner._full_block_table_clone(1, gpu, gpu[:2])
    assert other is not clone and torch.equal(other, clone)
    # A different row count or a new source tensor rebuilds.
    assert runner._full_block_table_clone(0, gpu, gpu[:4]).shape == (4, 4)
    gpu2 = gpu + 100
    assert torch.equal(runner._full_block_table_clone(0, gpu2, gpu2[:4]), gpu2[:4])


def test_bucketed_block_table_follows_the_source_tensor():
    runner = make_runner()
    gpu = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    idx = torch.arange(3)
    out = runner._bucketed_block_table(0, gpu, gpu[:2], idx)
    assert torch.equal(out, gpu[:2, :3])
    assert runner._bucketed_block_table(0, gpu, gpu[:2], idx) is out
    assert runner._bucketed_block_table(0, gpu, gpu[:2], torch.arange(5)).shape == (
        2,
        5,
    )
    gpu2 = gpu * 2
    assert torch.equal(
        runner._bucketed_block_table(0, gpu2, gpu2[:2], idx), gpu2[:2, :3]
    )


# --- state-pool tables -----------------------------------------------------------------------


def uncached_state_pool_table(req_ids, slots, padded, n_blocks):
    """Reference table, built row by row with torch."""
    table = torch.full((padded, n_blocks), -1, dtype=torch.int32)
    for i, req_id in enumerate(req_ids[:padded]):
        if req_id is None:
            continue
        table[i] = (
            1 + slots[req_id] * n_blocks + torch.arange(n_blocks, dtype=torch.int32)
        )
    return table


def make_pool_runner(req_ids, max_reqs=4):
    runner = make_runner()
    runner.input_batch = SimpleNamespace(req_ids=list(req_ids))
    runner.requests = {r: object() for r in req_ids if r is not None}
    runner._state_pool_slots = {}
    runner._state_pool_free = list(range(max_reqs))
    return runner


def test_state_pool_table_matches_and_is_reused_per_group():
    runner = make_pool_runner(["a", "b", None])
    table, full = runner._state_pool_table(4, 2, group_id=3)
    slots = dict(runner._state_pool_slots)
    assert slots == {"a": 0, "b": 1}
    expected = uncached_state_pool_table(["a", "b", None], slots, 4, 2)
    assert torch.equal(table, expected) and torch.equal(full, expected)
    assert table.data_ptr() != full.data_ptr()
    again = runner._state_pool_table(4, 2, group_id=3)
    assert again[0] is table and again[1] is full
    # Another group: same contents, its own tensors.
    other = runner._state_pool_table(4, 2, group_id=4)
    assert other[0] is not table and torch.equal(other[0], table)


def test_state_pool_table_rebuilds_when_slots_change():
    runner = make_pool_runner(["a", "b"])
    table, _ = runner._state_pool_table(2, 1, group_id=0)
    # "a" finishes (its state is dropped) and "c" takes the next free slot in row 0.
    del runner.requests["a"]
    runner.requests["c"] = object()
    runner.input_batch.req_ids = ["c", "b"]
    new, _ = runner._state_pool_table(2, 1, group_id=0)
    slots = runner._state_pool_slots
    assert "a" not in slots and slots["c"] == 2
    assert torch.equal(new, uncached_state_pool_table(["c", "b"], slots, 2, 1))
    assert not torch.equal(new, table)


def test_state_pool_table_exhausted_raises():
    runner = make_pool_runner(["a", "b"], max_reqs=1)
    with pytest.raises(RuntimeError, match="exhausted"):
        runner._state_pool_table(2, 1, group_id=0)


# --- guard evaluation stance -----------------------------------------------------------------


@pytest.mark.parametrize("env, skip", [(None, False), ("0", False), ("1", True)])
def test_model_forward_context_guard_stance(monkeypatch, env, skip):
    neuron_utils = pytest.importorskip("vllm_neuron.utils.neuron_utils")
    if env is None:
        monkeypatch.delenv("VLLM_NEURON_SKIP_GUARD_EVAL", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_SKIP_GUARD_EVAL", env)
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    cfg = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
    ctx = neuron_utils.model_forward_context(cfg)
    assert ctx.stance.stance == "fail_on_recompile"
    assert ctx.stance.skip_guard_eval_unsafe is skip
    with ctx:
        pass
