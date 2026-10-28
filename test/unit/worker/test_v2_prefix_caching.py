# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 prefix caching for recurrent-state models (Mamba "align"), on CPU.

The pieces below the engine: the checkpoint copy functions and their device-side
boundary test, the host candidates, the request state's manager -> kernel block split,
and the runner's specs, KV-cache layout and checkpoint bookkeeping with vLLM's own
KV-cache grouping. End-to-end runs against HF are in
test/unit/model/qwen3_5/test_v2_prefix_caching_e2e.py.
"""

import dataclasses

import pytest
import torch

pytest.importorskip("vllm")

import test_v2_recurrent_state as base
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_neuron.vllm.worker import hybrid_state
from vllm_neuron.vllm.worker.neuron_forward_adapter import build_attention_metadata
from vllm_neuron.vllm.worker.neuron_request_state import NeuronRequestState

# base: one bf16 KV head of 128, block 32; the real state page is 16 attention pages.
RATIO = base.REAL_PAGE // base.ATTN_PAGE
CKPT_BLOCK = RATIO * base.BLOCK  # 512 tokens


# ---------------------------------------------------------------- host candidates


def test_checkpoint_boundaries():
    cb = hybrid_state.checkpoint_boundaries
    assert cb(13, 3, 16) == [16]  # verify of positions 13 .. 15
    assert cb(14, 3, 16) == [16]  # the boundary at the step's last token + 1
    assert cb(16, 3, 16) == []
    assert cb(32, 16, 16) == [48]  # a chunk ending on a boundary
    assert cb(0, 40, 16) == [16, 32]  # all boundaries of a window
    # Async slack: the device count may be up to `slack` below the host count.
    assert cb(17, 3, 16, slack=2) == [16]
    assert cb(17, 3, 16) == []


# ---------------------------------------------------------------- copy functions


def _pools(pages=7, blocks=6, rows=2):
    """One state group, one layer, two halves: dedicated pool [pages, rows, 128] with
    page p filled with p, shared blocks [blocks, rows, 128] with block b filled with
    100 + b."""
    pool = [
        torch.arange(pages, dtype=torch.float32).view(-1, 1, 1).repeat(1, rows, 128)
        + 0.5 * h
        for h in range(2)
    ]
    ckpt = [
        (100 + torch.arange(blocks, dtype=torch.float32))
        .view(-1, 1, 1)
        .repeat(1, rows, 128)
        + 0.5 * h
        for h in range(2)
    ]
    return pool, ckpt


def _i(*values):
    return torch.tensor(values, dtype=torch.int64)


def test_save_copies_the_page_of_the_boundary_token():
    pages_per_slot = 3  # k = 2
    pool, ckpt = _pools()
    num_computed = torch.tensor([0, 49, 0], dtype=torch.int32)
    num_accepted = torch.tensor([1, 3, 1], dtype=torch.int32)
    # Slot 1 verified positions 46 .. 48 and accepted all three: the state after 48
    # tokens is that after its second token, page 1 + 1 * 3 + 1 = 5.
    reached = hybrid_state.save_state_checkpoints(
        pool,
        ckpt,
        [0, 0],
        _i(1),
        _i(48),
        _i(4).view(1, 1),
        torch.tensor([True]),
        num_computed,
        num_accepted,
        pages_per_slot,
    )
    assert int(reached) == 1
    assert bool((ckpt[0][4] == 5).all()) and bool((ckpt[1][4] == 5.5).all())
    # Other blocks, the null block included, are unchanged.
    for b in (0, 1, 2, 3, 5):
        assert bool((ckpt[0][b] == 100 + b).all())


@pytest.mark.parametrize(
    "num_computed, num_accepted, boundary, valid",
    [
        (47, 1, 48, True),  # accepted one token: 47 computed, the boundary not reached
        (52, 3, 48, True),  # the boundary was before the step (49 computed)
        (49, 3, 64, True),  # past the accepted tokens
        (49, 3, 48, False),  # padding row
    ],
)
def test_save_skips_rows_that_do_not_reach_the_boundary(
    num_computed, num_accepted, boundary, valid
):
    pool, ckpt = _pools()
    before = [c.clone() for c in ckpt]
    hybrid_state.save_state_checkpoints(
        pool,
        ckpt,
        [0, 0],
        _i(1),
        _i(boundary),
        _i(4).view(1, 1),
        torch.tensor([valid]),
        torch.tensor([0, num_computed], dtype=torch.int32),
        torch.tensor([1, num_accepted], dtype=torch.int32),
        3,
    )
    for c, b in zip(ckpt, before):
        assert torch.equal(c, b)


def test_save_with_a_null_destination_is_a_no_op():
    pool, ckpt = _pools()
    before = [c.clone() for c in ckpt]
    hybrid_state.save_state_checkpoints(
        pool,
        ckpt,
        [0, 0],
        _i(1),
        _i(32),
        _i(0).view(1, 1),
        torch.tensor([True]),
        torch.tensor([0, 32], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        1,
    )
    for c, b in zip(ckpt, before):
        assert torch.equal(c, b)


def test_restore_copies_checkpoints_and_skips_padding():
    pool, ckpt = _pools()
    before = [p.clone() for p in pool]
    hybrid_state.restore_state_checkpoints(
        ckpt, pool, [0, 0], _i(3, 0).view(1, 2), _i(4, 0), torch.tensor([True, False])
    )
    assert bool((pool[0][4] == 103).all()) and bool((pool[1][4] == 103.5).all())
    for p, b in zip(pool, before):
        assert torch.equal(p[:4], b[:4]) and torch.equal(p[5:], b[5:])


def test_checkpointer_entries_and_groups():
    pool_a, ckpt_a = _pools()
    pool_b, _ = _pools()
    # Layers of two state groups; both groups' layers on one shared buffer.
    checkpointer = hybrid_state.StateCheckpointer(
        {"l0": pool_a, "l1": pool_b},
        {"l0": ckpt_a, "l1": ckpt_a},
        [(0, ["l0"]), (2, ["l1"])],
        pages_per_slot=1,
        device=torch.device("cpu"),
        compiled=False,
    )
    assert checkpointer.group_ids == [0, 2]
    assert checkpointer.entry_groups == [0, 0, 1, 1]
    pool_b[0][3] = 42.0
    num_computed = torch.tensor([0, 0, 32], dtype=torch.int32)
    ones = torch.ones(3, dtype=torch.int32)
    # Slot 2 ended a chunk at 32: page 3 of each layer to its group's block.
    checkpointer.save([(2, 32, [1, 5])], num_computed, ones)
    assert bool((ckpt_a[0][1] == 3).all()) and bool((ckpt_a[0][5] == 42).all())
    checkpointer.restore([(0, [5, 1])])  # group 0 from block 5, group 1 from block 1
    assert bool((pool_a[0][1] == 42).all()) and bool((pool_b[0][1] == 3).all())
    assert (checkpointer.num_saves, checkpointer.num_restores) == (1, 1)
    snapshot = [t.clone() for t in pool_a + pool_b + ckpt_a]
    checkpointer.warm_up(num_computed, ones)  # padding only: no change
    for t, s in zip(pool_a + pool_b + ckpt_a, snapshot):
        assert torch.equal(t, s)


def test_shared_state_views_share_objects_per_buffer():
    raw = torch.zeros(4 * 2048, dtype=torch.int8)
    views = hybrid_state.shared_state_views(
        {"a": raw, "b": raw}, {"a": 2048, "b": 2048}
    )
    assert views["a"] is views["b"]
    half_a, half_b = views["a"]
    assert half_a.shape == (4, 2, 128)
    half_b[1].fill_(1.0)
    # Block 1 of the second half: bytes [P/2 * (4 + 1), P/2 * (4 + 2)) of the buffer.
    assert raw.view(torch.float32)[256 * 5 : 256 * 6].eq(1).all()


# ------------------------------------------------------------- kernel block split


def test_request_state_splits_manager_blocks():
    rs = NeuronRequestState(
        max_num_reqs=2,
        max_model_len=64,
        vocab_size=10,
        device=torch.device("cpu"),
        block_sizes=[16, 4],
    )
    rs.configure_block_sizes([16, 4], min_widths=[5, 16], kernel_ratios=[1, 4])
    rs.add_request(
        req_id="a",
        prompt_len=20,
        prefill_token_ids=list(range(20)),
        num_computed_tokens=0,
        max_tokens=4,
        block_ids=([0, 7], [3, 9]),
    )
    slot = rs.req_id_to_index["a"]
    assert rs.block_table(0)[slot, :2].tolist() == [0, 7]
    assert rs.block_table(1)[slot, :8].tolist() == [12, 13, 14, 15, 36, 37, 38, 39]
    assert rs.num_blocks(1, slot) == 8
    rs.update_cached_request("a", 16, ([], [2]))
    assert rs.block_table(1)[slot, 8:12].tolist() == [8, 9, 10, 11]
    with pytest.raises(AssertionError, match="overflow"):
        rs.update_cached_request("a", 16, ([], list(range(4, 10))))  # 32 wide


# ----------------------------------------------------------------- the runner


def _apc_runner(spec_k=0):
    runner = base._runner()
    runner.vllm_config.cache_config.enable_prefix_caching = True
    runner.vllm_config.cache_config.mamba_cache_mode = "align"
    runner._prefix_caching = True
    runner.num_speculative_tokens = spec_k
    runner.use_async_scheduling = False
    runner.request_state = NeuronRequestState(
        max_num_reqs=base.MAX_REQS,
        max_model_len=base.MAX_MODEL_LEN,
        vocab_size=1000,
        device=runner.device,
        block_sizes=[base.BLOCK],
        num_speculative_tokens=spec_k,
    )
    return runner


def _apc_config(runner):
    specs = runner.get_kv_cache_spec()
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    vllm_config = VllmConfig()
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    groups = get_kv_cache_groups(vllm_config, specs)
    return get_kv_cache_config_from_groups(
        vllm_config, groups, base.NUM_BLOCKS * base.REAL_PAGE * 2
    )


def test_apc_specs_carry_the_real_page_in_align_mode():
    runner = _apc_runner(spec_k=2)
    specs = runner.get_kv_cache_spec()
    state = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    assert len(state) == 6
    for name, spec in state.items():
        assert spec.mamba_cache_mode == "align" and spec.num_speculative_blocks == 2
        assert spec.page_size_bytes == base.REAL_PAGE
        assert spec.block_size == CKPT_BLOCK
        assert runner._state_real_pages[name][1] == base.REAL_PAGE
    # The pool holds the running state of every slot (with 1 + k pages).
    assert runner.state_pool_bytes() == 6 * (1 + base.MAX_REQS * 3) * base.REAL_PAGE
    # vLLM enlarges the attention block to the state page: one block size everywhere.
    config = _apc_config(runner)
    assert {g.kv_cache_spec.block_size for g in config.kv_cache_groups} == {CKPT_BLOCK}
    # The kernel block was recorded before the engine core changes the config.
    runner.vllm_config.cache_config.block_size = CKPT_BLOCK
    assert runner.get_kv_cache_spec() == specs


def test_apc_initialize_splits_attention_blocks_and_builds_checkpointer():
    runner = _apc_runner()
    config = _apc_config(runner)
    ratios = []
    build_clearer = runner._build_kv_block_clearer

    def record_clearer(kv_caches, ratio=1):
        ratios.append(ratio)
        return build_clearer(kv_caches, ratio)

    runner._build_kv_block_clearer = record_clearer
    runner.initialize_kv_cache(config)
    # The clearer zeroes every kernel block of a manager block (a partial clear left
    # stale bytes that changed FP8-KV outputs in the end-to-end test).
    assert ratios == [RATIO]
    model = runner.model
    num_blocks = config.num_blocks
    state, attn = base._state_and_attn_groups(runner)
    k, v = model.kv_caches["layers.3.self_attn"]
    assert k.shape == (num_blocks * RATIO, 1, base.BLOCK, 128)
    assert runner.block_sizes == [
        CKPT_BLOCK if g in state else base.BLOCK for g in range(len(runner.block_sizes))
    ]
    rs = runner.request_state
    for g in attn:
        assert rs.kernel_ratios[g] == RATIO
        assert rs.block_table(g).shape[1] == base.MAX_MODEL_LEN // base.BLOCK
    for g in state:
        assert rs.block_table(g).shape[1] == base.MAX_MODEL_LEN // CKPT_BLOCK + 1
    ckpt = runner._ckpt
    assert ckpt is not None and runner._ckpt_block_size == CKPT_BLOCK
    assert ckpt.group_ids == state and len(ckpt.pool) == 12
    # Checkpoint views alias the shared buffer at the attention layers' blocks; the
    # pool pages are separate.
    raw = k.untyped_storage().data_ptr()
    assert all(c.untyped_storage().data_ptr() == raw for c in ckpt.ckpt)
    assert all(p.untyped_storage().data_ptr() != raw for p in ckpt.pool)
    assert ckpt.ckpt[0].shape == (num_blocks, base.REAL_PAGE // 8 // 128, 128)
    # Manager block 2's checkpoint occupies the bytes of kernel blocks 2 * RATIO ..
    # of each K / V half.
    ckpt.ckpt[0][2].fill_(1.0)
    assert k[2 * RATIO : 3 * RATIO].float().abs().sum() > 0
    assert not k[: 2 * RATIO].any() and not v.any()


def test_apc_metadata_uses_kernel_blocks():
    runner = _apc_runner()
    runner.initialize_kv_cache(_apc_config(runner))
    state, (attn,) = base._state_and_attn_groups(runner)
    groups = runner._kv_cache_config.kv_cache_groups
    req = base._new_req(runner, "a", 600, state_block=10, attn_blocks=[20, 21])
    req.block_ids = tuple(
        [0, 30 + g] if g in state else [20, 21] for g in range(len(groups))
    )
    step = base._step(new=[req], scheduled={"a": 512})
    runner.update_states(step)
    model_input = runner._prepare_model_input(step)
    kwargs = runner._build_forward_adapter_kwargs(model_input)
    md = kwargs["attn_metadata"]
    attn_md = md["layers.3.self_attn"]
    assert attn_md["block_size"] == base.BLOCK
    # Manager blocks 20 and 21 as kernel blocks; the 512-token chunk touches only the
    # first manager block's, so the rest of the row is the -1 sentinel.
    row = attn_md["block_table_tensor"][0].tolist()
    assert row[:RATIO] == list(range(20 * RATIO, 21 * RATIO))
    assert set(row[RATIO:]) == {-1}
    assert runner.request_state.block_table(attn)[0, : 2 * RATIO].tolist() == list(
        range(20 * RATIO, 22 * RATIO)
    )
    gdn_md = md["layers.0.linear_attn"]
    assert gdn_md["block_size"] == CKPT_BLOCK
    # The state groups see their slot's pool page; their valid tokens are the
    # attention group's (the state table's first column is the null block).
    assert gdn_md["block_table_tensor"].tolist() == [[1]]
    assert torch.equal(gdn_md["slot_mapping"], attn_md["slot_mapping"])
    assert bool((gdn_md["slot_mapping"][:512] > 0).all())
    # Without block_sizes the metadata keeps the spec's block size (non-APC graphs).
    plain = build_attention_metadata(
        groups,
        model_input.block_tables,
        model_input.slot_mapping,
        max_query_len=512,
        slot_mappings=model_input.slot_mappings,
    )
    assert plain["layers.3.self_attn"]["block_size"] == CKPT_BLOCK


def test_apc_saves_and_restores_follow_the_scheduler():
    runner = _apc_runner()
    runner.initialize_kv_cache(_apc_config(runner))
    state, _ = base._state_and_attn_groups(runner)
    groups = runner._kv_cache_config.kv_cache_groups
    calls = []
    ckpt = runner._ckpt
    ckpt.save = lambda entries, nc, na: calls.append(("save", entries))
    ckpt.restore = lambda entries: calls.append(("restore", entries))

    def new(req_id, num_computed, ids):
        req = base._new_req(runner, req_id, 1100, state_block=0, attn_blocks=[])
        req.num_computed_tokens = num_computed
        req.block_ids = tuple(
            ids[g] if g in state else [40, 41, 42] for g in range(len(groups))
        )
        return req

    # A hit of one block: the state block at column 0 is restored to slot 0's page.
    req = new("a", CKPT_BLOCK, {g: [50 + g, 60 + g] for g in state})
    step = base._step(new=[req], scheduled={"a": CKPT_BLOCK})
    runner.update_states(step)
    runner._restore_prefix_checkpoints(step)
    assert calls == [("restore", [(0, [50 + g for g in state])])]
    # The chunk ends on the next boundary: save to column 1.
    model_input = runner._prepare_model_input(step)
    runner._save_prefix_checkpoints(step, model_input)
    assert calls[-1] == ("save", [(0, 2 * CKPT_BLOCK, [60 + g for g in state])])
    # A chunk not ending on a boundary saves nothing.
    calls.clear()
    step = base._step(cached=[("a", 2 * CKPT_BLOCK)], scheduled={"a": 64})
    runner.update_states(step)
    runner._save_prefix_checkpoints(step, runner._prepare_model_input(step))
    assert calls == []
    # A hit that is not on a block boundary is a contract violation.
    req = new("b", 100, {g: [70 + g] for g in state})
    step = base._step(new=[req], scheduled={"b": 64})
    runner.update_states(step)
    with pytest.raises(RuntimeError, match="checkpointed per"):
        runner._restore_prefix_checkpoints(step)


@pytest.mark.parametrize(
    "block_sizes, k, match",
    [
        ([512, 256], 0, "one block size"),
        ([500], 0, "multiple of the attention kernel"),
        ([32], 16, r"exceed 2 \* num_speculative_tokens"),
    ],
)
def test_apc_geometry_errors(block_sizes, k, match):
    runner = _apc_runner()
    runner.num_speculative_tokens = k
    runner._kernel_block_size = base.BLOCK
    with pytest.raises(ValueError, match=match):
        runner._prefix_cache_geometry(block_sizes)


def test_apc_rejects_state_specs_without_align_mode():
    runner = _apc_runner()
    specs = {
        n: dataclasses.replace(s, mamba_cache_mode="none")
        if isinstance(s, MambaSpec)
        else s
        for n, s in runner.get_kv_cache_spec().items()
    }
    with pytest.raises(NotImplementedError, match="'align'"):
        runner.initialize_kv_cache(_apc_config_from(specs))


def _apc_config_from(specs):
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    vllm_config = VllmConfig()
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    groups = get_kv_cache_groups(vllm_config, specs)
    return get_kv_cache_config_from_groups(
        vllm_config, groups, base.NUM_BLOCKS * base.REAL_PAGE * 2
    )


def test_non_apc_runner_has_no_checkpointer():
    runner = base._initialized_runner()
    assert runner._ckpt is None and not runner._prefix_caching
    assert all(r == 1 for r in runner.request_state.kernel_ratios)
    assert runner.block_sizes == [
        g.kv_cache_spec.block_size for g in runner._kv_cache_config.kv_cache_groups
    ]
