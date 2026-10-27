# SPDX-License-Identifier: Apache-2.0
"""Runner-independent recurrent-state helpers (vllm_neuron.vllm.worker.hybrid_state), CPU.

Shapes are Qwen3.8-27B at TP=4 (48 GDN layers with conv [3, 2560] + ssm [12, 128, 128] float32
state, 16 attention layers with one 256-wide KV head per rank). The KV cache groups are built
by vLLM's own get_kv_cache_groups, as the engine core does.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_neuron.model.kv_cache import StateLayerSpec
from vllm_neuron.vllm.worker import hybrid_state

KERNEL_BLOCK = 32
MAX_MODEL_LEN = 32768
ATTN_PAGE = 2 * KERNEL_BLOCK * 256 * 2  # K + V, one head of 256, bf16
STATE_BYTES = 3 * 2560 * 4 + 12 * 128 * 128 * 4


def _attn_specs(dtype=torch.bfloat16):
    return {
        f"layers.{i}.self_attn": FullAttentionSpec(
            block_size=KERNEL_BLOCK, num_kv_heads=1, head_size=256, dtype=dtype
        )
        for i in range(3, 64, 4)
    }


def _state_layers():
    return [
        StateLayerSpec(
            name=f"layers.{i}.linear_attn",
            shapes=((3, 2560), (12, 128, 128)),
            dtypes=(torch.float32, torch.float32),
        )
        for i in range(64)
        if i % 4 != 3
    ]


def _specs(placeholder_pages, *, dedicated=True, prefix_caching=False, k=0):
    return hybrid_state.build_state_cache_specs(
        _state_layers(),
        _attn_specs(),
        kernel_block=KERNEL_BLOCK,
        max_model_len=MAX_MODEL_LEN,
        prefix_caching=prefix_caching,
        dedicated=dedicated,
        num_spec_blocks=k,
        placeholder_pages=placeholder_pages,
    )


def _groups(state_specs):
    vllm_config = VllmConfig()
    vllm_config.cache_config.kv_cache_layout = "LBNHC"  # Neuron's only layout
    specs = dict(_attn_specs())
    specs.update(state_specs)
    return [
        (type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size, len(g.layer_names))
        for g in get_kv_cache_groups(vllm_config, specs)
    ]


def test_state_page_align_pages():
    assert hybrid_state.state_page_align_pages(32) == 8
    assert hybrid_state.state_page_align_pages(128) == 2
    assert hybrid_state.state_page_align_pages(256) == 1
    assert hybrid_state.state_page_align_pages(96) == 1  # does not divide 256
    assert hybrid_state.state_page_align_pages(512) == 1


def test_real_page_is_padded_power_of_two_attention_pages():
    _, real = _specs(placeholder_pages=1)
    assert set(real) == {layer.name for layer in _state_layers()}
    layout, page = real["layers.0.linear_attn"]
    assert layout.page_bytes == STATE_BYTES
    # 817152 bytes -> whole 256-token blocks (8 pages) -> 32 pages, a power of two.
    assert page == 32 * ATTN_PAGE
    assert page >= STATE_BYTES


def test_v1_placeholder_is_one_256_token_block():
    # V1's behavior: the placeholder page is one 256-token attention block, so vLLM enlarges
    # the attention block to 256 tokens (kernel blocks split it back to 32).
    specs, real = _specs(hybrid_state.state_page_align_pages(KERNEL_BLOCK))
    spec = specs["layers.0.linear_attn"]
    assert isinstance(spec, MambaSpec)
    assert spec.page_size_bytes == 8 * ATTN_PAGE
    assert spec.mamba_cache_mode == "none" and spec.block_size == MAX_MODEL_LEN
    assert spec.num_speculative_blocks == 0
    assert real["layers.0.linear_attn"][1] == 32 * ATTN_PAGE
    assert sorted(_groups(specs)) == sorted(
        [("FullAttentionSpec", 256, 16)] + [("MambaSpec", MAX_MODEL_LEN, 16)] * 3
    )


def test_v2_placeholder_keeps_attention_block_size():
    # V2's 1-page placeholder: no enlargement, so no manager -> kernel block split.
    specs, _ = _specs(placeholder_pages=1)
    assert specs["layers.0.linear_attn"].page_size_bytes == ATTN_PAGE
    assert sorted(_groups(specs)) == sorted(
        [("FullAttentionSpec", KERNEL_BLOCK, 16)]
        + [("MambaSpec", MAX_MODEL_LEN, 16)] * 3
    )


def test_shared_pool_reports_real_page():
    specs, real = _specs(placeholder_pages=1, dedicated=False)
    assert real == {}
    spec = specs["layers.0.linear_attn"]
    assert spec.page_size_bytes == 32 * ATTN_PAGE
    assert spec.shapes == ((3, 2560), (12, 128, 128))


def test_prefix_caching_uses_align_mode_with_enlarged_block():
    specs, _ = _specs(placeholder_pages=1, dedicated=False, prefix_caching=True, k=0)
    spec = specs["layers.0.linear_attn"]
    assert spec.mamba_cache_mode == "align"
    assert spec.block_size == 32 * KERNEL_BLOCK


def test_spec_blocks_are_carried():
    specs, _ = _specs(placeholder_pages=1, k=2)
    assert specs["layers.0.linear_attn"].num_speculative_blocks == 2


@pytest.mark.parametrize("k", [0, 1, 2])
def test_state_pool_pages(k):
    spec = MambaSpec(
        block_size=MAX_MODEL_LEN,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
        num_speculative_blocks=k,
    )
    assert hybrid_state.state_pool_pages(spec, 4, dedicated=True) == 1 + 4 * (1 + k)
    assert hybrid_state.state_pool_pages(spec, 4, dedicated=False) == 0
    align = MambaSpec(
        block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"
    )
    assert hybrid_state.state_pool_pages(align, 4, dedicated=True) == 0


def test_state_page_prefers_recorded_real_page():
    specs, real = _specs(placeholder_pages=1)
    name = "layers.0.linear_attn"
    assert hybrid_state.state_page(name, specs[name], real) == real[name]
    shared, _ = _specs(placeholder_pages=1, dedicated=False)
    layout, page = hybrid_state.state_page(name, shared[name], {})
    assert page == shared[name].page_size_bytes
    assert layout.page_bytes == STATE_BYTES


def test_allocate_state_pool_shapes_and_validation():
    _, real = _specs(placeholder_pages=1)
    pools = hybrid_state.allocate_state_pool(real, 5, torch.device("cpu"))
    assert set(pools) == set(real)
    a, b = pools["layers.0.linear_attn"]
    assert a.shape == b.shape == (5, 32 * ATTN_PAGE // 8)
    assert a.dtype == torch.float32 and a.data_ptr() != b.data_ptr()
    assert not a.any()
    layout, _ = real["layers.0.linear_attn"]
    with pytest.raises(ValueError, match="does not fit"):
        hybrid_state.allocate_state_pool({"x": (layout, 1024)}, 2, torch.device("cpu"))


def test_pool_pages_for_slot():
    row = hybrid_state.pool_pages_for_slot(2, 3, 5)
    assert row.dtype == np.int32
    assert row.tolist() == [7, 8, 9, 0, 0]
    assert hybrid_state.pool_pages_for_slot(0, 1, 1).tolist() == [1]
    # Narrower than the slot's pages: only the leading pages.
    assert hybrid_state.pool_pages_for_slot(1, 3, 2).tolist() == [4, 5]


def test_kv_block_clearer_chunks_and_pads_with_null_block():
    caches = [torch.ones(24, 1, 2, 4), torch.full((24, 1, 2, 4), 2.0)]
    calls = []

    def zero_fn(c, blocks):
        calls.append(blocks.tolist())
        return hybrid_state.zero_kv_blocks(c, blocks)

    clearer = hybrid_state.KVBlockClearer(
        caches, 1, torch.device("cpu"), zero_fn=zero_fn
    )
    clearer.clear([0, 11, 3, 3] + list(range(12, 20)))
    # 0 and duplicates dropped, sorted, chunks of KV_CLEAR_CHUNK padded with block 0.
    assert calls == [[3, 11, 12, 13, 14, 15, 16, 17], [18, 19, 0, 0, 0, 0, 0, 0]]
    for cache, value in zip(caches, (1.0, 2.0)):
        cleared = [0, 3] + list(range(11, 20))
        assert not cache[cleared].any()
        kept = [b for b in range(24) if b not in cleared]
        assert bool((cache[kept] == value).all())
    calls.clear()
    clearer.clear(None)
    clearer.clear([0])
    assert calls == []
    clearer.warm_up()
    assert calls == [[0] * hybrid_state.KV_CLEAR_CHUNK]


def test_kv_block_clearer_expands_manager_blocks_to_kernel_blocks():
    cache = torch.ones(32, 1, 2, 4)
    clearer = hybrid_state.KVBlockClearer(
        [cache], 4, torch.device("cpu"), zero_fn=hybrid_state.zero_kv_blocks
    )
    clearer.clear([2])
    assert not cache[8:12].any() and not cache[0:4].any()
    assert bool((cache[4:8] == 1).all()) and bool((cache[12:] == 1).all())
