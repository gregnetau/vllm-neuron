# SPDX-License-Identifier: Apache-2.0
import json
import pathlib

import pytest
import torch

RAW = json.loads(pathlib.Path(__file__).with_name("qwen3_8_27b_config.json").read_text())


@pytest.fixture(scope="module")
def cfg(qwen3_5_config):
    return qwen3_5_config.Qwen3_5Config.from_configs(RAW)


def test_real_model_state_sizes_tp4(state, state_cache, cfg):
    conv, ssm = state.gdn_state_shapes(cfg, 4)
    assert conv == (3, 2560) and ssm == (12, 128, 128)
    lay = state.gdn_page_layout(cfg, 4)
    assert lay.dtypes == (torch.float32, torch.float32)
    assert lay.offsets == (0, 30720)
    assert lay.page_bytes == 2560 * 3 * 4 + 12 * 128 * 128 * 4 == 817152
    # attention page at the plugin's default 32-token block: K+V, 1 KV head/rank, hd 256, bf16
    attn_page = 2 * 1 * 256 * 2 * 32
    padded = state_cache.padded_state_page_bytes(lay.page_bytes, attn_page)
    assert padded % attn_page == 0 and padded - lay.page_bytes < attn_page
    assert padded // attn_page == 25  # vLLM will scale attention block_size 32 -> 800
    # ~150 MiB of state per request across 4 ranks (48 GDN layers)
    total = 48 * lay.page_bytes * 4
    assert 140e6 < total < 160e6
    S, h = state_cache.state_half_elems(lay, padded)
    assert S == 204288 and h == 102400 and S <= 2 * h


def test_tp_must_divide(state, cfg):
    with pytest.raises(ValueError):
        state.gdn_state_shapes(cfg, 8)


def test_paired_halves_match_attention_block_bytes(state_cache):
    """State page i and attention block i share bytes; other indices never overlap."""
    N, page_bytes = 6, 4096  # attention: (2, N, heads=1, block=8, hd=128) bf16 -> 2*8*128*2 = 4096
    raw = torch.zeros(N * page_bytes, dtype=torch.int8)
    k, v = raw.view(torch.bfloat16).view(2, N, 1, 8, 128)
    a, b = state_cache.paired_half_views(raw, page_bytes)
    assert a.is_contiguous() and b.is_contiguous() and a.shape == (N, page_bytes // 8)
    a[3].fill_(1.0)
    b[3].fill_(2.0)
    for i in range(N):  # only attention block 3 sees the state page 3 bytes
        touched = bool(k[i].float().abs().sum() > 0) or bool(v[i].float().abs().sum() > 0)
        assert touched == (i == 3), i
    k[1].fill_(5.0)  # attention block 1 does not disturb state page 3
    assert torch.all(a[3] == 1.0) and torch.all(b[3] == 2.0)


def test_state_half_elems_rejects_overflow(state_cache):
    lay = state_cache.page_layout(((10,), (20,)), (torch.float32, torch.float32))
    with pytest.raises(ValueError):
        state_cache.state_half_elems(lay, 64)  # 30 elems > 2 halves of 8
    with pytest.raises(AssertionError):
        state_cache.state_half_elems(
            state_cache.page_layout(((4,),), (torch.bfloat16,)), 64)


def test_prefix_cache_state_block_tp4(state, state_cache, cfg):
    """With prefix caching the state block spans the tokens of one enlarged attention block:
    the runner pads the page to whole 256-token attention blocks (8 kernel blocks of 32)."""
    lay = state.gdn_page_layout(cfg, 4)
    attn_page = 2 * 1 * 256 * 2 * 32
    padded = state_cache.padded_state_page_bytes(lay.page_bytes, attn_page, align_pages=8)
    assert padded // attn_page * 32 == 1024
