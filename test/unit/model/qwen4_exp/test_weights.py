# SPDX-License-Identifier: Apache-2.0
"""TP sharding of the attention projections with replicated KV heads."""

import pytest
import torch

from qwen4_exp_testlib import real_text_config_dict


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_qkv_replicated_kv_heads(tp):
    from vllm_neuron.model.qwen3_5.weights import rotary_channel_permutation
    from vllm_neuron.model.qwen4_exp import weights as W
    from vllm_neuron.model.qwen4_exp.config import Qwen4ExpConfig

    cfg = Qwen4ExpConfig.from_configs(
        dict(
            real_text_config_dict(),
            hidden_size=16,
            head_dim=8,
            partial_rotary_factor=0.5,
        )
    )
    hd, H = cfg.head_dim, cfg.hidden_size
    q = torch.randn(cfg.num_attention_heads * 2 * hd, H)
    k = torch.randn(cfg.num_key_value_heads * hd, H)
    v = torch.randn(cfg.num_key_value_heads * hd, H)
    perm = rotary_channel_permutation(hd, cfg.rotary_dim)
    qh = cfg.num_attention_heads // tp
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    for r in range(tp):
        w = W.attention_qkv_device(q, k, v, cfg, r, tp).t()  # [rows, H]
        assert w.shape[0] == (qh + 2 * cfg.kv_heads_per_rank(tp)) * hd
        kq = w[qh * hd :].view(2, cfg.kv_heads_per_rank(tp), hd, H)
        # every Q head on this rank attends with the KV head(s) it holds
        for j in range(qh):
            kv_head = (r * qh + j) // group
            local = kv_head - cfg.kv_head_for_rank(r, tp)
            assert 0 <= local < cfg.kv_heads_per_rank(tp)
            torch.testing.assert_close(kq[0, local], k.view(-1, hd, H)[kv_head][perm])
            torch.testing.assert_close(kq[1, local], v.view(-1, hd, H)[kv_head])


def test_replicated_helpers():
    from vllm_neuron.model.qwen4_exp import weights as W

    w = torch.randn(6, 4)
    torch.testing.assert_close(W.linear_t(w), w.t())
    assert W.linear_t(w).is_contiguous()
    n = torch.randn(8, dtype=torch.bfloat16)
    g = W.folded_norm(n)
    assert g.dtype == torch.float32
    torch.testing.assert_close(g, 1.0 + n.float())
    c = torch.randn(12, 1, 4)
    torch.testing.assert_close(W.depthwise_conv(c), c[:, 0])
