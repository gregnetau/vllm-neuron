# SPDX-License-Identifier: Apache-2.0
"""Emulate the device-layout math (permuted q/k, folded norms, fused GDN in_proj, TP shards)
on CPU and compare with the HF-semantics reference."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

TP = 2


@pytest.fixture(scope="module")
def mods(qwen3_5_config, weights):
    from conftest import ROOT, _load

    gdn = _load("vllm_neuron.model.qwen3_5.gdn_ops", ROOT / "model/qwen3_5/gdn_ops.py")
    ref = _load("vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py")
    return gdn, ref


@pytest.fixture(scope="module")
def cfg(qwen3_5_config):
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=96, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    return qwen3_5_config.Qwen3_5Config.from_configs(hf_cfg.to_dict())


def _device_attention(x, positions, per_rank, cfg, weights, ref):
    """Mirror of Qwen3_5Attention.forward_prefill on one rank (returns o_proj partial)."""
    L = x.shape[0]
    Hr, Hkvr, hd = cfg.num_attention_heads // TP, cfg.num_key_value_heads // TP, cfg.head_dim
    qkv = x @ per_rank["qkv"]
    q, k, v = torch.split(qkv, [Hr * hd, Hkvr * hd, Hkvr * hd], dim=-1)
    gate = x @ per_rank["gate"]
    q, k, v = q.view(L, Hr, hd), k.view(L, Hkvr, hd), v.view(L, Hkvr, hd)
    eps = cfg.rms_norm_eps
    norm = lambda t, gamma: (t.float() * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps) * gamma.float()).type_as(t)
    q, k = norm(q, per_rank["q_norm"]), norm(k, per_rank["k_norm"])
    cos, sin = weights.kernel_rope_angles(positions, hd, cfg.rotary_dim, cfg.rope_theta)
    q, k = ref.apply_rope(q, cos, sin), ref.apply_rope(k, cos, sin)
    rep = Hr // Hkvr
    scores = torch.einsum("lhd,thd->hlt", q, k.repeat_interleave(rep, 1)) * hd**-0.5
    causal = torch.arange(L)[None, :] <= positions[:, None]
    o = torch.einsum("hlt,thd->lhd", scores.masked_fill(~causal[None], float("-inf")).softmax(-1), v.repeat_interleave(rep, 1))
    o = o * torch.sigmoid(gate.view(L, Hr, hd))
    return o.reshape(L, Hr * hd) @ per_rank["o"]


def test_full_attention_device_layout_matches_reference(cfg, weights, mods):
    _, ref = mods
    torch.manual_seed(0)
    H, Hkv, hd, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    w = {
        "q_proj": torch.randn(H * 2 * hd, D) * 0.2, "k_proj": torch.randn(Hkv * hd, D) * 0.2,
        "v_proj": torch.randn(Hkv * hd, D) * 0.2, "o_proj": torch.randn(D, H * hd) * 0.2,
        "q_norm": torch.randn(hd) * 0.1, "k_norm": torch.randn(hd) * 0.1,
    }
    x, pos = torch.randn(9, D), torch.arange(9)
    expected, _ = ref.full_attention_reference(x, w, cfg, pos, None)

    total = 0
    for r in range(TP):
        per_rank = {
            "qkv": weights.attention_qkv_device(w["q_proj"], w["k_proj"], w["v_proj"], cfg, r, TP),
            "gate": weights.attention_gate_device(w["q_proj"], cfg, r, TP),
            "o": weights.attention_o_device(w["o_proj"], cfg, r, TP),
            "q_norm": weights.attention_qk_norm_device(w["q_norm"], cfg),
            "k_norm": weights.attention_qk_norm_device(w["k_norm"], cfg),
        }
        total = total + _device_attention(x, pos, per_rank, cfg, weights, ref)
    torch.testing.assert_close(total, expected, rtol=1e-4, atol=1e-5)


def test_gdn_device_layout_matches_reference_incl_decode(cfg, weights, mods):
    gdn, _ = mods
    torch.manual_seed(1)
    D, Hk, Hv, Dk, Dv = cfg.hidden_size, cfg.linear_num_key_heads, cfg.linear_num_value_heads, 16, 16
    w = {
        "in_proj_qkv": torch.randn(cfg.conv_dim, D) * 0.2, "in_proj_z": torch.randn(cfg.value_dim, D) * 0.2,
        "in_proj_b": torch.randn(Hv, D) * 0.2, "in_proj_a": torch.randn(Hv, D) * 0.2,
        "conv1d": torch.randn(cfg.conv_dim, 1, 4) * 0.5, "A_log": torch.log(torch.rand(Hv) * 4 + 0.1),
        "dt_bias": torch.randn(Hv) * 0.2, "norm": torch.rand(Dv) + 0.5,
        "out_proj": torch.randn(D, cfg.value_dim) * 0.2,
    }
    x = torch.randn(40, D)
    expected, conv_ref, ssm_ref = gdn.gdn_layer_reference(x[:39], w, cfg, None, None)
    exp_dec, _, _ = gdn.gdn_layer_reference(x[39:], w, cfg, conv_ref, ssm_ref, decode=True)

    Hkr, Hvr = Hk // TP, Hv // TP
    kd, vd = cfg.key_dim // TP, cfg.value_dim // TP
    pre_total, dec_total = 0, 0
    for r in range(TP):
        inp = weights.gdn_in_proj_device(w["in_proj_qkv"], w["in_proj_z"], w["in_proj_b"], w["in_proj_a"], cfg, r, TP)
        conv_w = weights.gdn_conv_device(w["conv1d"], cfg, r, TP)
        A_log, dt_bias = (weights.gdn_head_vector_device(w[n], cfg, r, TP) for n in ("A_log", "dt_bias"))
        out_w = weights.gdn_out_proj_device(w["out_proj"], cfg, r, TP)

        def split(proj):
            return torch.split(proj, [2 * kd + vd, vd, Hvr, Hvr], dim=-1)

        # prefill (39 tokens, single sequence) using chunk op
        mixed, z, b, a = split(x[:39] @ inp)
        mixed, cs = gdn.causal_conv1d(mixed, conv_w, None)
        q, k, v = torch.split(mixed, [kd, kd, vd], dim=-1)
        q, k = q.reshape(39, Hkr, Dk).repeat_interleave(Hvr // Hkr, 1), k.reshape(39, Hkr, Dk).repeat_interleave(Hvr // Hkr, 1)
        g, beta = gdn.gdn_gating(a, b, A_log, dt_bias)
        o, ss = gdn.chunk_gated_delta_rule(q, k, v.reshape(39, Hvr, Dv), g, beta, None)
        o = gdn.gated_rmsnorm(o, z.reshape(39, Hvr, Dv), w["norm"], cfg.rms_norm_eps).reshape(39, vd)
        pre_total = pre_total + o @ out_w

        # decode (1 token) using the batched step ops with B=1 and stored state
        mixed, z, b, a = split(x[39:] @ inp)
        mixed, cs = gdn.causal_conv1d_step_batched(mixed, conv_w, cs[None])
        q, k, v = torch.split(mixed, [kd, kd, vd], dim=-1)
        q, k = q.reshape(1, Hkr, Dk).repeat_interleave(Hvr // Hkr, 1), k.reshape(1, Hkr, Dk).repeat_interleave(Hvr // Hkr, 1)
        g, beta = gdn.gdn_gating(a, b, A_log, dt_bias)
        o, _ = gdn.gated_delta_step_batched(q, k, v.reshape(1, Hvr, Dv), g, beta, ss[None])
        o = gdn.gated_rmsnorm(o, z.reshape(1, Hvr, Dv), w["norm"], cfg.rms_norm_eps).reshape(1, vd)
        dec_total = dec_total + o @ out_w

    torch.testing.assert_close(pre_total, expected, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(dec_total, exp_dec, rtol=1e-3, atol=1e-4)
