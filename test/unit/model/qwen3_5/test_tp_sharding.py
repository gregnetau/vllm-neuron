# SPDX-License-Identifier: Apache-2.0
"""TP sharding rules: shards reassemble, and per-rank GDN partial outputs sum to the full output."""

import copy

import pytest
import torch

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

TP = 2
P = "model.language_model.layers.0."


@pytest.fixture(scope="module")
def tiny(qwen3_5_config):
    torch.manual_seed(1)
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=64, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        vocab_size=100,
    )
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(hf_cfg.to_dict())
    layer = hf.Qwen3_5GatedDeltaNet(hf_cfg, 0).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        layer.norm.weight.copy_(torch.rand_like(layer.norm.weight) + 0.5)
        layer.A_log.copy_(torch.log(torch.rand_like(layer.A_log) * 4 + 0.1))
    return cfg, layer


def _hf_named(layer):
    return {P + "linear_attn." + n: p.data for n, p in layer.named_parameters()}


def test_gdn_tp_partial_sums_equal_unsharded(gdn_ops, weights, tiny):
    cfg, layer = tiny
    named = _hf_named(layer)
    x = torch.randn(70, cfg.hidden_size)

    def w_of(rank, tp):
        g = lambda n: weights.shard_weight(P + "linear_attn." + n, named[P + "linear_attn." + n], cfg, rank, tp)
        return {
            "in_proj_qkv": g("in_proj_qkv.weight"), "in_proj_z": g("in_proj_z.weight"),
            "in_proj_b": g("in_proj_b.weight"), "in_proj_a": g("in_proj_a.weight"),
            "conv1d": g("conv1d.weight"), "A_log": g("A_log"), "dt_bias": g("dt_bias"),
            "norm": g("norm.weight"), "out_proj": g("out_proj.weight"),
        }

    full, _, _ = gdn_ops.gdn_layer_reference(x, w_of(0, 1), cfg, None, None)
    rank_cfg = copy.copy(cfg)
    rank_cfg.linear_num_key_heads //= TP
    rank_cfg.linear_num_value_heads //= TP
    parts = [gdn_ops.gdn_layer_reference(x, w_of(r, TP), rank_cfg, None, None)[0] for r in range(TP)]
    torch.testing.assert_close(sum(parts), full, rtol=1e-4, atol=1e-5)


def test_attention_qproj_keeps_query_gate_pairs(weights, tiny):
    cfg, _ = tiny
    heads, hd = cfg.num_attention_heads, cfg.head_dim
    # tag every row with (head, 0=query|1=gate)
    q = torch.arange(heads * 2 * hd).reshape(heads, 2, hd).float()
    q_proj = q.reshape(heads * 2 * hd, 1).expand(-1, cfg.hidden_size).contiguous()
    shard = weights.shard_weight(P + "self_attn.q_proj.weight", q_proj, cfg, 1, TP)
    assert shard.shape[0] == heads // TP * 2 * hd
    assert torch.equal(shard[:, 0], q[heads // TP :].reshape(-1))  # rank 1 owns heads 2..3, in [q|gate] order


@pytest.mark.parametrize("leaf,dim", [
    ("mlp.gate_proj.weight", 0), ("mlp.up_proj.weight", 0), ("mlp.down_proj.weight", 1),
    ("self_attn.k_proj.weight", 0), ("self_attn.o_proj.weight", 1),
    ("linear_attn.in_proj_qkv.weight", None), ("linear_attn.out_proj.weight", 1),
])
def test_shards_cover_tensor_exactly(weights, tiny, leaf, dim):
    cfg, _ = tiny
    shape = {
        "mlp.gate_proj.weight": (64, 48), "mlp.up_proj.weight": (64, 48), "mlp.down_proj.weight": (48, 64),
        "self_attn.k_proj.weight": (32, 48), "self_attn.o_proj.weight": (48, 64),
        "linear_attn.in_proj_qkv.weight": (cfg.conv_dim, 48), "linear_attn.out_proj.weight": (48, cfg.value_dim),
    }[leaf]
    t = torch.randn(shape)
    shards = [weights.shard_weight(P + leaf, t, cfg, r, TP) for r in range(TP)]
    if dim is None:  # segmented q|k|v: same multiset of rows, each rank gets 1/TP of every segment
        assert sum(s.numel() for s in shards) == t.numel()
        assert torch.equal(torch.cat(shards).sort(dim=0).values, t.sort(dim=0).values)
    else:
        assert torch.equal(torch.cat(shards, dim=dim), t)


def test_indivisible_raises(weights, tiny):
    cfg, _ = tiny
    with pytest.raises(ValueError):
        weights.shard_weight(P + "mlp.up_proj.weight", torch.randn(65, 48), cfg, 0, TP)


def test_partial_rotary_permutation_equivalent_to_full_width_rope(weights, qwen3_5_config):
    """Permuted channels + full-width rotate_half (cos=1/sin=0 on pass dims) == partial rotary."""
    from conftest import ROOT, _load

    gdn_ops = _load("vllm_neuron.model.qwen3_5.gdn_ops", ROOT / "model/qwen3_5/gdn_ops.py")
    ref = _load("vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py")
    D, R, L, H = 256, 64, 11, 3
    cfg = qwen3_5_config.Qwen3_5Config(head_dim=D, partial_rotary_factor=R / D, rope_theta=1e7)
    torch.manual_seed(0)
    q, k = torch.randn(L, H, D), torch.randn(L, H, D)
    pos = torch.arange(5, 5 + L)

    cos, sin = ref.rope_cos_sin(pos, cfg)
    q_ref, k_ref = ref.apply_rope(q, cos, sin), ref.apply_rope(k, cos, sin)
    scores_ref = torch.einsum("lhd,thd->hlt", q_ref, k_ref)

    perm = weights.rotary_channel_permutation(D, R)
    assert sorted(perm.tolist()) == list(range(D))
    kcos, ksin = weights.kernel_rope_angles(pos, D, R, cfg.rope_theta)
    q_k, k_k = ref.apply_rope(q[..., perm], kcos, ksin), ref.apply_rope(k[..., perm], kcos, ksin)
    torch.testing.assert_close(torch.einsum("lhd,thd->hlt", q_k, k_k), scores_ref, rtol=1e-4, atol=1e-4)
    # the roped tensors are the reference ones, permuted
    torch.testing.assert_close(q_k, q_ref[..., perm], rtol=1e-5, atol=1e-5)

    # row-permuting q_proj (query part only) reproduces permuted queries; gate untouched
    hidden = torch.randn(L, 32)
    w = torch.randn(H * 2 * D, 32)
    qg = (hidden @ w.t()).view(L, H, 2 * D)
    wp = weights.permute_head_rows(w, H, D, perm, per_head_width=2 * D, offset=0)
    qgp = (hidden @ wp.t()).view(L, H, 2 * D)
    torch.testing.assert_close(qgp[..., :D], qg[..., :D][..., perm])
    torch.testing.assert_close(qgp[..., D:], qg[..., D:])


def test_fold_zero_centered_norm(weights, qwen3_5_config):
    from conftest import ROOT, _load

    ref = _load("vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py")
    x, w = torch.randn(4, 16), torch.randn(16) * 0.1
    gamma = weights.fold_zero_centered_norm(w)
    plain = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    torch.testing.assert_close(plain, ref.rmsnorm_zero_centered(x, w, 1e-6), rtol=1e-5, atol=1e-6)
