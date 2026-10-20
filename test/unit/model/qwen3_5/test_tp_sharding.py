# SPDX-License-Identifier: Apache-2.0
"""TP sharding of the model's weight loaders (``weights.*_device``): shards reassemble, and
per-rank GDN partial outputs sum to the full output."""

import copy

import pytest
import torch

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

TP = 2


@pytest.fixture(scope="module")
def tiny(qwen3_5_config):
    torch.manual_seed(1)
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
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
    return {n: p.data for n, p in layer.named_parameters()}


def test_gdn_tp_partial_sums_equal_unsharded(gdn_ops, weights, tiny):
    """Per-rank shards from the model's GDN weight loaders (device layout transposed back to
    HF "out x in") give partial outputs that sum to the unsharded layer output."""
    cfg, layer = tiny
    w = _hf_named(layer)
    x = torch.randn(70, cfg.hidden_size)

    def w_of(rank):
        heads = lambda n: weights.gdn_in_proj_heads_device(w[n], rank, TP).t()
        vec = lambda n: weights.gdn_head_vector_device(w[n], cfg, rank, TP)
        return {
            "in_proj_qkv": weights.gdn_in_proj_qkv_device(
                w["in_proj_qkv.weight"], cfg, rank, TP
            ).t(),
            "in_proj_z": heads("in_proj_z.weight"),
            "in_proj_b": heads("in_proj_b.weight"),
            "in_proj_a": heads("in_proj_a.weight"),
            "conv1d": weights.gdn_conv_device(w["conv1d.weight"], cfg, rank, TP),
            "A_log": vec("A_log"),
            "dt_bias": vec("dt_bias"),
            "norm": w["norm.weight"],
            "out_proj": weights.gdn_out_proj_device(
                w["out_proj.weight"], cfg, rank, TP
            ).t(),
        }

    full_w = {n.removesuffix(".weight"): t for n, t in w.items()}
    full, _, _ = gdn_ops.gdn_layer_reference(x, full_w, cfg, None, None)
    rank_cfg = copy.copy(cfg)
    rank_cfg.linear_num_key_heads //= TP
    rank_cfg.linear_num_value_heads //= TP
    parts = [
        gdn_ops.gdn_layer_reference(x, w_of(r), rank_cfg, None, None)[0]
        for r in range(TP)
    ]
    torch.testing.assert_close(sum(parts), full, rtol=1e-4, atol=1e-5)


def test_attention_shards_keep_query_gate_pairs(weights, tiny):
    """q_proj is [query | gate] per head: a rank's query (qkv) and gate columns come from the
    same heads, and the gate columns keep their channel order."""
    cfg, _ = tiny
    heads, kv, hd, D = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.hidden_size,
    )
    # tag every row with (head, 0=query|1=gate)
    q = torch.arange(heads * 2 * hd).reshape(heads, 2, hd).float()
    q_proj = q.reshape(heads * 2 * hd, 1).expand(-1, D).contiguous()
    k_proj = v_proj = torch.zeros(kv * hd, D)
    perm = weights.rotary_channel_permutation(hd, cfg.rotary_dim)
    mine = q[heads // TP :]  # rank 1 owns the second half of the heads
    gate = weights.attention_gate_device(q_proj, cfg, 1, TP)
    assert gate.shape == (D, heads // TP * hd)
    assert torch.equal(gate[0], mine[:, 1].reshape(-1))
    qkv = weights.attention_qkv_device(q_proj, k_proj, v_proj, cfg, 1, TP)
    assert qkv.shape == (D, (heads + 2 * kv) // TP * hd)
    assert torch.equal(qkv[0, : heads // TP * hd], mine[:, 0][:, perm].reshape(-1))


# (device-layout helper, HF tensor shape, HF dim that is split; None: q | k | v segments)
_SHARDED = {
    "mlp_gate_up_device": ((64, 48), 0),
    "mlp_down_device": ((48, 64), 1),
    "attention_o_device": ((48, 64), 1),
    "gdn_in_proj_heads_device": ((96, 48), 0),
    "gdn_out_proj_device": ((48, 96), 1),
    "gdn_in_proj_qkv_device": (None, None),
}


@pytest.mark.parametrize("fn", list(_SHARDED))
def test_shards_cover_tensor_exactly(weights, tiny, fn):
    cfg, _ = tiny
    shape, dim = _SHARDED[fn]
    t = torch.randn(shape or (cfg.conv_dim, 48))
    f = getattr(weights, fn)
    takes_cfg = fn not in (
        "mlp_gate_up_device",
        "mlp_down_device",
        "gdn_in_proj_heads_device",
    )
    # device layout is [in, out]: transpose back to HF "out x in"
    shards = [(f(t, cfg, r, TP) if takes_cfg else f(t, r, TP)).t() for r in range(TP)]
    if (
        dim is None
    ):  # segmented q|k|v: same multiset of rows, each rank gets 1/TP of every segment
        assert sum(s.numel() for s in shards) == t.numel()
        assert torch.equal(torch.cat(shards).sort(dim=0).values, t.sort(dim=0).values)
    else:
        assert torch.equal(torch.cat(shards, dim=dim), t)


def test_indivisible_raises(weights, tiny):
    with pytest.raises(ValueError):
        weights.mlp_gate_up_device(torch.randn(65, 48), 0, TP)


def test_partial_rotary_permutation_equivalent_to_full_width_rope(
    weights, qwen3_5_config
):
    """Permuted channels + full-width rotate_half (cos=1/sin=0 on pass dims) == partial rotary."""
    from conftest import ROOT, _load

    _load("vllm_neuron.model.qwen3_5.gdn_ops", ROOT / "model/qwen3_5/gdn_ops.py")
    ref = _load(
        "vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py"
    )
    D, R, L, H = 256, 64, 11, 3
    cfg = qwen3_5_config.Qwen3_5Config(
        head_dim=D,
        partial_rotary_factor=R / D,
        rope_theta=1e7,
        num_attention_heads=H,
        num_key_value_heads=1,
    )
    torch.manual_seed(0)
    q, k = torch.randn(L, H, D), torch.randn(L, H, D)
    pos = torch.arange(5, 5 + L)

    cos, sin = ref.rope_cos_sin(pos, cfg)
    q_ref, k_ref = ref.apply_rope(q, cos, sin), ref.apply_rope(k, cos, sin)
    scores_ref = torch.einsum("lhd,thd->hlt", q_ref, k_ref)

    perm = weights.rotary_channel_permutation(D, R)
    assert sorted(perm.tolist()) == list(range(D))
    inv_freq = weights.rope_inv_freq(R, cfg.rope_theta)
    kcos, ksin = weights.kernel_rope_angles_from_inv_freq(pos, inv_freq, D, R)
    q_k, k_k = (
        ref.apply_rope(q[..., perm], kcos, ksin),
        ref.apply_rope(k[..., perm], kcos, ksin),
    )
    torch.testing.assert_close(
        torch.einsum("lhd,thd->hlt", q_k, k_k), scores_ref, rtol=1e-4, atol=1e-4
    )
    # the roped tensors are the reference ones, permuted
    torch.testing.assert_close(q_k, q_ref[..., perm], rtol=1e-5, atol=1e-5)

    # the loaded q/k columns produce permuted queries/keys; the gate columns are untouched
    hidden = torch.randn(L, 32)
    w_q, w_k, w_v = torch.randn(H * 2 * D, 32), torch.randn(D, 32), torch.randn(D, 32)
    qg = (hidden @ w_q.t()).view(L, H, 2 * D)
    qkv = hidden @ weights.attention_qkv_device(w_q, w_k, w_v, cfg, 0, 1)
    torch.testing.assert_close(qkv[:, : H * D].view(L, H, D), qg[..., :D][..., perm])
    torch.testing.assert_close(qkv[:, H * D : H * D + D], (hidden @ w_k.t())[..., perm])
    torch.testing.assert_close(qkv[:, H * D + D :], hidden @ w_v.t())
    gate = hidden @ weights.attention_gate_device(w_q, cfg, 0, 1)
    torch.testing.assert_close(gate.view(L, H, D), qg[..., D:])


def test_fold_zero_centered_norm(weights, qwen3_5_config):
    from conftest import ROOT, _load

    ref = _load(
        "vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py"
    )
    x, w = torch.randn(4, 16), torch.randn(16) * 0.1
    gamma = weights.fold_zero_centered_norm(w)
    plain = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma
    torch.testing.assert_close(
        plain, ref.rmsnorm_zero_centered(x, w, 1e-6), rtol=1e-5, atol=1e-6
    )
