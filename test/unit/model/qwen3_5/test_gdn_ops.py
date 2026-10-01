# SPDX-License-Identifier: Apache-2.0
"""GDN reference ops vs the Hugging Face Qwen3.5 implementation."""

import pytest
import torch
import torch.nn.functional as F

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

torch.manual_seed(0)
H, DK, DV = 4, 16, 16


def _rand_inputs(L, dtype=torch.float32):
    q = torch.randn(L, H, DK, dtype=dtype)
    k = torch.randn(L, H, DK, dtype=dtype)
    v = torch.randn(L, H, DV, dtype=dtype)
    g = -torch.rand(L, H) * 0.5
    beta = torch.rand(L, H)
    return q, k, v, g, beta


def _hf_ref(fn, q, k, v, g, beta, state=None, **kw):
    out, s = fn(
        q[None], k[None], v[None], g[None], beta[None],
        initial_state=None if state is None else state[None],
        output_final_state=True, use_qk_l2norm_in_kernel=True, **kw,
    )
    return out[0], s[0]


@pytest.mark.parametrize("L", [1, 7, 64, 130])
def test_recurrent_matches_hf(gdn_ops, L):
    q, k, v, g, beta = _rand_inputs(L)
    s0 = torch.randn(H, DK, DV) * 0.1
    out, s = gdn_ops.recurrent_gated_delta_rule(q, k, v, g, beta, s0)
    ref_out, ref_s = _hf_ref(hf.torch_recurrent_gated_delta_rule, q, k, v, g, beta, s0)
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s, ref_s, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("L", [1, 5, 64, 65, 200])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_matches_hf(gdn_ops, L, with_state):
    q, k, v, g, beta = _rand_inputs(L)
    s0 = torch.randn(H, DK, DV) * 0.1 if with_state else None
    out, s = gdn_ops.chunk_gated_delta_rule(q, k, v, g, beta, s0)
    ref_out, ref_s = _hf_ref(hf.torch_chunk_gated_delta_rule, q, k, v, g, beta, s0)
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(s, ref_s, rtol=1e-3, atol=1e-4)


def test_chunk_equals_recurrent_and_carries_state(gdn_ops):
    """Splitting a prompt into chunked-prefill pieces must not change results."""
    q, k, v, g, beta = _rand_inputs(150)
    full, s_full = gdn_ops.recurrent_gated_delta_rule(q, k, v, g, beta, None)
    s = None
    outs = []
    for lo, hi in ((0, 37), (37, 101), (101, 150)):
        o, s = gdn_ops.chunk_gated_delta_rule(
            q[lo:hi], k[lo:hi], v[lo:hi], g[lo:hi], beta[lo:hi], s
        )
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs), full, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(s, s_full, rtol=1e-3, atol=1e-4)


def test_causal_conv1d_matches_hf_and_carries_state(gdn_ops):
    C, K, L = 12, 4, 50
    x = torch.randn(L, C)
    w = torch.randn(C, K)
    # HF full-sequence path: silu(conv1d(x))[:L] with left zero padding
    conv = torch.nn.Conv1d(C, C, K, groups=C, bias=False, padding=K - 1)
    conv.weight.data = w.unsqueeze(1)
    ref = F.silu(conv(x.t()[None])[:, :, :L])[0].t()

    y_full, st_full = gdn_ops.causal_conv1d(x, w, None)
    torch.testing.assert_close(y_full, ref, rtol=1e-5, atol=1e-6)

    st, ys = None, []
    for lo, hi in ((0, 2), (2, 3), (3, 30), (30, 50)):  # includes L < K-1 pieces
        y, st = gdn_ops.causal_conv1d(x[lo:hi], w, st)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys), y_full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(st, st_full)

    # One decode step agrees with the HF single-token update semantics
    # (written inline: the helper's name differs across transformers versions).
    x_new = torch.randn(1, C)
    window = torch.cat([st_full, x_new.t()], dim=-1)  # [C, K]
    ref_step = F.silu((window * w).sum(-1))[None]
    y_step, st_step = gdn_ops.causal_conv1d(x_new, w, st_full)
    torch.testing.assert_close(y_step, ref_step, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(st_step, window[:, 1:])


def test_gated_rmsnorm_matches_hf(gdn_ops):
    x, gate = torch.randn(9, DV), torch.randn(9, DV)
    norm = hf.Qwen3_5RMSNormGated(DV, eps=1e-6)
    norm.weight.data = torch.rand(DV) + 0.5
    torch.testing.assert_close(
        gdn_ops.gated_rmsnorm(x, gate, norm.weight.data, 1e-6), norm(x, gate)
    )


def test_layer_reference_matches_hf_with_chunked_prefill_then_decode(gdn_ops):
    """End-to-end GDN block: chunked prefill (2 pieces) + token decode == HF full forward."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        hidden_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=64, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        vocab_size=100,
    )
    layer = hf.Qwen3_5GatedDeltaNet(cfg, 0).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        layer.norm.weight.copy_(torch.rand_like(layer.norm.weight) + 0.5)
        layer.A_log.copy_(torch.log(torch.rand_like(layer.A_log) * 4 + 0.1))

    L_pre, L_dec = 90, 6
    x = torch.randn(1, L_pre + L_dec, 48)
    with torch.no_grad():
        ref = layer(x)[0]

    w = {
        "in_proj_qkv": layer.in_proj_qkv.weight.data, "in_proj_z": layer.in_proj_z.weight.data,
        "in_proj_b": layer.in_proj_b.weight.data, "in_proj_a": layer.in_proj_a.weight.data,
        "conv1d": layer.conv1d.weight.data, "A_log": layer.A_log.data,
        "dt_bias": layer.dt_bias.data, "norm": layer.norm.weight.data,
        "out_proj": layer.out_proj.weight.data,
    }
    xs = x[0]
    cs = ss = None
    outs = []
    for lo, hi in ((0, 50), (50, L_pre)):  # chunked prefill
        o, cs, ss = gdn_ops.gdn_layer_reference(xs[lo:hi], w, cfg, cs, ss)
        outs.append(o)
    for t in range(L_pre, L_pre + L_dec):  # decode
        o, cs, ss = gdn_ops.gdn_layer_reference(xs[t : t + 1], w, cfg, cs, ss, decode=True)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs), ref, rtol=1e-3, atol=1e-4)


def test_batched_decode_steps_match_per_request_recurrent(gdn_ops):
    B, C, K = 3, 10, 4
    xs = torch.randn(B, C)
    w = torch.randn(C, K)
    cs = torch.randn(B, C, K - 1)
    y, ns = gdn_ops.causal_conv1d_step_batched(xs, w, cs)
    for b in range(B):
        yb, nsb = gdn_ops.causal_conv1d(xs[b : b + 1], w, cs[b])
        torch.testing.assert_close(y[b], yb[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ns[b], nsb)

    q, k, v = torch.randn(B, H, DK), torch.randn(B, H, DK), torch.randn(B, H, DV)
    g, beta = -torch.rand(B, H), torch.rand(B, H)
    S = torch.randn(B, H, DK, DV) * 0.1
    out, S2 = gdn_ops.gated_delta_step_batched(q, k, v, g, beta, S)
    for b in range(B):
        o, sb = gdn_ops.recurrent_gated_delta_rule(
            q[b : b + 1], k[b : b + 1], v[b : b + 1], g[b : b + 1], beta[b : b + 1], S[b]
        )
        torch.testing.assert_close(out[b], o[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(S2[b], sb, rtol=1e-5, atol=1e-6)


def test_padding_tokens_are_identity_for_state(gdn_ops):
    """beta=0, g=0 on padded tail leaves the state (and valid outputs) unchanged."""
    q, k, v, g, beta = _rand_inputs(100)
    n = 70
    o_ref, s_ref = gdn_ops.chunk_gated_delta_rule(q[:n], k[:n], v[:n], g[:n], beta[:n], None)
    g_m, beta_m = g.clone(), beta.clone()
    g_m[n:], beta_m[n:] = 0.0, 0.0
    o, s = gdn_ops.chunk_gated_delta_rule(q, k, v, g_m, beta_m, None)
    torch.testing.assert_close(o[:n], o_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s, s_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("C", [16, 64])
def test_inverse_unit_lower_matches_triangular_solve(gdn_ops, C):
    """The matmul-only inverse equals solve_triangular (which Neuron cannot lower)."""
    torch.manual_seed(2)
    m = torch.tril(torch.randn(3, 5, C, C) * 0.3, diagonal=-1)
    eye = torch.eye(C)
    ref = torch.linalg.solve_triangular(eye + m, eye.expand_as(m), upper=False, unitriangular=True)
    torch.testing.assert_close(gdn_ops.inverse_unit_lower(m, eye), ref, rtol=1e-4, atol=1e-4)


def test_chunk_rule_graph_has_no_linalg_ops(gdn_ops):
    """Trace-safety: no custom-call-lowered linear algebra in the chunked prefill path."""
    from torch.fx.experimental.proxy_tensor import make_fx

    q, k, v, g, beta = _rand_inputs(70)
    gm = make_fx(lambda q, k, v, g, beta: gdn_ops.chunk_gated_delta_rule(q, k, v, g, beta, None)[0])(
        q, k, v, g, beta)
    targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    assert targets, "empty graph"
    assert not any("linalg" in t or "triangular" in t for t in targets), sorted(targets)
