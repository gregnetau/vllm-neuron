# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator causality checks of the per-token components of the MTP verify step
(decode with S = 1 + k > 1 tokens per request) at Qwen3.8-27B TP=4 per-rank shapes.

Each verify row holds S tokens of one request; token j may depend only on tokens <= j of its
own row (and on nothing of other rows). Every test runs a kernel once, perturbs only the inputs
of the later tokens (or of other requests) and reruns: the earlier tokens' outputs (and the GDN
state pages written for them) must not change. Correctness is checked against PyTorch
references (the GDN reference uses the sequence (prefill) form: causal_conv1d over the row and
recurrent_gated_delta_rule, starting from the page's state).

Covered: gdn_decode_fp8, gdn_decode_batched, the vendored attention_block_tkg verify attention
(FP8 and BF16 KV, mask from gen_attention_decode_mask, incl. the flash-tiled long-context path),
fp8_mlp_decode, fp8_matvec, the output-gate projection; and at layer level, one S = 3 verify
step of Qwen3_5Attention.forward_decode / Qwen3_5GatedDeltaNet.decode_fused / _decode_spec
against three S = 1 plain decode steps.

Kernels run through libtorch_neuronx_lite's CPU simulator entry (the same conversion the
CPU-mode dispatch of wrap_nki uses; it maps torch float8_e4m3fn to NKI float8_e4m3), LNC=2.

The simulator runs take several minutes, so the module runs only with RUN_NKI_SIM=1:

    RUN_NKI_SIM=1 pytest test/unit/model/qwen3_5/test_verify_causality_sim.py
"""

import importlib
import json
import os
import pathlib

import pytest
import torch

if os.environ.get("RUN_NKI_SIM") != "1":
    pytest.skip(
        "slow NKI-simulator tests (several minutes); set RUN_NKI_SIM=1 to run",
        allow_module_level=True,
    )

nki = pytest.importorskip("nki")
pytest.importorskip("nkilib")
pytest.importorskip("ml_dtypes")

F8 = torch.float8_e4m3fn
D, KC, EPS = 128, 4, 1e-6
TP = 4
S_VERIFY = 3  # 1 + k with k = 2 draft tokens

_CFG = json.loads(
    (pathlib.Path(__file__).parent / "qwen3_8_27b_config.json").read_text()
)["text_config"]
H = _CFG["hidden_size"]  # 5120
HK = _CFG["linear_num_key_heads"] // TP  # 4
HV = _CFG["linear_num_value_heads"] // TP  # 12
C = 2 * HK * D + HV * D  # 2560
R = 1024  # rows per page half (test_gdn_kernels_sim CONFIGS["27b_tp4"])
I_RANK = _CFG["intermediate_size"] // TP  # 4352
NQ = _CFG["num_attention_heads"] // TP  # 6
DH = _CFG["head_dim"]  # 256


@pytest.fixture(autouse=True)
def _sim_env(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")


def _sim(kernel, **kwargs):
    try:
        from libtorch_neuronx_lite.nki.nki_cpu_sim import simulate_nki_kernel
    except ImportError as e:
        pytest.skip(f"libtorch_neuronx_lite simulator unavailable: {e}")
    out = simulate_nki_kernel(kernel, 2, kwargs)
    if isinstance(out, tuple):
        return tuple(torch.as_tensor(o) for o in out)
    return torch.as_tensor(out)


def _module(name):
    try:
        return importlib.import_module(f"vllm_neuron.model.qwen3_5.{name}")
    except ImportError as e:
        pytest.skip(f"{name} unavailable: {e}")


def _q8(w):
    s = w.abs().max() / 240
    return (w / s).clamp(-240, 240).to(F8), float(s)


def _fq(x, scale):
    """FP8 quantize-dequantize-free: the fp8 values (as float) of x / scale."""
    return (x / scale).clamp(-240, 240).to(F8).float()


def _sc(v, n=1):
    return torch.full((128, n), float(v))


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


def _rn(g):
    return lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc


# ---------------------------------------------------------------------------
# GDN: pages, slots, reference
# ---------------------------------------------------------------------------
def _read(sa, sb, slot):
    page = torch.cat([sa[slot], sb[slot]], dim=0)  # [2R, 128]
    ssm = page[: HV * D].reshape(HV, D, D)
    conv = page[HV * D : HV * D + (KC - 1) * C // D].reshape(KC - 1, C).t()  # [C, K-1]
    return conv.clone(), ssm.clone()


def _slots(NR, S, state_in, seed):
    """Distinct pages per request (S blocks each; page 0 = null page)."""
    g = torch.Generator().manual_seed(seed)
    bt = (1 + torch.randperm(NR * S, generator=g)).reshape(NR, S)
    sin = bt[:, state_in].clone()
    return sin.to(torch.int32), bt.to(torch.int32)


def _gdn_reference(gdn_ops, mixed, z, b, a, w, sa, sb, slots_in, slots_out):
    """Sequence-form (prefill) reference per request: causal_conv1d over the row from the
    page's conv state, recurrent_gated_delta_rule from its SSM state. Returns
    (out [T, Hv*D] f32, {page: (conv, ssm)} written after each token)."""
    NR, S = slots_out.shape
    rep = HV // HK
    outs = torch.zeros(NR * S, HV * D)
    pages = {}
    for r in range(NR):
        conv0, ssm0 = _read(sa, sb, int(slots_in[r]))
        rows = slice(r * S, (r + 1) * S)
        m, _ = gdn_ops.causal_conv1d(mixed[rows].float(), w["conv_w"].float(), conv0)
        q, k, v = torch.split(m, [HK * D, HK * D, HV * D], dim=-1)
        q = q.reshape(S, HK, D).repeat_interleave(rep, 1)
        k = k.reshape(S, HK, D).repeat_interleave(rep, 1)
        gg, beta = gdn_ops.gdn_gating(a[rows], b[rows], w["A_log"], w["dt_bias"])
        o, _ = gdn_ops.recurrent_gated_delta_rule(
            q, k, v.reshape(S, HV, D), gg, beta, ssm0
        )
        o = gdn_ops.gated_rmsnorm(
            o, z[rows].float().reshape(S, HV, D), w["norm_w"].float(), EPS
        )
        outs[rows] = o.reshape(S, -1)
        # state after token j: rerun the prefix (the reference has no per-token state output)
        for j in range(S):
            pre = slice(r * S, r * S + j + 1)
            _, conv_j = gdn_ops.causal_conv1d(
                mixed[pre].float(), w["conv_w"].float(), conv0
            )
            _, ssm_j = gdn_ops.recurrent_gated_delta_rule(
                q[: j + 1],
                k[: j + 1],
                v[: j + 1].reshape(j + 1, HV, D),
                gg[: j + 1],
                beta[: j + 1],
                ssm0,
            )
            pages[int(slots_out[r, j])] = (conv_j, ssm_j)
    return outs, pages


def _gdn_weights(seed):
    g = torch.Generator().manual_seed(seed)
    rn = _rn(g)
    return {
        "conv_w": rn(C, KC, sc=0.5).bfloat16(),
        "A_log": rn(HV, sc=0.5),
        "dt_bias": rn(HV, sc=0.5),
        "norm_w": (1 + rn(D, sc=0.1)).bfloat16(),
    }


def _gdn_pages(n_pages, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_pages, R, D, generator=g) * 0.1, torch.randn(
        n_pages, R, D, generator=g
    ) * 0.1


def _check_pages(sa, sb, pages, atol_rel):
    for p, (conv, ssm) in pages.items():
        k_conv, k_ssm = _read(sa, sb, p)
        assert _rel(k_ssm, ssm) < atol_rel, ("ssm", p, _rel(k_ssm, ssm))
        assert _rel(k_conv, conv.float()) < atol_rel, ("conv", p)


def _assert_causal(name, out0, out1, sa0, sb0, sa1, sb1, slots_out, j, tol):
    """Tokens <= j of every request (and the pages written for them) unchanged; tokens > j
    must change (the perturbation reached the kernel)."""
    NR, S = slots_out.shape
    for r in range(NR):
        for t in range(S):
            d = (out1[r * S + t].float() - out0[r * S + t].float()).abs().max().item()
            if t <= j:
                assert d <= tol, (
                    f"{name}: request {r} token {t} changed by {d} (perturbed > {j})"
                )
            else:
                assert d > 0, f"{name}: perturbation of token {t} had no effect"
            if t <= j:
                p = int(slots_out[r, t])
                ds = max(
                    (sa1[p] - sa0[p]).abs().max().item(),
                    (sb1[p] - sb0[p]).abs().max().item(),
                )
                assert ds <= tol, (
                    f"{name}: page of request {r} token {t} changed by {ds}"
                )


# ---------------------------------------------------------------------------
# 1. gdn_decode_fp8: whole GDN sub-layer
# ---------------------------------------------------------------------------
def _fp8_gdn_inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    rn = _rn(g)
    w = _gdn_weights(seed + 1)
    wqkv, sq = _q8(rn(H, C, sc=0.02))
    wz, sz = _q8(rn(H, HV * D, sc=0.02))
    wout, so = _q8(rn(HV * D, H, sc=0.02))
    w_b, w_a = rn(H, HV, sc=0.02).bfloat16(), rn(H, HV, sc=0.02).bfloat16()
    w.update(
        x=(rn(T, H) * 2).bfloat16(),
        ln_w=1 + rn(H, sc=0.1),
        w_qkv=wqkv,
        qkv_s=sq,
        w_z=wz,
        z_s=sz,
        w_out=wout,
        out_s=so,
        w_b=w_b,
        w_a=w_a,
        in_s=0.05,
        out_in_s=0.02,
    )
    return w


def _run_fp8_gdn(GK, w, x, sa, sb, slots_in, slots_out):
    out, na, nb = _sim(
        GK.gdn_decode_fp8_kernel,
        x=x,
        ln_w=w["ln_w"],
        w_qkv=w["w_qkv"],
        w_z=w["w_z"],
        w_ba=GK.gdn_ba_decode_layout(w["w_b"], w["w_a"]),
        qkv_scale=_sc(w["qkv_s"]),
        z_scale=_sc(w["z_s"]),
        in_scale=_sc(w["in_s"]),
        conv_w=w["conv_w"],
        A_log=w["A_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        state_a=sa.clone(),
        state_b=sb.clone(),
        slots_in=slots_in,
        slots_out=slots_out.contiguous(),
        w_out=w["w_out"],
        out_w_scale=_sc(w["out_s"]),
        out_in_scale=_sc(w["out_in_s"]),
        ln_eps=EPS,
        eps=EPS,
    )
    return (out[0].float() + out[1].float()), na.float(), nb.float()


def _fp8_gdn_reference(gdn_ops, w, sa, sb, slots_in, slots_out):
    x = w["x"].float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w["ln_w"]
    xq = _fq(xn, w["in_s"])
    mixed = xq @ w["w_qkv"].float() * (w["qkv_s"] * w["in_s"])
    z = xq @ w["w_z"].float() * (w["z_s"] * w["in_s"])
    xb = xn.bfloat16().float()
    b, a = xb @ w["w_b"].float(), xb @ w["w_a"].float()
    o, pages = _gdn_reference(gdn_ops, mixed, z, b, a, w, sa, sb, slots_in, slots_out)
    out = _fq(o, w["out_in_s"]) @ w["w_out"].float() * (w["out_s"] * w["out_in_s"])
    return out, pages


@pytest.mark.parametrize("NR,state_in", [(1, 0), (4, 1), (4, 2)])
def test_gdn_decode_fp8_verify_causal(gdn_ops, NR, state_in):
    GK = _module("gdn_kernels")
    S = S_VERIFY
    T = NR * S
    w = _fp8_gdn_inputs(T, seed=10 + NR)
    sa, sb = _gdn_pages(1 + T + 1, seed=NR)
    slots_in, slots_out = _slots(NR, S, state_in, seed=NR)

    out0, a0, b0 = _run_fp8_gdn(GK, w, w["x"], sa, sb, slots_in, slots_out)
    ref, pages = _fp8_gdn_reference(gdn_ops, w, sa, sb, slots_in, slots_out)
    assert torch.isfinite(out0).all()
    for t in range(T):  # per-token so a token mix-up cannot hide in a global max
        assert _rel(out0[t], ref[t]) < 6e-2, (t, _rel(out0[t], ref[t]))
    _check_pages(a0, b0, pages, 2e-2)

    g = torch.Generator().manual_seed(99)
    for j in range(S - 1):
        x1 = w["x"].clone()
        for r in range(NR):
            x1[r * S + j + 1 : (r + 1) * S] = (
                torch.randn(S - j - 1, H, generator=g) * 2
            ).bfloat16()
        out1, a1, b1 = _run_fp8_gdn(GK, w, x1, sa, sb, slots_in, slots_out)
        _assert_causal("gdn_decode_fp8", out0, out1, a0, b0, a1, b1, slots_out, j, 0.0)


# ---------------------------------------------------------------------------
# 2. gdn_decode_batched (BF16 projections done outside)
# ---------------------------------------------------------------------------
def _batched_inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    rn = _rn(g)
    w = _gdn_weights(seed + 1)
    w.update(
        mixed=rn(T, C).bfloat16(),
        z=rn(T, HV * D).bfloat16(),
        b=rn(T, HV).bfloat16(),
        a=rn(T, HV).bfloat16(),
    )
    return w


def _run_batched(GK, w, inp, sa, sb, slots_in, slots_out):
    out, na, nb = _sim(
        GK.gdn_decode_batched_kernel,
        mixed=inp["mixed"],
        z=inp["z"],
        b=inp["b"],
        a=inp["a"],
        conv_w=w["conv_w"],
        A_log=w["A_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        state_a=sa.clone(),
        state_b=sb.clone(),
        slots_in=slots_in,
        slots_out=slots_out.contiguous(),
        eps=EPS,
    )
    return out.float(), na.float(), nb.float()


@pytest.mark.parametrize("NR,state_in", [(1, 0), (4, 1), (4, 2)])
def test_gdn_decode_batched_verify_causal(gdn_ops, NR, state_in):
    GK = _module("gdn_kernels")
    S = S_VERIFY
    T = NR * S
    w = _batched_inputs(T, seed=20 + NR)
    sa, sb = _gdn_pages(1 + T + 1, seed=NR + 5)
    slots_in, slots_out = _slots(NR, S, state_in, seed=NR + 5)
    inp = {k: w[k] for k in ("mixed", "z", "b", "a")}

    out0, a0, b0 = _run_batched(GK, w, inp, sa, sb, slots_in, slots_out)
    ref, pages = _gdn_reference(
        gdn_ops, w["mixed"], w["z"], w["b"], w["a"], w, sa, sb, slots_in, slots_out
    )
    for t in range(T):
        assert _rel(out0[t], ref[t]) < 2e-2, (t, _rel(out0[t], ref[t]))
    _check_pages(a0, b0, pages, 1e-3)

    g = torch.Generator().manual_seed(98)
    for j in range(S - 1):
        inp1 = {k: v.clone() for k, v in inp.items()}
        for r in range(NR):
            rows = slice(r * S + j + 1, (r + 1) * S)
            for t in inp1.values():  # every per-token input of the later tokens
                t[rows] = torch.randn(t[rows].shape, generator=g).to(t.dtype)
        out1, a1, b1 = _run_batched(GK, w, inp1, sa, sb, slots_in, slots_out)
        _assert_causal(
            "gdn_decode_batched", out0, out1, a0, b0, a1, b1, slots_out, j, 0.0
        )


# ---------------------------------------------------------------------------
# 3. verify attention (vendored attention_block_tkg, FP8 KV, head_dim 256)
# ---------------------------------------------------------------------------
def _attn_setup(B, blk, nblk, seed, fp8=True):
    g = torch.Generator().manual_seed(seed)
    rn = _rn(g)
    S = S_VERIFY
    S_ctx = blk * nblk
    w = rn(H, (NQ + 2) * DH, sc=0.02)
    wq, ws = _q8(w) if fp8 else (w.bfloat16(), None)
    kv = F8 if fp8 else torch.bfloat16
    n_pages = 1 + B * nblk
    st = {
        "fp8": fp8,
        "X": rn(B, S, H).bfloat16(),
        "W": wq,
        "ws": ws,
        "isc": 0.05,
        "qn": (1 + rn(1, DH, sc=0.1)).bfloat16(),
        "kn": (1 + rn(1, DH, sc=0.1)).bfloat16(),
        "Kc": rn(n_pages, blk, DH).clamp(-240, 240).to(kv),
        "Vc": rn(n_pages, blk, DH).clamp(-240, 240).to(kv),
        "bt": (1 + torch.randperm(B * nblk, generator=g))
        .reshape(B, nblk)
        .to(torch.int32),
        "pos0": torch.tensor([S_ctx // 2 + 5 + 7 * b for b in range(B)]),
        "blk": blk,
        "S_ctx": S_ctx,
    }
    pos = (st["pos0"][:, None] + torch.arange(S)).float()  # [B, S]
    ang = pos[None] * torch.logspace(0, -3, DH // 2)[:, None, None]  # [Dh/2, B, S]
    st["cos"], st["sin"], st["pos"] = ang.cos().bfloat16(), ang.sin().bfloat16(), pos
    return st


def _attn_mask(st, B):
    import vllm_neuron.functional as NF

    return NF.gen_attention_decode_mask(
        pos_ids=st["pos"].reshape(1, -1),
        bs=B,
        q_head=NQ,
        s_active=S_VERIFY,
        s_prior=st["S_ctx"],
        start_pos=None,
        block_len=st["blk"],
    )


def _run_attn(st, X, Kc, Vc, mask):
    from nkilib.core.utils.common_types import QuantizationType

    from vllm_neuron.model.qwen3_5.attention_block_tkg import (
        attention_block_tkg_k_rows_kernel,
    )

    one = torch.ones(1, 1, dtype=torch.bfloat16)
    quant = {}
    if st["fp8"]:  # static FP8 QKV and FP8 KV (Qwen3_5Attention._decode_quant_kwargs)
        quant = {
            "quantization_type_qkv": QuantizationType.STATIC,
            "weight_dequant_scale_qkv": _sc(st["ws"], 3),
            "input_dequant_scale_qkv": _sc(st["isc"]),
            "k_scale": one,
            "v_scale": one,
        }
    attn, k_new, v_new = _sim(
        attention_block_tkg_k_rows_kernel,
        X=X,
        X_hidden_dim_actual=H,
        rmsnorm_X_enabled=False,
        W_qkv=st["W"],
        bias_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=st["qn"],
        rmsnorm_QK_pre_rope_W_K=st["kn"],
        rmsnorm_QK_post_rope_enabled=False,
        cos=st["cos"],
        sin=st["sin"],
        rope_contiguous_layout=True,
        K_cache_transposed=False,
        active_blocks_table=st["bt"],
        K_cache=Kc,
        V_cache=Vc,
        attention_mask=mask,
        softmax_scale=DH**-0.5,
        sink=None,
        update_cache=False,
        W_out=None,
        bias_out=None,
        transposed_out=False,
        out_in_sb=False,
        **quant,
    )
    return attn.float(), k_new.float(), v_new.float()


def _attn_reference(st, B):
    S, blk = S_VERIFY, st["blk"]
    isc = st["isc"]
    if st["fp8"]:
        qkv = _fq(st["X"].float(), isc) @ st["W"].float() * (st["ws"] * isc)
    else:
        qkv = st["X"].float() @ st["W"].float()
    q = qkv[..., : NQ * DH].reshape(B, S, NQ, DH)
    k = qkv[..., NQ * DH : (NQ + 1) * DH].reshape(B, S, DH)
    v = qkv[..., (NQ + 1) * DH :].reshape(B, S, DH)

    def rms(t, wt):
        return (
            t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * wt.float().view(-1)
        )

    c = st["cos"].float().permute(1, 2, 0)  # [B, S, Dh/2]
    s_ = st["sin"].float().permute(1, 2, 0)

    def rope(t, c, s_):
        t1, t2 = t[..., : DH // 2], t[..., DH // 2 :]
        return torch.cat([t1 * c - t2 * s_, t2 * c + t1 * s_], -1)

    q = rope(rms(q, st["qn"]), c[:, :, None], s_[:, :, None])
    k = rope(rms(k, st["kn"]), c, s_)
    out = torch.zeros(B, NQ, DH, S)
    for b in range(B):
        P = int(st["pos0"][b])
        idx = [(int(st["bt"][b, p // blk]), p % blk) for p in range(P)]
        kp = torch.stack([st["Kc"][i, o].float() for i, o in idx])
        vp = torch.stack([st["Vc"][i, o].float() for i, o in idx])
        for t in range(S):
            kk, vv = torch.cat([kp, k[b, : t + 1]]), torch.cat([vp, v[b, : t + 1]])
            out[b, :, :, t] = torch.softmax(q[b, t] @ kk.T * DH**-0.5, -1) @ vv
    return out, k, v


# (B, block_size, blocks per request): batch- and s_prior-sharded LNC splits, short and
# longer contexts (S_ctx 256 .. 32768; above 8K per LNC shard the prior is flash-tiled).
ATTN_CASES = [
    (1, 32, 8),
    (4, 32, 8),
    (1, 32, 64),
    (4, 32, 64),
    (4, 128, 16),
    (4, 16, 512),
    (1, 128, 256),
]


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("B,blk,nblk", ATTN_CASES)
def test_verify_attention_causal(B, blk, nblk, fp8):
    S = S_VERIFY
    st = _attn_setup(B, blk, nblk, seed=B * 100 + nblk, fp8=fp8)
    mask = _attn_mask(st, B)
    a0, k0, v0 = _run_attn(st, st["X"], st["Kc"], st["Vc"], mask)
    assert a0.shape == (B, NQ, DH, S) and k0.shape == (B * S, DH)

    ra, rk, rv = _attn_reference(st, B)
    tol = 0.1 if fp8 else 2e-2  # FP8 (qkv input, K/V) rounding: ~3-5 %; BF16 ~0.5 %
    for b in range(B):
        for t in range(S):
            assert _rel(a0[b, ..., t], ra[b, ..., t]) < tol, (b, t)
    assert _rel(k0.view(B, S, DH), rk) < tol and _rel(v0.view(B, S, DH), rv) < tol

    # later tokens' Q/K/V (all derive from X) must not reach earlier tokens
    g = torch.Generator().manual_seed(7)
    for j in range(S - 1):
        X1 = st["X"].clone()
        X1[:, j + 1 :] = torch.randn(B, S - j - 1, H, generator=g).bfloat16()
        a1, _, _ = _run_attn(st, X1, st["Kc"], st["Vc"], mask)
        for b in range(B):
            for t in range(S):
                d = (a1[b, ..., t] - a0[b, ..., t]).abs().max().item()
                assert (d == 0) == (t <= j), (b, t, j, d)

    # the active tokens' K/V come from the new K/V, not from the (stale) cache slots at
    # positions >= pos0, which forward_decode writes only after attention
    Kc2, Vc2 = st["Kc"].clone(), st["Vc"].clone()
    for b in range(B):
        for p in range(int(st["pos0"][b]), st["S_ctx"]):
            i, o = int(st["bt"][b, p // blk]), p % blk
            Kc2[i, o] = (
                (torch.randn(DH, generator=g) * 10).clamp(-240, 240).to(Kc2.dtype)
            )
            Vc2[i, o] = (
                (torch.randn(DH, generator=g) * 10).clamp(-240, 240).to(Vc2.dtype)
            )
    a2, _, _ = _run_attn(st, st["X"], Kc2, Vc2, mask)
    assert torch.equal(a2, a0)


# ---------------------------------------------------------------------------
# 4. fp8_mlp_decode / fp8_matvec over 3*B tokens: token independence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [3, 12])
def test_fp8_mlp_decode_token_independent(T):
    FK = _module("fp8_kernels")
    g = torch.Generator().manual_seed(T)
    rn = _rn(g)
    (gw, gs), (uw, us), (dw, ds) = (
        _q8(rn(H, I_RANK, sc=0.02)),
        _q8(rn(H, I_RANK, sc=0.02)),
        _q8(rn(I_RANK, H, sc=0.02)),
    )
    lw = 1 + rn(H, sc=0.1)
    gu_in, dn_in = 0.02, 0.002

    def run(x):
        p = _sim(
            FK.fp8_mlp_decode_kernel,
            x=x,
            ln_w=lw,
            gate=gw,
            up=uw,
            down=dw,
            gate_scale=_sc(gs),
            up_scale=_sc(us),
            down_scale=_sc(ds),
            gate_up_in_scale=_sc(gu_in),
            down_in_scale=_sc(dn_in),
            eps=EPS,
        )
        return p[0].float() + p[1].float()

    x = (rn(T, H) * 2).bfloat16()
    y0 = run(x)
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * lw
    xq = _fq(xn, gu_in)
    act = torch.nn.functional.silu(xq @ gw.float() * (gs * gu_in)) * (
        xq @ uw.float() * (us * gu_in)
    )
    ref = _fq(act, dn_in) @ dw.float() * (ds * dn_in)
    for t in range(T):
        assert _rel(y0[t], ref[t]) < 5e-2, (t, _rel(y0[t], ref[t]))
    for victim in (0, T - 1):
        x1 = x.clone()
        x1[victim] = (
            torch.randn(H, generator=g) * 20
        ).bfloat16()  # large: would move a dynamic scale
        y1 = run(x1)
        for t in range(T):
            d = (y1[t] - y0[t]).abs().max().item()
            assert (d == 0) == (t != victim), (victim, t, d)


@pytest.mark.parametrize("T", [3, 12])
@pytest.mark.parametrize("K,N", [(H, 2560), (NQ * DH, H)])
def test_fp8_matvec_token_independent(T, K, N):
    FK = _module("fp8_kernels")
    g = torch.Generator().manual_seed(T + K)
    w, s = _q8(torch.randn(K, N, generator=g) * 0.02)
    isc = 0.02

    def run(x):
        return _sim(
            FK.fp8_matvec_kernel, x=x, w=w, w_scale=_sc(s), in_scale=_sc(isc)
        ).float()

    x = torch.randn(T, K, generator=g).bfloat16()
    y0 = run(x)
    ref = _fq(x.float(), isc) @ w.float() * (s * isc)
    for t in range(T):
        assert _rel(y0[t], ref[t]) < 2e-2, t
    for victim in (0, T - 1):
        x1 = x.clone()
        x1[victim] = (torch.randn(K, generator=g) * 20).bfloat16()
        y1 = run(x1)
        for t in range(T):
            d = (y1[t] - y0[t]).abs().max().item()
            assert (d == 0) == (t != victim), (victim, t, d)


# ---------------------------------------------------------------------------
# 5. Output-gate projection of the verify attention (NF.qkv_proj BF16 TKG kernel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [3, 12])
def test_gate_proj_decode_token_independent(T):
    try:
        import vllm_neuron.functional as NF
    except ImportError as e:
        pytest.skip(f"vllm_neuron.functional unavailable: {e}")
    g = torch.Generator().manual_seed(T)
    N = NQ * DH
    w = (torch.randn(H, N, generator=g) * 0.02).bfloat16()

    def run(x):  # as Qwen3_5Attention._gate_proj_decode
        return (
            NF.qkv_proj(
                hidden=x.unsqueeze(0),
                qkv_weights=w,
                bias=None,
                d_head=128,
                num_q_heads=N // 128 - 2,
                num_kv_heads=1,
            )
            .squeeze(0)
            .float()
        )

    x = torch.randn(T, H, generator=g).bfloat16()
    y0 = run(x)
    ref = x.float() @ w.float()
    for t in range(T):
        assert _rel(y0[t], ref[t]) < 2e-2, t
    x1 = x.clone()
    x1[T - 1] = torch.randn(H, generator=g).bfloat16()
    y1 = run(x1)
    for t in range(T):
        assert ((y1[t] - y0[t]).abs().max().item() == 0) == (t != T - 1), t


# ---------------------------------------------------------------------------
# 6. Layer level: one S = 3 verify step == three S = 1 decode steps (the model's own
#    forward_decode / decode_fused / _decode_spec glue around the kernels above)
# ---------------------------------------------------------------------------
class _TP1:
    world_size, rank_in_group, device_group = 1, 0, None


def _model(monkeypatch, fp8):
    import types

    M = _module("model")
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    monkeypatch.setattr(M, "get_tp_group", lambda: _TP1())
    # 27B per-rank shapes at world size 1
    cfg = Qwen3_5Config(
        num_hidden_layers=4,
        num_attention_heads=NQ,
        num_key_value_heads=1,
        linear_num_key_heads=HK,
        linear_num_value_heads=HV,
        neuron_config=types.SimpleNamespace(
            quantization="fp8" if fp8 else None, modules_to_not_convert=None
        ),
    )
    return M, cfg


def _set_fp8(mod, name, shape, g, sc=0.02):
    w = torch.randn(*shape, generator=g) * sc
    if getattr(mod, name + "_weight").dtype == F8:
        q, s = _q8(w)
        getattr(mod, name + "_weight").data = q
        getattr(mod, name + "_weight_scale").data.fill_(s)
    else:
        getattr(mod, name + "_weight").data = w.bfloat16()


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("B", [1, 2])
def test_attention_layer_verify_equals_sequential_decode(monkeypatch, B, fp8):
    M, cfg = _model(monkeypatch, fp8)
    L = M.Qwen3_5Attention(cfg, 3)
    g = torch.Generator().manual_seed(B)
    with torch.no_grad():
        _set_fp8(L, "qkv_proj", (H, (NQ + 2) * DH), g)
        _set_fp8(L, "o_proj", (NQ * DH, H), g)
        L.gate_weight.data = (torch.randn(H, NQ * DH, generator=g) * 0.02).bfloat16()
        if fp8:
            L.qkv_input_scale.fill_(0.05)
            L.o_input_scale.fill_(0.01)
        L.q_norm.weight.data = (1 + 0.1 * torch.randn(DH, generator=g)).bfloat16()
        L.k_norm.weight.data = (1 + 0.1 * torch.randn(DH, generator=g)).bfloat16()
    S, blk, nblk = S_VERIFY, 32, 8
    kv = F8 if fp8 else torch.bfloat16
    kc0 = torch.randn(1 + B * nblk, 1, blk, DH, generator=g).clamp(-240, 240).to(kv)
    vc0 = torch.randn(1 + B * nblk, 1, blk, DH, generator=g).clamp(-240, 240).to(kv)
    bt = (1 + torch.randperm(B * nblk, generator=g)).reshape(B, nblk).to(torch.int32)
    L.runner_clears_kv_blocks = True
    if fp8:
        L.k_scale = L.v_scale = torch.ones(1, 1, dtype=torch.bfloat16)
    rot = M.Qwen3_5RotaryEmbedding(cfg)

    def step(x, pos):  # pos [B, s]
        slot = bt.long().gather(1, pos.long() // blk) * blk + pos.long() % blk
        md = {
            "slot_mapping": slot.view(-1),
            "block_size": blk,
            "block_table_tensor": bt,
            "max_blocks_per_seq": nblk,
            "max_query_len": pos.shape[1],
            "decode_token_threshold": 8,
        }
        p = pos.reshape(-1)
        with torch.no_grad():
            return L.forward_decode(
                x, p, rot(p, torch.bfloat16), {"layers.3.self_attn": md}
            ).float()

    x = torch.randn(B * S, H, generator=g).bfloat16()
    pos = (
        torch.tensor([100 + 13 * b for b in range(B)])[:, None] + torch.arange(S)
    ).int()
    L.k_cache, L.v_cache = kc0.clone(), vc0.clone()
    yv = step(x, pos)
    kv_verify = (L.k_cache.clone(), L.v_cache.clone())
    L.k_cache, L.v_cache = kc0.clone(), vc0.clone()
    ys = torch.stack(
        [
            step(x.view(B, S, H)[:, j].contiguous(), pos[:, j : j + 1].contiguous())
            for j in range(S)
        ],
        dim=1,
    ).reshape(B * S, H)
    # same kernels, different shapes (LNC split / reduction order): bf16-ulp level only
    for t in range(B * S):
        assert _rel(yv[t], ys[t]) < 1e-2, (t, _rel(yv[t], ys[t]))
    assert torch.equal(L.k_cache.float(), kv_verify[0].float())
    assert torch.equal(L.v_cache.float(), kv_verify[1].float())


@pytest.mark.parametrize("path", ["fused_fp8", "batched_fp8", "batched_bf16"])
@pytest.mark.parametrize("B", [1, 4])
def test_gdn_layer_verify_equals_sequential_decode(monkeypatch, B, path):
    """decode_fused (gdn_decode_fp8) / _decode_spec (gdn_decode_batched) for one S = 3 verify
    step from state_in vs three S = 1 decode steps on a scratch page seeded from state_in."""
    fp8 = path.endswith("fp8")
    M, cfg = _model(monkeypatch, fp8)
    G = M.Qwen3_5GatedDeltaNet(cfg, 0)
    g = torch.Generator().manual_seed(B + 7)
    with torch.no_grad():
        _set_fp8(G, "in_proj_qkv", (H, G.conv_dim), g)
        _set_fp8(G, "in_proj_z", (H, G.value_dim), g)
        _set_fp8(G, "out_proj", (G.value_dim, H), g)
        if fp8:
            G.in_proj_input_scale.fill_(0.05)
            G.out_proj_input_scale.fill_(0.02)
        G.in_proj_b_weight.data = (torch.randn(H, HV, generator=g) * 0.02).bfloat16()
        G.in_proj_a_weight.data = (torch.randn(H, HV, generator=g) * 0.02).bfloat16()
        G.conv_weight.data = (torch.randn(C, KC, generator=g) * 0.5).bfloat16()
        G.A_log.data = torch.randn(HV, generator=g) * 0.5
        G.dt_bias.data = torch.randn(HV, generator=g) * 0.5
        G.norm_weight.data = (1 + 0.1 * torch.randn(D, generator=g)).bfloat16()
    G.prepare_decode_layout()
    ln = M.Qwen3_5RMSNorm(H, EPS, torch.float32)
    ln.weight.data = 1 + 0.1 * torch.randn(H, generator=g)
    monkeypatch.setattr(G, "_use_decode_kernel", lambda x, max_requests=1: True)
    if path == "fused_fp8":
        assert G.use_fused_decode(torch.zeros(B * S_VERIFY, H))
    else:  # project row by row: CPU torch matmul rounding depends on the row count
        project = G._project

        def per_row(h):
            parts = [project(h[i : i + 1]) for i in range(h.shape[0])]
            return tuple(torch.cat(p, dim=0) for p in zip(*parts))

        monkeypatch.setattr(G, "_project", per_row)

    def run(x, md):
        with torch.no_grad():
            if path == "fused_fp8":
                return G.decode_fused(x, md, ln).float()
            h = ln(x)
            return (
                G._decode_spec(h, md) if "state_in" in md else G._decode(h, md)
            ).float()

    S = S_VERIFY
    n_pages = 1 + B * S + B
    sa0 = torch.randn(n_pages, R * D, generator=g) * 0.1
    sb0 = torch.randn(n_pages, R * D, generator=g) * 0.1
    bt = (1 + torch.randperm(B * S, generator=g)).reshape(B, S).to(torch.int32)
    state_in = torch.tensor([b % S for b in range(B)], dtype=torch.int32)
    x = (torch.randn(B * S, H, generator=g) * 2).bfloat16()

    sa, sb = sa0.clone(), sb0.clone()
    G.bind_state(sa, sb)
    yv = run(x, {"block_table_tensor": bt, "state_in": state_in})
    va, vb = sa.clone(), sb.clone()

    sa, sb = sa0.clone(), sb0.clone()
    scratch = torch.arange(1 + B * S, n_pages, dtype=torch.int32)
    for b in range(B):
        sa[scratch[b]], sb[scratch[b]] = (
            sa0[bt[b, state_in[b]]],
            sb0[bt[b, state_in[b]]],
        )
    G.bind_state(sa, sb)
    used_b = (HV * D - R + (KC - 1) * C // D) * D  # ssm tail + conv rows in half B
    ys = []
    for j in range(S):
        ys.append(
            run(
                x.view(B, S, H)[:, j].contiguous(),
                {"block_table_tensor": scratch.view(B, 1)},
            )
        )
        for b in range(
            B
        ):  # the page written for token j == the state after j plain decode steps
            p = int(bt[b, j])  # fp32 state; matmul order differs with the token count
            assert _rel(sa[scratch[b]], va[p]) < 1e-3, (b, j)
            assert _rel(sb[scratch[b], :used_b], vb[p, :used_b]) < 1e-3, (b, j)
    ys = torch.stack(ys, dim=1).reshape(B * S, H)
    for t in range(B * S):
        assert _rel(yv[t], ys[t]) < 1e-2, (t, _rel(yv[t], ys[t]))
