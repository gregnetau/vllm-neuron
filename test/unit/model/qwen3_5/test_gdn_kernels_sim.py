# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator checks of the GDN decode kernels (gdn_kernels.py) against the PyTorch ops.

Per-rank head configurations of the shipped checkpoints at their TP sizes; state pages in the
model's [ssm | conv | pad] layout over two halves (state.py). Padded batch rows carry -1
output slots and must not write any page.
"""

import importlib

import pytest
import torch

nki = pytest.importorskip("nki")
if not hasattr(nki, "simulate"):
    pytest.skip("nki.simulate unavailable", allow_module_level=True)
pytest.importorskip("nkilib")

D, KC, EPS = 128, 4, 1e-6

# (name, Hk per rank, Hv per rank, rows per page half)
CONFIGS = {
    "2b_tp2": (8, 8, 1024),  # Qwen3.5-2B: 16 / 16 heads, TP=2
    "35b_tp4": (4, 8, 1024),  # Qwen3.5-35B-A3B: 16 / 32 heads, TP=4
    "27b_tp4": (4, 12, 1024),  # Qwen3.8-27B: 16 / 48 heads, TP=4 (heads straddle halves)
}


@pytest.fixture(autouse=True)
def _lnc2(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")


@pytest.fixture
def GK():
    """Imported per test: the conftest drops qwen3_5 modules after each test."""
    try:
        return importlib.import_module("vllm_neuron.model.qwen3_5.gdn_kernels")
    except ImportError as e:  # vllm_neuron not importable (no vLLM)
        pytest.skip(f"gdn_kernels unavailable: {e}")


def _inputs(Hk, Hv, R, T, n_pages, seed):
    g = torch.Generator().manual_seed(seed)
    C = 2 * Hk * D + Hv * D
    rn = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
    return {
        "mixed": rn(T, C).bfloat16(),
        "z": rn(T, Hv * D).bfloat16(),
        "b": rn(T, Hv).bfloat16(),
        "a": rn(T, Hv).bfloat16(),
        "conv_w": rn(C, KC, sc=0.5).bfloat16(),
        "A_log": rn(Hv, sc=0.5),
        "dt_bias": rn(Hv, sc=0.5),
        "norm_w": (1 + rn(D, sc=0.1)).bfloat16(),
        "state_a": rn(n_pages, R, D, sc=0.1),
        "state_b": rn(n_pages, R, D, sc=0.1),
    }


def _read(sa, sb, slot, Hk, Hv):
    page = torch.cat([sa[slot], sb[slot]], dim=0)  # [2R, 128]
    C = 2 * Hk * D + Hv * D
    ssm = page[: Hv * D].reshape(Hv, D, D)
    conv = page[Hv * D : Hv * D + (KC - 1) * C // D].reshape(KC - 1, C).t()  # [C, K-1]
    return conv.clone(), ssm.clone()


def _write(sa, sb, slot, conv, ssm, Hv):
    R = sa.shape[1]
    page = torch.cat([sa[slot], sb[slot]], dim=0)
    page[: Hv * D] = ssm.reshape(-1, D)
    page[Hv * D : Hv * D + conv.numel() // D] = conv.t().reshape(-1, D)
    sa[slot], sb[slot] = page[:R], page[R:]


def _reference(gdn_ops, inp, slots_in, slots_out, Hk, Hv):
    """Float32 PyTorch reference (gdn_ops) of the batched decode step."""
    sa, sb = inp["state_a"].clone(), inp["state_b"].clone()
    NR, S = slots_out.shape
    mixed, z = inp["mixed"].float(), inp["z"].float()
    w = inp["conv_w"].float()
    rep = Hv // Hk
    outs = torch.zeros(NR * S, Hv * D)
    for r in range(NR):
        conv, ssm = _read(sa, sb, int(slots_in[r]), Hk, Hv)
        for j in range(S):
            t = r * S + j
            m, conv = gdn_ops.causal_conv1d_step_batched(
                mixed[t : t + 1], w, conv.unsqueeze(0)
            )
            conv = conv[0]
            q, k, v = torch.tensor_split(m[0], [Hk * D, 2 * Hk * D])
            q = q.reshape(Hk, D).repeat_interleave(rep, 0)[None]
            k = k.reshape(Hk, D).repeat_interleave(rep, 0)[None]
            gg, beta = gdn_ops.gdn_gating(
                inp["a"][t : t + 1], inp["b"][t : t + 1], inp["A_log"], inp["dt_bias"]
            )
            o, ssm4 = gdn_ops.gated_delta_step_batched(
                q, k, v.reshape(1, Hv, D), gg, beta, ssm[None]
            )
            ssm = ssm4[0]
            o = gdn_ops.gated_rmsnorm(
                o[0], z[t].reshape(Hv, D), inp["norm_w"].float(), EPS
            )
            outs[t] = o.reshape(-1)
            if int(slots_out[r, j]) >= 0:
                _write(sa, sb, int(slots_out[r, j]), conv, ssm, Hv)
    return outs, sa, sb


def _run_batched(GK, inp, slots_in, slots_out):
    k = GK.gdn_decode_batched_kernel
    out, sa, sb = nki.simulate(k[2])(
        inp["mixed"],
        inp["z"],
        inp["b"],
        inp["a"],
        inp["conv_w"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_w"],
        inp["state_a"].clone(),
        inp["state_b"].clone(),
        slots_in.to(torch.int32),
        slots_out.to(torch.int32).contiguous(),
        EPS,
    )
    return torch.as_tensor(out).float(), torch.as_tensor(sa), torch.as_tensor(sb)


def _slots(NR, S, n_pad, state_in, seed):
    """Distinct pages per request (1 + k blocks each, page 0 = null page); the last n_pad
    rows are padding (-1 everywhere; they read the null page)."""
    g = torch.Generator().manual_seed(seed)
    pages = 1 + torch.randperm(NR * S, generator=g)
    bt = pages.reshape(NR, S)
    bt[NR - n_pad :] = -1
    sin = torch.where(
        bt[:, 0] >= 0,
        bt.gather(1, torch.full((NR, 1), state_in)).view(NR),
        torch.zeros(NR, dtype=bt.dtype),
    )
    return sin, bt


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


CASES = [
    # (config, requests, tokens per request, padded rows, state_in)
    ("2b_tp2", 1, 1, 0, 0),
    ("2b_tp2", 2, 1, 0, 0),
    ("2b_tp2", 4, 1, 1, 0),
    ("2b_tp2", 8, 1, 3, 0),
    ("2b_tp2", 1, 3, 0, 0),
    ("2b_tp2", 2, 3, 0, 2),
    ("2b_tp2", 4, 3, 1, 1),
    ("2b_tp2", 8, 3, 2, 2),
    ("2b_tp2", 8, 4, 1, 3),
    ("35b_tp4", 1, 1, 0, 0),
    ("35b_tp4", 4, 1, 1, 0),
    ("35b_tp4", 8, 1, 0, 0),
    ("35b_tp4", 2, 3, 1, 1),
    ("35b_tp4", 8, 3, 2, 2),
    ("27b_tp4", 4, 1, 1, 0),
    ("27b_tp4", 8, 3, 1, 2),
]


@pytest.mark.parametrize("cfg,NR,S,n_pad,state_in", CASES)
def test_gdn_decode_batched_matches_reference(GK, gdn_ops, cfg, NR, S, n_pad, state_in):
    Hk, Hv, R = CONFIGS[cfg]
    T = NR * S
    inp = _inputs(Hk, Hv, R, T, n_pages=1 + NR * S + 2, seed=NR * 10 + S)
    slots_in, slots_out = _slots(NR, S, n_pad, state_in, seed=NR + S)
    ref_out, ref_a, ref_b = _reference(gdn_ops, inp, slots_in, slots_out, Hk, Hv)
    out, sa, sb = _run_batched(GK, inp, slots_in, slots_out)

    real = slice(0, (NR - n_pad) * S)
    assert torch.isfinite(out).all()
    assert _rel(out[real], ref_out[real]) < 1e-2
    # every page: written ones match, all others (incl. the null page and the padded rows'
    # would-be targets) are untouched
    assert _rel(sa, ref_a) < 1e-3 and _rel(sb, ref_b) < 1e-3
    written = set(slots_out.flatten().tolist())
    for p in set(range(sa.shape[0])) - written:
        assert torch.equal(sa[p], inp["state_a"][p]), p
        assert torch.equal(sb[p], inp["state_b"][p]), p


def test_gdn_decode_single_matches_reference(GK, gdn_ops):
    """The existing B=1 kernel against the same reference (harness check)."""
    Hk, Hv, R = CONFIGS["2b_tp2"]
    inp = _inputs(Hk, Hv, R, 1, n_pages=4, seed=1)
    slots = torch.tensor([2])
    ref_out, ref_a, ref_b = _reference(gdn_ops, inp, slots, slots.view(1, 1), Hk, Hv)
    out, sa, sb = nki.simulate(GK.gdn_decode_kernel[2])(
        inp["mixed"],
        inp["z"],
        inp["b"],
        inp["a"],
        inp["conv_w"],
        inp["A_log"],
        inp["dt_bias"],
        inp["norm_w"],
        inp["state_a"].clone(),
        inp["state_b"].clone(),
        slots.to(torch.int32),
        EPS,
    )
    assert _rel(torch.as_tensor(out).float(), ref_out) < 1e-2
    assert _rel(torch.as_tensor(sa), ref_a) < 1e-3
    assert _rel(torch.as_tensor(sb), ref_b) < 1e-3


@pytest.mark.parametrize("S", [1, 3])
def test_model_batched_decode_matches_torch_path(monkeypatch, S):
    """Qwen3_5GatedDeltaNet._decode / _decode_spec through gdn_decode_batched (simulated) vs the
    PyTorch path: slot plumbing (state_in select, padded rows) and the in-place state update."""
    from test_model_smoke import _install_stubs

    mod = _install_stubs()
    from conftest import ROOT, _load

    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    try:
        GKs = _load(
            "vllm_neuron.model.qwen3_5.gdn_kernels",
            ROOT / "model/qwen3_5/gdn_kernels.py",
        )
    except ImportError as e:
        pytest.skip(f"gdn_kernels unavailable: {e}")
    torch.manual_seed(3)
    cfg = Qwen3_5Config(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=96,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=128,
        torch_dtype=torch.bfloat16,
    )
    g = mod.Qwen3_5GatedDeltaNet(cfg, 0)
    with torch.no_grad():
        for name, p in g.named_parameters():
            p.copy_(
                torch.rand_like(p) + 0.5
                if name in ("A_log", "norm_weight")
                else torch.randn_like(p) * (0.5 if name == "conv_weight" else 0.06)
            )
    n_pages, half = 16, 512 * 128  # ssm rows 512 in half A, conv rows 24 in half B
    sa0 = torch.randn(n_pages, half) * 0.1
    sb0 = torch.randn(n_pages, half) * 0.1
    sa0[0], sb0[0] = 0.0, 0.0  # the null page

    B, k1 = 4, max(S, 2)
    bt = torch.tensor([[3, 4, 5], [7, 8, 9], [11, 12, 13], [-1, -1, -1]])[:, :k1]
    md = {"block_table_tensor": bt.to(torch.int32)}
    if S > 1:
        md["state_in"] = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    x = torch.randn(B * S, 256).to(torch.bfloat16)
    x[(B - 1) * S :] = 0  # padded rows (the model zeroes them after every layer)

    def simulated(mixed, z, b, a, conv_w, A_log, dt_bias, norm_w, sa, sb, si, so, eps):
        out, na, nb = nki.simulate(GKs.gdn_decode_batched_kernel[2])(
            mixed, z, b, a, conv_w, A_log, dt_bias, norm_w, sa, sb, si, so, eps
        )
        sa.copy_(torch.as_tensor(na))
        sb.copy_(torch.as_tensor(nb))
        return torch.as_tensor(out)

    monkeypatch.setattr(GKs, "gdn_decode_batched", simulated)
    monkeypatch.setattr(g, "_use_decode_kernel", lambda x, max_requests=1: True)
    res = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VLLM_NEURON_GDN_BATCHED_DECODE", mode)
        assert g._use_batched_kernel(x) == (mode == "1")
        sa, sb = sa0.clone(), sb0.clone()
        g.bind_state(sa, sb)
        with torch.no_grad():
            if S > 1:
                out = g._decode_spec(x, md)
            elif mode == "1":
                out = g._decode(x, md)
            else:
                out = g._decode_torch(x, md)
        res[mode] = (out.float(), sa, sb)
    (o0, a0, b0), (o1, a1, b1) = res["0"], res["1"]
    real = slice(0, (B - 1) * S)
    assert torch.isfinite(o1).all()
    assert _rel(o1[real], o0[real]) < 3e-2
    cr = 3 * 8 * 128  # conv rows of half B; the torch path also zeroes the page padding
    for p in range(1, n_pages):  # page 0: the torch path writes zeros, the kernel skips
        assert _rel(a1[p], a0[p]) < 3e-2 and _rel(b1[p, :cr], b0[p, :cr]) < 3e-2, p
    assert torch.equal(a1[0], sa0[0]) and torch.equal(b1[0], sb0[0])
    written = set(bt[: B - 1, :S].flatten().tolist())
    for p in set(range(n_pages)) - written:
        assert torch.equal(a1[p], sa0[p]) and torch.equal(b1[p], sb0[p]), p

