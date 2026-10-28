# SPDX-License-Identifier: Apache-2.0
"""Device check and timing of the batched GDN decode kernel (gdn_kernels.gdn_decode_batched)
against the PyTorch decode path (Qwen3_5GatedDeltaNet._decode_torch / _decode_spec) for one
GDN layer at one rank's shapes, B requests x S tokens (S > 1: MTP verify with state_in).

Both paths run the same work after the input projections (identical in both, so left out):
state read, causal conv, gating, delta rule, state write-back, gated RMSNorm. Reports
  * numerics of the kernel and of the PyTorch path on device against the PyTorch path on CPU
    (output and every state page; with B > 1 the last row is padding, -1 in the block table,
    and must not write any page),
  * the in-graph time per layer: chains of 1 and 8 layers, each consuming the previous output.

Usage (on one NeuronCore, LNC=2; nothing else may hold the device):
    python3 bench_gdn_decode.py                         # all configs, B 1/4/8, S 1/3
    python3 bench_gdn_decode.py --config 35b_tp4 --batch 4 --tokens 1 3 --impl kernel
"""

import argparse
import time

import torch

# one compiled run() per (config, B, S, impl)
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config.accumulated_recompile_limit = 4096
from torch import nn

D, KC, EPS = 128, 4, 1e-6
# per-rank (key heads, value heads): Qwen3.5-2B TP=2, Qwen3.5-35B-A3B TP=4, Qwen3.8-27B TP=4
CONFIGS = {"2b_tp2": (8, 8), "35b_tp4": (4, 8), "27b_tp4": (4, 12)}
ROWS = 1024  # rows of 128 floats per page half


def _layer(Hk, Hv, n_pages, seed=0):
    """A Qwen3_5GatedDeltaNet with only what the decode path reads (no TP group, no config)."""
    from vllm_neuron.model.qwen3_5.model import Qwen3_5GatedDeltaNet as G

    g = G.__new__(G)
    nn.Module.__init__(g)
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s, sc=1.0: torch.randn(*s, generator=gen) * sc
    g.Hk, g.Hv, g.Dk, g.Dv, g.K = Hk, Hv, D, D, KC
    g.key_dim, g.value_dim = Hk * D, Hv * D
    g.conv_dim = 2 * g.key_dim + g.value_dim
    g.conv_numel = g.conv_dim * (KC - 1)
    g.ssm_numel = Hv * D * D
    g.state_numel = g.conv_numel + g.ssm_numel
    g.dtype = torch.bfloat16
    g.config = type("Cfg", (), {"rms_norm_eps": EPS})()
    g.conv_weight = rn(g.conv_dim, KC, sc=0.5).bfloat16()
    g.A_log = rn(Hv, sc=0.5)
    g.dt_bias = rn(Hv, sc=0.5)
    g.norm_weight = (1 + rn(D, sc=0.1)).bfloat16()
    g.state_a = rn(n_pages, ROWS, D, sc=0.1)
    g.state_b = rn(n_pages, ROWS, D, sc=0.1)
    g.state_a[0], g.state_b[0] = 0.0, 0.0  # the null page
    return g


def _to(g, dev):
    for n in ("conv_weight", "A_log", "dt_bias", "norm_weight", "state_a", "state_b"):
        setattr(g, n, getattr(g, n).to(dev))
    return g


def _torch_path(g, mixed, z, b, a, bt, state_in):
    """Qwen3_5GatedDeltaNet._decode_torch / _decode_spec after the projections."""
    from vllm_neuron.model.qwen3_5 import gdn_ops

    B, T = bt.shape[0], mixed.shape[0]
    S = T // B
    bt_raw = bt.long()
    bt_c = bt_raw.clamp(min=0)
    if state_in is None:
        slots_in = bt_c[:, 0]
    else:
        slots_in = bt_c.gather(1, state_in.long().view(B, 1)).view(B)
    mixed, b, a = mixed.view(B, S, -1), b.view(B, S, -1), a.view(B, S, -1)
    conv, ssm = g._read_state(slots_in)
    outs = []
    for j in range(S):
        m_j, conv = gdn_ops.causal_conv1d_step_batched(mixed[:, j], g.conv_weight, conv)
        q, k, v = g._qkv_heads(m_j, B)
        gg, beta = gdn_ops.gdn_gating(a[:, j], b[:, j], g.A_log, g.dt_bias)
        o, ssm = gdn_ops.gated_delta_step_batched(q, k, v, gg, beta, ssm)
        g._write_state(bt_raw[:, j], conv, ssm)
        outs.append(o)
    o = torch.stack(outs, dim=1).reshape(T, g.Hv, g.Dv)
    o = gdn_ops.gated_rmsnorm(o, z.reshape(T, g.Hv, g.Dv), g.norm_weight, EPS)
    return o.reshape(T, g.value_dim)


def _kernel_path(g, mixed, z, b, a, bt, state_in):
    """Qwen3_5GatedDeltaNet._decode_batched after the projections."""
    from vllm_neuron.model.qwen3_5.gdn_kernels import gdn_decode_batched

    bt = bt.to(torch.int32)
    B, S = bt.shape[0], mixed.shape[0] // bt.shape[0]
    slots_out = bt[:, :S].contiguous()
    if state_in is None:
        slots_in = bt[:, 0]
    else:
        cols = torch.arange(bt.shape[1], device=bt.device).view(1, -1)
        sel = state_in.long().view(B, 1) == cols
        slots_in = torch.where(sel, bt, torch.zeros_like(bt)).sum(dim=1)
    slots_in = slots_in.clamp(min=0).to(torch.int32)
    return gdn_decode_batched(
        mixed, z, b, a, g.conv_weight, g.A_log, g.dt_bias, g.norm_weight,
        g.state_a, g.state_b, slots_in, slots_out, EPS,
    )  # fmt: skip


def _single_path(g, mixed, z, b, a, bt, state_in):
    """gdn_kernels.gdn_decode (B = 1, one token: the default for that case)."""
    from vllm_neuron.model.qwen3_5.gdn_kernels import gdn_decode

    slots = bt[:, 0].clamp(min=0).to(torch.int32)
    return gdn_decode(
        mixed, z, b, a, g.conv_weight, g.A_log, g.dt_bias, g.norm_weight,
        g.state_a, g.state_b, slots, EPS,
    )  # fmt: skip


PATHS = {"kernel": _kernel_path, "torch": _torch_path, "single": _single_path}


def _inputs(Hk, Hv, B, S, pad, seed=1):
    gen = torch.Generator().manual_seed(seed)
    T, C = B * S, 2 * Hk * D + Hv * D
    k1 = max(S, 2)  # block table [B, 1 + k]
    bt = (1 + torch.arange(B * k1)).view(B, k1).to(torch.int32)
    if pad:
        bt[-1] = -1
    state_in = None
    if S > 1:
        state_in = torch.randint(0, k1, (B,), generator=gen).to(torch.int32)
    x = [
        torch.randn(T, C, generator=gen).bfloat16(),
        torch.randn(T, Hv * D, generator=gen).bfloat16(),
        torch.randn(T, Hv, generator=gen).bfloat16(),
        torch.randn(T, Hv, generator=gen).bfloat16(),
    ]
    if pad:
        for t in x:
            t[(B - 1) * S :] = 0
    return x, bt, state_in, 1 + B * k1 + 1


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()


def _compile(fn):
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(fn, backend=get_compile_backend_name(), dynamic=False)


def check(cfg, B, S, impls, dev):
    Hk, Hv = CONFIGS[cfg]
    pad = B > 1
    x, bt, state_in, n_pages = _inputs(Hk, Hv, B, S, pad)
    ref_g = _layer(Hk, Hv, n_pages)
    state0 = (ref_g.state_a.clone(), ref_g.state_b.clone())
    with torch.no_grad():
        ref = _torch_path(ref_g, *x, bt, state_in)
    T = B * S
    real = slice(0, (B - 1) * S if pad else T)
    written = set(bt[: B - 1 if pad else B, :S].flatten().tolist())
    cr = 3 * (2 * Hk + Hv)  # conv rows (half B holds the end of the ssm rows + conv)
    xd = [t.to(dev) for t in x]
    btd = bt.to(dev)
    sid = None if state_in is None else state_in.to(dev)
    for impl in impls:
        if impl == "single" and (B, S) != (1, 1):
            continue
        path = PATHS[impl]
        g = _to(_layer(Hk, Hv, n_pages), dev)

        def f(m, z, b, a, bt_, si, g=g, path=path):
            return path(g, m, z, b, a, bt_, si)

        if dev.type != "cpu":
            f = _compile(f)
        with torch.no_grad():
            out = f(*xd, btd, sid).cpu()
        sa, sb = g.state_a.cpu(), g.state_b.cpu()
        err_o = _rel(out[real], ref[real])
        err_s = 0.0
        stray = []
        for p in range(n_pages):
            if p in written:
                lim = max(0, Hv * D - ROWS) + cr
                err_s = max(
                    err_s,
                    _rel(sa[p], ref_g.state_a[p]),
                    _rel(sb[p, :lim], ref_g.state_b[p, :lim]),
                )
            elif impl != "torch" and not (
                torch.equal(sa[p], state0[0][p]) and torch.equal(sb[p], state0[1][p])
            ):
                # the torch path writes zeros to the null page; the kernel must not
                stray.append(p)
        finite = bool(torch.isfinite(out).all())

        # time: chain of n layers on the same state pages, each layer's z = f(previous out)
        def chain(n, g=g, path=path):
            def run(m, z, b, a, bt_, si):
                for _ in range(n):
                    o = path(g, m, z, b, a, bt_, si)
                    z = (z.float() + 1e-3 * o.float()).to(z.dtype)
                return z

            return _compile(run)

        def timed(fc, reps=20):
            with torch.no_grad():
                fc(*xd, btd, sid).cpu()
                t = time.perf_counter()
                for _ in range(reps):
                    fc(*xd, btd, sid).cpu()
            return (time.perf_counter() - t) / reps

        per = float("nan")
        if dev.type != "cpu":  # --cpu-sim checks numerics only
            per = (timed(chain(8)) - timed(chain(1))) / 7
        ok = err_o < 3e-2 and err_s < 3e-2 and finite and not stray
        print(
            f"{cfg:8s} B {B} S {S} {impl:6s}: {per * 1e6:8.1f} us/layer  rel err out {err_o:.1e} "
            f"state {err_s:.1e} finite {finite} stray pages {stray}  {'OK' if ok else 'FAIL'}",
            flush=True,
        )


def _use_simulator():
    import os

    import nki

    from vllm_neuron.model.qwen3_5 import gdn_kernels as GK

    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

    def wrap(kernel):
        def call(*args):
            out, na, nb = nki.simulate(kernel[2])(*args)
            args[8].copy_(torch.as_tensor(na))  # state_a, state_b: updated in place
            args[9].copy_(torch.as_tensor(nb))
            return torch.as_tensor(out)

        return call

    GK.gdn_decode_batched = wrap(GK.gdn_decode_batched_kernel)
    GK.gdn_decode = wrap(GK.gdn_decode_kernel)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--batch", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 3], help="tokens per request"
    )
    ap.add_argument(
        "--impl",
        nargs="+",
        default=["kernel", "torch", "single"],
        choices=["kernel", "torch", "single"],
        help="kernel: gdn_decode_batched, torch: the PyTorch path, "
        "single: gdn_decode (B = S = 1 only)",
    )
    ap.add_argument(
        "--cpu-sim",
        action="store_true",
        help="numerics only, on CPU with the NKI simulator (no device, no timing)",
    )
    args = ap.parse_args()
    if args.cpu_sim:
        _use_simulator()
        dev = torch.device("cpu")
    else:
        import libtorch_neuronx_lite  # noqa: F401

        dev = torch.device("neuron:0")
    for cfg in args.config:
        for S in args.tokens:
            for B in args.batch:
                check(cfg, B, S, args.impl, dev)


if __name__ == "__main__":
    main()
