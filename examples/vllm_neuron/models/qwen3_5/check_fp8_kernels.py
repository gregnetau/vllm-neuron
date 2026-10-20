# SPDX-License-Identifier: Apache-2.0
"""Device check of the Qwen3.5 static-FP8 decode kernels (fp8_kernels.py) on one NeuronCore.

Compares fp8_matvec and fp8_mlp_decode against a PyTorch FP8 emulation (same quantization)
and reports the in-graph time per call: chains of 1 and 8 calls with distinct weights, each
call consuming the previous output (like decoder layers).

Usage (27B TP=4 shard shapes):
    python3 check_fp8_kernels.py [--tokens 1]
"""

import argparse
import time

import torch
import torch.nn.functional as F

H, I, EPS = 5120, 4352, 1e-6
MATVEC_SHAPES = [(5120, 2560), (5120, 1536), (1536, 5120)]
F8 = torch.float8_e4m3fn


def _q8(w):
    s = w.abs().max() / 240
    return (w / s).clamp(-240, 240).to(F8), float(s)


def _fq(x, scale):
    return (x / scale).clamp(-240, 240).to(F8).float()


def _timed(fn, x, n_layers=8, reps=10):
    from vllm_neuron.envs import get_compile_backend_name

    opts = {"alias_meta_to_neuron": True, "compiler_args": [
        "--auto-cast=none", "-O1",
        "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
        "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]}

    def chain(n):
        def f(h):
            for i in range(n):
                h = fn(h, i)
            return h
        return torch.compile(f, backend=get_compile_backend_name(), dynamic=False, options=opts)

    def run(f):
        with torch.no_grad():
            f(x).cpu()
            t = time.perf_counter()
            for _ in range(reps):
                f(x).cpu()
        return (time.perf_counter() - t) / reps

    return (run(chain(n_layers)) - run(chain(1))) / (n_layers - 1)


def check_matvec(T, dev):
    from vllm_neuron.envs import get_compile_backend_name
    from vllm_neuron.model.qwen3_5.fp8_kernels import fp8_matvec

    for K, N in MATVEC_SHAPES:
        ws = [_q8(torch.randn(K, N) * 0.02) for _ in range(8)]
        wd = [w.to(dev) for w, _ in ws]
        wsc = [torch.full((128, 1), s).to(dev) for _, s in ws]
        isc = torch.full((128, 1), 0.02).to(dev)
        x = torch.randn(T, K).to(torch.bfloat16)
        ref = _fq(x.float(), 0.02) @ ws[0][0].float() * (ws[0][1] * 0.02)
        out = torch.compile(lambda h: fp8_matvec(h, wd[0], wsc[0], isc), backend=get_compile_backend_name(),
                            dynamic=False)(x.to(dev)).cpu().float()
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        step = lambda h, i: h * (1 + 1e-9 * fp8_matvec(h, wd[i], wsc[i], isc)[:, :1])
        per = _timed(step, x.to(dev))
        print(f"fp8_matvec     K {K:5d} N {N:5d} T {T}: rel err {err:.1e}  {per * 1e6:6.1f} us  "
              f"{K * N / per / 1e9:4.0f} GB/s")


def check_mlp(T, dev):
    from vllm_neuron.envs import get_compile_backend_name
    from vllm_neuron.model.qwen3_5.fp8_kernels import fp8_mlp_decode

    sc = lambda v: torch.full((128, 1), v).to(dev)
    layers = []
    for _ in range(8):
        (g, gs), (u, us), (d, ds) = (_q8(torch.randn(H, I) * 0.02), _q8(torch.randn(H, I) * 0.02),
                                     _q8(torch.randn(I, H) * 0.02))
        layers.append(dict(g=g, u=u, d=d, gs=gs, us=us, ds=ds, lw=1 + 0.1 * torch.randn(H)))
    gu_in, dn_in = 0.02, 0.002
    gu_t, dn_t = sc(gu_in), sc(dn_in)
    D = [dict(g=L["g"].to(dev), u=L["u"].to(dev), d=L["d"].to(dev), gs=sc(L["gs"]), us=sc(L["us"]),
              ds=sc(L["ds"]), lw=L["lw"].to(dev)) for L in layers]

    def mlp(h, i):
        d = D[i]
        return fp8_mlp_decode(h, d["lw"], d["g"], d["u"], d["d"], d["gs"], d["us"], d["ds"], gu_t, dn_t,
                              EPS)

    x = (torch.randn(T, H) * 2).to(torch.bfloat16)
    L = layers[0]
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * L["lw"]
    xq = _fq(xn, gu_in)
    a = F.silu(xq @ L["g"].float() * (L["gs"] * gu_in)) * (xq @ L["u"].float() * (L["us"] * gu_in))
    ref = _fq(a, dn_in) @ L["d"].float() * (L["ds"] * dn_in)
    out = torch.compile(lambda h: mlp(h, 0), backend=get_compile_backend_name(), dynamic=False)(x.to(dev))
    out = out.cpu().float()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    per = _timed(lambda h, i: h + mlp(h, i).to(h.dtype), x.to(dev))
    print(f"fp8_mlp_decode H {H} I {I} T {T}: rel err {err:.1e}  {per * 1e6:6.1f} us  "
          f"{3 * H * I / per / 1e9:4.0f} GB/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=1)
    args = ap.parse_args()
    import libtorch_neuronx_lite  # noqa: F401

    dev = torch.device("neuron:0")
    torch.manual_seed(0)
    check_matvec(args.tokens, dev)
    check_mlp(args.tokens, dev)


if __name__ == "__main__":
    main()
