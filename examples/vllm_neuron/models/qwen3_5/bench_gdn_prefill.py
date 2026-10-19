# SPDX-License-Identifier: Apache-2.0
"""Device timing of the Gated DeltaNet prefill recurrence (gdn_ops.chunk_gated_delta_rule) at
one rank's Qwen3.8-27B shapes (TP=4: 12 value heads, Dk = Dv = 128) for a prefill chunk.

Reports the in-graph time per call (chains of 1 and N calls, each consuming the previous
output, like decoder layers) and the error against a float64 CPU reference.

Usage:
    python3 bench_gdn_prefill.py [--tokens 1024] [--impl ref]
"""

import argparse
import time

import torch

H, D = 12, 128


def _inputs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, D, generator=g).to(torch.bfloat16)
    k = torch.randn(T, H, D, generator=g).to(torch.bfloat16)
    v = torch.randn(T, H, D, generator=g).to(torch.bfloat16)
    gate = -torch.rand(T, H, generator=g) * 0.5  # log decay per token, <= 0
    beta = torch.rand(T, H, generator=g)
    s0 = torch.randn(H, D, D, generator=g) * 0.1
    return q, k, v, gate, beta, s0


def _timed(fn, args, n_layers=8, reps=5):
    from vllm_neuron.envs import get_compile_backend_name

    def chain(n):
        def f(q, k, v, g, b, s):
            out = q
            for _ in range(n):
                o, s = fn(out, k, v, g, b, s)
                out = (q.float() + 1e-3 * o.float()).to(q.dtype)
            return out, s
        return torch.compile(f, backend=get_compile_backend_name(), dynamic=False)

    def run(f):
        with torch.no_grad():
            f(*args)[0].cpu()
            t = time.perf_counter()
            for _ in range(reps):
                f(*args)[0].cpu()
        return (time.perf_counter() - t) / reps

    return (run(chain(n_layers)) - run(chain(1))) / (n_layers - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--impl", default="ref", help="gdn_ops function name")
    ap.add_argument("--chunk-size", type=int, default=None, help="delta-rule chunk (default gdn_ops')")
    args = ap.parse_args()
    import libtorch_neuronx_lite  # noqa: F401
    from vllm_neuron.envs import get_compile_backend_name
    from vllm_neuron.model.qwen3_5 import gdn_ops

    fn = getattr(gdn_ops, "chunk_gated_delta_rule" if args.impl == "ref" else args.impl)
    if args.chunk_size:
        import functools

        fn = functools.partial(fn, chunk_size=args.chunk_size)
    dev = torch.device("neuron:0")
    q, k, v, g, b, s0 = _inputs(args.tokens)
    ref_o, ref_s = gdn_ops.recurrent_gated_delta_rule(q.double(), k.double(), v.double(), g.double(),
                                                      b.double(), s0.double())
    dargs = tuple(t.to(dev) for t in (q, k, v, g, b, s0))
    o, s = torch.compile(fn, backend=get_compile_backend_name(), dynamic=False)(*dargs)
    o, s = o.cpu().double(), s.cpu().double()
    err_o = ((o - ref_o).norm() / ref_o.norm()).item()
    err_s = ((s - ref_s).norm() / ref_s.norm()).item()
    per = _timed(fn, dargs)
    print(f"{args.impl} chunk {args.chunk_size or gdn_ops.CHUNK_SIZE} T {args.tokens}: rel err out {err_o:.1e} state {err_s:.1e}  {per * 1e3:7.2f} ms/layer"
          f"  ({48 * per * 1e3:6.1f} ms for 48 GDN layers)", flush=True)


if __name__ == "__main__":
    main()
