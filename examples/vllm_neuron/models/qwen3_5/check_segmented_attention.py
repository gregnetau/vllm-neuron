# SPDX-License-Identifier: Apache-2.0
"""Device check of segmented prefill attention at Qwen3.5 / Qwen3.8 shapes (head_dim 256).

Compares the nkilib segmented attention kernel (NF.segmented_attention) with a float32
PyTorch reference on one rank's shapes of the 27B model (6 query heads, one KV head,
head_dim 256), for one 1024-token chunk after a range of cached prefixes, and reports the
in-graph time per call next to the PyTorch fallback (Qwen3_5Attention._attend_cached_torch).

With ``--fp8-kv`` the cache is FP8 (E4M3) and the kernel is the model's vendored variant
(attention_segmented_cte.py) that reads the unpacked FP8 K cache.

Usage:
    python3 check_segmented_attention.py [--max-model-len 32768] [--n-valid 1000] [--fp8-kv]
"""

import argparse
import time

import torch

NH, NKH, DH = 6, 1, 256


def reference(q, kc, vc, bt, prior, n_valid, block_size, scale):
    """q [Nh, T, Dh] -> [Nh, Dh, T] in float32: causal attention over prior + n_valid keys."""
    T = q.shape[1]
    blocks = bt.clamp(min=0).long()
    S = blocks.shape[0] * block_size
    k = kc[blocks].transpose(0, 1).reshape(NKH, S, DH).float().repeat_interleave(NH // NKH, 0)
    v = vc[blocks].transpose(0, 1).reshape(NKH, S, DH).float().repeat_interleave(NH // NKH, 0)
    key_pos, q_pos = torch.arange(S), prior + torch.arange(T)
    allowed = (key_pos[None] < prior + n_valid) & (key_pos[None] <= q_pos[:, None])
    scores = (q.float() @ k.transpose(1, 2)) * scale
    scores = torch.where(allowed[None], scores, torch.full((), -1e30))
    return (torch.softmax(scores, -1) @ v).transpose(1, 2)


def _timed(fn, args, reps=10):
    with torch.no_grad():
        fn(*args).cpu()
        t = time.perf_counter()
        for _ in range(reps):
            fn(*args).cpu()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--block-size", type=int, default=32, help="kernel block size (serving: 32)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--n-valid", type=int, default=1000, help="valid tokens in the chunk")
    ap.add_argument("--priors", type=str, default=None,
                    help="comma-separated cached prefix lengths (default: 0, 1 chunk, half, max)")
    ap.add_argument("--no-torch", action="store_true", help="skip timing the PyTorch fallback")
    ap.add_argument("--fp8-kv", action="store_true", help="FP8 KV cache (vendored kernel)")
    args = ap.parse_args()

    import libtorch_neuronx_lite  # noqa: F401

    from vllm_neuron import functional as NF
    from vllm_neuron.envs import get_compile_backend_name

    dev = torch.device("neuron:0")
    torch.manual_seed(0)
    bs, T, L = args.block_size, args.chunk, args.max_model_len
    max_blocks = L // bs
    num_blocks = max_blocks + 8
    scale = DH**-0.5
    kv_dtype = torch.float8_e4m3fn if args.fp8_kv else torch.bfloat16
    kc = torch.randn(num_blocks, NKH, bs, DH).to(kv_dtype)
    vc = torch.randn(num_blocks, NKH, bs, DH).to(kv_dtype)
    kc[0] = 0
    vc[0] = 0  # null block
    kd, vd = kc.to(dev), vc.to(dev)
    seg_kernel = None
    if args.fp8_kv:
        from vllm_neuron.model.qwen3_5.attention_segmented_cte import (
            attention_segmented_cte_fp8_kernel,
        )

        seg_kernel = attention_segmented_cte_fp8_kernel

    def kernel(q, bt, prior):
        return NF.segmented_attention(q, k_cache=kd, v_cache=vd, block_tables=bt, prior_tokens=prior,
                                      block_size=bs, kv_segment_size=T, scale=scale, tp_q=True,
                                      tp_out=True, kernel=seg_kernel)

    def torch_fallback(q, bt, prior):
        # Same computation as Qwen3_5Attention._attend_cached_torch (gather the whole table).
        blocks = bt[0].long().clamp(min=0)
        S = blocks.shape[0] * bs
        k = kd.index_select(0, blocks).transpose(0, 1).reshape(NKH, S, DH).float()
        v = vd.index_select(0, blocks).transpose(0, 1).reshape(NKH, S, DH).float()
        key_pos, q_pos = torch.arange(S, device=q.device), prior.reshape(1) + torch.arange(T, device=q.device)
        allowed = (key_pos[None] < prior.reshape(1) + args.n_valid) & (key_pos[None] <= q_pos[:, None])
        k, v = k.repeat_interleave(NH // NKH, 0), v.repeat_interleave(NH // NKH, 0)
        s = torch.where(allowed[None], (q.float() @ k.transpose(1, 2)) * scale, torch.full((), -1e30, device=q.device))
        return (torch.softmax(s, -1) @ v).transpose(1, 2).to(q.dtype)

    backend = get_compile_backend_name()
    kernel_c = torch.compile(kernel, backend=backend, dynamic=False)
    torch_c = torch.compile(torch_fallback, backend=backend, dynamic=False)

    priors = ([int(p) for p in args.priors.split(",")] if args.priors
              else sorted({0, T, (L // 2 // T) * T, L - T}))
    for prior in priors:
        used = -(-(prior + T) // bs)
        perm = torch.randperm(num_blocks - 1)[:used] + 1
        bt = torch.full((1, max_blocks), -1, dtype=torch.int32)
        bt[0, :used] = perm.to(torch.int32)
        q = torch.randn(NH, T, DH).to(torch.bfloat16)
        p = torch.tensor([[prior]], dtype=torch.int32)
        ref = reference(q, kc, vc, bt[0], prior, args.n_valid, bs, scale)[..., :args.n_valid]
        dargs = (q.to(dev), bt.to(dev), p.to(dev))
        out = kernel_c(*dargs).cpu().float()[..., :args.n_valid]
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        t_k = _timed(kernel_c, dargs)
        line = (f"max_model_len {L:6d} prior {prior:6d}: rel err {err:.1e} cos {cos:.6f}  "
                f"kernel {t_k * 1e3:7.2f} ms")
        if not args.no_torch:
            t_t = _timed(torch_c, dargs)
            line += f"  torch fallback {t_t * 1e3:7.2f} ms"
        print(line, flush=True)


if __name__ == "__main__":
    main()
