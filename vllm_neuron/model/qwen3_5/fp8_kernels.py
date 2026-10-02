# SPDX-License-Identifier: Apache-2.0
"""Static-FP8 decode NKI kernels for Qwen3.5 / Qwen3.8 (Trn2, per-tensor scales).

Decode is weight-bandwidth bound. These kernels stream FP8 weights through the Tensor Engine
in double-row mode (256-deep contraction per instruction), which keeps up with HBM where
single-row matmuls do not, and keep a whole sub-layer in one kernel so weight streaming does
not stall at kernel boundaries.

``fp8_matvec``: y[T, N] = x[T, K] @ W[K, N] (T <= 32), W in the standard [K, N] layout.
``fp8_mlp_decode``: RMSNorm + gate/up + SiLU-mul + down for T <= 32 tokens; returns the two
logical-core partial sums [2, T, H] (the caller adds them with the residual).

Common scheme:
  * Activations are quantized in-kernel with the static input scale (clamped to +-240, Trn2
    FP8 E4M3) and are the stationary operand, zero-padded to 32 columns (double-row minimum).
  * The x contraction index is k = p * KT + kt (partition p, step kt), so x loads and the
    per-partition weight rows are contiguous.
  * Results accumulate in PSUM in passes of up to 4096 columns (8 banks); weight DMAs cover
    whole rows of the pass (long runs) and are chunked along K (~1 MiB) so matmuls start early.
  * The MLP splits the intermediate dimension over the two logical-core programs: each computes
    gate/up for its half, transposes the activation onto partitions (i = p * ICT + j) and
    multiplies by the matching rows of down, giving a partial sum per core (no core barrier).
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

FP8_MAX = 240.0
MAX_TOKENS = 32
_P = 128
_NB = 512            # matmul columns (one PSUM bank of fp32)
_SB = 4096           # columns per PSUM pass (8 banks)
_CHUNK_BYTES = 1 << 20


def _sb(shape, dtype=nl.float32):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _ranges(n, step):
    """[[start, end)] covering [0, n) in pieces of step."""
    out = []
    s = 0
    for _ in range(n):
        if s >= n:
            break
        out.append([s, min(n, s + step)])
        s = s + step
    return out


def _scale(src):
    t = _sb((_P, 1))
    nisa.dma_copy(dst=t, src=src[0:_P, 0:1])
    return t


def _quantize_cols(dst, src, inv, T):
    """dst[:, :, t] (fp8, [P, KT, TP]) = clamp(src[:, t, :] * inv) for t < T; padding zeroed."""
    nisa.memset(dst, 0.0)
    for t in range(T):
        tmp = _sb((_P, src.shape[2]))
        nisa.tensor_scalar(dst=tmp, data=src[:, t, :], op0=nl.multiply, operand0=inv,
                           op1=nl.minimum, operand1=FP8_MAX)
        nisa.tensor_scalar(dst=dst[:, :, t], data=tmp, op0=nl.maximum, operand0=-FP8_MAX)


def _stream_pass(ps, subs, xq, wv, c0, KT, sw, wdtype):
    """ps[:, si, :nb] = sum_k xq[:, k, :].T @ W[k, sub si] with W = wv[:, :, c0:c0 + sw] ([P, KT, *] HBM).
    Pairs of k steps use double row; an odd last step uses a single-row matmul."""
    kc_max = max(2, (_CHUNK_BYTES // (_P * sw)) // 2 * 2)
    chunks = _ranges(KT, kc_max)
    for ci in range(len(chunks)):
        k0 = chunks[ci][0]
        kc = chunks[ci][1] - k0
        wt = _sb((_P, kc, sw), wdtype)
        nisa.dma_copy(dst=wt, src=wv[:, k0:k0 + kc, nl.ds(c0, sw)])
        for j in range(kc // 2):
            for si in range(len(subs)):
                n0 = subs[si][0]
                nb = subs[si][1] - n0
                nisa.nc_matmul(dst=ps[:, si, 0:nb], stationary=xq[:, k0 + 2 * j:k0 + 2 * j + 2, :],
                               moving=wt[:, 2 * j:2 * j + 2, n0:n0 + nb], accumulate=(k0 + 2 * j) > 0,
                               perf_mode=nisa.matmul_perf_mode.double_row)
        if kc % 2 == 1:
            for si in range(len(subs)):
                n0 = subs[si][0]
                nb = subs[si][1] - n0
                nisa.nc_matmul(dst=ps[:, si, 0:nb], stationary=xq[:, k0 + kc - 1, :],
                               moving=wt[:, kc - 1, n0:n0 + nb], accumulate=(k0 + kc - 1) > 0)


def _evacuate(dst, ps, subs, osc, T):
    """dst[T, sw] = ps[0:T] * osc."""
    for si in range(len(subs)):
        n0 = subs[si][0]
        nb = subs[si][1] - n0
        nisa.tensor_scalar(dst=dst[:, n0:n0 + nb], data=ps[0:T, si, 0:nb], op0=nl.multiply,
                           operand0=osc[0:T, :])


def _load_x(x, KT):
    """x [T, K] -> [P, T, KT] fp32 with k = p * KT + kt."""
    T, K = x.shape
    xs = _sb((_P, T, KT), x.dtype)
    nisa.dma_copy(dst=xs, src=x.reshape((T, _P, KT)).ap(pattern=[[KT, _P], [K, T], [1, KT]], offset=0))
    xf = _sb((_P, T, KT))
    nisa.tensor_copy(dst=xf, src=xs)
    return xf


@nki.jit
def fp8_matvec_kernel(x, w, w_scale, in_scale):
    """x [T, K] bf16, w [K, N] fp8 (K % 128 == 0, N even), w_scale / in_scale [128, 1] f32
    -> [T, N] bf16. rev 5"""
    T, K = x.shape
    _, N = w.shape
    KT = K // _P
    _, n_prgs, prg_id = get_verified_program_sharding_info("fp8_matvec", (0, 1))
    Nc = N // n_prgs
    nbase = prg_id * Nc
    out = nl.ndarray((T, N), dtype=x.dtype, buffer=nl.shared_hbm)

    isc = _scale(in_scale)
    inv = _sb((_P, 1))
    nisa.reciprocal(dst=inv, data=isc)
    osc = _sb((_P, 1))
    nisa.tensor_tensor(dst=osc, data1=_scale(w_scale), data2=isc, op=nl.multiply)

    xq = _sb((_P, KT, MAX_TOKENS), w.dtype)
    _quantize_cols(xq, _load_x(x, KT), inv, T)

    wv = w.reshape((_P, KT, N))
    passes = _ranges(Nc, _SB)
    for pi in range(len(passes)):
        s0 = passes[pi][0]
        sw = passes[pi][1] - s0
        subs = _ranges(sw, _NB)
        ps = nl.ndarray((MAX_TOKENS, len(subs), _NB), dtype=nl.float32, buffer=nl.psum)
        _stream_pass(ps, subs, xq, wv, nbase + s0, KT, sw, w.dtype)
        o = _sb((T, sw), x.dtype)
        _evacuate(o, ps, subs, osc, T)
        nisa.dma_copy(dst=out[:, nl.ds(nbase + s0, sw)], src=o)
    return out


@nki.jit
def fp8_mlp_decode_kernel(x, ln_w, gate, up, down, gate_scale, up_scale, down_scale,
                          gate_up_in_scale, down_in_scale, eps):
    """x [T, H] bf16 (pre-norm), ln_w [H] f32, gate/up [H, I] fp8, down [I, H] fp8, scales
    [128, 1] f32 -> [2, T, H] f32 per-core partial sums of the MLP output. rev 1"""
    T, H = x.shape
    _, I = gate.shape
    KT = H // _P
    TP = MAX_TOKENS
    _, n_prgs, prg_id = get_verified_program_sharding_info("fp8_mlp_decode", (0, 1))
    Ic = I // n_prgs
    ICT = Ic // _P
    ibase = prg_id * Ic
    out = nl.ndarray((n_prgs, T, H), dtype=nl.float32, buffer=nl.shared_hbm)

    gu_in = _scale(gate_up_in_scale)
    dn_in = _scale(down_in_scale)
    inv_gu = _sb((_P, 1))
    nisa.reciprocal(dst=inv_gu, data=gu_in)
    inv_dn = _sb((_P, 1))
    nisa.reciprocal(dst=inv_dn, data=dn_in)
    og = _sb((_P, 1))
    nisa.tensor_tensor(dst=og, data1=_scale(gate_scale), data2=gu_in, op=nl.multiply)
    ou = _sb((_P, 1))
    nisa.tensor_tensor(dst=ou, data1=_scale(up_scale), data2=gu_in, op=nl.multiply)
    od = _sb((_P, 1))
    nisa.tensor_tensor(dst=od, data1=_scale(down_scale), data2=dn_in, op=nl.multiply)

    # ---- RMSNorm: sum of squares over k (free kt, then partitions via a ones matmul)
    xf = _load_x(x, KT)
    lw = _sb((_P, KT))
    nisa.dma_copy(dst=lw, src=ln_w.reshape((_P, KT)))
    sq = _sb((_P, T, KT))
    nisa.tensor_tensor(dst=sq, data1=xf, data2=xf, op=nl.multiply)
    part = _sb((_P, T))
    nisa.tensor_reduce(dst=part, op=nl.add, data=sq, axis=2)
    ones = _sb((_P, _P))
    nisa.memset(ones, 1.0)
    tot = nl.ndarray((_P, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot, stationary=ones, moving=part)
    rstd = _sb((_P, T))
    nisa.activation(dst=rstd, op=nl.rsqrt, data=tot, scale=1.0 / H, bias=eps)
    xn = _sb((_P, T, KT))
    for t in range(T):
        nisa.tensor_scalar(dst=xn[:, t, :], data=xf[:, t, :], op0=nl.multiply, operand0=rstd[:, t:t + 1])
        nisa.tensor_tensor(dst=xn[:, t, :], data1=xn[:, t, :], data2=lw, op=nl.multiply)
    xq = _sb((_P, KT, TP), gate.dtype)
    _quantize_cols(xq, xn, inv_gu, T)

    # ---- gate / up for this core's intermediate half -> [T, Ic]
    subs = _ranges(Ic, _NB)
    gv = gate.reshape((_P, KT, I))
    uv = up.reshape((_P, KT, I))
    g = _sb((T, Ic))
    ps_g = nl.ndarray((TP, len(subs), _NB), dtype=nl.float32, buffer=nl.psum)
    _stream_pass(ps_g, subs, xq, gv, ibase, KT, Ic, gate.dtype)
    _evacuate(g, ps_g, subs, og, T)
    u = _sb((T, Ic))
    ps_u = nl.ndarray((TP, len(subs), _NB), dtype=nl.float32, buffer=nl.psum)
    _stream_pass(ps_u, subs, xq, uv, ibase, KT, Ic, up.dtype)
    _evacuate(u, ps_u, subs, ou, T)

    # ---- transpose to [P, ICT, T] (local i = p * ICT + j), act = silu(g) * u, quantize
    gt = _sb((_P, ICT, T))
    ut = _sb((_P, ICT, T))
    g3 = g.reshape((T, _P, ICT))
    u3 = u.reshape((T, _P, ICT))
    for j in range(ICT):
        tp = nl.ndarray((_P, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=g3[:, :, j])
        nisa.tensor_copy(dst=gt[:, j, :], src=tp)
        tq = nl.ndarray((_P, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tq, data=u3[:, :, j])
        nisa.tensor_copy(dst=ut[:, j, :], src=tq)
    e = _sb((_P, ICT, T))
    nisa.activation(dst=e, op=nl.exp, data=gt, scale=-1.0)
    nisa.tensor_scalar(dst=e, data=e, op0=nl.add, operand0=1.0)
    r = _sb((_P, ICT, T))
    nisa.reciprocal(dst=r, data=e)
    a = _sb((_P, ICT, T))
    nisa.tensor_tensor(dst=a, data1=gt, data2=r, op=nl.multiply)
    nisa.tensor_tensor(dst=a, data1=a, data2=ut, op=nl.multiply)
    nisa.tensor_scalar(dst=a, data=a, op0=nl.multiply, operand0=inv_dn, op1=nl.minimum, operand1=FP8_MAX)
    aq = _sb((_P, ICT, TP), down.dtype)
    nisa.memset(aq, 0.0)
    nisa.tensor_scalar(dst=aq[:, :, 0:T], data=a, op0=nl.maximum, operand0=-FP8_MAX)

    # ---- down: this core's rows of down -> partial [T, H]
    dv = down.reshape((n_prgs, _P, ICT, H))
    passes = _ranges(H, _SB // 2 if H > _SB else H)
    for pi in range(len(passes)):
        s0 = passes[pi][0]
        sw = passes[pi][1] - s0
        dsubs = _ranges(sw, _NB)
        ps = nl.ndarray((TP, len(dsubs), _NB), dtype=nl.float32, buffer=nl.psum)
        _stream_pass(ps, dsubs, aq, dv[prg_id], s0, ICT, sw, down.dtype)
        o = _sb((T, sw))
        _evacuate(o, ps, dsubs, od, T)
        nisa.dma_copy(dst=out[prg_id, :, s0:s0 + sw], src=o)
    return out


def can_use_fp8_matvec(x, w) -> bool:
    T, K = x.shape
    return T <= MAX_TOKENS and K % _P == 0 and w.shape[1] % 2 == 0 and w.dtype != x.dtype


def can_use_fp8_mlp_decode(x, gate) -> bool:
    T, H = x.shape
    return T <= MAX_TOKENS and H % _P == 0 and gate.shape[1] % (2 * _P) == 0 and gate.dtype != x.dtype


def fp8_matvec(x, w, w_scale, in_scale):
    """Torch entry (inside the compiled graph): x [T, K] @ w [K, N] -> [T, N] in x.dtype."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(fp8_matvec_kernel)[2](x, w, w_scale, in_scale)


def fp8_mlp_decode(x, ln_w, gate, up, down, gate_scale, up_scale, down_scale, gate_up_in_scale,
                   down_in_scale, eps):
    """Torch entry: MLP(RMSNorm(x)) for decode tokens -> [T, H] float32 (core partials summed)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    parts = wrap_nki(fp8_mlp_decode_kernel)[2](x, ln_w, gate, up, down, gate_scale, up_scale, down_scale,
                                               gate_up_in_scale, down_in_scale, eps)
    return parts[0] + parts[1]
