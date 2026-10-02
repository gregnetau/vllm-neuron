# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the Gated DeltaNet (linear attention) layers of Qwen3.5 / Qwen3.8.

``gdn_decode``: one fused decode step after the input projections (B requests x 1 token):
gather each request's state page, causal conv + SiLU, q/k L2-norm, gating, the gated delta
rule, gated RMSNorm, and the in-place state write-back. Returns the normed output ready for
``out_proj``. The pure-torch reference is ``Qwen3_5GatedDeltaNet._decode_torch``.

State page (per layer, per request; see state.py / model._read_state): rows of 128 float32
split over two halves ``state_a``/``state_b`` [pages, R, 128] (R % 128 == 0, so no head
straddles the halves):
  rows [h*128 + dk]        recurrent state of head h, row dk, columns dv (S[Dk, Dv])
  rows [Hv*128 + j*NCB + f] conv state stored [K-1, C]: tap j, channels f*128 .. f*128+127
Channel c of the conv input lives on partition c % 128, column c // 128, so column f of the
q/k/v region is exactly one head (head_dim == 128).

Delta rule per head with the *old* state S (decay e = exp(g)):
  S' = e*S + k (x) delta,  delta = beta * (v - e * S^T k),  o = S'^T q = e * S^T q + delta * (k.q)
so both mat-vecs use S ([S^T k, S^T q] in one matmul), and S' is one rank-1 matmul + axpy.

Page DMAs use a runtime scalar page offset with hardware descriptor generation (vector-offset
indirect DMAs generate descriptors on GpSimd per row). silu/sigmoid are exp + vector reciprocal
and the code is ordered softplus -> exp -> rsqrt to limit activation-table reloads.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

_L2_EPS = 1e-6


def _sb(shape, dtype=nl.float32):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)


def _ps(shape):
    return nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)


def _head_runs(h0, nh, D, R):
    """Heads [h0, h0+nh) -> [[half, first_head, count]] runs that stay inside one half."""
    hpr = R // D  # heads per half
    runs = []
    h = h0
    for _ in range(nh):
        if h >= h0 + nh:
            break
        half = h // hpr
        end = min(h0 + nh, (half + 1) * hpr)
        runs.append([half, h, end - h])
        h = end
    return runs


def _page_ap(view, slot, offset, pattern):
    """Access pattern into page ``slot`` (runtime [1, 1] int32 SBUF) of view [P, R*D]."""
    return view.ap(pattern=pattern, offset=offset, scalar_offset=slot, indirect_dim=0)


def _heads_dma(tile3, h0, nh, D, R, va, vb, slot, write):
    """Gather/scatter heads [h0, h0+nh) of a page <-> tile3 [128, nh, D] (partition = dk).
    Hardware DGE with a scalar page offset: no per-row descriptors on GpSimd."""
    runs = _head_runs(h0, nh, D, R)
    for ri in range(len(runs)):
        run = runs[ri]
        view = va
        if run[0] == 1:
            view = vb
        hfirst = run[1]
        cnt = run[2]
        hbm = _page_ap(view, slot, (hfirst - run[0] * (R // D)) * D * D, [[D, D], [D * D, cnt], [1, D]])
        sl = tile3[:, hfirst - h0:hfirst - h0 + cnt, :]
        if write:
            nisa.dma_copy(dst=hbm, src=sl, dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip)
        else:
            nisa.dma_copy(dst=sl, src=hbm, dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip)


def _bcast_load(src, n, offset):
    """HBM vector src[offset:offset+n] -> [128, n] f32 tile, same values on every partition."""
    raw = _sb((128, n), src.dtype)
    nisa.dma_copy(dst=raw, src=src.ap(pattern=[[0, 128], [1, n]], offset=offset))
    out = _sb((128, n))
    nisa.tensor_copy(dst=out, src=raw)
    return out


def _cols_load(src, ncols, offset, D):
    """HBM row-major vector, column-blocked: tile[p, f] = src[offset + f*D + p] -> f32."""
    raw = _sb((D, ncols), src.dtype)
    nisa.dma_copy(dst=raw, src=src.ap(pattern=[[1, D], [D, ncols]], offset=offset))
    out = _sb((D, ncols))
    nisa.tensor_copy(dst=out, src=raw)
    return out


def _sigmoid(x, n):
    """1 / (1 + exp(-x)): exp on the scalar engine, reciprocal on the vector engine."""
    e = _sb((128, n))
    nisa.activation(dst=e, op=nl.exp, data=x, scale=-1.0)
    nisa.tensor_scalar(dst=e, data=e, op0=nl.add, operand0=1.0)
    r = _sb((128, n))
    nisa.reciprocal(dst=r, data=e)
    return r


def _rows_load_t(src, nrows, offset, D):
    """HBM rows src[offset + r*D : ... + D] for r < nrows -> [D, nrows] f32 (column r = row r).
    One contiguous DMA plus an on-chip transpose instead of an element-strided DMA."""
    raw = _sb((nrows, D), src.dtype)
    nisa.dma_copy(dst=raw, src=src.ap(pattern=[[D, nrows], [1, D]], offset=offset))
    f32 = _sb((nrows, D))
    nisa.tensor_copy(dst=f32, src=raw)
    ps = _ps((D, nrows))
    nisa.nc_transpose(dst=ps, data=f32)
    out = _sb((D, nrows))
    nisa.tensor_copy(dst=out, src=ps)
    return out


def _silu(x, n):
    y = _sb((128, n))
    nisa.tensor_tensor(dst=y, data1=x, data2=_sigmoid(x, n), op=nl.multiply)
    return y


@nki.jit
def gdn_decode_kernel(mixed, z, b, a, conv_w, A_log, dt_bias, norm_w, state_a, state_b, slots,
                      eps=1e-6):
    """mixed [B, C] / z [B, Hv*D] / b, a [B, Hv] (bf16 projections); conv_w [C, K] bf16;
    A_log, dt_bias [Hv] f32; norm_w [D]; state_a/b [P, R, 128] f32 (updated in place);
    slots [B] int32. Returns (out [B, Hv*D] bf16, state_a, state_b)."""
    # rev 11: the NKI compile cache keys on this function's source only; bump on helper edits.
    B, C = mixed.shape
    D = norm_w.shape[0]
    Hv = z.shape[1] // D
    K = conv_w.shape[1] * 128 // C if conv_w.shape[0] == 128 else conv_w.shape[1]
    Hk = (C - Hv * D) // (2 * D)
    rep = Hv // Hk
    P, R, W = state_a.shape
    NCB = C // D            # channel blocks (columns)
    CR = (K - 1) * NCB      # conv-state rows
    CONV0 = Hv * D          # first conv row in the page
    assert D == 128 and W == D and C % D == 0 and R % D == 0
    assert CONV0 % R + CR <= R, "conv rows must not straddle the page halves"

    _, n_prgs, prg_id = get_verified_program_sharding_info("gdn_decode", (0, 1))
    assert Hv % n_prgs == 0
    hp = Hv // n_prgs
    h0 = prg_id * hp

    va = state_a.reshape((P, R * D))   # page-major views: a runtime slot selects the page
    vb = state_b.reshape((P, R * D))
    conv_view = va
    if CONV0 >= R:
        conv_view = vb
    conv_off = (CONV0 % R) * D
    out = nl.ndarray((B, Hv * D), dtype=z.dtype, buffer=nl.shared_hbm)

    # ---- request-independent constants
    ones = _sb((128, 128))
    nisa.memset(ones, 1.0)
    w_raw = _sb((128, NCB, K), conv_w.dtype)  # w[p, f, j] = conv_w[f*128 + p, j]
    if conv_w.shape[0] == 128:  # pre-laid out [128, NCB * K] (contiguous per partition)
        nisa.dma_copy(dst=w_raw, src=conv_w.reshape((128, NCB, K)))
    else:
        nisa.dma_copy(dst=w_raw, src=conv_w.ap(pattern=[[K, 128], [D * K, NCB], [1, K]], offset=0))
    w = _sb((128, NCB, K))
    nisa.tensor_copy(dst=w, src=w_raw)
    normw = _cols_load(norm_w, 1, 0, D)            # [128, 1]
    alog = _bcast_load(A_log, hp, h0)              # own heads only
    dtb = _bcast_load(dt_bias, hp, h0)
    neg_c = _sb((128, hp))                         # -exp(A_log)
    nisa.activation(dst=neg_c, op=nl.exp, data=alog)
    nisa.tensor_scalar(dst=neg_c, data=neg_c, op0=nl.multiply, operand0=-1.0)
    oh_i = _sb((hp, hp), nl.int32)                 # one-hot rows: onehot[p, i] = (p == i)
    nisa.iota(oh_i, [[-1, hp]], offset=0, channel_multiplier=1)
    oh_f = _sb((hp, hp))
    nisa.tensor_copy(dst=oh_f, src=oh_i)
    onehot = _sb((hp, hp))
    nisa.tensor_scalar(dst=onehot, data=oh_f, op0=nl.equal, operand0=0.0)

    for bi in range(B):
        slot = _sb((1, 1), nl.int32)
        nisa.dma_copy(dst=slot, src=slots[bi:bi + 1].reshape((1, 1)))

        # ---- loads: state tiles (one DMA per half), conv rows, projections
        S = _sb((128, hp, D))
        _heads_dma(S, h0, hp, D, R, va, vb, slot, write=False)
        cs_rows = _sb((CR, D))
        nisa.dma_copy(dst=cs_rows, src=_page_ap(conv_view, slot, conv_off, [[D, CR], [1, D]]),
                      dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip)
        x = _rows_load_t(mixed, NCB, bi * C, D)     # x[p, f] = mixed[bi, f*128 + p]
        a_b = _bcast_load(a, hp, bi * Hv + h0)
        b_b = _bcast_load(b, hp, bi * Hv + h0)
        zl = _rows_load_t(z, hp, bi * Hv * D + h0 * D, D)

        # ---- gating first (softplus table, then everything exp-based, then rsqrt):
        # decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
        ga = _sb((128, hp))
        nisa.tensor_tensor(dst=ga, data1=a_b, data2=dtb, op=nl.add)
        sp = _sb((128, hp))
        nisa.activation(dst=sp, op=nl.softplus, data=ga)
        nisa.tensor_tensor(dst=sp, data1=sp, data2=neg_c, op=nl.multiply)
        decay = _sb((128, hp))
        nisa.activation(dst=decay, op=nl.exp, data=sp)
        beta = _sigmoid(b_b, hp)
        zs = _silu(zl, hp)

        # ---- causal conv (channels on partitions) + SiLU
        cs_ps = _ps((128, CR))
        nisa.nc_transpose(dst=cs_ps, data=cs_rows)
        cs = _sb((128, CR))
        nisa.tensor_copy(dst=cs, src=cs_ps)
        y = _sb((128, NCB))
        nisa.tensor_tensor(dst=y, data1=x, data2=w[:, :, K - 1], op=nl.multiply)
        for j in range(K - 1):
            t = _sb((128, NCB))
            nisa.tensor_tensor(dst=t, data1=cs[:, j * NCB:(j + 1) * NCB], data2=w[:, :, j], op=nl.multiply)
            nisa.tensor_tensor(dst=y, data1=y, data2=t, op=nl.add)
        xs = _silu(y, NCB)

        if prg_id == 0:  # new conv state: drop the oldest tap, append x
            ncs = _sb((128, CR))
            nisa.tensor_copy(dst=ncs[:, 0:(K - 2) * NCB], src=cs[:, NCB:(K - 1) * NCB])
            nisa.tensor_copy(dst=ncs[:, (K - 2) * NCB:(K - 1) * NCB], src=x)
            ncs_ps = _ps((CR, 128))
            nisa.nc_transpose(dst=ncs_ps, data=ncs)
            ncs_rows = _sb((CR, D))
            nisa.tensor_copy(dst=ncs_rows, src=ncs_ps)
            nisa.dma_copy(dst=_page_ap(conv_view, slot, conv_off, [[D, CR], [1, D]]), src=ncs_rows,
                          dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip)

        # ---- q, k L2 norm (sum over Dk = partitions via ones-matmul), (sum + eps)^-0.5
        qk = xs[:, 0:2 * Hk]
        sq = _sb((128, 2 * Hk))
        nisa.tensor_tensor(dst=sq, data1=qk, data2=qk, op=nl.multiply)
        ssum = _ps((128, 2 * Hk))
        nisa.nc_matmul(dst=ssum, stationary=ones, moving=sq)
        rinv = _sb((128, 2 * Hk))
        nisa.activation(dst=rinv, op=nl.rsqrt, data=ssum, bias=_L2_EPS)
        qkn = _sb((128, 2 * Hk))
        nisa.tensor_tensor(dst=qkn, data1=qk, data2=rinv, op=nl.multiply)
        nisa.tensor_scalar(dst=qkn[:, 0:Hk], data=qkn[:, 0:Hk], op0=nl.multiply, operand0=float(D) ** -0.5)
        kq = _sb((128, Hk))
        nisa.tensor_tensor(dst=kq, data1=qkn[:, 0:Hk], data2=qkn[:, Hk:2 * Hk], op=nl.multiply)
        kqdot_ps = _ps((128, Hk))
        nisa.nc_matmul(dst=kqdot_ps, stationary=ones, moving=kq)
        kqm = _sb((128, Hk, 2))                     # matmul moving operand per key head: [k, q]
        nisa.tensor_copy(dst=kqm[:, :, 0], src=qkn[:, Hk:2 * Hk])
        nisa.tensor_copy(dst=kqm[:, :, 1], src=qkn[:, 0:Hk])
        ke = _sb((128, hp))                         # k and k.q expanded to this program's v-heads
        kqd = _sb((128, hp))
        for i in range(hp):
            kh = (h0 + i) // rep
            nisa.tensor_copy(dst=ke[:, i:i + 1], src=qkn[:, Hk + kh:Hk + kh + 1])
            nisa.tensor_copy(dst=kqd[:, i:i + 1], src=kqdot_ps[:, kh:kh + 1])
        v = xs[:, 2 * Hk + h0:2 * Hk + h0 + hp]

        # ---- delta rule, all of this program's heads at once
        mv = _ps((128, hp, 2))                      # [:, i, 0] = S_i^T k, [:, i, 1] = S_i^T q
        for i in range(hp):
            kh = (h0 + i) // rep
            nisa.nc_matmul(dst=mv[:, i, :], stationary=S[:, i, :], moving=kqm[:, kh, :])
        t = _sb((128, hp))
        nisa.tensor_tensor(dst=t, data1=mv[:, :, 0], data2=decay, op=nl.multiply)
        nisa.tensor_tensor(dst=t, data1=v, data2=t, op=nl.subtract)
        delta = _sb((128, hp))
        nisa.tensor_tensor(dst=delta, data1=t, data2=beta, op=nl.multiply)
        o = _sb((128, hp))
        nisa.tensor_tensor(dst=o, data1=mv[:, :, 1], data2=decay, op=nl.multiply)
        t2 = _sb((128, hp))
        nisa.tensor_tensor(dst=t2, data1=delta, data2=kqd, op=nl.multiply)
        nisa.tensor_tensor(dst=o, data1=o, data2=t2, op=nl.add)

        # rank-1 updates: outer[dk, i*D + dv] = k_i[dk] * delta_i[dv] via a block-diagonal moving
        dT_ps = _ps((hp, 128))
        nisa.nc_transpose(dst=dT_ps, data=delta)
        dT = _sb((hp, 128))
        nisa.tensor_copy(dst=dT, src=dT_ps)
        kT_ps = _ps((hp, 128))
        nisa.nc_transpose(dst=kT_ps, data=ke)
        kT = _sb((hp, 128))
        nisa.tensor_copy(dst=kT, src=kT_ps)
        mbd = _sb((hp, hp, D))
        for i in range(hp):
            nisa.tensor_scalar(dst=mbd[:, i, :], data=dT, op0=nl.multiply, operand0=onehot[:, i:i + 1])
        S_new = _sb((128, hp, D))
        HG = 512 // D                               # heads per PSUM bank (512 f32)
        for g0 in range(0, hp, HG):
            ng = min(HG, hp - g0)
            outer = _ps((128, ng, D))
            nisa.nc_matmul(dst=outer, stationary=kT, moving=mbd[:, g0:g0 + ng, :])
            for i in range(ng):
                nisa.scalar_tensor_tensor(dst=S_new[:, g0 + i, :], data=S[:, g0 + i, :], op0=nl.multiply,
                                          operand0=decay[:, g0 + i:g0 + i + 1], op1=nl.add,
                                          operand1=outer[:, i, :])
        _heads_dma(S_new, h0, hp, D, R, va, vb, slot, write=True)

        # ---- gated RMSNorm over Dv (partitions): o * (mean(o^2) + eps)^-0.5 * w * silu(z)
        o2 = _sb((128, hp))
        nisa.tensor_tensor(dst=o2, data1=o, data2=o, op=nl.multiply)
        osum = _ps((128, hp))
        nisa.nc_matmul(dst=osum, stationary=ones, moving=o2)
        rr = _sb((128, hp))
        nisa.activation(dst=rr, op=nl.rsqrt, data=osum, scale=1.0 / D, bias=eps)
        on = _sb((128, hp))
        nisa.tensor_tensor(dst=on, data1=o, data2=rr, op=nl.multiply)
        nisa.tensor_scalar(dst=on, data=on, op0=nl.multiply, operand0=normw)
        res = _sb((128, hp))
        nisa.tensor_tensor(dst=res, data1=on, data2=zs, op=nl.multiply)
        res_ps = _ps((hp, 128))                     # rows = heads: one contiguous store
        nisa.nc_transpose(dst=res_ps, data=res)
        res_t = _sb((hp, 128), out.dtype)
        nisa.tensor_copy(dst=res_t, src=res_ps)
        nisa.dma_copy(dst=out.ap(pattern=[[D, hp], [1, D]], offset=bi * Hv * D + h0 * D), src=res_t)

    return out, state_a, state_b


def gdn_decode(mixed, z, b, a, conv_w, A_log, dt_bias, norm_w, state_a, state_b, slots, eps):
    """Torch entry (inside the compiled graph). Returns out [B, Hv*D]; states updated in place."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    out, _, _ = wrap_nki(gdn_decode_kernel)[2](
        mixed, z, b, a, conv_w, A_log, dt_bias, norm_w, state_a, state_b, slots, eps)
    return out
