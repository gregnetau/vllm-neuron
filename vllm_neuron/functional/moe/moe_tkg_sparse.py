# SPDX-License-Identifier: Apache-2.0
"""Decode MoE that runs only the local experts some token selected (dynamic expert loop).

``moe_tkg`` has two decode modes, and neither fits wide expert parallelism with few tokens:

  * selective runs one (token, expert) slot at a time over all T * K slots. Under EP most slots
    belong to other ranks, but each one still loads a full expert's weights.
  * all-expert loads every local expert once, whether or not any token picked it.

This kernel takes a compact list of the local experts that at least one token selected and runs
a hardware dynamic loop (``nl.fori_loop`` with a register bound) over it. Each iteration loads
one expert's weights and computes it for all T tokens; the per-token affinity zeroes the tokens
that did not pick it. Weight traffic is one load per distinct selected local expert.

The two physical cores of an LNC=2 core split the list in blocks of ``unroll`` entries (core c
runs blocks c, c+2, ...), so there is no cross-core traffic inside the loop. Each core writes its fp32 partial sum and the
caller adds the two.

Per expert (weights stationary on the PE, the T <= 128 tokens moving):

    gu[h, j]  (j < I: gate, j >= I: up)   SBUF [128, H/128, 2I], partition p <- rows p*H1 .. +H1
    g^T, u^T  = gu^T x^T                  PSUM [128, 2I/128, T]  (i = tile*128 + p)
    a^T       = silu(g^T * s_g) * (u^T * s_u)           (per-channel FP8 scales s, bf16)
    dn[i, h]                              SBUF [128, I/128, H],  partition p <- rows tile*128 + p
    y^T       = dn^T a^T                  PSUM [128, H/128, T]   (h = tile*128 + p)
    acc      += y^T * s_d[h] * w[t]                             (fp32)

Weight DMAs read long contiguous runs (H/128 * 2I elements per partition for gate/up, H for
down), the reason for the gate/up layout above.

trn2 hardware / compiler constraints the kernel is written around:

  * DMAs move contiguous runs only: a DMA that scatters 2- or 4-byte elements into SBUF returns
    corrupted data, so transposed operands are loaded row-contiguous and transposed on the PE.
  * One whole PSUM bank per matmul accumulation group: groups sharing a bank as slices of one
    tile accumulate wrong sums. fp32 transposes get a bank of their own (they may not share a
    bank with matmuls).
  * No stride-0 broadcast over a middle (free) dim: the token weights are applied per h tile.
  * Per-expert weights are addressed through ``.select(0, idx)`` views of [E, ...] with an SBUF
    index; a flat ``scalar_offset`` into the flattened tensor reads wrong data (seen at E = 64).
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

__all__ = ["moe_tkg_sparse", "sparse_expert_plan"]

P = 128
GU_CHUNK = 4  # gate/up rows per partition per DMA (one SBUF tile each)
PSUM_F = 512  # fp32 elements per partition in one PSUM bank
N_BANKS = 5  # PSUM banks the matmul accumulation groups rotate over


@nki.jit
def moe_tkg_sparse_kernel(
    hidden_input: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_list: nl.NkiTensor,
    num_iters: nl.NkiTensor,
    affinities: nl.NkiTensor,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    unroll: int = 2,
):
    """Sum over the listed local experts of affinity * expert(x), split over the LNC=2 cores.

    Args:
        hidden_input: [T, H] bf16, T <= 128 (tokens sit on partitions in the transposes).
        expert_gate_up_weights: [E_L, H, 2, I] bf16 or float8_e4m3fn.
        expert_down_weights: [E_L, I, H].
        expert_list: [S, 1] int32 local expert ids, the selected ones first.
        num_iters: [1, 1] int32 loop trips, ceil(n / (2 * unroll)) for n selected experts
            (S >= 2 * unroll * trips; the entries past n are unselected experts).
        affinities: [S, T] fp32. Row j: each token's weight for expert_list[j] (0 if the token
            did not select it).
        expert_gate_up_weights_scale / expert_down_weights_scale: [E_L, 2, I] / [E_L, H] fp32
            per-output-channel dequant scales (FP8), or None.
        unroll: experts per core per loop trip. Each trip issues all its weight DMAs before
            its compute, so with unroll >= 2 one expert's loads overlap another's matmuls (the
            loop boundary itself does not overlap).

    Returns:
        [2, T, H] fp32 per-core partial sums.
    """
    T, H = hidden_input.shape
    E, _, _, I = expert_gate_up_weights.shape
    I2 = 2 * I
    H1, NI, NI2 = H // P, I // P, I2 // P
    NC = (H1 + GU_CHUNK - 1) // GU_CHUNK
    fp8 = expert_gate_up_weights_scale is not None
    _, n_prgs, core = get_verified_program_sharding_info("moe_tkg_sparse", (0, 1))
    wdt = expert_gate_up_weights.dtype
    if str(wdt) in (
        "float8e4",
        "float8_e4m3fn",
    ):  # e4m3fn bytes read as trn2 e4m3 (|w| <= 240)
        wdt = nl.float8_e4m3
    idt = hidden_input.dtype

    output = nl.ndarray((n_prgs, T, H), dtype=nl.float32, buffer=nl.shared_hbm)
    # Per-expert views, selected at runtime by an SBUF index (not a flat scalar_offset).
    gu_view = expert_gate_up_weights.reshape((E, P, H1 * I2))
    dn_view = expert_down_weights.reshape((E, NI, P, H))
    if fp8:
        sgu_view = expert_gate_up_weights_scale.reshape((E, NI2, P))
        sdn_view = expert_down_weights_scale.reshape((E, H1, P))

    # x [T, H] -> x_sb[p, k, t] = x[t, p*H1 + k] (moving operand of h-tile k: x_sb[:, k, :]).
    # Contiguous load + PE transposes (no element-scattering DMA).
    x_rows = nl.ndarray((T, H), dtype=idt, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_rows, src=hidden_input)
    x_sb = nl.ndarray((P, H1, T), dtype=idt, buffer=nl.sbuf)
    xt_banks = [nl.ndarray((P, PSUM_F), dtype=idt, buffer=nl.psum) for _ in range(2)]
    for k in range(H1):
        xt = xt_banks[k % 2]
        nisa.nc_transpose(
            dst=xt[:, 0:T], data=x_rows.ap(pattern=[[H, T], [H1, P]], offset=k)
        )
        nisa.tensor_copy(dst=x_sb[:, k, :], src=xt[:, 0:T])
    acc = nl.ndarray((P, H1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(acc, 0.0)

    # PSUM: one whole bank per matmul accumulation group, round-robin, and a separate bank for
    # the fp32 transposes. 5 + 1 + 2 (x) = 8 banks.
    banks = [
        nl.ndarray((P, PSUM_F), dtype=nl.float32, buffer=nl.psum)
        for _ in range(N_BANKS)
    ]
    tp = nl.ndarray((P, PSUM_F), dtype=nl.float32, buffer=nl.psum)

    # Loop-carried buffers live outside the dynamic loop (NCC_IGCA108); one set per unrolled
    # expert, and the weights in separate tiles per DMA so the matmuls start on the first ones.
    def _buffers():
        b = dict(
            w_gu=[
                nl.ndarray(
                    (P, min(GU_CHUNK, H1 - c * GU_CHUNK) * I2),
                    dtype=wdt,
                    buffer=nl.sbuf,
                )
                for c in range(NC)
            ],
            w_dn=[nl.ndarray((P, H), dtype=wdt, buffer=nl.sbuf) for _ in range(NI)],
            gu=nl.ndarray((P, NI2, T), dtype=nl.float32, buffer=nl.sbuf),
            g=nl.ndarray((P, NI, T), dtype=nl.float32, buffer=nl.sbuf),
            a=nl.ndarray((P, NI, T), dtype=idt, buffer=nl.sbuf),
            y=nl.ndarray((P, H1, T), dtype=nl.float32, buffer=nl.sbuf),
            aff=nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf),
            eid=nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf),
        )
        if fp8:
            b["s_rows"] = nl.ndarray((H1, P), dtype=nl.float32, buffer=nl.sbuf)
            b["s_gu"] = nl.ndarray((P, NI2), dtype=nl.float32, buffer=nl.sbuf)
            b["s_dn"] = nl.ndarray((P, H1), dtype=nl.float32, buffer=nl.sbuf)
        return b

    bufs = [_buffers() for _ in range(unroll)]
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(slot, core * unroll)
    n_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sb, src=num_iters)
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, n_sb)

    def _scales_T(dst, src_view, rows, eid_ix, b):
        """dst[p, j] = scale[e, j*128 + p]: load [rows, 128] contiguous, transpose on the PE."""
        nisa.dma_copy(
            dst=b["s_rows"][0:rows, :],
            src=src_view.select(0, eid_ix),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.nc_transpose(dst=tp[:, 0:rows], data=b["s_rows"][0:rows, :])
        nisa.tensor_copy(dst=dst, src=tp[:, 0:rows])

    def _load_index(u, b):
        """Expert id and per-token weights of list entry slot + u."""
        nisa.dma_copy(
            dst=b["eid"],
            src=expert_list.ap(
                pattern=[[1, 1], [1, 1]], offset=u, scalar_offset=slot, indirect_dim=0
            ),
        )
        # The same T-run on every partition.
        nisa.dma_copy(
            dst=b["aff"],
            src=affinities.ap(
                pattern=[[0, P], [1, T]],
                offset=u * T,
                scalar_offset=slot,
                indirect_dim=0,
            ),
        )

    def _load_weights(b):
        eid_ix = b["eid"].ap(pattern=[[1, 1], [1, 1]], offset=0)
        # Gate/up: partition p <- rows [p*H1, (p+1)*H1) (contiguous runs), GU_CHUNK rows a DMA.
        gu_e = gu_view.select(0, eid_ix)
        for c in range(NC):
            n_el = b["w_gu"][c].shape[1]
            nisa.dma_copy(
                dst=b["w_gu"][c],
                src=gu_e[:, c * GU_CHUNK * I2 : c * GU_CHUNK * I2 + n_el],
                dge_mode=nisa.dge_mode.hwdge,
            )
        # Down: tile r, partition p <- row r*128 + p.
        dn_e = dn_view.select(0, eid_ix)
        for r in range(NI):
            nisa.dma_copy(dst=b["w_dn"][r], src=dn_e[r], dge_mode=nisa.dge_mode.hwdge)
        if fp8:
            _scales_T(b["s_gu"], sgu_view, NI2, eid_ix, b)
            _scales_T(b["s_dn"], sdn_view, H1, eid_ix, b)

    def _compute(b):
        """acc += w * expert(x)."""
        # g^T/u^T [i, t] = sum_h gu[h, i] x[t, h], times the per-channel scale.
        for j in range(NI2):
            bank = banks[j % N_BANKS]
            for k in range(H1):
                c, kk = k // GU_CHUNK, k % GU_CHUNK
                nisa.nc_matmul(
                    bank[:, 0:T],
                    stationary=b["w_gu"][c][:, kk * I2 + j * P : kk * I2 + (j + 1) * P],
                    moving=x_sb[:, k, :],
                    accumulate=k > 0,
                )
            if fp8:
                nisa.tensor_scalar(
                    dst=b["gu"][:, j, :],
                    data=bank[:, 0:T],
                    op0=nl.multiply,
                    operand0=b["s_gu"][:, j : j + 1],
                )
            else:
                nisa.tensor_copy(dst=b["gu"][:, j, :], src=bank[:, 0:T])
        nisa.activation(dst=b["g"], op=nl.silu, data=b["gu"][:, 0:NI, :])
        nisa.tensor_tensor(
            dst=b["a"], data1=b["g"], data2=b["gu"][:, NI:NI2, :], op=nl.multiply
        )

        # y^T [h, t] = sum_i dn[i, h] a[t, i]; y *= s_d[h] * w[t]. w is applied per h tile
        # (no stride-0 broadcast of it over the tiles).
        for n in range(H1):
            bank = banks[(NI2 + n) % N_BANKS]
            for r in range(NI):
                nisa.nc_matmul(
                    bank[:, 0:T],
                    stationary=b["w_dn"][r][:, n * P : (n + 1) * P],
                    moving=b["a"][:, r, :],
                    accumulate=r > 0,
                )
            if fp8:
                nisa.scalar_tensor_tensor(
                    dst=b["y"][:, n, :],
                    data=bank[:, 0:T],
                    op0=nl.multiply,
                    operand0=b["s_dn"][:, n : n + 1],
                    op1=nl.multiply,
                    operand1=b["aff"],
                )
            else:
                nisa.tensor_tensor(
                    dst=b["y"][:, n, :],
                    data1=bank[:, 0:T],
                    data2=b["aff"],
                    op=nl.multiply,
                )
        nisa.tensor_tensor(dst=acc, data1=acc, data2=b["y"], op=nl.add)

    def _body(_):
        # All DMAs of the trip are issued before any compute, so no engine queue holds a later
        # expert's loads behind an earlier expert's compute.
        for u in range(unroll):
            _load_index(u, bufs[u])
        for u in range(unroll):
            _load_weights(bufs[u])
        for u in range(unroll):
            _compute(bufs[u])
        nisa.tensor_scalar(dst=slot, data=slot, op0=nl.add, operand0=n_prgs * unroll)

    nl.fori_loop(0, n_reg, _body)

    # acc [h0, h1, t] -> out [t, h1*128 + h0]
    out_sb = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    for n in range(H1):
        nisa.nc_transpose(dst=tp[0:T, 0:P], data=acc[:, n, :])
        nisa.tensor_copy(dst=out_sb[:, n * P : (n + 1) * P], src=tp[0:T, 0:P])
    nisa.dma_copy(dst=output[core], src=out_sb)
    return output


def moe_tkg_sparse(
    hidden_input: Tensor,
    expert_gate_up_weights: Tensor,
    expert_down_weights: Tensor,
    expert_list: Tensor,
    num_iters: Tensor,
    affinities: Tensor,
    expert_gate_up_weights_scale: Optional[Tensor] = None,
    expert_down_weights_scale: Optional[Tensor] = None,
    unroll: int = 2,
) -> Tensor:
    """Torch entry: runs :func:`moe_tkg_sparse_kernel` on both cores of an LNC=2 core (the
    list inputs come from :func:`sparse_expert_plan`) and returns the [T, H] fp32 sum of the
    two per-core partials."""
    out = wrap_nki(moe_tkg_sparse_kernel)[2](
        hidden_input=hidden_input,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_list=expert_list,
        num_iters=num_iters,
        affinities=affinities,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        unroll=unroll,
    )
    return out[0] + out[1]


def sparse_expert_plan(affinities: Tensor, max_experts: int, unroll: int = 2):
    """Torch-side plan for :func:`moe_tkg_sparse` from dense local affinities [T, E_L] fp32
    (zero = not selected). Returns ``(expert_list [S, 1] int32, num_iters [1, 1] int32,
    affinities [S, T] fp32)``: the selected experts first in ascending id order, then the rest.
    ``S`` is ``max_experts`` rounded up to a multiple of 2 * unroll, capped at E_L (which must
    itself be a multiple).

    Static shapes and dense ops only (a cumsum and a one-hot matmul rather than sort/topk and a
    gather, which are slow on device)."""
    T, E_L = affinities.shape
    step = 2 * unroll
    S = min(E_L, -(-max_experts // step) * step)
    assert E_L % step == 0, f"E_L={E_L} must be a multiple of {step}"
    dev = affinities.device
    used = (affinities != 0).any(dim=0).float()  # [E_L]
    e = torch.arange(E_L, device=dev, dtype=torch.float32)
    tri = (e.unsqueeze(1) < e.unsqueeze(0)).float()  # [e', e]: e' < e
    before = used @ tri  # [E_L] selected experts below e
    n = used.sum()
    pos = torch.where(
        used > 0, before, n + e - before
    )  # position of expert e in the list
    j = torch.arange(S, device=dev, dtype=torch.float32)
    onehot = (pos.unsqueeze(0) == j.unsqueeze(1)).float()  # [S, E_L]
    expert_list = (onehot @ e).to(torch.int32).reshape(S, 1)
    aff = onehot @ affinities.t()  # [S, T]
    n_i = n.to(torch.int32).clamp(max=S).reshape(1, 1)
    num_iters = ((n_i + step - 1) // step).to(torch.int32)
    return expert_list, num_iters, aff
