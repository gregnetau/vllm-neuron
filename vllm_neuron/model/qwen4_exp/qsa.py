# SPDX-License-Identifier: Apache-2.0
"""
QSA (Qwen Sparse Attention) indexer and token selection for Qwen3.8-Flash-Next
=============================================================================

<-- MODEL-SPECIFIC: exact port of HF ``Qwen4ExpTextQSAIndexer`` (transformers 5.17
``modular_qwen4_exp.py``) as pure, static-shape, traceable functions. The attention layer
(``model.py``) composes them; nothing here owns weights or caches.

Math (per attention layer, ``R`` = compress ratio 4, ``D`` = 128, ``nh`` = 4 heads):

  * ``index_qk_proj`` [H -> (nh + 1) * D] gives nh query heads and one raw key per token.
  * Queries: zero-centered RMSNorm (gamma = 1 + w), then NeoX rope on the first
    ``rotary_dim`` (64) channels at the token position (same inv_freq as the main attention).
  * Raw keys go to a paged key cache (the attention layer's side cache). Each step the compressed keys are recomputed
    from the cached raw keys: block b = rope(k_norm(mean_fp32(raw[R*b : R*b + R]).to(dtype)),
    position R*b).
  * Score(q, b) = sum_h relu(q_h . kc_b) / sqrt(D). A query at position p sees the complete
    blocks b < (p + 1) // R, takes the top ``budget_blocks`` (512) of them (all if fewer),
    expands each to its R tokens and always adds the open tail [(p + 1) // R * R, p].
  * The core attention then attends only to the selected tokens (AND causal).

Static shapes: everything is sized by the context bucket ``S`` (cached-token slots, a
multiple of the block size). If ``S // R <= budget_blocks`` every visible block is selected
and QSA is exactly dense causal attention; ``qsa_is_dense`` is the trace-time (Python) check
that lets the caller skip the indexer (the raw keys must still be written).

Top-k: exact, without sort/topk. ``topk_block_mask`` finds the k-th largest score by
bisection on the IEEE bit pattern (non-negative fp32 scores order like their int32 bits:
31 compare+count passes), then breaks ties at that value by lower block index with a second
bisection on the index (ceil(log2(N)) passes). Exactly min(budget, visible) blocks are
selected per query. Tie rule vs HF: HF calls ``torch.topk`` per query, whose choice among
equal scores is an implementation detail (on CPU it is not lowest-index-first in general).
Ties only matter when fewer than ``budget`` visible blocks have a score above the k-th
value (in practice: zero scores from the ReLU), and then any choice among the tied blocks
is an equally valid top-k; this module always takes the lowest block indices.

cos/sin convention (indexer only): HF layout ``[N, rotary_dim]`` = cat(cos(f), cos(f)),
f = pos * inv_freq, built by ``qsa_rope_cos_sin`` from integer positions. It is NOT the
permuted-channel layout of the Qwen3.5 main attention (``kernel_rope_angles_from_inv_freq``).

Dtypes follow HF: projection / rope in the model dtype, norms in fp32 (output cast back),
the block mean in fp32 (cast back), scores in fp32.
"""

import math

import torch
import torch.nn.functional as F

from vllm_neuron.model.kv_cache import paged_slot_indices

__all__ = [
    "qsa_is_dense",
    "qsa_rope_cos_sin",
    "apply_qsa_rope",
    "zero_centered_rms_norm",
    "index_qk",
    "write_raw_keys",
    "gather_raw_keys",
    "write_compressed_keys",
    "compressed_keys",
    "index_scores",
    "topk_block_mask",
    "select_blocks",
    "token_mask",
    "qsa_token_mask",
    "decode_mask_from_token_mask",
    "compact_slots",
    "topk_block_ids",
    "compact_decode_selection",
    "compact_decode_mask",
]


def qsa_is_dense(S: int, budget_blocks: int, ratio: int = 4) -> bool:
    """True if a context bucket of ``S`` token slots always selects every visible block.

    Python-level (static shapes): when True the selection equals the causal mask, so the
    caller can skip scoring and run its dense path. Raw keys must still be written.
    """
    return S // ratio <= budget_blocks


# =============================================================================
# Section 1: norm / rope
# =============================================================================


def zero_centered_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm over the last dim with gamma = 1 + w (pass the folded gamma), HF numerics:
    normalise and scale in fp32, cast back to ``x.dtype``."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * gamma.float()).to(x.dtype)


def qsa_rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin ``[N, rotary_dim]`` in the HF layout cat(f, f), f = pos * inv_freq.

    positions: ``[N]`` integer token positions (exact in fp32 up to 2**24).
    inv_freq: ``[rotary_dim // 2]`` fp32 (``qwen3_5.weights.rope_inv_freq(64, theta)``).
    Angles are fp32 (as HF), the result is cast to ``dtype`` (the model dtype, as HF).
    """
    ang = positions.float()[:, None] * inv_freq.float()[None, :]
    full = torch.cat([ang, ang], dim=-1)
    return full.cos().to(dtype), full.sin().to(dtype)


def apply_qsa_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """NeoX rotate_half rope on the first ``R = cos.shape[-1]`` channels of ``x``.

    x: ``[N, ..., D]``; cos/sin: ``[N, R]`` (broadcast over the middle dims). Computed in
    ``x.dtype`` like HF ``apply_rotary_pos_emb``.
    """
    R = cos.shape[-1]
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (R,)
    cos, sin = cos.to(x.dtype).view(shape), sin.to(x.dtype).view(shape)
    xr, xp = x[..., :R], x[..., R:]
    x1, x2 = xr[..., : R // 2], xr[..., R // 2 :]
    rot = xr * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat([rot, xp], dim=-1)


# =============================================================================
# Section 2: projection and the raw-key cache
# =============================================================================


def index_qk(
    hidden: torch.Tensor,
    w_qk: torch.Tensor,
    q_gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    n_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer projection for the step's tokens.

    hidden: ``[T, H]``.
    w_qk: ``[H, (n_heads + 1) * D]`` (plugin ``[in, out]`` convention, i.e. the transposed HF
        ``index_qk_proj.weight``; columns = n_heads query heads, then the key head).
        Replicated across TP ranks (it is small: 2560 x 640).
    q_gamma: ``[D]`` folded q_layernorm gamma (1 + w).
    cos, sin: ``[T, rotary_dim]`` from ``qsa_rope_cos_sin`` at the token positions.

    Returns ``(q_idx [T, n_heads, D] normed + roped, k_raw [T, D] un-normed, un-roped)``,
    both in ``hidden.dtype``. ``k_raw`` is what goes into the raw-key cache.
    """
    T = hidden.shape[0]
    qk = hidden @ w_qk.to(hidden.dtype)
    D = qk.shape[-1] // (n_heads + 1)
    q = qk[:, : n_heads * D].reshape(T, n_heads, D)
    k_raw = qk[:, n_heads * D :]
    q = apply_qsa_rope(zero_centered_rms_norm(q, q_gamma, eps), cos, sin)
    return q, k_raw


def write_raw_keys(
    key_cache: torch.Tensor, k_raw: torch.Tensor, slot_mapping: torch.Tensor
):
    """Scatter ``k_raw [T, D]`` into the paged raw-key cache in place.

    key_cache: ``[num_blocks, 1, block_size, D]`` or ``[num_blocks, block_size, D]``.
    slot_mapping: ``[T]`` flat cache slots; -1 / out of range entries are redirected to the
        null block (``paged_slot_indices``).
    """
    # Index the bound tensor itself, not a view: on the XLA route an in-place write through a
    # view (e.g. squeeze) is not seen by later reads of the base in the same graph.
    bs = key_cache.shape[-2]
    blk, pos = paged_slot_indices(slot_mapping, bs, key_cache.shape[0])
    blk, pos = blk.long(), pos.long()
    if key_cache.dim() == 4:
        key_cache.index_put_(
            (blk, torch.zeros_like(blk), pos), k_raw.to(key_cache.dtype)
        )
    else:
        key_cache.index_put_((blk, pos), k_raw.to(key_cache.dtype))


def gather_raw_keys(key_cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Gather the cached raw keys of each request: ``[B, S, D]``, S = n_blk * block_size.

    block_table: ``[B, n_blk]`` (-1 = unused -> null block; those slots are never visible). Slots past a request's length hold stale data (possibly NaN from
    recycled pages); ``select_blocks`` masks them with ``where``, never arithmetic.
    """
    cache = key_cache.squeeze(1) if key_cache.dim() == 4 else key_cache
    B, n_blk = block_table.shape
    bt = block_table.clamp(min=0).reshape(-1).long()
    return cache.index_select(0, bt).reshape(B, n_blk * cache.shape[1], cache.shape[-1])


# =============================================================================
# Section 3: compressed keys, scores, top-k
# =============================================================================


def _compress_blocks(
    raw: torch.Tensor,
    starts: torch.Tensor,
    k_gamma: torch.Tensor,
    inv_freq: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compressed keys of blocks: raw ``[N, ..., ratio, D]`` (the block's raw keys), starts
    ``[N]`` (block start positions) -> ``[N, ..., D]``: rope(k_norm(mean_fp32(raw)), start)."""
    pooled = raw.float().mean(dim=-2).to(raw.dtype)
    normed = zero_centered_rms_norm(pooled, k_gamma, eps)
    cos, sin = qsa_rope_cos_sin(starts, inv_freq, raw.dtype)
    return apply_qsa_rope(normed, cos, sin)


def write_compressed_keys(
    ckey_cache: torch.Tensor,
    key_cache: torch.Tensor,
    q_pos: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_gamma: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    ratio: int = 4,
    eps: float = 1e-6,
):
    """Update the compressed-key cache for the blocks of this step's tokens, in place.

    ckey_cache: ``[num_blocks, 1, block_size / ratio, D]``, entry ``j`` of a cache block holds
        the compressed key of its tokens ``[ratio * j, ratio * j + ratio)``.
    key_cache: the raw-key cache ``[num_blocks, 1, block_size, D]``, this step's keys already
        written (``write_raw_keys``).
    q_pos, slot_mapping: ``[T]`` positions / cache slots of this step's tokens.

    Each token recomputes the entry of its own block from the cached raw keys, so a block's
    entry is final once its last token is written (the tokens of one block in one step
    write identical values). Entries of incomplete blocks are garbage and never visible.
    Values equal ``compressed_keys`` over the gathered raw keys.
    """
    bs = key_cache.shape[-2]
    blk, pos = paged_slot_indices(slot_mapping, bs, key_cache.shape[0])
    blk, pos = blk.long(), pos.long()
    start = pos - pos % ratio  # block start within the cache block
    offs = start.unsqueeze(-1) + torch.arange(ratio, device=pos.device)  # [T, ratio]
    b2 = blk.unsqueeze(-1).expand_as(offs)
    raw = key_cache[b2, torch.zeros_like(b2), offs]  # [T, ratio, D]
    p = q_pos.reshape(-1).to(torch.int32)
    kc = _compress_blocks(raw, p - p % ratio, k_gamma, inv_freq, eps)
    ckey_cache.index_put_(
        (blk, torch.zeros_like(blk), start // ratio), kc.to(ckey_cache.dtype)
    )


def compressed_keys(
    k_raw_ctx: torch.Tensor,
    k_gamma: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    ratio: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compressed block keys ``[B, S // ratio, D]`` from the raw keys ``[B, S, D]``.

    kc[b] = rope(k_norm(mean_fp32(raw[ratio*b : ratio*b + ratio]).to(dtype)), ratio*b), in
    ``k_raw_ctx.dtype``. Every block of the bucket is computed (the partial / unwritten
    ones are garbage and are masked by visibility in ``select_blocks``).
    """
    B, S, D = k_raw_ctx.shape
    nb = S // ratio
    starts = torch.arange(nb, device=k_raw_ctx.device, dtype=torch.int32) * ratio
    # rope broadcasts over dim 0: put the block axis first
    raw = k_raw_ctx.reshape(B, nb, ratio, D).transpose(0, 1)
    return _compress_blocks(raw, starts, k_gamma, inv_freq, eps).transpose(0, 1)


def _visible_blocks(q_pos: torch.Tensor, nb: int, ratio: int) -> torch.Tensor:
    """Complete blocks visible to each query, ``(p + 1) // ratio`` clamped to [0, nb]. int32."""
    p1 = q_pos.to(torch.int32) + 1
    return torch.div(p1, ratio, rounding_mode="floor").clamp(0, nb)


def index_scores(
    q_idx: torch.Tensor, kc: torch.Tensor, q_pos: torch.Tensor, ratio: int = 4
) -> torch.Tensor:
    """fp32 block scores ``[B, Tq, NB]``: sum_h relu(q_h . kc) / sqrt(D) for visible blocks,
    exactly -1 for invisible ones (also replaces any NaN/Inf from stale cache slots).

    q_idx: ``[B, Tq, nh, D]``; kc: ``[B, NB, D]``; q_pos: ``[B, Tq]`` token positions.
    """
    B, Tq, nh, D = q_idx.shape
    nb = kc.shape[1]
    s = torch.matmul(q_idx.reshape(B, Tq * nh, D).float(), kc.float().transpose(1, 2))
    s = torch.relu(s).reshape(B, Tq, nh, nb).sum(dim=2) / math.sqrt(D)
    vis = _visible_blocks(q_pos, nb, ratio)
    blk = torch.arange(nb, device=kc.device, dtype=torch.int32)
    visible = blk.view(1, 1, nb) < vis.unsqueeze(-1)
    return torch.where(visible, s, torch.full((), -1.0, device=s.device))


def _bits_to_float(b: torch.Tensor) -> torch.Tensor:
    """fp32 value of the non-negative IEEE bit patterns ``b`` (int32 in [0, 2**31)), without
    a bitcast (``Tensor.view(dtype)`` does not lower on the XLA route): exact arithmetic,
    mantissa in [1, 2) (or [0, 1) for subnormals) times 2**(e - 127), built from the exponent
    bits with exact power-of-two multiplies (no constant tensors: they do not trace).
    Exponent 255 (inf / NaN patterns) maps to +inf. Subnormal results (< 2**-126) may flush
    to 0 on device; scores that small are 0 in practice."""
    e = b >> 23
    m = (b & 0x7FFFFF) + ((e > 0).to(b.dtype) << 23)
    e = e.clamp(min=1)  # subnormals: m * 2**-23 * 2**-126
    one = torch.ones_like(b, dtype=torch.float32)
    p = one
    for j in range(7):  # 2**(e & 127), exact up to 2**127
        p = p * torch.where(((e >> j) & 1) > 0, one * float(2 ** (1 << j)), one)
    hi = (e >> 7) > 0  # e >= 128: * 2; else * 2**-127 (as two normal factors)
    p = (
        p
        * torch.where(hi, one * 2.0, one * 2.0**-63)
        * torch.where(hi, one, one * 2.0**-64)
    )
    return (m.to(torch.float32) * (2.0**-23)) * p


def _radix_search(
    count_ok, nbits: int, rows_like: torch.Tensor, r: int
) -> torch.Tensor:
    """Largest x in [0, 2**nbits) with ``count_ok(x)`` True, given it is True at 0 and
    monotone (True then False). ``count_ok(t [..., m]) -> bool [..., m]``. int32."""
    lo = torch.zeros_like(rows_like)
    b = nbits
    while b > 0:
        rr = min(r, b)
        step = 1 << (b - rr)
        j = torch.arange(1, 1 << rr, device=lo.device, dtype=torch.int32) * step
        ok = count_ok(lo.unsqueeze(-1) + j)  # [..., 2**rr - 1], monotone in j
        lo = lo + ok.sum(-1, dtype=torch.int32) * step
        b -= rr
    return lo


def _bitcast_lowers() -> bool:
    """Whether ``Tensor.view(torch.int32)`` lowers: the native route (torch.compile) and CPU;
    not the XLA route."""
    from vllm_neuron import envs

    return envs.is_native_backend()


def topk_block_mask(
    scores: torch.Tensor,
    k: torch.Tensor,
    radix_bits: int = 1,
    int_bits: bool | None = None,
) -> torch.Tensor:
    """Exact top-k as a bool mask, without sort: ``[..., N]`` -> ``[..., N]``.

    scores: fp32, ``>= 0`` (or -0.0) where eligible, negative where not (never selected).
    k: int32 ``[...]`` per-row count, at most the number of eligible entries.
    radix_bits: bits resolved per pass (2**r - 1 thresholds tested at once). 1 = bisection,
        the default: on trn2 r = 5 was ~10x slower for decode (B=8, 64k context) than r = 1.
    int_bits: compare the scores' int32 bit patterns with each threshold (one bitcast)
        instead of rebuilding every threshold as fp32 (``_bits_to_float``, ~130 ops per pass
        and 87% of the selection's ops: the native route's compiler front end then spends
        hours on a 4096-token prefill graph). The same order for scores >= 0, so the same
        mask; negative scores and -0.0 have negative patterns and are never counted. None:
        where the bitcast lowers (``_bitcast_lowers``).
    Selects every score above the k-th largest value v and, among the scores equal to v,
    the lowest-index ones, so exactly k entries per row. All ops are compare / count /
    where on static shapes: ceil(31 / r) + ceil(log2(N) / r) passes, each testing
    2**r - 1 thresholds over the row.
    """
    N = scores.shape[-1]
    r = radix_bits
    k = k.to(torch.int32)
    s = scores.unsqueeze(-2)  # [..., 1, N]

    # 1) v = the k-th largest value. For x >= 0 the int32 bit pattern is monotone in x, so
    #    v's bits = the largest t in [0, 2**31) with count(scores >= t) >= k. Patterns at or
    #    above +inf map to +inf and count 0. k == 0 gives +inf: nothing selected.
    if int_bits is None:
        int_bits = _bitcast_lowers()
    if int_bits:
        s_bits = scores.contiguous().view(torch.int32).unsqueeze(-2)  # [..., 1, N]

        def value_ok(t):
            return (s_bits >= t.unsqueeze(-1)).sum(-1, dtype=torch.int32) >= k.unsqueeze(-1)

    else:

        def value_ok(t):
            thr = _bits_to_float(t).unsqueeze(-1)  # [..., m, 1]
            return (s >= thr).sum(-1, dtype=torch.int32) >= k.unsqueeze(-1)

    v = _bits_to_float(_radix_search(value_ok, 31, k, r)).unsqueeze(-1)
    gt = scores > v
    eq = scores == v
    need = k - gt.sum(-1, dtype=torch.int32)
    # 2) ties at v, lowest index first: I = the index of the need-th tie = the largest i with
    #    count(eq & idx < i) < need (need >= 1); select eq & idx <= I.
    idx = torch.arange(N, device=scores.device, dtype=torch.int32)
    eq_i = eq.unsqueeze(-2)

    def index_ok(t):
        c = (eq_i & (idx < t.unsqueeze(-1))).sum(-1, dtype=torch.int32)
        return c < need.unsqueeze(-1)

    nbits = max(1, (N - 1).bit_length())
    last = _radix_search(index_ok, nbits, k, r)
    return gt | (eq & (idx <= last.unsqueeze(-1)) & (need > 0).unsqueeze(-1))


def select_blocks(
    q_idx: torch.Tensor,
    kc: torch.Tensor,
    q_pos: torch.Tensor,
    budget_blocks: int,
    ratio: int = 4,
) -> torch.Tensor:
    """Selected blocks ``bool [B, Tq, NB]``: the top min(budget_blocks, visible) visible
    blocks per query (ties -> lower block index).

    q_idx: ``[B, Tq, nh, D]`` (from ``index_qk``); kc: ``[B, NB, D]`` (``compressed_keys``);
    q_pos: ``[B, Tq]`` token positions. Prefill: B = 1, Tq = chunk. Decode: Tq = 1 (1 + k
    for MTP verify).
    """
    nb = kc.shape[1]
    scores = index_scores(q_idx, kc, q_pos, ratio)
    k = _visible_blocks(q_pos, nb, ratio).clamp(max=budget_blocks)
    return topk_block_mask(scores, k)


def token_mask(
    block_sel: torch.Tensor, q_pos: torch.Tensor, S: int, ratio: int = 4
) -> torch.Tensor:
    """Allowed keys ``bool [B, Tq, S]``: tokens of the selected blocks, plus the open tail
    [(p + 1) // ratio * ratio, p], AND causal (key position <= p).

    block_sel: ``[B, Tq, S // ratio]``; q_pos: ``[B, Tq]``. Positions are cache slots of
    the request (slot j = token j).
    """
    B, Tq, nb = block_sel.shape
    p = q_pos.to(torch.int32).unsqueeze(-1)  # [B, Tq, 1]
    j = torch.arange(S, device=block_sel.device, dtype=torch.int32)
    blocks = block_sel.unsqueeze(-1).expand(B, Tq, nb, ratio).reshape(B, Tq, nb * ratio)
    if nb * ratio < S:
        blocks = torch.cat([blocks, blocks.new_zeros(B, Tq, S - nb * ratio)], dim=-1)
    tail_start = torch.div(p + 1, ratio, rounding_mode="floor") * ratio
    return (blocks | (j >= tail_start)) & (j <= p)


def qsa_token_mask(
    q_idx: torch.Tensor,
    k_raw_ctx: torch.Tensor,
    q_pos: torch.Tensor,
    k_gamma: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    budget_blocks: int,
    ratio: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Full selection: raw cached keys ``[B, S, D]`` + queries -> ``bool [B, Tq, S]``.

    Equals ``compressed_keys`` -> ``select_blocks`` -> ``token_mask``. When
    ``qsa_is_dense(S, budget_blocks, ratio)`` it short-circuits to the causal mask.
    """
    S = k_raw_ctx.shape[1]
    if qsa_is_dense(S, budget_blocks, ratio):
        j = torch.arange(S, device=q_pos.device, dtype=torch.int32)
        return j <= q_pos.to(torch.int32).unsqueeze(-1)
    kc = compressed_keys(k_raw_ctx, k_gamma, inv_freq, ratio=ratio, eps=eps)
    sel = select_blocks(q_idx, kc, q_pos, budget_blocks, ratio)
    return token_mask(sel, q_pos, S, ratio)


# =============================================================================
# Section 4: decode-kernel mask
# =============================================================================


def nki_mask_block_len(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int, lnc: int
) -> int:
    """Block length of the decode-mask layout produced by nkilib ``gen_mask_tkg_hbm`` (and read
    by ``attention_block_tkg``): nkilib ``resize_cache_block_len_for_attention_tkg_kernel``
    without its prints, i.e. including the flash-attention s_prior tiling constraint.

    The plugin's ``attention_decode_mask._resize_block_len`` (used by the torch mask and the
    torch attention fallback) does not apply that constraint: at s_prior = 65536, B = 8, 3 heads
    it gives 128 while the NKI kernels use 64.
    """
    from nkilib.core.attention.attention_tkg_utils import (
        is_s_prior_sharded,
        uses_flash_attention,
    )

    p_max = 128
    n_prgs = (
        lnc
        if lnc > 1 and is_s_prior_sharded(bs, q_head, s_active, s_prior, p_max)
        else 1
    )
    divisor = s_prior // (n_prgs * p_max)
    per_shard = s_prior // n_prgs
    use_fa, fa_tile = uses_flash_attention(True, per_shard)
    if use_fa:
        divisor = math.gcd(divisor, fa_tile // p_max)
        if per_shard % fa_tile:
            divisor = math.gcd(divisor, (per_shard % fa_tile) // p_max)
    return math.gcd(block_len, divisor)


def decode_mask_from_token_mask(
    tok_mask: torch.Tensor,
    q_pos: torch.Tensor,
    base_mask: torch.Tensor,
    block_len: int,
) -> torch.Tensor:
    """AND a QSA token mask into the pre-generated decode mask of ``NF.attention_decode``.

    tok_mask: ``bool [B, S_tkg, S_ctx]`` over the request's cache slots (``token_mask``).
    q_pos: ``[B, S_tkg]`` positions of the active tokens (consecutive per request).
    base_mask: ``NF.gen_attention_decode_mask(...)`` output, ``[S_ctx, B, q_heads, S_tkg]``
        in the kernel's partition-major block-KV layout (legacy, pre QK-swap: pass the
        result to ``attention_decode`` as is; it transposes for QK-swap itself).
    block_len: the cache block length given to ``gen_attention_decode_mask``.

    The kernel places the S_tkg active tokens' K/V in the LAST S_tkg mask slots
    (S_ctx - S_tkg + i holds active token i at position q_pos[b, 0] + i), so those slots take
    the selection of the active positions; the other slots are cache positions. Returns the
    product, same shape / dtype / layout as ``base_mask``.
    """
    S_ctx, B, H, S_tkg = base_mask.shape
    # Active slots: gather tok_mask at q_pos[b, 0] + i (clamped: padded rows are don't-care).
    steps = torch.arange(S_tkg, device=q_pos.device, dtype=torch.int32)
    act_pos = (q_pos[:, :1].to(torch.int32) + steps).clamp(0, S_ctx - 1).long()
    act = torch.gather(tok_mask, 2, act_pos.unsqueeze(1).expand(B, S_tkg, S_tkg))
    seq = torch.cat([tok_mask[..., : S_ctx - S_tkg], act], dim=-1)  # [B, S_tkg, S_ctx]
    return base_mask * _to_decode_mask_layout(seq, q_pos, base_mask, block_len)


def _to_decode_mask_layout(
    seq: torch.Tensor, q_pos: torch.Tensor, base_mask: torch.Tensor, block_len: int
) -> torch.Tensor:
    """bool ``[B, S_tkg, S_ctx]`` over the kernel's sequential slots -> ``base_mask``'s layout
    ``[S_ctx, B, 1, S_tkg]`` (broadcast over heads) and dtype."""
    from vllm_neuron.functional.attention.attention_decode_mask import (
        P_MAX,
        _can_use_kernel,
        _resize_block_len,
    )
    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    S_ctx, B, H, S_tkg = base_mask.shape
    # Sequential -> kernel layout. Within each fold of P_MAX * bl slots the kernel stores
    # [bl (block offset), P_MAX (partition)] while the sequence is [P_MAX, bl]
    # (inverse of the torch fallback in attention_decode._torch_attention_decode_impl). The
    # fold size is that of the generator gen_attention_decode_mask used (same predicate).
    if _can_use_kernel(q_pos, B, S_tkg, S_ctx):
        bl = nki_mask_block_len(block_len, B, H, S_tkg, S_ctx, get_logical_nc_config())
    else:
        bl = _resize_block_len(block_len, B, H, S_tkg, S_ctx)
    if bl > 0:
        seq = seq.reshape(B, S_tkg, S_ctx // (P_MAX * bl), P_MAX, bl).transpose(-1, -2)
    seq = seq.reshape(B, 1, S_tkg, S_ctx).permute(3, 0, 1, 2)  # [S_ctx, B, 1, S_tkg]
    return seq.to(base_mask.dtype)


# =============================================================================
# Section 5: compact decode (attend only to the selected tokens)
# =============================================================================


def compact_slots(budget_blocks: int, ratio: int = 4) -> int:
    """Compact context length for ``compact_decode_selection``, a multiple of 256 slots: the
    selected blocks, the open tail block and the kernel's active slot (2,304 for 512 blocks)."""
    need = (budget_blocks + 1) * ratio + 1
    return -(-need // 256) * 256


def topk_block_ids(
    scores: torch.Tensor, budget_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The selected blocks as an index list: ``(idx int32 [B, k], valid bool [B, k])``.

    scores: ``[B, 1, NB]`` (``index_scores``: >= 0 where visible, -1 elsewhere). Takes the top
    ``k = budget_blocks`` scores with ``vllm_neuron.functional.topk`` (the NKI top-k kernel on
    device, ``torch.topk`` elsewhere); the min(k, visible) entries with a score >= 0 are valid,
    the others are invisible blocks. Among equal scores it may pick other blocks than
    ``topk_block_mask`` (which takes the lowest indices); both are an exact top-k. Order is
    unspecified.
    """
    from vllm_neuron.functional.topk import topk

    B, _, NB = scores.shape
    vals, idx = topk(scores.reshape(B, NB), budget_blocks, dim=-1, gather_dim=-1)
    return idx.to(torch.int32).clamp(0, NB - 1), vals >= 0


def compact_decode_selection(
    idx: torch.Tensor,
    valid: torch.Tensor,
    q_pos: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    n_slots: int,
    ratio: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode selection as a compact paged context for ``NF.attention_decode``.

    The K/V caches ``[num_blocks, block_size, Dh]`` are read as ``[num_blocks * block_size /
    ratio, ratio, Dh]``, i.e. in pages of one QSA block, so the kernel's block table can list
    exactly the selected blocks: no K/V copy, and the kernel reads ``n_slots`` slots instead of
    the whole context bucket.

    idx, valid: ``[B, k]`` selected block indices in any order (``topk_block_ids``), one
        decode token per request.
    q_pos: ``[B, 1]`` decode positions; block_table: ``[B, n_blk]`` cache blocks of
        ``block_size`` tokens (-1 = unused).
    n_slots: compact context length (``compact_slots``); a multiple of 128.

    Returns ``(table int32 [B, n_slots / ratio], seq bool [B, 1, n_slots])``:
      * table: page ids of the selected blocks, then the open tail block, then page 0 (the
        null block) as padding;
      * seq: allowed compact slots in sequence order: the tokens of the selected blocks and of
        the tail before ``q_pos`` (the kernel takes the token at ``q_pos`` from its own new
        K/V, in the last slot), and the last slot iff the token at ``q_pos`` is selected.
    Attention is invariant to the order of the keys, so this equals the masked full-context
    decode (``token_mask`` + ``decode_mask_from_token_mask``) of the same selection.
    """
    B, k = idx.shape
    dev = idx.device
    n_pages = n_slots // ratio
    if n_pages < k + 2 or n_slots % ratio:
        raise ValueError(f"compact context {n_slots} too small for {k} blocks")
    ppb = block_size // ratio  # QSA pages per cache block
    bt = block_table.clamp(min=0).long()
    n_blk = bt.shape[1]
    p = q_pos.reshape(B, 1).long()

    def page_of(blk):  # QSA block index -> page id of the K/V cache read in QSA pages
        cb = torch.gather(
            bt, 1, torch.div(blk, ppb, rounding_mode="floor").clamp(max=n_blk - 1)
        )
        return cb * ppb + blk % ppb

    # Built per page / slot with where (no concatenation: neuronx-cc failed on that at B=8).
    tail = torch.div(p + 1, ratio, rounding_mode="floor")  # [B, 1] open block
    e = torch.arange(n_pages, device=dev).view(1, n_pages)
    blk = F.pad(idx.long(), (0, n_pages - k))  # [B, n_pages]
    ok = F.pad(valid, (0, n_pages - k))
    blk = torch.where(e == k, tail, blk)
    ok = (ok & (e < k)) | (e == k)
    table = torch.where(ok, page_of(blk), torch.zeros_like(blk)).to(torch.int32)

    t = torch.arange(ratio, device=dev).view(1, 1, ratio)
    tok_pos = blk.unsqueeze(-1) * ratio + t  # [B, n_pages, ratio]
    allowed = (ok.unsqueeze(-1) & (tok_pos < p.unsqueeze(-1))).reshape(B, n_slots)
    own_blk = torch.div(p, ratio, rounding_mode="floor")
    own = (valid & (idx.long() == own_blk)).any(-1, keepdim=True)
    act_ok = own | ((p + 1) % ratio != 0)  # [B, 1]
    last = torch.arange(n_slots, device=dev).view(1, n_slots) == n_slots - 1
    seq = torch.where(last, act_ok, allowed)
    return table, seq.view(B, 1, n_slots)


def compact_decode_mask(
    seq: torch.Tensor, q_pos: torch.Tensor, base_mask: torch.Tensor, block_len: int
) -> torch.Tensor:
    """``compact_decode_selection``'s ``seq`` in the layout / dtype of ``base_mask`` (the
    ``gen_attention_decode_mask`` output for the compact context with every slot visible,
    i.e. positions ``n_slots - 1``). block_len: the compact page length (``ratio``)."""
    return base_mask * _to_decode_mask_layout(seq, q_pos, base_mask, block_len)
