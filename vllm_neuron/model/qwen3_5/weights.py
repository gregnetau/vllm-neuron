# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.8 dense text-tower checkpoint layout and TP sharding rules
==========================================================================

Pure-torch (no vLLM / Neuron imports) so the sharding rules are unit-testable on CPU.
The model's weight loaders (``model._attach_layer_loaders``) apply the ``*_device``
helpers below to full HF tensors.

Checkpoint prefixes (``Qwen3_5ForConditionalGeneration``): text weights live under
``model.language_model.*`` (``lm_head.weight`` is top level). ``model.visual.*`` is not
loaded (text only); ``mtp.*`` is loaded only by the MTP draft model (``mtp.py``).

Sharding (TP degree ``tp``; every sharded dim must divide evenly):
  full attention   q_proj   rows, per-head blocks of [query(head_dim) | gate(head_dim)]
                   k/v_proj rows by KV head (replicated when tp > KV heads: ranks
                   r*rep .. r*rep+rep-1 hold KV head r), o_proj cols by Q head
                   q_norm / k_norm replicated (head_dim only)
  linear attention in_proj_qkv rows: q | k | v segments each sharded by its own heads
                   conv1d channels: same q | k | v segmentation
                   in_proj_z / in_proj_a / in_proj_b / A_log / dt_bias: by value head
                   out_proj cols by value head; norm (head_v_dim) replicated
  MLP              gate/up rows, down cols
  MoE (qwen3_5_moe) routed experts [E, 2I, H] / [E, H, I]: every rank holds every expert,
                   sharded on I (or, with expert parallelism, a contiguous expert range);
                   shared expert like the MLP; router and shared_expert_gate replicated
  embed_tokens / lm_head vocab rows (generic sharding loaders in model.py); layernorms
                   and the final norm replicated
Head grouping is preserved: value heads per rank = key heads per rank * (Hv/Hk), so
``repeat_interleave`` of q/k inside a rank matches the unsharded pairing.
"""

import torch

TEXT_PREFIX = "model.language_model."


def _chunk(t: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    size = t.shape[dim]
    if size % tp:
        raise ValueError(f"dim {dim} of {tuple(t.shape)} not divisible by tp={tp}")
    step = size // tp
    return t.narrow(dim, rank * step, step).contiguous()


def _segments(t: torch.Tensor, sizes: list[int], dim: int, rank: int, tp: int):
    parts = torch.split(t, sizes, dim=dim)
    return torch.cat([_chunk(p, dim, rank, tp) for p in parts], dim=dim)


# ---------------------------------------------------------------------------
# Load-time transforms so Qwen3.5 attention can reuse Qwen3's fused Neuron kernels
# ---------------------------------------------------------------------------
def rotary_channel_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Per-head channel order that turns partial rotary into full-width rotate_half.

    Qwen3.5 rotates only the first ``rotary_dim`` channels, pairing ``i`` with
    ``i + rotary_dim/2``. The fused Neuron kernels rotate all ``head_dim`` channels,
    pairing ``i`` with ``i + head_dim/2``. Reordering channels to

        [rot[0:R/2], pass[0:(D-R)/2], rot[R/2:R], pass[(D-R)/2:D-R]]

    puts every rotary pair at distance ``D/2``; pass-through channels get cos=1, sin=0
    (see :func:`kernel_rope_angles_from_inv_freq`). Applying the same permutation to q and k leaves
    q.k and the per-head QK RMSNorm (weights permuted alike) unchanged; v is untouched.
    Returns ``perm`` with ``new[j] = old[perm[j]]``.
    """
    half_r = rotary_dim // 2
    half_pass = (head_dim - rotary_dim) // 2
    if head_dim - rotary_dim != 2 * half_pass or rotary_dim != 2 * half_r:
        raise ValueError("head_dim and rotary_dim must be even")
    idx = torch.arange(head_dim)
    rot_a, rot_b = idx[:half_r], idx[half_r:rotary_dim]
    pass_a, pass_b = (
        idx[rotary_dim : rotary_dim + half_pass],
        idx[rotary_dim + half_pass :],
    )
    return torch.cat([rot_a, pass_a, rot_b, pass_b])


def rope_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    """Inverse frequencies for the rotary channels, [rotary_dim / 2] float32 (CPU)."""
    return 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )


def kernel_rope_angles_from_inv_freq(
    positions: torch.Tensor, inv_freq: torch.Tensor, head_dim: int, rotary_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [L, head_dim] for the permuted layout (identity on pass-through channels).

    Creates no device-specific tensors (shapes/devices derive from the arguments), so it is
    safe to trace: the model keeps ``inv_freq`` as a buffer that moves with the module.
    """
    ang = positions.float()[:, None] * inv_freq[None, :]  # [L, R/2]
    zeros = ang.new_zeros(ang.shape[0], (head_dim - rotary_dim) // 2)
    half = torch.cat([ang, zeros], dim=-1)  # [L, D/2]
    full = torch.cat([half, half], dim=-1)
    return full.cos(), full.sin()


def fold_zero_centered_norm(weight: torch.Tensor) -> torch.Tensor:
    """``x * (1 + w)`` -> standard RMSNorm gamma ``1 + w`` (computed in float32)."""
    return (1.0 + weight.float()).to(weight.dtype)


# ---------------------------------------------------------------------------
# Device-layout parameters (what the Neuron model actually stores per TP rank)
#
# All take full HF tensors (nn.Linear "out x in" layout) and return this rank's
# parameter in the model's [in, out] layout. Kept pure-torch so the model's loaders
# are unit-testable against the reference implementation on CPU.
# ---------------------------------------------------------------------------
def _heads(w: torch.Tensor, n: int, width: int) -> torch.Tensor:
    return w.view(n, width, -1)


def attention_qkv_device(q_proj, k_proj, v_proj, cfg, rank: int, tp: int):
    """[hidden, q_r | k_r | v_r]; q/k channels permuted for full-width rotate_half."""
    H, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    perm = rotary_channel_permutation(hd, cfg.rotary_dim)
    q = _heads(q_proj, H, 2 * hd)[:, :hd]  # drop the gate rows
    q = _chunk(q[:, perm], 0, rank, tp).reshape(-1, q_proj.shape[1])
    kv_rank, kv_tp = cfg.kv_head_shard(rank, tp)  # replicated KV heads: shared shard
    k = _chunk(_heads(k_proj, Hkv, hd)[:, perm], 0, kv_rank, kv_tp).reshape(
        -1, k_proj.shape[1]
    )
    v = _chunk(_heads(v_proj, Hkv, hd), 0, kv_rank, kv_tp).reshape(-1, v_proj.shape[1])
    return torch.cat([q, k, v], dim=0).t().contiguous()


def attention_gate_device(q_proj, cfg, rank: int, tp: int):
    """[hidden, q_size_r]: the output-gate rows of q_proj (channel order untouched)."""
    H, hd = cfg.num_attention_heads, cfg.head_dim
    g = _chunk(_heads(q_proj, H, 2 * hd)[:, hd:], 0, rank, tp)
    return g.reshape(-1, q_proj.shape[1]).t().contiguous()


def attention_o_device(o_proj, cfg, rank: int, tp: int):
    """[q_size_r, hidden]."""
    return _chunk(o_proj, 1, rank, tp).t().contiguous()


def attention_qk_norm_device(norm_weight, cfg):
    """Folded (1 + w) gamma with the same channel permutation as q/k."""
    perm = rotary_channel_permutation(cfg.head_dim, cfg.rotary_dim)
    return fold_zero_centered_norm(norm_weight)[perm].contiguous()


def gdn_in_proj_qkv_device(in_proj_qkv, cfg, rank: int, tp: int):
    """[hidden, conv_dim_r]: this rank's q | k | v columns."""
    seg = [cfg.key_dim, cfg.key_dim, cfg.value_dim]
    return _segments(in_proj_qkv, seg, 0, rank, tp).t().contiguous()


def gdn_in_proj_heads_device(w, rank: int, tp: int):
    """in_proj_z / in_proj_b / in_proj_a -> [hidden, cols_r] for this rank's value heads."""
    return _chunk(w, 0, rank, tp).t().contiguous()


def gdn_conv_device(conv1d_weight, cfg, rank: int, tp: int):
    """[conv_dim_r, K] depthwise conv taps (q | k | v channel segments sharded)."""
    seg = [cfg.key_dim, cfg.key_dim, cfg.value_dim]
    return _segments(conv1d_weight.squeeze(1), seg, 0, rank, tp).contiguous()


def gdn_out_proj_device(out_proj, cfg, rank: int, tp: int):
    """[value_dim_r, hidden]."""
    return _chunk(out_proj, 1, rank, tp).t().contiguous()


def gdn_head_vector_device(vec, cfg, rank: int, tp: int):
    """A_log / dt_bias -> this rank's value heads."""
    return _chunk(vec, 0, rank, tp).contiguous()


def mlp_gate_up_device(w, rank: int, tp: int):
    """[hidden, intermediate_r]."""
    return _chunk(w, 0, rank, tp).t().contiguous()


def mlp_down_device(w, rank: int, tp: int):
    """[intermediate_r, hidden]."""
    return _chunk(w, 1, rank, tp).t().contiguous()


# ---------------------------------------------------------------------------
# Sparse MoE experts (qwen3_5_moe). Kernel layout (nkilib moe_cte / moe_block_tkg):
# gate_up [E_local, H, 2, I_r] (index 0 = gate, 1 = up), down [E_local, I_r, H].
#
# ``src`` may be a tensor or a checkpoint slice (``get_shape()`` + ``__getitem__``), so only
# this rank's rows are read: the backbone stores one fused tensor per layer (HF
# ``Qwen3_5MoeExperts``: gate_up_proj [E, 2I, H] whose first I rows are gate, down_proj
# [E, H, I]); the MTP layer stores per-expert ``nn.Linear`` weights.
# ---------------------------------------------------------------------------
def _shape(src) -> tuple[int, ...]:
    return tuple(src.get_shape()) if hasattr(src, "get_shape") else tuple(src.shape)


def moe_gate_up_device(src, rank: int, tp: int) -> torch.Tensor:
    """Fused gate_up_proj [E, 2I, H] -> [E, H, 2, I/tp]."""
    E, two_i, H = _shape(src)
    inter = two_i // 2
    if inter % tp:
        raise ValueError(f"moe_intermediate_size={inter} not divisible by tp={tp}")
    step = inter // tp
    lo = rank * step
    # One half at a time: the transposed copy of the full shard doubles the peak otherwise.
    gate = src[:, lo : lo + step, :]
    out = torch.empty(E, H, 2, step, dtype=gate.dtype)
    out[:, :, 0] = gate.transpose(1, 2)
    del gate
    out[:, :, 1] = src[:, inter + lo : inter + lo + step, :].transpose(1, 2)
    return out


def moe_down_device(src, rank: int, tp: int, chunk_experts: int = 32) -> torch.Tensor:
    """Fused down_proj [E, H, I] -> [E, I/tp, H]. Read in groups of whole experts: slicing the
    last dim of a checkpoint slice directly is one small read per (expert, row)."""
    E, _, inter = _shape(src)
    if inter % tp:
        raise ValueError(f"moe_intermediate_size={inter} not divisible by tp={tp}")
    step = inter // tp
    lo = rank * step
    parts = []
    for first in range(0, E, chunk_experts):
        group = src[first : min(first + chunk_experts, E)]
        parts.append(group[:, :, lo : lo + step].transpose(1, 2))
    return torch.cat(parts, dim=0)


def moe_gate_up_device_per_expert(gates, ups, rank: int, tp: int) -> torch.Tensor:
    """Per-expert gate_proj / up_proj [I, H] (lists over experts) -> [E, H, 2, I/tp]."""
    return torch.stack(
        [
            torch.stack(
                [
                    mlp_gate_up_device(g[:], rank, tp),
                    mlp_gate_up_device(u[:], rank, tp),
                ],
                dim=1,
            )
            for g, u in zip(gates, ups)
        ]
    )


def moe_down_device_per_expert(downs, rank: int, tp: int) -> torch.Tensor:
    """Per-expert down_proj [H, I] (list over experts) -> [E, I/tp, H]."""
    return torch.stack([mlp_down_device(d[:], rank, tp) for d in downs])


# ---------------------------------------------------------------------------
# Per-tensor static FP8 (trn2). Values are kept within +-240 so float8_e4m3fn bytes are
# identical to the legacy e4m3 format trn2 kernels use (the runner adds
# --experimental-unsafe-fp8e4m3fn-as-fp8e4m3). Scales are dequant multipliers:
# W ~= W_fp8 * weight_scale, x_fp8 = clamp(x / input_scale).
FP8_MAX = 240.0
SCALE_PARTITIONS = (
    128  # kernels take scalar scales replicated across partitions [128, 1]
)


def fp8_scale(amax: float | torch.Tensor) -> float:
    return max(float(amax), 1e-12) / FP8_MAX


def fp8_weight_scale(w_full: torch.Tensor) -> float:
    """Per-tensor scale from the FULL (unsharded) checkpoint tensor: identical on every rank."""
    return fp8_scale(w_full.abs().amax().float())


def fp8_quantize(w: torch.Tensor, scale: float) -> torch.Tensor:
    return (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def fp8_quantize_parts(
    w: torch.Tensor, scales: list[float], sizes: list[int]
) -> torch.Tensor:
    """Quantize consecutive column blocks of w [H, sum(sizes)] with their own scales (fused QKV)."""
    blocks = torch.split(w, sizes, dim=1)
    return torch.cat([fp8_quantize(b, s) for b, s in zip(blocks, scales)], dim=1)


def fp8_scale_tile(scale: float, n: int = 1) -> torch.Tensor:
    return torch.full((SCALE_PARTITIONS, n), scale, dtype=torch.float32)
