# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.8 dense text-tower checkpoint layout and TP sharding rules
==========================================================================

Pure-torch (no vLLM / Neuron imports) so the sharding rules are unit-testable on CPU.
Operates on HF-named tensors ("out x in" nn.Linear layout); the model loader composes
these shards into its fused/transposed device parameters.

Checkpoint prefixes (``Qwen3_5ForConditionalGeneration``): text weights live under
``model.language_model.*`` (``lm_head.weight`` is top level). ``model.visual.*`` and
``mtp.*`` are ignored (text-only, MTP off).

Sharding (TP degree ``tp``; every sharded dim must divide evenly, no KV replication):
  full attention   q_proj   rows, per-head blocks of [query(head_dim) | gate(head_dim)]
                   k/v_proj rows by KV head, o_proj cols by Q head
                   q_norm / k_norm replicated (head_dim only)
  linear attention in_proj_qkv rows: q | k | v segments each sharded by its own heads
                   conv1d channels: same q | k | v segmentation
                   in_proj_z / in_proj_a / in_proj_b / A_log / dt_bias: by value head
                   out_proj cols by value head; norm (head_v_dim) replicated
  MLP              gate/up rows, down cols;   embeddings, lm_head, layernorms: see below
Head grouping is preserved: value heads per rank = key heads per rank * (Hv/Hk), so
``repeat_interleave`` of q/k inside a rank matches the unsharded pairing.
"""

import re

import torch

TEXT_PREFIX = "model.language_model."
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def is_text_weight(name: str) -> bool:
    """True for weights the text-only model loads (skips vision tower and MTP head)."""
    return name == "lm_head.weight" or name.startswith(TEXT_PREFIX)


def expected_text_weight_names(cfg) -> set[str]:
    """Every HF text-tower tensor name the model needs, from the config."""
    names = {
        f"{TEXT_PREFIX}embed_tokens.weight",
        f"{TEXT_PREFIX}norm.weight",
    }
    if not cfg.tie_word_embeddings:
        names.add("lm_head.weight")
    for i, layer_type in enumerate(cfg.layer_types):
        p = f"{TEXT_PREFIX}layers.{i}."
        names |= {
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
        }
        if layer_type == "full_attention":
            names |= {
                p + f"self_attn.{n}.weight"
                for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
            }
        else:
            names |= {p + f"linear_attn.{n}.weight" for n in (
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                "conv1d", "out_proj", "norm",
            )}
            names |= {p + "linear_attn.A_log", p + "linear_attn.dt_bias"}
    return names


def _chunk(t: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    size = t.shape[dim]
    if size % tp:
        raise ValueError(f"dim {dim} of {tuple(t.shape)} not divisible by tp={tp}")
    step = size // tp
    return t.narrow(dim, rank * step, step).contiguous()


def _segments(t: torch.Tensor, sizes: list[int], dim: int, rank: int, tp: int):
    parts = torch.split(t, sizes, dim=dim)
    return torch.cat([_chunk(p, dim, rank, tp) for p in parts], dim=dim)


def shard_weight(
    name: str, tensor: torch.Tensor, cfg, rank: int, tp: int
) -> torch.Tensor:
    """Return this rank's slice of an HF-named text weight (replicated ones unchanged)."""
    if tp == 1:
        return tensor
    if name in ("lm_head.weight", f"{TEXT_PREFIX}embed_tokens.weight"):
        return _chunk(tensor, 0, rank, tp)  # vocab-sharded
    m = _LAYER_RE.match(name)
    if m is None:
        return tensor  # final norm
    leaf = m.group(2)

    if leaf.endswith("layernorm.weight") or leaf in (
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "linear_attn.norm.weight",
    ):
        return tensor
    # dense MLP
    if leaf in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        return _chunk(tensor, 0, rank, tp)
    if leaf == "mlp.down_proj.weight":
        return _chunk(tensor, 1, rank, tp)
    # full attention: per-head blocks are contiguous, so a plain row chunk keeps
    # [query | gate] pairs together.
    if leaf in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight"):
        return _chunk(tensor, 0, rank, tp)
    if leaf == "self_attn.o_proj.weight":
        return _chunk(tensor, 1, rank, tp)
    # linear attention
    seg = [cfg.key_dim, cfg.key_dim, cfg.value_dim]
    if leaf == "linear_attn.in_proj_qkv.weight":
        return _segments(tensor, seg, 0, rank, tp)
    if leaf == "linear_attn.conv1d.weight":  # [C, 1, K]
        return _segments(tensor, seg, 0, rank, tp)
    if leaf in ("linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.A_log", "linear_attn.dt_bias"):
        return _chunk(tensor, 0, rank, tp)
    if leaf == "linear_attn.out_proj.weight":
        return _chunk(tensor, 1, rank, tp)
    raise KeyError(f"No sharding rule for {name}")


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
    (see :func:`kernel_rope_angles`). Applying the same permutation to q and k leaves
    q.k and the per-head QK RMSNorm (weights permuted alike) unchanged; v is untouched.
    Returns ``perm`` with ``new[j] = old[perm[j]]``.
    """
    half_r = rotary_dim // 2
    half_pass = (head_dim - rotary_dim) // 2
    if head_dim - rotary_dim != 2 * half_pass or rotary_dim != 2 * half_r:
        raise ValueError("head_dim and rotary_dim must be even")
    idx = torch.arange(head_dim)
    rot_a, rot_b = idx[:half_r], idx[half_r:rotary_dim]
    pass_a, pass_b = idx[rotary_dim : rotary_dim + half_pass], idx[rotary_dim + half_pass :]
    return torch.cat([rot_a, pass_a, rot_b, pass_b])


def permute_head_rows(
    weight: torch.Tensor, num_heads: int, head_dim: int, perm: torch.Tensor,
    per_head_width: int | None = None, offset: int = 0,
) -> torch.Tensor:
    """Apply ``perm`` to the channel order inside each head's block of output rows.

    ``weight`` is ``[num_heads * per_head_width, in]`` (``per_head_width`` defaults to
    ``head_dim``; q_proj uses ``2 * head_dim`` with the query at ``offset=0`` and the
    gate untouched after it). Only rows ``[offset, offset + head_dim)`` of each head
    block are permuted.
    """
    width = per_head_width or head_dim
    w = weight.view(num_heads, width, -1).clone()
    w[:, offset : offset + head_dim] = w[:, offset : offset + head_dim][:, perm]
    return w.view(weight.shape)


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


def kernel_rope_angles(
    positions: torch.Tensor, head_dim: int, rotary_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper (tests / CPU use): builds inv_freq on positions' device."""
    inv_freq = rope_inv_freq(rotary_dim, theta).to(positions.device)
    return kernel_rope_angles_from_inv_freq(positions, inv_freq, head_dim, rotary_dim)


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
    k = _chunk(_heads(k_proj, Hkv, hd)[:, perm], 0, rank, tp).reshape(-1, k_proj.shape[1])
    v = _chunk(_heads(v_proj, Hkv, hd), 0, rank, tp).reshape(-1, v_proj.shape[1])
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


def gdn_in_proj_device(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, cfg, rank, tp):
    """[hidden, qkv_r | z_r | b_r | a_r] fused input projection."""
    seg = [cfg.key_dim, cfg.key_dim, cfg.value_dim]
    parts = [
        _segments(in_proj_qkv, seg, 0, rank, tp),
        _chunk(in_proj_z, 0, rank, tp),
        _chunk(in_proj_b, 0, rank, tp),
        _chunk(in_proj_a, 0, rank, tp),
    ]
    return torch.cat(parts, dim=0).t().contiguous()


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
# Per-tensor static FP8 (trn2). Values are kept within +-240 so float8_e4m3fn bytes are
# identical to the legacy e4m3 format trn2 kernels use (the runner adds
# --experimental-unsafe-fp8e4m3fn-as-fp8e4m3). Scales are dequant multipliers:
# W ~= W_fp8 * weight_scale, x_fp8 = clamp(x / input_scale).
FP8_MAX = 240.0
SCALE_PARTITIONS = 128  # kernels take scalar scales replicated across partitions [128, 1]


def fp8_scale(amax: float | torch.Tensor) -> float:
    return max(float(amax), 1e-12) / FP8_MAX


def fp8_weight_scale(w_full: torch.Tensor) -> float:
    """Per-tensor scale from the FULL (unsharded) checkpoint tensor: identical on every rank."""
    return fp8_scale(w_full.abs().amax().float())


def fp8_quantize(w: torch.Tensor, scale: float) -> torch.Tensor:
    return (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def fp8_quantize_parts(w: torch.Tensor, scales: list[float], sizes: list[int]) -> torch.Tensor:
    """Quantize consecutive column blocks of w [H, sum(sizes)] with their own scales (fused QKV)."""
    blocks = torch.split(w, sizes, dim=1)
    return torch.cat([fp8_quantize(b, s) for b, s in zip(blocks, scales)], dim=1)


def fp8_scale_tile(scale: float, n: int = 1) -> torch.Tensor:
    return torch.full((SCALE_PARTITIONS, n), scale, dtype=torch.float32)
