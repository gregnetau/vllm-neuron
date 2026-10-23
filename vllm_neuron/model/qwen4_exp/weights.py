# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next checkpoint layout and TP sharding rules
===========================================================

Pure-torch (no vLLM / Neuron imports) so the sharding rules are unit-testable on CPU. The
Gated DeltaNet layers use the Qwen3.5 rules unchanged (``qwen3_5/weights.py``); this module adds
what differs:

  full attention   2 KV heads. For tp <= 2 they are sharded; for larger tp each rank holds one
                   replicated KV head (rank r: head ``cfg.kv_head_for_rank(r, tp)``), which keeps
                   the Qwen3.5 "one KV head per rank" kernels and block views.
  QSA indexer      index_qk_proj and its q/k norms replicated (4 + 1 heads x 128).
  gated residual   hc_norm, input_mix_weight_down/up, block_inject_weight replicated.
  PLE              key_proj / value_proj / norms / dilated conv replicated (the n-gram table
                   itself never reaches the device, see ``ngram.py``).

All ``*_device`` helpers take full HF tensors (nn.Linear "out x in") and return this rank's
parameter in the model's [in, out] layout.
"""

import torch

from vllm_neuron.model.qwen3_5.weights import (  # noqa: F401  (re-exported)
    TEXT_PREFIX,
    _chunk,
    _heads,
    fold_zero_centered_norm,
    rotary_channel_permutation,
)


def _kv_rows(w: torch.Tensor, cfg, rank: int, tp: int, perm=None) -> torch.Tensor:
    """This rank's KV-head rows of k_proj / v_proj (sharded, or one replicated head)."""
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim
    heads = _heads(w, nkv, hd)
    if perm is not None:
        heads = heads[:, perm]
    n = cfg.kv_heads_per_rank(tp)
    h0 = cfg.kv_head_for_rank(rank, tp)
    return heads[h0 : h0 + n].reshape(-1, w.shape[1])


def attention_qkv_device(q_proj, k_proj, v_proj, cfg, rank: int, tp: int):
    """[hidden, q_r | k_r | v_r]; q/k channels permuted for full-width rotate_half."""
    H, hd = cfg.num_attention_heads, cfg.head_dim
    perm = rotary_channel_permutation(hd, cfg.rotary_dim)
    q = _heads(q_proj, H, 2 * hd)[:, :hd]  # drop the gate rows
    q = _chunk(q[:, perm], 0, rank, tp).reshape(-1, q_proj.shape[1])
    k = _kv_rows(k_proj, cfg, rank, tp, perm)
    v = _kv_rows(v_proj, cfg, rank, tp)
    return torch.cat([q, k, v], dim=0).t().contiguous()


def linear_t(w: torch.Tensor) -> torch.Tensor:
    """Replicated nn.Linear weight [out, in] -> [in, out]."""
    return w.t().contiguous()


def folded_norm(w: torch.Tensor) -> torch.Tensor:
    """Zero-centered RMSNorm weight -> float32 gamma (1 + w)."""
    return fold_zero_centered_norm(w.float())


def depthwise_conv(w: torch.Tensor) -> torch.Tensor:
    """nn.Conv1d depthwise weight [C, 1, K] -> [C, K]."""
    return w.squeeze(1).contiguous()
