# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch Qwen3.5 dense text model (single sequence, explicit per-layer state)
=================================================================================

The behavioural spec for the Neuron model: no paging, no TP, no vLLM imports. Full
attention layers keep a growing dense K/V; linear-attention layers carry
(conv_state, ssm_state). Used as the oracle for on-device correctness tests
(alongside HF) and to pin the exact semantics the device kernels must reproduce.

Text-only positions: the three mrope rows are identical, so interleaved mrope reduces
to plain rotary on the first ``rotary_dim`` channels of each head.
"""

import torch
import torch.nn.functional as F

from . import gdn_ops


def rmsnorm_zero_centered(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Qwen3.5 RMSNorm: ``x / rms(x) * (1 + weight)`` computed in float32."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * (1.0 + weight.float())).type_as(x)


def rope_cos_sin(positions: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """positions [L] -> cos, sin [L, rotary_dim] (float32)."""
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, cfg.rotary_dim, 2, dtype=torch.float32) / cfg.rotary_dim)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x [L, H, D]; rotates the first cos.shape[-1] channels."""
    rd = cos.shape[-1]
    cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
    rot, rest = x[..., :rd], x[..., rd:]
    return torch.cat([rot * cos + _rotate_half(rot) * sin, rest], dim=-1)


def full_attention_reference(hidden, w, cfg, positions, kv):
    """hidden [L, D]; kv = (k [T, Hkv, hd], v [T, Hkv, hd]) or None. Returns (out, kv)."""
    L = hidden.shape[0]
    H, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qg = (hidden @ w["q_proj"].t()).view(L, H, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    k = (hidden @ w["k_proj"].t()).view(L, Hkv, hd)
    v = (hidden @ w["v_proj"].t()).view(L, Hkv, hd)
    q = rmsnorm_zero_centered(q, w["q_norm"], cfg.rms_norm_eps)
    k = rmsnorm_zero_centered(k, w["k_norm"], cfg.rms_norm_eps)
    cos, sin = rope_cos_sin(positions, cfg)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if kv is not None:
        k, v = torch.cat([kv[0], k]), torch.cat([kv[1], v])
    T = k.shape[0]
    rep = H // Hkv
    kk, vv = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("lhd,thd->hlt", q.float(), kk.float()) * hd**-0.5
    causal = torch.arange(T)[None, :] <= (positions[:, None])  # key pos <= query pos
    scores = scores.masked_fill(~causal[None], float("-inf"))
    o = torch.einsum("hlt,thd->lhd", scores.softmax(-1), vv.float()).to(hidden.dtype)
    o = o * torch.sigmoid(gate)
    return o.reshape(L, H * hd) @ w["o_proj"].t(), (k, v)


def mlp_reference(x, w):
    return (F.silu(x @ w["gate_proj"].t()) * (x @ w["up_proj"].t())) @ w["down_proj"].t()


class Qwen3_5Reference:
    """weights: HF-named state dict; ``prefix`` is the text-tower key prefix, e.g.
    ``"model.language_model."`` (ConditionalGeneration) or ``"model."`` (CausalLM)."""

    def __init__(self, cfg, weights: dict[str, torch.Tensor], prefix: str, lm_head: str = "lm_head.weight"):
        self.cfg, self.w, self.p = cfg, weights, prefix
        self.lm_head = weights.get(lm_head, weights[f"{prefix}embed_tokens.weight"])
        self.state: list = [None] * cfg.num_hidden_layers  # kv or (conv, ssm)
        self.pos = 0

    def _sub(self, i: int, group: str, names):
        return {n: self.w[f"{self.p}layers.{i}.{group}.{n}.weight"] for n in names}

    def forward(self, input_ids: torch.Tensor, decode: bool = False) -> torch.Tensor:
        """Process ``input_ids`` [L] continuing from stored state; returns logits [L, V]."""
        cfg = self.cfg
        L = input_ids.shape[0]
        positions = torch.arange(self.pos, self.pos + L)
        x = F.embedding(input_ids, self.w[f"{self.p}embed_tokens.weight"])
        for i, kind in enumerate(cfg.layer_types):
            lp = f"{self.p}layers.{i}."
            h = rmsnorm_zero_centered(x, self.w[lp + "input_layernorm.weight"], cfg.rms_norm_eps)
            if kind == "full_attention":
                aw = self._sub(i, "self_attn", ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"))
                h, self.state[i] = full_attention_reference(h, aw, cfg, positions, self.state[i])
            else:
                g = lambda n: self.w[lp + "linear_attn." + n]
                lw = {n: g(n + ".weight") for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "out_proj")}
                lw.update(A_log=g("A_log"), dt_bias=g("dt_bias"), norm=g("norm.weight"))
                conv, ssm = self.state[i] or (None, None)
                h, conv, ssm = gdn_ops.gdn_layer_reference(h, lw, cfg, conv, ssm, decode=decode)
                self.state[i] = (conv, ssm)
            x = x + h
            h = rmsnorm_zero_centered(x, self.w[lp + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
            x = x + mlp_reference(h, self._sub(i, "mlp", ("gate_proj", "up_proj", "down_proj")))
        x = rmsnorm_zero_centered(x, self.w[f"{self.p}norm.weight"], cfg.rms_norm_eps)
        self.pos += L
        return x @ self.lm_head.t()
