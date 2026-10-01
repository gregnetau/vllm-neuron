# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.8 dense hybrid (Gated DeltaNet + full attention), text only
=========================================================================

BF16, or static FP8 MLPs (``quantization="fp8"``). Numerics of the weight transforms, GDN
ops, partial-rotary trick and TP sharding are unit-tested on CPU (``test/unit/model/qwen3_5``).

Adapted from ``vllm_neuron/model/qwen3/model.py`` (dense, GQA, QK-norm). Differences:

  * 48 of 64 layers are Gated DeltaNet linear attention. Each keeps a per-request conv
    state and recurrent state in a state page (see ``model/state_cache.py``), addressed by
    the first block id of the layer's (Mamba) block-table group.
  * The 16 full-attention layers have an output gate (q_proj emits [query | gate] per
    head), head_dim 256, partial rotary (64 channels) and zero-centered RMSNorm.
    Partial rotary is expressed as full-width rotate_half by permuting q/k channels at
    load time (``weights.rotary_channel_permutation``); zero-centered norms are folded to
    ``1 + w`` so the fused Neuron kernels can be reused. The gate is applied to the
    attention output *before* o_proj, so decode calls ``attention_decode`` without W_out.
  * GDN prefill runs through the pure-PyTorch ops in ``gdn_ops`` (traced by torch.compile;
    the chunk loop unrolls). Decode uses the fused NKI kernel in ``gdn_kernels`` when it
    applies, else the same torch ops.

Layout conventions follow Qwen3: prefill is sequence-parallel (residual stream is
[T/tp, D]; all-gather before / reduce-scatter after each mixer and MLP), decode is
replicated with all-reduce.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3.5-specific. Change when porting.
"""

import json
import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, StateLayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from . import gdn_ops, weights as W
from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config
from .state import gdn_state_shapes

logger = logging.getLogger(__name__)

CKPT = W.TEXT_PREFIX  # "model.language_model."

def _loader(fn) -> SafetensorsWeightLoader:
    """Loader from ``fn(tensors: list[Tensor], rank) -> Tensor`` over full tensors."""
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: fn([s[:] for s in slices], rank)
    )


def _fp8_loaders(device_fn, scale_cols: int = 1):
    """(weight loader, weight-scale loader) for a static-FP8 projection. The per-tensor scale
    comes from the full checkpoint tensor, so every rank uses the same scale."""
    weight = _loader(lambda t, r: W.fp8_quantize(device_fn(t[0], r), W.fp8_weight_scale(t[0])))
    scale = _loader(lambda t, r: W.fp8_scale_tile(W.fp8_weight_scale(t[0]), scale_cols))
    return weight, scale


def _fp8_param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False)


def _scale_param(n: int = 1) -> nn.Parameter:
    return nn.Parameter(torch.ones(W.SCALE_PARTITIONS, n), requires_grad=False)


def _fp8_linear(x, weight, weight_scale, input_scale, d_head: int = 128):
    """x [T, H] @ weight [H, N] in static FP8 through the fused-QKV projection kernel
    (N split as (N/d_head - 2) + 1 + 1 heads; one scale for all three parts)."""
    from nkilib.core.utils.common_types import QuantizationType

    n_heads = weight.shape[1] // d_head
    return NF.qkv_proj(
        hidden=x.unsqueeze(0), qkv_weights=weight, bias=None, d_head=d_head,
        num_q_heads=n_heads - 2, num_kv_heads=1, quantization_type=QuantizationType.STATIC,
        qkv_w_scale=weight_scale, qkv_in_scale=input_scale,
    ).squeeze(0)


def _fp8_out_proj(x, weight, weight_scale, input_scale, n_heads: int, d_head: int):
    """x [T, n_heads*d_head] @ weight [n_heads*d_head, H] in static FP8 (output-projection kernel)."""
    from nkilib.core.utils.common_types import QuantizationType

    T = x.shape[0]
    active = x.reshape(T, n_heads, d_head).permute(1, 2, 0).unsqueeze(0)  # [1, N, D, T]
    return NF.o_proj(active, weight, None, quantization_type=QuantizationType.STATIC,
                     weight_scales=weight_scale, input_scales=input_scale).reshape(T, -1)


# =============================================================================
# Section 1: RMSNorm / RoPE
# <-- MODEL-SPECIFIC: zero-centered RMSNorm, folded to gamma = 1 + w at load time.
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """Standard RMSNorm whose ``weight`` holds the folded gamma (1 + checkpoint weight)."""

    def __init__(self, size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(dtype)


def _apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """q [Nh, T, Dh], k [Nkv, T, Dh], cos/sin [T, Dh] (permuted-channel layout)."""
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class Qwen3_5RotaryEmbedding(nn.Module):
    """cos/sin for the permuted-channel layout: rotary angles on the first R/2 channels of
    each half, identity (cos=1, sin=0) elsewhere. Text-only: the 3 mrope rows are equal.

    ``inv_freq`` is a non-persistent buffer: it moves to the device with the model and the
    traced graph creates no device-specific tensors (which would bake the trace device in).
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.head_dim, self.rotary_dim = config.head_dim, config.rotary_dim
        self.register_buffer(
            "inv_freq", W.rope_inv_freq(config.rotary_dim, config.rope_theta),
            persistent=False,
        )

    def forward(self, position_ids: torch.Tensor, device, dtype):
        cos, sin = W.kernel_rope_angles_from_inv_freq(
            position_ids, self.inv_freq, self.head_dim, self.rotary_dim
        )
        return cos.to(dtype), sin.to(dtype)


# =============================================================================
# Section 2: Full attention (16 layers)
# =============================================================================


class Qwen3_5Attention(nn.Module):
    """GQA attention with output gate, per-head QK-norm, partial rotary.

    >>> PARALLELISM: TP <<< Q/K/V/gate heads sharded; o_proj row-parallel.
    <-- MODEL-SPECIFIC: head_dim 256 (> 128: flash/o_proj kernels fall back to PyTorch;
        # attention_decode is called with d_head=256).
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.scaling = config.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        config.validate_tp(self.world_size)

        self.num_attention_heads_per_rank = config.num_attention_heads // self.world_size
        self.num_key_value_heads_per_rank = config.num_key_value_heads // self.world_size
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size, self.kv_size = q_size, kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=self.dtype)
        )
        self.gate_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )
        # Kernel-facing gamma (folded + channel-permuted), model dtype.
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        self.k_cache = self.v_cache = None
        self.kv_blocks_cover_context = False  # set by the runner (set_kv_blocks_cover_context)
        self.k_scale = self.v_scale = None
        self.k_scale_float = self.v_scale_float = 1.0

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata=None):
        name = f"layers.{self.layer_idx}.self_attn"
        md = attn_metadata[name]
        if md["max_query_len"] <= md["decode_token_threshold"]:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        if self.world_size > 1:  # >>> PARALLELISM: all-gather from SP <<<
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    def _gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """sigmoid(gate) laid out [Nh, Dh, T] (matches flash_attention tp_out=True)."""
        T = hidden_states.shape[0]
        gate = (hidden_states @ self.gate_weight).view(
            T, self.num_attention_heads_per_rank, self.head_dim
        )
        return torch.sigmoid(gate.float()).permute(1, 2, 0)

    def _write_kv(self, k, v, slot_mapping, block_size):
        """k, v [Nkv, T, Dh] -> paged cache (same scheme as Qwen3)."""
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (k.reshape(-1, self.head_dim) * self.k_scale).clamp(
                -FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)
            v_flat = (v.reshape(-1, self.head_dim) * self.v_scale).clamp(
                -FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.v_cache.dtype)
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        nkh = self.num_key_value_heads_per_rank
        head_idx = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(
            slot_mapping.shape[0])
        blk_idx = block_indices.repeat(nkh)
        pos_idx = position_indices.repeat(nkh)
        self.k_cache.index_put_((blk_idx, head_idx, pos_idx), k_flat)
        self.v_cache.index_put_((blk_idx, head_idx, pos_idx), v_flat)

    def _zero_fresh_blocks(self, blocks: torch.Tensor):
        """Zero whole KV blocks [N] about to receive their first token (0 = null block, harmless).
        The pool is shared with GDN fp32 state pages; a recycled page read as bf16 can hold
        NaN/Inf, and the decode kernel reads the full block (0 * NaN survives the mask)."""
        zeros = torch.zeros((blocks.shape[0],) + tuple(self.k_cache.shape[1:]),
                            dtype=self.k_cache.dtype, device=self.k_cache.device)
        self.k_cache.index_put_((blocks,), zeros)
        self.v_cache.index_put_((blocks,), zeros.to(self.v_cache.dtype))

    def _attend_cached_torch(self, q, md, block_size):
        """Prefill attention over the request's cached context + this chunk (already written to
        the cache), in torch. q [Nh, T, Dh] -> [Nh, Dh, T]. One request per prefill step.
        Keys are valid iff pos < cached + n_valid (padded query rows see only valid keys too), and
        invalid V rows are zeroed: unwritten cache can hold NaN/Inf from recycled state pages."""
        Nh, T, Dh = q.shape
        nkh = self.num_key_value_heads_per_rank
        bt = md["block_table_tensor"][0].long().clamp(min=0)  # -1 (unused) -> null block; masked
        prior = md["cached_seq_len"].reshape(-1)[:1].long()
        n_valid = (md["slot_mapping"] > 0).sum().reshape(1)
        S = bt.shape[0] * block_size
        kc = self.k_cache.index_select(0, bt).transpose(0, 1).reshape(nkh, S, Dh)
        vc = self.v_cache.index_select(0, bt).transpose(0, 1).reshape(nkh, S, Dh)
        key_pos = torch.arange(S, device=q.device)
        q_pos = prior + torch.arange(T, device=q.device)
        key_ok = key_pos < prior + n_valid                                  # [S]
        allowed = key_ok[None, :] & (key_pos[None, :] <= q_pos[:, None])   # [T, S]
        vc = torch.where(key_ok[None, :, None], vc.float(), torch.zeros((), device=q.device))
        kc = kc.float().repeat_interleave(self.num_key_value_groups, dim=0)
        vc = vc.repeat_interleave(self.num_key_value_groups, dim=0)
        scores = torch.matmul(q.float(), kc.transpose(1, 2)) * self.scaling  # [Nh, T, S]
        scores = torch.where(allowed[None], scores, torch.full((), -1e30, device=q.device))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, vc).transpose(1, 2).to(q.dtype)          # [Nh, Dh, T]

    def forward_prefill(self, hidden_states, positions, position_embeddings, attn_metadata=None):
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        hidden_states = hidden_states.to(self.dtype)
        T = hidden_states.shape[0]

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0), qkv_weights=self.qkv_proj_weight, bias=None
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(T, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(T, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(T, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        q, k = self.q_norm(q), self.k_norm(k)  # QK-norm before RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        md = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        slot_mapping, block_size = md["slot_mapping"], md["block_size"]
        # Blocks starting at/after the cached prefix hold no valid data yet -> clear them.
        # Unused table entries are the -1 read sentinel (runner); map them to the null block.
        bt = md["block_table_tensor"][0].long().clamp(min=0)
        starts = torch.arange(bt.shape[0], device=bt.device) * block_size
        prior = md["cached_seq_len"].reshape(-1)[:1].to(starts.dtype)
        self._zero_fresh_blocks(torch.where(starts >= prior, bt, torch.zeros_like(bt)))
        self._write_kv(k, v, slot_mapping, block_size)

        kv_segment_size = md.get("kv_segment_size")
        if kv_segment_size and self.head_dim > 128:
            # The segmented attention kernel needs head_dim <= 128 (27B: 256).
            attn_output = self._attend_cached_torch(q, md, block_size)
        elif kv_segment_size:
            attn_output = NF.segmented_attention(
                q, k_cache=self.k_cache, v_cache=self.v_cache,
                block_tables=md["block_table_tensor"], prior_tokens=md.get("cached_seq_len"),
                block_size=block_size, kv_segment_size=kv_segment_size,
                scale=self.scaling, tp_q=True, tp_out=True,
            )
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2), k.transpose(1, 2), v,
                scale=self.scaling, tp_q=False, tp_out=True,
            )  # [Nh, Dh, T]

        # <-- MODEL-SPECIFIC: output gate before o_proj
        attn_output = (attn_output.float() * self._gate(hidden_states)).to(self.dtype)
        out = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None).squeeze(0)
        if self.world_size > 1:  # >>> PARALLELISM: reduce-scatter to SP layout <<<
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    def forward_decode(self, hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        slot_mapping, block_size = md["slot_mapping"], md["block_size"]
        block_table = md["block_table_tensor"]
        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        hidden_states = hidden_states.to(self.dtype)
        S_ctx = md["max_blocks_per_seq"] * block_size
        nkh = self.num_key_value_heads_per_rank

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_k = cos[:, :half_d].view(B, S_decode, half_d).permute(2, 0, 1).contiguous().to(self.dtype)
        sin_k = sin[:, :half_d].view(B, S_decode, half_d).permute(2, 0, 1).contiguous().to(self.dtype)
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B * S_decode).to(torch.float32), bs=B,
            q_head=self.num_attention_heads_per_rank, s_active=S_decode, s_prior=S_ctx,
            start_pos=None, block_len=block_size,
        )
        k_cache = self.k_cache.squeeze(1) if self.k_cache.dim() == 4 and nkh else self.k_cache
        v_cache = self.v_cache.squeeze(1) if self.v_cache.dim() == 4 and nkh else self.v_cache

        # A token at block offset 0 starts a fresh block: clear it before the kernel reads it.
        # Writing the cache ahead of the kernel copies the whole cache, so this is skipped
        # when every block of a request is allocated (and cleared) at prefill.
        if not self.kv_blocks_cover_context:
            blk_new = (slot_mapping // block_size).long()
            self._zero_fresh_blocks(torch.where(slot_mapping % block_size == 0, blk_new,
                                                torch.zeros_like(blk_new)))

        # Unused table entries are the runner's -1 sentinel; the kernel skips OOB reads but the
        # skipped SBUF tile keeps stale (possibly NaN) data, which survives the mask (0*NaN).
        # Point them at the null block instead, which every step keeps zeroed.
        active_blocks = block_table.clamp(min=0).to(torch.int32)

        # W_out=None: the gate must multiply the attention output before o_proj.
        # Output layout [B, Nh, Dh, S].
        attn, K_new, V_new = NF.attention_decode(
            X=hidden_states.view(B, S_decode, hidden), X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False, W_qkv=self.qkv_proj_weight, bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False, cos=cos_k, sin=sin_k, rope_contiguous_layout=True,
            K_cache_transposed=False, active_blocks_table=active_blocks,
            K_cache=k_cache, V_cache=v_cache, attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float, sink=None, update_cache=False,
            W_out=None, bias_out=None, transposed_out=False, out_in_sb=False,
            k_scale=self.k_scale, v_scale=self.v_scale,
            attention_dp=1, attention_dp_group=None, attention_dp_rank=0, kv_needs_a2a=False,
        )

        gate = (hidden_states @ self.gate_weight).view(
            B, S_decode, self.num_attention_heads_per_rank, self.head_dim
        ).permute(0, 2, 3, 1)  # [B, Nh, Dh, S]
        attn = (attn.float() * torch.sigmoid(gate.float()) / self.v_scale_float).to(self.dtype)
        output = NF.o_proj(attn, self.o_proj_weight, None).reshape(B * S_decode, hidden)

        # Manual KV cache update (same as Qwen3)
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]
        k_new = (K_new.permute(1, 2, 0).reshape(B, nkh, S_decode, self.head_dim)
                 .transpose(0, 1).reshape(nkh, B * S_decode, self.head_dim))
        v_new = V_new.transpose(0, 1).reshape(-1, self.head_dim)
        head_idx = torch.arange(nkh, dtype=torch.long, device=hidden_states.device
                                ).repeat_interleave(num_tokens)
        blk_idx, pos_idx = block_indices.repeat(nkh), position_indices.repeat(nkh)
        self.k_cache.index_put_((blk_idx, head_idx, pos_idx), k_new.reshape(-1, self.head_dim).to(self.k_cache.dtype))
        self.v_cache.index_put_((blk_idx, head_idx, pos_idx), v_new.to(self.v_cache.dtype))

        if self.world_size > 1:  # >>> PARALLELISM: TP all-reduce <<<
            self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 3: Gated DeltaNet linear attention (48 layers)
# =============================================================================


class Qwen3_5GatedDeltaNet(nn.Module):
    """Linear-attention mixer with per-request recurrent state.

    >>> PARALLELISM: TP <<< value/key heads sharded (Hk/tp, Hv/tp); out_proj row-parallel.
    State pages (bound by the runner, per layer): two contiguous float32 halves
    ``state_a``/``state_b`` [pages, h]. A page's flat state is ``cat(A[i], B[i])[:S]`` =
    recurrent state [Hv/tp, Dk, Dv] then conv state stored [K-1, conv_dim/tp] (see state.py;
    _read_state/_write_state take conv as [B, C, K-1]). A request's page = first block id of
    its block table.
    """

    # Static-FP8 input scales: buffer -> calibration modules (relative to the layer).
    FP8_INPUT_SCALES = {
        "in_proj_input_scale": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
        "out_proj_input_scale": ("linear_attn.out_proj",),
    }

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        config.validate_tp(self.world_size)

        tp = self.world_size
        self.Hk, self.Hv = config.linear_num_key_heads // tp, config.linear_num_value_heads // tp
        self.Dk, self.Dv = config.linear_key_head_dim, config.linear_value_head_dim
        self.K = config.linear_conv_kernel_dim
        self.key_dim, self.value_dim = self.Hk * self.Dk, self.Hv * self.Dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # Separate projections: column slices of a fused output at unaligned offsets
        # (b/a are Hv wide) are not lowered correctly.
        H = self.hidden_size
        # Static FP8 for in_proj_qkv / in_proj_z / out_proj; in_proj_b / in_proj_a stay BF16.
        self.fp8 = config.fp8_quantized(f"layers.{layer_idx}.linear_attn")
        if self.fp8:
            self.in_proj_qkv_weight = _fp8_param(H, self.conv_dim)
            self.in_proj_z_weight = _fp8_param(H, self.value_dim)
            self.out_proj_weight = _fp8_param(self.value_dim, self.hidden_size)
            self.in_proj_qkv_weight_scale = _scale_param(3)
            self.in_proj_z_weight_scale = _scale_param(3)
            self.out_proj_weight_scale = _scale_param(1)
            for name in self.FP8_INPUT_SCALES:
                self.register_buffer(name, torch.ones(W.SCALE_PARTITIONS, 1), persistent=False)
        else:
            self.in_proj_qkv_weight = nn.Parameter(torch.empty(H, self.conv_dim, dtype=self.dtype))
            self.in_proj_z_weight = nn.Parameter(torch.empty(H, self.value_dim, dtype=self.dtype))
            self.out_proj_weight = nn.Parameter(torch.empty(self.value_dim, self.hidden_size, dtype=self.dtype))
        self.in_proj_b_weight = nn.Parameter(torch.empty(H, self.Hv, dtype=self.dtype))
        self.in_proj_a_weight = nn.Parameter(torch.empty(H, self.Hv, dtype=self.dtype))
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, self.K, dtype=self.dtype))
        self.A_log = nn.Parameter(torch.empty(self.Hv, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.Hv, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.ones(self.Dv, dtype=self.dtype))

        self.conv_numel = self.conv_dim * (self.K - 1)
        self.ssm_numel = self.Hv * self.Dv * self.Dk
        self.state_numel = self.conv_numel + self.ssm_numel
        self.state_a = self.state_b = None  # bound by bind_state_cache

    # -- state pages ----------------------------------------------------------
    # Pages are gathered/scattered as [B, rows, _STATE_ROW] (same bytes as [pages, half]):
    # a single [B, half] row lands on one partition and re-laying it out to [Hv, Dk, Dv]
    # becomes per-element DMA. The 3-D view is made in bind_state, outside the graph, since
    # index_copy_ through an in-graph .view() is not lowered reliably.
    _STATE_ROW = 128

    def bind_state(self, state_a: torch.Tensor, state_b: torch.Tensor):
        R, half = self._STATE_ROW, state_a.shape[1]
        if half % R == 0 and self.conv_numel % R == 0 and self.ssm_numel % R == 0:
            state_a, state_b = state_a.view(-1, half // R, R), state_b.view(-1, half // R, R)
        self.state_a, self.state_b = state_a, state_b

    def _read_state(self, slots: torch.Tensor):
        """slots [B] long -> conv [B, C, K-1] f32, ssm [B, Hv, Dk, Dv] f32."""
        B = slots.shape[0]
        page = torch.cat(
            [self.state_a.index_select(0, slots), self.state_b.index_select(0, slots)], dim=1
        )  # [B, 2*rows, R] or [B, 2*half]
        # page = [ssm | conv | pad] (state.py)
        if page.dim() == 3:
            R = self._STATE_ROW
            cr, sr = self.conv_numel // R, self.ssm_numel // R
            ssm, conv = page[:, :sr], page[:, sr : sr + cr]
        else:
            ssm, conv = page[:, : self.ssm_numel], page[:, self.ssm_numel : self.state_numel]
        return (conv.reshape(B, self.K - 1, self.conv_dim).transpose(1, 2),
                ssm.reshape(B, self.Hv, self.Dk, self.Dv))

    def _write_state(self, slots: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor):
        B = slots.shape[0]
        if self.state_a.dim() == 3:
            R, rows = self._STATE_ROW, self.state_a.shape[1]
            conv = conv.transpose(1, 2).float()  # stored [K-1, C]
            page = torch.cat([ssm.reshape(B, -1, R).float(), conv.reshape(B, -1, R)], dim=1)
            page = torch.nn.functional.pad(page, (0, 0, 0, 2 * rows - page.shape[1]))
        else:
            rows = self.state_a.shape[1]
            conv = conv.transpose(1, 2).float()  # stored [K-1, C]
            page = torch.cat([ssm.reshape(B, -1).float(), conv.reshape(B, -1)], dim=-1)
            page = torch.nn.functional.pad(page, (0, 2 * rows - self.state_numel))
        self.state_a.index_copy_(0, slots, page[:, :rows])
        self.state_b.index_copy_(0, slots, page[:, rows:])

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata=None):
        md = attn_metadata[f"layers.{self.layer_idx}.linear_attn"]
        is_decode = md["max_query_len"] <= md["decode_token_threshold"]
        if is_decode:
            out = self._decode(hidden_states, md)
            if self.world_size > 1:  # >>> PARALLELISM: TP all-reduce <<<
                self.tp_group.all_reduce(out)
            return out
        if self.world_size > 1:  # >>> PARALLELISM: all-gather from SP <<<
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = self._prefill(hidden_states, md)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    # -- helpers -------------------------------------------------------------
    def _project(self, hidden_states):
        b, a = hidden_states @ self.in_proj_b_weight, hidden_states @ self.in_proj_a_weight
        if self.fp8:
            return (_fp8_linear(hidden_states, self.in_proj_qkv_weight, self.in_proj_qkv_weight_scale,
                                self.in_proj_input_scale, self.Dk),
                    _fp8_linear(hidden_states, self.in_proj_z_weight, self.in_proj_z_weight_scale,
                                self.in_proj_input_scale, self.Dv), b, a)
        return hidden_states @ self.in_proj_qkv_weight, hidden_states @ self.in_proj_z_weight, b, a

    def _out_proj(self, o):
        if self.fp8:
            return _fp8_out_proj(o, self.out_proj_weight, self.out_proj_weight_scale,
                                 self.out_proj_input_scale, self.Hv, self.Dv)
        return o @ self.out_proj_weight

    def _conv(self, x):
        """Depthwise causal conv as K shifted multiply-adds (not grouped conv1d).
        x [K-1+T, C] -> [T, C]."""
        T = x.shape[0] - (self.K - 1)
        w = self.conv_weight.to(x.dtype)
        y = x[0:T] * w[:, 0]
        for j in range(1, self.K):
            y = y + x[j : j + T] * w[:, j]
        return y

    def _qkv_heads(self, mixed, n):
        # tensor_split, not torch.split with a size list: split_with_sizes drops the row
        # stride when lowered (each chunk becomes flat[offset:offset+numel]).
        q, k, v = torch.tensor_split(mixed, [self.key_dim, 2 * self.key_dim], dim=-1)
        rep = self.Hv // self.Hk
        # Key head j serves value heads j*rep .. j*rep+rep-1 (== repeat_interleave on heads).
        q = q.reshape(n, self.Hk, 1, self.Dk).expand(n, self.Hk, rep, self.Dk).reshape(n, self.Hv, self.Dk)
        k = k.reshape(n, self.Hk, 1, self.Dk).expand(n, self.Hk, rep, self.Dk).reshape(n, self.Hv, self.Dk)
        return q, k, v.reshape(n, self.Hv, self.Dv)

    def _finish(self, o, z, n):
        o = gdn_ops.gated_rmsnorm(o, z.reshape(n, self.Hv, self.Dv),
                                  self.norm_weight, self.config.rms_norm_eps)
        return self._out_proj(o.reshape(n, self.value_dim))

    # -- prefill: ONE request per step (like the attention path), padded to a bucket ----
    def _prefill(self, hidden_states, md):
        T = hidden_states.shape[0]
        hidden_states = hidden_states.to(self.dtype)
        slot = md["block_table_tensor"][0, :1].long().clamp(min=0)  # [1]: this request's state page
        # Padding tokens are written to the null block (slot 0); real slots are >= block_size.
        valid = md["slot_mapping"] > 0  # [T]
        n_valid = valid.sum()
        # New requests must start from zero state (pages are recycled, never cleared).
        has_state = (md["cached_seq_len"].reshape(()) > 0)

        mixed, z, b, a = self._project(hidden_states)
        conv0, ssm0 = self._read_state(slot)
        conv0, ssm0 = conv0[0], ssm0[0]
        conv0 = torch.where(has_state, conv0, torch.zeros_like(conv0))
        ssm0 = torch.where(has_state, ssm0, torch.zeros_like(ssm0))

        # Causal conv over [state | tokens]; the new state is the last K-1 *valid* inputs.
        xt = torch.cat([conv0.t().to(mixed.dtype), mixed], dim=0)  # [K-1+T, C]
        idx = n_valid + torch.arange(self.K - 1, device=xt.device)
        new_conv = xt.index_select(0, idx).t()  # [C, K-1]
        mixed = torch.nn.functional.silu(self._conv(xt))

        q, k, v = self._qkv_heads(mixed, T)
        g, beta = gdn_ops.gdn_gating(a, b, self.A_log, self.dt_bias)
        # Padded tail tokens become identity updates (no decay, no write).
        vmask = valid.to(g.dtype)[:, None]
        g, beta = g * vmask, beta * vmask
        o, new_ssm = gdn_ops.chunk_gated_delta_rule(q, k, v, g, beta, ssm0)

        self._write_state(slot, new_conv.unsqueeze(0), new_ssm.unsqueeze(0))
        return self._finish(o, z, T)

    # -- decode: B requests x 1 token, each with its own state page ---------------------
    def _use_decode_kernel(self, x: torch.Tensor) -> bool:
        """Fused NKI decode (gdn_kernels.gdn_decode) needs the [pages, rows, 128] state view
        and head_dim == 128; NKI availability per can_run_kernel (VLLM_NEURON_DISABLE_NKI_KERNELS)."""
        if x.device.type == "cpu" or self.state_a is None or self.state_a.dim() != 3:
            return False
        if x.shape[0] != 1:  # B > 1 needs a core barrier around the conv-state update
            return False
        if not (self.Dk == self.Dv == self._STATE_ROW == self.state_a.shape[-1]):
            return False
        if self.state_a.shape[1] % self._STATE_ROW:  # kernel needs heads aligned to the halves
            return False
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        return can_run_kernel(x)

    def _decode(self, hidden_states, md):
        if not self._use_decode_kernel(hidden_states):
            return self._decode_torch(hidden_states, md)
        from .gdn_kernels import gdn_decode

        slots = md["block_table_tensor"][:, 0].clamp(min=0).to(torch.int32)  # padded rows (-1) -> null block 0
        assert hidden_states.shape[0] == slots.shape[0], "GDN decode supports one token per request"
        mixed, z, b, a = self._project(hidden_states.to(self.dtype))
        o = gdn_decode(mixed, z, b, a, self.conv_weight, self.A_log, self.dt_bias, self.norm_weight,
                       self.state_a, self.state_b, slots, self.config.rms_norm_eps)
        return self._out_proj(o)

    def _decode_torch(self, hidden_states, md):
        slots = md["block_table_tensor"][:, 0].long().clamp(min=0)  # [B]; padded rows (-1) -> null block 0
        B = slots.shape[0]
        assert hidden_states.shape[0] == B, "GDN decode supports one token per request"
        hidden_states = hidden_states.to(self.dtype)

        mixed, z, b, a = self._project(hidden_states)
        conv0, ssm0 = self._read_state(slots)  # [B, C, K-1], [B, Hv, Dk, Dv]
        mixed, new_conv = gdn_ops.causal_conv1d_step_batched(mixed, self.conv_weight, conv0)
        q, k, v = self._qkv_heads(mixed, B)
        g, beta = gdn_ops.gdn_gating(a, b, self.A_log, self.dt_bias)
        o, new_ssm = gdn_ops.gated_delta_step_batched(q, k, v, g, beta, ssm0)

        self._write_state(slots, new_conv, new_ssm)
        return self._finish(o, z, B)


# =============================================================================
# Section 4: Dense MLP, decoder layer
# =============================================================================


class Qwen3_5MLP(nn.Module):
    """SiLU-gated MLP, TP intermediate sharding (identical structure to Qwen3MLP)."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        inter = config.intermediate_size // self.world_size
        self.gate_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inter, dtype=config.torch_dtype))
        self.up_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inter, dtype=config.torch_dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, config.hidden_size, dtype=config.torch_dtype))

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        output = NF.mlp(hidden_states, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight)
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)
        return output


class Qwen3_5MLPStaticFP8(nn.Module):
    """Per-tensor static-FP8 MLP with the post-attention RMSNorm fused in (as Llama static FP8).

    Weights are quantized at load from the BF16 checkpoint (scale = amax(full W) / 240);
    activation scales come from offline calibration (fp8_activation_scales_path). The
    decoder layer must skip its own post_attention_layernorm and pass ``ln_w``.
    """

    FP8_INPUT_SCALES = {
        "gate_up_input_scale": ("mlp.gate_proj", "mlp.up_proj"),
        "down_input_scale": ("mlp.down_proj",),
    }

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.eps = config.rms_norm_eps
        inter, H, f8 = config.intermediate_size // self.world_size, config.hidden_size, torch.float8_e4m3fn
        self.gate_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=f8), requires_grad=False)
        self.up_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=f8), requires_grad=False)
        self.down_proj_weight = nn.Parameter(torch.empty(inter, H, dtype=f8), requires_grad=False)
        # [128, 1] fp32 dequant scales (scalar replicated per partition). Weight scales are
        # parameters so the sharded checkpoint loader computes them; input scales are not in
        # the checkpoint (calibration file), so they are buffers set after loading.
        for name in ("gate_weight_scale", "up_weight_scale", "down_weight_scale"):
            setattr(self, name, nn.Parameter(torch.ones(W.SCALE_PARTITIONS, 1), requires_grad=False))
        for name in self.FP8_INPUT_SCALES:
            self.register_buffer(name, torch.ones(W.SCALE_PARTITIONS, 1), persistent=False)

    def _kernel_ok(self, hidden_states: torch.Tensor, n_tokens: int) -> bool:
        """Whether NF.mlp runs the FP8 kernel for n_tokens rows; its PyTorch fallback ignores
        the quantization scales (e.g. CTE with intermediate/rank > 4096 and hidden < 7168)."""
        from nkilib.core.utils.common_types import QuantizationType

        from vllm_neuron.functional.mlp import _can_use_kernel
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        probe = torch.empty((n_tokens, self.gate_proj_weight.shape[0]), device="meta")
        return can_run_kernel(hidden_states) and _can_use_kernel(
            probe, self.gate_proj_weight, quantization_type=QuantizationType.STATIC)

    def _forward_dequant(self, hidden_states, is_prefill: bool, ln_w: torch.Tensor):
        """BF16 MLP over dequantized FP8 weights (weight-only FP8) with the fused norm."""
        x = hidden_states.float()
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * ln_w.float()).to(torch.bfloat16)
        if is_prefill and self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        deq = lambda w, s: w.to(torch.bfloat16) * s[0, 0].to(torch.bfloat16)
        h = F.silu(x @ deq(self.gate_proj_weight, self.gate_weight_scale)) * (
            x @ deq(self.up_proj_weight, self.up_weight_scale))
        return h @ deq(self.down_proj_weight, self.down_weight_scale)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool, ln_w: torch.Tensor) -> torch.Tensor:
        from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
        from nkilib.core.utils.common_types import NormType, QuantizationType

        n_tokens = hidden_states.shape[0] * (self.world_size if is_prefill else 1)
        if not self._kernel_ok(hidden_states, n_tokens):
            output = self._forward_dequant(hidden_states, is_prefill, ln_w)
            return self._reduce(output, is_prefill)
        ln_w = ln_w.to(torch.bfloat16).view(1, -1)
        hidden_states = hidden_states.to(torch.bfloat16)
        if is_prefill or hidden_states.shape[0] > TKG_BS_SEQLEN_THRESHOLD:
            # CTE path cannot fuse norm + quant: quantize first (prefill gathers fp8 rows).
            hidden_states = NF.rmsnorm_quant(
                hidden_states, ln_w=ln_w, input_dequant_scale=self.gate_up_input_scale,
                eps=self.eps, quantization_type=QuantizationType.STATIC)
            if is_prefill and self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            norm_type, mlp_ln_w = NormType.NO_NORM, None
        else:
            norm_type, mlp_ln_w = NormType.RMS_NORM, ln_w  # TKG fuses norm + quant
        output = NF.mlp(
            hidden_states, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight,
            eps=self.eps, ln_w=mlp_ln_w, norm_type=norm_type,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_weight_scale, up_w_scale=self.up_weight_scale,
            down_w_scale=self.down_weight_scale, gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale, output_dtype="bfloat16",
        )
        return self._reduce(output, is_prefill)

    def _reduce(self, output, is_prefill: bool):
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)
        return output


def _mixer_key(config: Qwen3_5Config, i: int) -> str:
    kind = "self_attn" if config.layer_types[i] == FULL_ATTENTION else "linear_attn"
    return f"layers.{i}.{kind}"


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Folded gamma; fp32 to avoid rounding (1 + w) to bf16 (computation is fp32 anyway).
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps, torch.float32)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps, torch.float32)
        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        self.mlp_fp8 = config.fp8_quantized(f"layers.{layer_idx}.mlp")
        self.mlp = Qwen3_5MLPStaticFP8(config) if self.mlp_fp8 else Qwen3_5MLP(config)
        self.mixer_key = _mixer_key(config, layer_idx)

    @property
    def mixer(self):
        return self.self_attn if self.layer_type == FULL_ATTENTION else self.linear_attn

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata=None):
        md = attn_metadata[self.mixer_key]
        is_decode = md["max_query_len"] <= md["decode_token_threshold"]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mixer(hidden_states, positions, position_embeddings, attn_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.mlp_fp8:  # norm is fused into the fp8 MLP kernels
            hidden_states = self.mlp(hidden_states, is_prefill=not is_decode,
                                     ln_w=self.post_attention_layernorm.weight)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        return residual + hidden_states


# =============================================================================
# Section 5: Backbone and LM head
# =============================================================================


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=config.hidden_size,
            dtype=config.torch_dtype, tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps, torch.float32)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)
        self.first_mixer_key = _mixer_key(config, 0)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size, is_storage_transposed=False, pad_shard=True),
        )

    def forward(self, input_ids, positions, attn_metadata=None, rank=None,
                inputs_embeds=None, is_token_ids=None):
        md0 = attn_metadata[self.first_mixer_key]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        if (is_prefill and self.world_size > 1 and inputs_embeds is not None
                and is_token_ids is not None):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]
        hidden_states = NF.merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids)

        position_embeddings = self.rotary_emb(positions, device=hidden_states.device, dtype=hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions=positions,
                                  position_embeddings=position_embeddings, attn_metadata=attn_metadata)
        hidden_states = self.norm(hidden_states)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states, []


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 dense hybrid text model + column-parallel LM head (untied)."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config if config.neuron_config else None)
        debug_logits = config.neuron_config is not None and config.neuron_config.debug_logits_dir is not None
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0) or debug_logits

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config, tp_group=self.tp_group.device_group)
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(shard_dim=0, shard_size=config.vocab_size // self.world_size,
                                   num_shards=self.world_size, is_storage_transposed=False))
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)

        self._setup_weight_loaders()

    @torch.no_grad()
    def forward(self, input_ids, positions, inputs_embeds=None, is_token_ids=None,
                attn_metadata=None, sampling_positions=None, sampling_params=None,
                spec_decode_metadata=None, logit_mask=None, rank=None, **kwargs):
        # ``kwargs`` swallows the runner's rotary_position_ids: text-only mrope rows are equal,
        # so the 1D cache positions carry the same information.
        positions = positions.to(torch.int32)
        md0 = attn_metadata[self.model.first_mixer_key]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP.")

        hidden_states, _ = self.model(input_ids, positions, attn_metadata=attn_metadata, rank=rank,
                                      inputs_embeds=inputs_embeds, is_token_ids=is_token_ids)
        logits = self.lm_head(torch.index_select(hidden_states, dim=0, index=sampling_positions))
        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1) if self.tp_group is not None else logits
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(Qwen3_5Config.from_configs(hf_config, neuron_config))

    # ── Multimodal-rope protocol (text only) ─────────────────────────────

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features) -> tuple[torch.Tensor, int]:
        """Text-only: identical positions on all three mrope rows, zero delta."""
        return torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1).clone(), 0

    # ── KV cache / recurrent state ───────────────────────────────────────

    def get_kv_spec(self):
        cfg, tp = self.config, self.world_size
        layers, state_layers = [], []
        conv_shape, ssm_shape = gdn_state_shapes(cfg, tp)
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type == FULL_ATTENTION:
                attn = layer.self_attn
                layers.append(LayerSpec(
                    name=f"layers.{i}.self_attn", num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim, dtype=attn.dtype, sliding_window_size=None, chunk_size=None))
            else:
                state_layers.append(StateLayerSpec(
                    name=f"layers.{i}.linear_attn", shapes=(conv_shape, ssm_shape),
                    dtypes=(cfg.mamba_ssm_dtype, cfg.mamba_ssm_dtype)))
        return KVSpec(layers=layers, state_layers=state_layers)

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != FULL_ATTENTION:
                continue
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache, layer.self_attn.v_cache = kv_caches[name][0], kv_caches[name][1]

    def set_kv_blocks_cover_context(self, value: bool):
        for layer in self.model.layers:
            if layer.layer_type == FULL_ATTENTION:
                layer.self_attn.kv_blocks_cover_context = value

    def bind_state_cache(self, state_caches):
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != LINEAR_ATTENTION:
                continue
            name = f"layers.{i}.linear_attn"
            if name not in state_caches:
                raise KeyError(f"State cache for layer {name} not initialized")
            layer.linear_attn.bind_state(*state_caches[name])

    # ── Weight loading ───────────────────────────────────────────────────

    def _setup_weight_loaders(self):
        """Attach per-parameter transforms (HF checkpoint tensors -> this rank's layout)."""
        cfg, tp = self.config, self.world_size
        for i, layer in enumerate(self.model.layers):
            p = f"{CKPT}layers.{i}."
            fold = _loader(lambda t, r: W.fold_zero_centered_norm(t[0]).float())
            set_weight_loader(layer.input_layernorm.weight, fold)
            set_weight_loader(layer.post_attention_layernorm.weight, fold)
            mlp = layer.mlp
            if layer.mlp_fp8:
                q = lambda dev: _loader(lambda t, r: W.fp8_quantize(dev(t[0], r, tp), W.fp8_weight_scale(t[0])))
                scale = _loader(lambda t, r: W.fp8_scale_tile(W.fp8_weight_scale(t[0])))
                set_weight_loader(mlp.gate_proj_weight, q(W.mlp_gate_up_device))
                set_weight_loader(mlp.up_proj_weight, q(W.mlp_gate_up_device))
                set_weight_loader(mlp.down_proj_weight, q(W.mlp_down_device))
                for sname in ("gate_weight_scale", "up_weight_scale", "down_weight_scale"):
                    set_weight_loader(getattr(mlp, sname), scale)
            else:
                set_weight_loader(mlp.gate_proj_weight, _loader(lambda t, r: W.mlp_gate_up_device(t[0], r, tp)))
                set_weight_loader(mlp.up_proj_weight, _loader(lambda t, r: W.mlp_gate_up_device(t[0], r, tp)))
                set_weight_loader(mlp.down_proj_weight, _loader(lambda t, r: W.mlp_down_device(t[0], r, tp)))
            if layer.layer_type == FULL_ATTENTION:
                a = layer.self_attn
                set_weight_loader(a.qkv_proj_weight, _loader(
                    lambda t, r: W.attention_qkv_device(t[0], t[1], t[2], cfg, r, tp)))
                set_weight_loader(a.gate_weight, _loader(lambda t, r: W.attention_gate_device(t[0], cfg, r, tp)))
                set_weight_loader(a.o_proj_weight, _loader(lambda t, r: W.attention_o_device(t[0], cfg, r, tp)))
                qk = _loader(lambda t, r: W.attention_qk_norm_device(t[0], cfg))
                set_weight_loader(a.q_norm.weight, qk)
                set_weight_loader(a.k_norm.weight, qk)
            else:
                g = layer.linear_attn
                qkv_dev = lambda t, r: W.gdn_in_proj_qkv_device(t, cfg, r, tp)
                heads_dev = lambda t, r: W.gdn_in_proj_heads_device(t, r, tp)
                out_dev = lambda t, r: W.gdn_out_proj_device(t, cfg, r, tp)
                if g.fp8:
                    for param, dev_fn, cols in (("in_proj_qkv", qkv_dev, 3), ("in_proj_z", heads_dev, 3),
                                                ("out_proj", out_dev, 1)):
                        wl, sl = _fp8_loaders(dev_fn, cols)
                        set_weight_loader(getattr(g, f"{param}_weight"), wl)
                        set_weight_loader(getattr(g, f"{param}_weight_scale"), sl)
                else:
                    set_weight_loader(g.in_proj_qkv_weight, _loader(lambda t, r: qkv_dev(t[0], r)))
                    set_weight_loader(g.in_proj_z_weight, _loader(lambda t, r: heads_dev(t[0], r)))
                    set_weight_loader(g.out_proj_weight, _loader(lambda t, r: out_dev(t[0], r)))
                head_cols = _loader(lambda t, r: heads_dev(t[0], r))
                for wp in (g.in_proj_b_weight, g.in_proj_a_weight):
                    set_weight_loader(wp, head_cols)
                set_weight_loader(g.conv_weight, _loader(lambda t, r: W.gdn_conv_device(t[0], cfg, r, tp)))
                vec = _loader(lambda t, r: W.gdn_head_vector_device(t[0], cfg, r, tp).float())
                set_weight_loader(g.A_log, vec)
                set_weight_loader(g.dt_bias, vec)
        set_weight_loader(self.model.norm.weight, _loader(lambda t, r: W.fold_zero_centered_norm(t[0]).float()))

    def _weight_mappings(self) -> dict:
        """Param name -> checkpoint key(s). Multiple keys are passed to the loader in order."""
        m = {
            "model.embed_tokens.weight": f"{CKPT}embed_tokens.weight",
            "model.norm.weight": f"{CKPT}norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i, layer in enumerate(self.model.layers):
            pre, ck = f"model.layers.{i}.", f"{CKPT}layers.{i}."
            m[pre + "input_layernorm.weight"] = ck + "input_layernorm.weight"
            m[pre + "post_attention_layernorm.weight"] = ck + "post_attention_layernorm.weight"
            m[pre + "mlp.gate_proj_weight"] = ck + "mlp.gate_proj.weight"
            m[pre + "mlp.up_proj_weight"] = ck + "mlp.up_proj.weight"
            m[pre + "mlp.down_proj_weight"] = ck + "mlp.down_proj.weight"
            if layer.mlp_fp8:
                for part in ("gate", "up", "down"):
                    m[pre + f"mlp.{part}_weight_scale"] = ck + f"mlp.{part}_proj.weight"
            if layer.layer_type == FULL_ATTENTION:
                s, c = pre + "self_attn.", ck + "self_attn."
                m[s + "qkv_proj_weight"] = [c + "q_proj.weight", c + "k_proj.weight", c + "v_proj.weight"]
                m[s + "gate_weight"] = c + "q_proj.weight"
                m[s + "o_proj_weight"] = c + "o_proj.weight"
                m[s + "q_norm.weight"] = c + "q_norm.weight"
                m[s + "k_norm.weight"] = c + "k_norm.weight"
            else:
                s, c = pre + "linear_attn.", ck + "linear_attn."
                for part in ("qkv", "z", "b", "a"):
                    m[s + f"in_proj_{part}_weight"] = c + f"in_proj_{part}.weight"
                m[s + "conv_weight"] = c + "conv1d.weight"
                m[s + "A_log"] = c + "A_log"
                m[s + "dt_bias"] = c + "dt_bias"
                m[s + "norm_weight"] = c + "norm.weight"
                m[s + "out_proj_weight"] = c + "out_proj.weight"
                if layer.linear_attn.fp8:
                    for part in ("in_proj_qkv", "in_proj_z", "out_proj"):
                        m[s + f"{part}_weight_scale"] = c + f"{part}.weight"
        return m

    def _materialize_buffers(self) -> None:
        """The runner builds the model on the meta device and loads parameters with assign=True.
        Non-persistent buffers are not in the checkpoint, so re-create them with real data."""
        self.model.rotary_emb.inv_freq = W.rope_inv_freq(
            self.config.rotary_dim, self.config.rope_theta
        )

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        self._materialize_buffers()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self._weight_mappings(), device).state_dict
        # Cast to each parameter's own dtype (norm gammas / A_log / dt_bias stay float32).
        params = dict(self.named_parameters())
        for name, tensor in rank_sharded.items():
            target = params[name].dtype if name in params else self.config.torch_dtype
            if tensor.dtype != target:
                rank_sharded[name] = tensor.to(target)
        self.load_state_dict(rank_sharded, strict=False, assign=True)
        self._load_fp8_activation_scales(checkpoint_path)

    def _fp8_modules(self):
        """(layer index, module) for every static-FP8 module with input scales."""
        for i, layer in enumerate(self.model.layers):
            if layer.mlp_fp8:
                yield i, layer.mlp
            if layer.layer_type == LINEAR_ATTENTION and layer.linear_attn.fp8:
                yield i, layer.linear_attn

    def _load_fp8_activation_scales(self, checkpoint_path: str | None) -> None:
        """Static-FP8 input scales (amax / 240) from the offline calibration JSON."""
        modules = list(self._fp8_modules())
        if not modules:
            return
        nc = self.config.neuron_config
        path = nc.fp8_activation_scales_path or os.path.join(checkpoint_path or "", "fp8_act_amax.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"quantization='fp8' needs calibrated activation scales; {path} not found. "
                "Set neuron_config.fp8_activation_scales_path (see calibrate_fp8.py).")
        with open(path) as f:
            amax = json.load(f)
        for i, mod in modules:
            dev = next(mod.parameters()).device
            for buf, keys in mod.FP8_INPUT_SCALES.items():
                missing = [k for k in keys if f"layers.{i}.{k}" not in amax]
                if missing:
                    raise KeyError(f"{path} has no calibration entry for layers.{i}.{missing[0]}")
                a = max(amax[f"layers.{i}.{k}"]["amax"] for k in keys)
                setattr(mod, buf, W.fp8_scale_tile(W.fp8_scale(a)).to(dev))
        logger.info("Loaded static-FP8 activation scales for %d modules from %s", len(modules), path)

    def load_weights_lite(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        """CPU-compile path: no KV-cache scales to load for BF16; only buffers need real data."""
        self._materialize_buffers()
        self._load_fp8_activation_scales(checkpoint_path)
