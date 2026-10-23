# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next (qwen4_exp), text only
=========================================

Built on the Qwen3.5 hybrid (``vllm_neuron/model/qwen3_5/model.py``): the Gated DeltaNet and
gated-attention mixers are the Qwen3.5 classes with small overrides. What is new:

  * Gated residual (hyper-connections). The residual stream is ``hc_count`` (4) streams of
    ``hidden_size``, [T, 4 * H]. Each sub-layer reads its input as a gated mean of the
    group-RMS-normalised streams and writes its output back into every stream with a per-stream
    gate (``Qwen4ExpGatedResidual``). This replaces the input/post-attention RMSNorms; there is
    no final norm, the last read (``hyper_connection_mixer``) feeds the LM head.
  * Sparse MoE MLP on every layer (``moe.py``): 512 experts split across the TP ranks (expert
    parallel inside the TP group), FP8 experts with ``quantization="fp8"``.
  * PLE on layer 1 (0-based): hashed n-gram embeddings. The 51B-parameter table and the int64
    hash stay on the host (``ngram.py``); every step the runner passes the gathered rows as
    ``ple_embeds`` [T, ple_embed_dim] (``build_host_inputs``). On device, ``Qwen4ExpPLE`` gates
    them per stream against the residual and adds a dilated causal conv (state: the last 9
    conv inputs per request, a separate recurrent-state layer ``layers.1.ple``).
  * QSA on the attention layers (``qsa.py``, ``Qwen4ExpAttention``). Raw indexer keys go to a
    side cache per layer, addressed like its K/V cache (``bind_index_cache``). When the static
    context bucket is <= 2048 slots every block is selected and the dense Qwen3.5 paths run
    unchanged; otherwise prefill attends in torch with the QSA token mask and decode ANDs it
    into the decode kernel's mask.
  * GDN output gate is sigmoid (``output_gate_type``), and the 2 KV heads are replicated for
    tp > 2 (one KV head per rank, so the Qwen3.5 kernels and KV views apply).

Layout conventions follow Qwen3.5: prefill is sequence-parallel (the residual is [T/tp, 4H];
mixers and the MoE all-gather their [T/tp, H] input and reduce-scatter the output), decode is
replicated with all-reduce.
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, StateLayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import gdn_ops
from vllm_neuron.model.qwen3_5 import model as Q
from vllm_neuron.model.qwen3_5 import weights as W35
from vllm_neuron.model.qwen3_5.state import gdn_state_shapes
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from . import qsa
from . import weights as W
from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen4ExpConfig
from .moe import Qwen4ExpSparseMoe, attach_moe_loaders, moe_weight_mappings

logger = logging.getLogger(__name__)

CKPT = W.TEXT_PREFIX  # "model.language_model."
_loader = Q._loader


def group_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, group: int, eps: float):
    """RMSNorm over consecutive groups of ``group`` channels; ``gamma`` is the folded (1 + w)
    float32 weight over all channels. Computed in float32, returned in x's dtype (as HF)."""
    shape = x.shape
    x32 = x.float().reshape(*shape[:-1], -1, group)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32.reshape(shape) * gamma).to(x.dtype)


# =============================================================================
# Section 1: Gated residual (hyper-connections)
# =============================================================================


class Qwen4ExpGatedResidual(nn.Module):
    """Read / write of the widened residual stream around one sub-layer.

    read(x [T, hc*H]) -> block input [T, H] (gated mean over the normalised streams) and, with
    ``use_combine``, the write gates [T, hc] = 2 * sigmoid(W_inject xn / hc).
    write(x, out [T, H], gates) -> x + out * gate per stream (``x`` is the un-normalised input).

    >>> PARALLELISM: replicated weights; per-token math, so it runs on the sequence-parallel
    shard in prefill and on replicated rows in decode. <<<
    The down-projection and the write-gate projection read the same normalised input, so they
    are one [hc*H, lowrank + hc] matmul.
    """

    def __init__(self, config: Qwen4ExpConfig, use_combine: bool = True):
        super().__init__()
        self.hc, self.H = config.hc_count, config.hidden_size
        self.r = config.hc_lowrank
        self.eps = config.rms_norm_eps
        self.use_combine = use_combine
        hcH, dt = config.hc_hidden_size, config.torch_dtype
        self.norm_weight = nn.Parameter(torch.ones(hcH, dtype=torch.float32))
        n_out = self.r + (self.hc if use_combine else 0)
        self.down_inject_weight = nn.Parameter(torch.empty(hcH, n_out, dtype=dt))
        self.up_weight = nn.Parameter(torch.empty(self.r, hcH, dtype=dt))

    def read(self, x: torch.Tensor):
        T = x.shape[0]
        xn = group_rmsnorm(x, self.norm_weight, self.H, self.eps)
        di = xn @ self.down_inject_weight
        low = F.silu(di[:, : self.r] / self.hc)
        gate = torch.sigmoid(low @ self.up_weight)
        block_in = (gate.view(T, self.hc, self.H) * xn.view(T, self.hc, self.H)).mean(
            dim=1
        )
        if not self.use_combine:
            return block_in
        inject = 2 * torch.sigmoid(di[:, self.r :] / self.hc)
        return block_in, inject

    @staticmethod
    def write(x: torch.Tensor, out: torch.Tensor, inject: torch.Tensor):
        T = x.shape[0]
        return x + (out.to(x.dtype).unsqueeze(1) * inject.unsqueeze(-1)).reshape(T, -1)


# =============================================================================
# Section 2: PLE (device side of the n-gram embedding)
# =============================================================================


class Qwen4ExpPLE(nn.Module):
    """Gate host-gathered n-gram embeddings into every residual stream (HF Qwen4ExpTextPLELayer).

    ple_embeds [T, E] (E = ple_embed_dim, the 16 hashed heads' rows concatenated):
        key [T, hc, H] = norm_k(E @ key_proj), value [T, H] = E @ value_proj
        g_s = sigmoid(signed_sqrt(<key_s, norm_q(x_s)> / sqrt(H)))     per stream s
        v_s = g_s * value;   out = x + v + silu(conv(norm_c(v)))
    The conv is depthwise over hc*H channels, kernel K=4, dilation ngram_size=3, causal; a
    request's state is its last (K-1)*3 = 9 conv inputs, kept float32 in a state page.

    >>> PARALLELISM: replicated. Prefill all-gathers the residual (the conv mixes neighbouring
    tokens across sequence-parallel shards) and returns this rank's shard. <<<
    """

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc, self.H = config.hc_count, config.hidden_size
        self.C = config.hc_hidden_size
        self.K, self.dil = config.ple_conv_kernel_size, config.ngram_size
        self.L = config.ple_conv_state_len
        self.eps = config.rms_norm_eps
        self.dtype = config.torch_dtype
        E = config.ple_embed_dim
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        # [key (hc*H) | value (H)]: both read the embeddings.
        self.key_value_weight = nn.Parameter(
            torch.empty(E, self.C + self.H, dtype=self.dtype)
        )
        for n in ("norm_key_weight", "norm_query_weight", "norm_conv_weight"):
            setattr(self, n, nn.Parameter(torch.ones(self.C, dtype=torch.float32)))
        self.conv_weight = nn.Parameter(torch.empty(self.C, self.K, dtype=self.dtype))
        self.state_numel = self.L * self.C
        self.state_a = self.state_b = None  # bound by bind_state_cache

    @property
    def state_name(self) -> str:
        return f"layers.{self.layer_idx}.ple"

    # -- state pages: [L, C] float32 over the two halves (state_cache.paired_half_views),
    # viewed as rows of 128 floats like the GDN pages: index_copy_ of whole [half] rows (one
    # partition each) does not lower (NCC_ITEN406), and the 3-D view must be made outside the graph.
    _STATE_ROW = 128

    def bind_state(self, state_a: torch.Tensor, state_b: torch.Tensor):
        R, half = self._STATE_ROW, state_a.shape[1]
        if half % R == 0 and self.state_numel % R == 0:
            state_a, state_b = (
                state_a.view(-1, half // R, R),
                state_b.view(-1, half // R, R),
            )
        self.state_a, self.state_b = state_a, state_b

    def _read_state(self, slots: torch.Tensor) -> torch.Tensor:
        """slots [B] long -> [B, L, C] float32."""
        B = slots.shape[0]
        page = torch.cat(
            [self.state_a.index_select(0, slots), self.state_b.index_select(0, slots)],
            dim=1,
        )  # [B, 2*rows, R] or [B, 2*half]
        return page.reshape(B, -1)[:, : self.state_numel].reshape(B, self.L, self.C)

    def _write_state(self, slots: torch.Tensor, state: torch.Tensor):
        """slots [B] (-1 for padded rows: zeros go to the null page)."""
        B = slots.shape[0]
        pad = (slots < 0).view(B, 1, 1)
        page = state.float().reshape(B, self.L, self.C)
        page = torch.where(
            pad, torch.zeros((), dtype=page.dtype, device=page.device), page
        )
        slots = slots.clamp(min=0)
        if self.state_a.dim() == 3:
            R, rows = self._STATE_ROW, self.state_a.shape[1]
            page = page.reshape(B, -1, R)
            page = F.pad(page, (0, 0, 0, 2 * rows - page.shape[1]))
            self.state_a.index_copy_(0, slots, page[:, :rows])
            self.state_b.index_copy_(0, slots, page[:, rows:])
        else:
            rows = self.state_a.shape[1]
            page = F.pad(page.reshape(B, -1), (0, 2 * rows - self.state_numel))
            self.state_a.index_copy_(0, slots, page[:, :rows])
            self.state_b.index_copy_(0, slots, page[:, rows:])

    def _gated(self, x: torch.Tensor, emb: torch.Tensor):
        """-> (gated values v [T, C], conv input norm_c(v) [T, C])."""
        T = x.shape[0]
        kv = emb.to(self.dtype) @ self.key_value_weight
        key = group_rmsnorm(kv[:, : self.C], self.norm_key_weight, self.H, self.eps)
        value = kv[:, self.C :]
        query = group_rmsnorm(x, self.norm_query_weight, self.H, self.eps)
        gate = (key.view(T, self.hc, self.H) * query.view(T, self.hc, self.H)).sum(
            -1, keepdim=True
        ) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        v = (torch.sigmoid(gate) * value.unsqueeze(1)).reshape(T, self.C)
        return v, group_rmsnorm(v, self.norm_conv_weight, self.H, self.eps)

    def _conv(self, u: torch.Tensor) -> torch.Tensor:
        """Dilated depthwise causal conv as K shifted multiply-adds. u [L+T, C] -> [T, C]."""
        T = u.shape[0] - self.L
        w = self.conv_weight.to(u.dtype)
        y = u[0:T] * w[:, 0]
        for j in range(1, self.K):
            y = y + u[j * self.dil : j * self.dil + T] * w[:, j]
        return y

    def forward(self, x, ple_embeds, attn_metadata, is_decode: bool, rank=None):
        md = attn_metadata[self.state_name]
        if is_decode:
            return self._decode(x, ple_embeds, md)
        return self._prefill(x, ple_embeds, md, rank)

    def _decode(self, x, emb, md):
        """B requests x S tokens each (S = 1, or 1 + k in an MTP verify step). With S > 1 the
        state starts from block ``state_in`` and the state after token j goes to block j (as
        the GDN layers do)."""
        bt_raw = md["block_table_tensor"].long()  # [B, n]; -1: padded row
        B = bt_raw.shape[0]
        S = x.shape[0] // B
        assert x.shape[0] == B * S, "PLE decode: the same number of tokens per request"
        if "state_in" in md:
            slots_in = (
                bt_raw.clamp(min=0).gather(1, md["state_in"].long().view(B, 1)).view(B)
            )
        else:
            slots_in = bt_raw[:, 0].clamp(min=0)
        v, vn = self._gated(x, emb)
        state = self._read_state(slots_in).to(x.dtype)  # [B, L, C]
        w = self.conv_weight.to(x.dtype)
        vn = vn.view(B, S, self.C)
        ys = []
        for j in range(S):
            u = torch.cat([state, vn[:, j : j + 1]], dim=1)  # [B, L+1, C]
            y = u[:, 0] * w[:, 0]
            for i in range(1, self.K):
                y = y + u[:, i * self.dil] * w[:, i]
            state = u[:, 1:]
            self._write_state(bt_raw[:, j], state)
            ys.append(y)
        y = torch.stack(ys, dim=1).reshape(B * S, self.C)
        return x + v + F.silu(y)

    def _prefill(self, x, emb, md, rank):
        """One request per step, padded to the bucket; x is this rank's shard [T/tp, C]."""
        if self.world_size > 1:  # >>> PARALLELISM: all-gather from SP <<<
            x = self.tp_group.all_gather(x, dim=0)
        T = x.shape[0]
        slot = md["block_table_tensor"][0, :1].long().clamp(min=0)
        valid = (md["slot_mapping"] > 0).view(
            T, 1
        )  # padding rows go to the null slot 0
        n_valid = valid.sum()
        has_state = md["cached_seq_len"].reshape(()) > 0

        v, vn = self._gated(x, emb)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        v, vn = torch.where(valid, v, zero), torch.where(valid, vn, zero)
        state = self._read_state(slot)[0].to(x.dtype)
        state = torch.where(has_state, state, torch.zeros_like(state))
        u = torch.cat([state, vn], dim=0)  # [L + T, C]
        new_state = u.index_select(0, n_valid + torch.arange(self.L, device=u.device))
        self._write_state(slot, new_state.unsqueeze(0))
        out = x + v + F.silu(self._conv(u))
        if self.world_size > 1:  # >>> PARALLELISM: back to this rank's SP shard <<<
            # One rank-invariant SPMD graph: the rank is a runtime int32 tensor.
            n = T // self.world_size
            r = rank.reshape(()) if rank is not None else self.tp_group.rank_in_group
            idx = r * n + torch.arange(n, dtype=torch.int32, device=out.device)
            out = out.index_select(0, idx)
        return out


# =============================================================================
# Section 3: Mixers (Qwen3.5 classes with Qwen3.8-Flash-Next overrides)
# =============================================================================


class Qwen4ExpGatedDeltaNet(Q.Qwen3_5GatedDeltaNet):
    """Qwen3.5 GDN with the configured output-gate activation (sigmoid for this model).

    The fused FP8 decode kernel applies an input RMSNorm that this model does not have (the
    gated-residual read produces the block input), so only the BF16 decode kernel is used, for
    up to ``DECODE_KERNEL_MAX_REQUESTS`` requests: each logical-core program writes only the
    conv-state rows of its own channels (``own_conv_rows``), so B > 1 needs no core barrier."""

    DECODE_KERNEL_MAX_REQUESTS = 16

    def _finish(self, o, z, n):
        o = gdn_ops.gated_rmsnorm(
            o,
            z.reshape(n, self.Hv, self.Dv),
            self.norm_weight,
            self.config.rms_norm_eps,
            gate_act=self.config.output_gate_type,
        )
        return self._out_proj(o.reshape(n, self.value_dim))

    def use_fused_decode(self, x: torch.Tensor) -> bool:
        return False

    def _decode(self, hidden_states, md):
        if self.Hk % 2 or not self._use_decode_kernel(
            hidden_states, max_requests=self.DECODE_KERNEL_MAX_REQUESTS
        ):
            return self._decode_torch(hidden_states, md)
        from vllm_neuron.model.qwen3_5.gdn_kernels import gdn_decode

        # Padded batch rows keep -1: the kernel skips their page reads and writes (the null
        # block, which attention may read as KV, never receives float32 state); their output
        # rows are undefined, so zero them.
        slots = md["block_table_tensor"][:, 0].to(torch.int32)
        assert hidden_states.shape[0] == slots.shape[0], (
            "GDN decode supports one token per request"
        )
        mixed, z, b, a = self._project(hidden_states.to(self.dtype))
        o = gdn_decode(
            mixed,
            z,
            b,
            a,
            self.conv_weight,
            self.A_log,
            self.dt_bias,
            self.norm_weight,
            self.state_a,
            self.state_b,
            slots,
            self.config.rms_norm_eps,
            gate_sigmoid=self.config.output_gate_type == "sigmoid",
            own_conv_rows=True,
        )
        if slots.shape[0] > 1:
            o = torch.where(
                (slots >= 0).view(-1, 1),
                o,
                torch.zeros((), dtype=o.dtype, device=o.device),
            )
        return self._out_proj(o)


class Qwen4ExpAttention(Q.Qwen3_5Attention):
    """Qwen3.5 gated attention with replicated KV heads (tp > num_key_value_heads) and QSA.

    QSA (``qsa.py``): a replicated indexer (4 query heads + 1 raw key x 128) scores mean-pooled
    4-token key blocks; each query attends to its top ``block_topk`` (512) complete blocks plus
    the open tail. Every step writes this step's raw indexer keys into a side cache that shares
    this layer's block table and slot mapping (``bind_index_cache``).

    The context bucket ``S`` (block-table width x block size, static) picks the path:
      * ``qsa_is_dense(S)`` (S <= 2048): every visible block is selected, QSA == dense causal
        attention, so the Qwen3.5 paths run unchanged (segmented / flash prefill, decode kernel).
      * otherwise prefill attends over the cached context in torch with the QSA token mask
        (``_attend_cached_qsa``), and decode ANDs the token mask into the pre-generated mask of
        the decode kernel (``qsa.decode_mask_from_token_mask``); the CPU / no-kernel decode
        applies it in ``_decode_torch``. MTP verify rows (S_tkg > 1) use the same masks.
    """

    FP8_INPUT_SCALES = {}

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.config = config
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.scaling = config.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        config.validate_tp(self.world_size)
        self.num_attention_heads_per_rank = (
            config.num_attention_heads // self.world_size
        )
        self.num_key_value_heads_per_rank = config.kv_heads_per_rank(self.world_size)
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size, self.kv_size = q_size, kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        H = self.hidden_size
        self.fp8 = False
        self.gate_weight = nn.Parameter(torch.empty(H, q_size, dtype=self.dtype))
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(H, q_size + 2 * kv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(torch.empty(q_size, H, dtype=self.dtype))
        self.q_norm = Q.Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Q.Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # QSA indexer (replicated): [4 query heads | 1 key head] x 128, zero-centered norms.
        n_idx = (
            config.indexer_n_heads + config.indexer_kv_heads
        ) * config.indexer_head_dim
        self.index_qk_weight = nn.Parameter(torch.empty(H, n_idx, dtype=self.dtype))
        self.index_q_norm_weight = nn.Parameter(
            torch.ones(config.indexer_head_dim, dtype=torch.float32)
        )
        self.index_k_norm_weight = nn.Parameter(
            torch.ones(config.indexer_head_dim, dtype=torch.float32)
        )
        # Indexer rope: same inv_freq as the main attention, HF cos/sin layout (qsa.py).
        self.register_buffer(
            "index_inv_freq",
            W35.rope_inv_freq(config.rotary_dim, config.rope_theta),
            persistent=False,
        )
        # The raw indexer keys live in a side cache addressed like this layer's K/V cache
        # (same block ids, slots and block table): [num_blocks, 1, block_size, D] in the model
        # dtype, allocated by ``bind_index_cache`` outside vLLM's shared KV pool. A KV group
        # of its own in the shared pool would bind a second set of graph inputs over the same
        # bytes, which neuronx-cc's graph I/O check (NCC_EVRF009) counts twice. The runner
        # keeps the side cache out of the KV budget (``kv_side_bytes_per_token_per_attn_layer``).
        self.md_key = f"layers.{layer_idx}.self_attn"  # KV spec name and metadata key
        self.index_cache = None  # plain attribute: tracing swaps it

        self.k_cache = self.v_cache = None
        self.runner_clears_kv_blocks = False
        self.k_scale = self.v_scale = None
        self.k_scale_float = self.v_scale_float = 1.0
        self._decode_token_mask = None  # QSA mask of the decode step in flight

    def materialize_buffers(self):
        self.index_inv_freq = W35.rope_inv_freq(
            self.config.rotary_dim, self.config.rope_theta
        )

    def kv_layer_spec(self) -> LayerSpec:
        """This layer's K/V cache; the QSA indexer keys are a side cache (``bind_kv``)."""
        return LayerSpec(
            name=self.md_key,
            num_kv_heads=self.num_key_value_heads_per_rank,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def bind_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        """Bind the runner's K/V cache, then allocate the indexer side cache next to it."""
        self.k_cache, self.v_cache = k_cache, v_cache
        self.bind_index_cache()
        if self.kv_fp8:
            for name in ("k_scale", "v_scale"):
                setattr(
                    self,
                    name,
                    torch.ones(1, 1, dtype=torch.bfloat16, device=k_cache.device),
                )
            self.k_scale_float = self.v_scale_float = 1.0

    def bind_index_cache(self):
        """Allocate the indexer side cache to match the bound V cache [num_blocks, nkh, bs, Dh]
        (V is never packed). Zero-filled; stale keys of reused blocks are never visible."""
        v = self.v_cache
        self.index_cache = torch.zeros(
            v.shape[0],
            1,
            v.shape[-2],
            self.config.indexer_head_dim,
            dtype=self.dtype,
            device=v.device,
        )

    # ── QSA indexer ──────────────────────────────────────────────────────

    def _qsa_dense(self, attn_metadata) -> bool:
        """Static: True if the step's context bucket makes QSA dense (all blocks selected)."""
        md = attn_metadata[self.md_key]
        S = md["block_table_tensor"].shape[1] * md["block_size"]
        return qsa.qsa_is_dense(
            S, self.config.block_topk, self.config.indexer_compress_ratio
        )

    def _index_keys(self, x: torch.Tensor, attn_metadata) -> None:
        """Write this step's raw indexer keys (x [T, H]); used when the selection is dense."""
        D = self.config.indexer_head_dim
        k_raw = x @ self.index_qk_weight[:, self.config.indexer_n_heads * D :]
        qsa.write_raw_keys(
            self.index_cache, k_raw, attn_metadata[self.md_key]["slot_mapping"]
        )

    def _qsa_index(self, x, q_pos, attn_metadata, S: int):
        """Indexer for this step: write the raw keys, then score inputs. x [T, H] (rows
        request-major, B requests x Tq tokens), q_pos [B, Tq] int32 ->
        (q_idx [B, Tq, nh, D] normed + roped, kc [B, S/4, D] compressed keys)."""
        cfg = self.config
        md = attn_metadata[self.md_key]
        B, Tq = q_pos.shape
        cos, sin = qsa.qsa_rope_cos_sin(q_pos.reshape(-1), self.index_inv_freq, x.dtype)
        q_idx, k_raw = qsa.index_qk(
            x,
            self.index_qk_weight,
            self.index_q_norm_weight,
            cos,
            sin,
            n_heads=cfg.indexer_n_heads,
            eps=cfg.rms_norm_eps,
        )
        # Raw keys first: a query at p = 4b + 3 selects among blocks that include itself.
        qsa.write_raw_keys(self.index_cache, k_raw, md["slot_mapping"])
        kctx = qsa.gather_raw_keys(self.index_cache, md["block_table_tensor"])
        if kctx.shape[1] != S:
            raise ValueError(
                f"QSA: the block table spans {kctx.shape[1]} slots, expected {S}"
            )
        kc = qsa.compressed_keys(
            kctx,
            self.index_k_norm_weight,
            self.index_inv_freq,
            ratio=cfg.indexer_compress_ratio,
            eps=cfg.rms_norm_eps,
        )
        return q_idx.view(B, Tq, cfg.indexer_n_heads, -1), kc

    def _qsa_select(self, q_idx, kc, q_pos, S: int) -> torch.Tensor:
        """bool [B, Tq, S] allowed keys (selected blocks + tail, causal)."""
        cfg = self.config
        sel = qsa.select_blocks(
            q_idx, kc, q_pos, cfg.block_topk, cfg.indexer_compress_ratio
        )
        return qsa.token_mask(sel, q_pos, S, cfg.indexer_compress_ratio)

    def _qsa_token_mask(self, x, q_pos, attn_metadata, S: int) -> torch.Tensor:
        """Write this step's raw keys, then select: bool [B, Tq, S]."""
        q_idx, kc = self._qsa_index(x, q_pos, attn_metadata, S)
        return self._qsa_select(q_idx, kc, q_pos, S)

    # ── Prefill ──────────────────────────────────────────────────────────

    def forward_prefill(
        self, hidden_states, positions, position_embeddings, attn_metadata=None
    ):
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        if self._qsa_dense(attn_metadata):
            self._index_keys(hidden_states.to(self.dtype), attn_metadata)
            return super().forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        return self._prefill_qsa(hidden_states, position_embeddings, attn_metadata)

    def _prefill_qsa(self, hidden_states, position_embeddings, attn_metadata):
        """Qwen3.5 ``forward_prefill`` with the attention over the cached context in torch and
        the QSA token mask ANDed into ``allowed``. One request per prefill step."""
        hidden_states = hidden_states.to(self.dtype)
        T = hidden_states.shape[0]
        qkv = Q.NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            **self._qkv_quant_kwargs(),
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(T, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(T, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(T, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = Q.apply_rotary_pos_emb(q, k, cos, sin)

        md = attn_metadata[self.md_key]
        slot_mapping, block_size = md["slot_mapping"].clamp(min=0), md["block_size"]
        bt = md["block_table_tensor"][0].long().clamp(min=0)
        prior = md["cached_seq_len"].reshape(-1)[:1].to(torch.int32)
        if not self.runner_clears_kv_blocks:
            starts = torch.arange(bt.shape[0], device=bt.device) * block_size
            self._zero_fresh_blocks(
                torch.where(starts >= prior.to(starts.dtype), bt, torch.zeros_like(bt))
            )
        self._write_kv(k, v, slot_mapping, block_size)

        S = bt.shape[0] * block_size
        q_pos = prior + torch.arange(T, device=q.device, dtype=torch.int32)
        q_idx, kc_idx = self._qsa_index(
            hidden_states, q_pos.view(1, T), attn_metadata, S
        )
        attn_output = self._attend_cached_qsa(
            q, md, block_size, q_idx[0], kc_idx, q_pos
        )

        attn_output = (attn_output.float() * self._gate(hidden_states)).to(self.dtype)
        out = self._o_proj(attn_output.unsqueeze(0)).squeeze(0)
        if self.world_size > 1:  # >>> PARALLELISM: reduce-scatter to SP layout <<<
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    # Sparse prefill works in query blocks to bound transient memory (a 4096-token chunk at
    # 64k context): QSA selection in blocks whose fp32 indexer scores [rows*4, S/4] fit
    # QSA_SELECT_BYTES (few large blocks: the exact top-k unrolls ~45 passes per block, which
    # dominates graph size), and attention in sub-blocks whose fp32 [Nh, rows, S] scores fit
    # QSA_PREFILL_SCORE_BYTES.
    QSA_SELECT_BYTES = 128 << 20
    QSA_PREFILL_SCORE_BYTES = 96 << 20
    QSA_PREFILL_MIN_ROWS = 128

    @staticmethod
    def _rows_within(T: int, row_bytes: int, budget: int, min_rows: int) -> int:
        """Largest divisor of T with rows * row_bytes <= budget (at least min_rows if T allows)."""
        rows = min(T, max(budget // max(row_bytes, 1), min(T, min_rows), 1))
        while T % rows:
            rows -= 1
        return rows

    def _attend_cached_qsa(self, q, md, block_size, q_idx, kc_idx, q_pos):
        """Sparse prefill attention over the cached context + this chunk, in query blocks.

        q [Nh, T, Dh]; q_idx [T, nh, D] / kc_idx [1, S/4, D] indexer inputs; q_pos [T] int32
        -> [Nh, Dh, T]. Per selection block: QSA top-k blocks; per attention sub-block: the
        token mask (selected blocks + tail, causal) and masked softmax attention over the
        context. Keys are valid iff pos < cached + n_valid; invalid V rows are zeroed
        (unwritten cache can hold NaN/Inf from recycled pages). K/V are gathered once (fp32
        [nkh, S, Dh]); GQA without repeat (grouped matmul)."""
        cfg = self.config
        Nh, T, Dh = q.shape
        nkh, G = self.num_key_value_heads_per_rank, self.num_key_value_groups
        bt = md["block_table_tensor"][0].long().clamp(min=0)
        prior = md["cached_seq_len"].reshape(-1)[:1].long()
        n_valid = (md["slot_mapping"] > 0).sum().reshape(1)
        S = bt.shape[0] * block_size
        NB = kc_idx.shape[1]
        kc = (
            self.k_cache.index_select(0, bt).transpose(0, 1).reshape(nkh, S, Dh).float()
        )
        vc = (
            self.v_cache.index_select(0, bt).transpose(0, 1).reshape(nkh, S, Dh).float()
        )
        if self.kv_fp8:  # stored x * (1/s)
            kc, vc = kc / self.k_scale_float, vc / self.v_scale_float
        key_ok = torch.arange(S, device=q.device) < prior + n_valid  # [S]
        zero = torch.zeros((), device=q.device)
        vc = torch.where(key_ok[None, :, None], vc, zero)
        kct = kc.transpose(1, 2)  # [nkh, Dh, S]
        neg = torch.full((), -1e30, device=q.device)
        sel_rows = self._rows_within(
            T,
            4 * cfg.indexer_n_heads * NB,
            self.QSA_SELECT_BYTES,
            self.QSA_PREFILL_MIN_ROWS,
        )
        att_rows = self._rows_within(
            sel_rows,
            4 * Nh * S,
            self.QSA_PREFILL_SCORE_BYTES,
            self.QSA_PREFILL_MIN_ROWS,
        )
        outs = []
        for s0 in range(0, T, sel_rows):
            qp = q_pos[s0 : s0 + sel_rows].view(1, sel_rows)
            sel = qsa.select_blocks(
                q_idx[s0 : s0 + sel_rows].unsqueeze(0),
                kc_idx,
                qp,
                cfg.block_topk,
                cfg.indexer_compress_ratio,
            )  # [1, sel_rows, NB]
            for a0 in range(0, sel_rows, att_rows):
                r0 = s0 + a0
                tok = qsa.token_mask(
                    sel[:, a0 : a0 + att_rows],
                    qp[:, a0 : a0 + att_rows],
                    S,
                    cfg.indexer_compress_ratio,
                )[0]
                allowed = tok & key_ok[None, :]  # [rows, S] (tok is causal)
                qb = q[:, r0 : r0 + att_rows].float().reshape(nkh, G * att_rows, Dh)
                scores = torch.matmul(qb, kct) * self.scaling  # [nkh, G*rows, S]
                scores = torch.where(allowed.repeat(G, 1)[None], scores, neg)
                o = torch.matmul(torch.softmax(scores, dim=-1), vc)  # [nkh, G*rows, Dh]
                outs.append(o.reshape(Nh, att_rows, Dh))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out.transpose(1, 2).to(q.dtype)  # [Nh, Dh, T]

    # ── Decode ───────────────────────────────────────────────────────────

    def forward_decode(
        self, hidden_states, positions, position_embeddings, attn_metadata
    ):
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        md = attn_metadata[self.md_key]
        B = md["block_table_tensor"].shape[0]
        S_q = hidden_states.shape[0] // B
        tok = None
        if not self._qsa_dense(attn_metadata):
            S = md["max_blocks_per_seq"] * md["block_size"]
            q_pos = positions.view(B, S_q).to(torch.int32)
            tok = self._qsa_token_mask(
                hidden_states.to(self.dtype), q_pos, attn_metadata, S
            )
        else:
            self._index_keys(hidden_states.to(self.dtype), attn_metadata)
        if not can_run_kernel(hidden_states):
            return self._decode_torch(
                hidden_states, positions, position_embeddings, attn_metadata, tok
            )
        # The Qwen3.5 fused-kernel decode, with the QSA selection ANDed into its mask
        # (_restrict_decode_mask); tok is None while every block is selected.
        self._decode_token_mask = tok
        try:
            return super().forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        finally:
            self._decode_token_mask = None

    def _restrict_decode_mask(self, attention_mask, positions, block_size):
        """ANDs the step's QSA token mask (set by forward_decode) into the decode-kernel mask."""
        if self._decode_token_mask is None:
            return attention_mask
        return qsa.decode_mask_from_token_mask(
            self._decode_token_mask, positions, attention_mask, block_size
        )

    def _decode_torch(
        self, hidden_states, positions, position_embeddings, attn_metadata, tok=None
    ):
        """Decode attention in torch (CPU / kernels off): B requests x S tokens each, over the
        paged cache after writing this step's K/V. ``tok`` [B, S_q, S]: QSA token mask."""
        md = attn_metadata[self.md_key]
        slot_mapping, block_size = md["slot_mapping"].clamp(min=0), md["block_size"]
        bt = md["block_table_tensor"].long().clamp(min=0)  # [B, nblk]
        B = bt.shape[0]
        T = hidden_states.shape[0]
        S_q = T // B
        nkh, Dh = self.num_key_value_heads_per_rank, self.head_dim
        x = hidden_states.to(self.dtype)
        q, k, v = torch.tensor_split(
            x @ self.qkv_proj_weight, self.qkv_split_indices, -1
        )
        q = q.view(T, -1, Dh).transpose(0, 1)
        k = k.view(T, nkh, Dh).transpose(0, 1)
        v = v.view(T, nkh, Dh).transpose(0, 1)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = Q.apply_rotary_pos_emb(q, k, cos, sin)
        self._write_kv(k, v, slot_mapping, block_size)

        S = bt.shape[1] * block_size
        kc = self.k_cache.index_select(0, bt.reshape(-1))  # [B*nblk, nkh, bs, Dh]
        vc = self.v_cache.index_select(0, bt.reshape(-1))
        kc = (
            kc.view(B, -1, nkh, block_size, Dh)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S, Dh)
        )
        vc = (
            vc.view(B, -1, nkh, block_size, Dh)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S, Dh)
        )
        kc = kc.float().repeat_interleave(self.num_key_value_groups, dim=1)
        vc = vc.float().repeat_interleave(self.num_key_value_groups, dim=1)
        qb = q.float().view(-1, B, S_q, Dh).transpose(0, 1)  # [B, Nh, S_q, Dh]
        scores = (
            torch.matmul(qb, kc.transpose(-1, -2)) * self.scaling
        )  # [B, Nh, S_q, S]
        q_pos = positions.view(B, 1, S_q, 1).long()
        allowed = torch.arange(S, device=q.device).view(1, 1, 1, S) <= q_pos
        if tok is not None:
            allowed = allowed & tok.view(B, 1, S_q, S)
        scores = torch.where(allowed, scores, torch.full((), -1e30, device=q.device))
        attn = torch.matmul(torch.softmax(scores, -1), vc)  # [B, Nh, S_q, Dh]
        attn = attn.transpose(0, 1).reshape(-1, T, Dh)  # [Nh, T, Dh]
        gate = (
            torch.sigmoid((x @ self.gate_weight).float())
            .view(T, -1, Dh)
            .transpose(0, 1)
        )
        attn = (attn * gate).transpose(0, 1).reshape(T, -1).to(self.dtype)
        out = attn @ self.o_proj_weight
        if self.world_size > 1:  # >>> PARALLELISM: TP all-reduce <<<
            self.tp_group.all_reduce(out)
        return out


# =============================================================================
# Section 4: Decoder layer, backbone
# =============================================================================


class Qwen4ExpDecoderLayer(nn.Module):
    """One decoder layer: [PLE] -> gated-residual read -> mixer (GDN or QSA attention) -> write,
    then read -> sparse MoE -> write. ``layer_type`` overrides the config's (MTP layers)."""

    def __init__(
        self, config: Qwen4ExpConfig, layer_idx: int, layer_type: str | None = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type or config.layer_types[layer_idx]
        if self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen4ExpAttention(config, layer_idx)
        else:
            self.linear_attn = Qwen4ExpGatedDeltaNet(config, layer_idx)
        self.ple = (
            Qwen4ExpPLE(config, layer_idx) if layer_idx in config.ple_layers else None
        )
        self.attn_hc = Qwen4ExpGatedResidual(config)
        self.mlp_hc = Qwen4ExpGatedResidual(config)
        self.mlp = Qwen4ExpSparseMoe(config, layer_idx)
        self.mixer_key = Q._mixer_key(config, layer_idx, self.layer_type)

    @property
    def mixer(self):
        return self.self_attn if self.layer_type == FULL_ATTENTION else self.linear_attn

    def forward(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata=None,
        ple_embeds=None,
        rank=None,
    ):
        md = attn_metadata[self.mixer_key]
        is_decode = md["max_query_len"] <= md["decode_token_threshold"]
        if self.ple is not None:
            hidden_states = self.ple(
                hidden_states, ple_embeds, attn_metadata, is_decode, rank=rank
            )
        x, inject = self.attn_hc.read(hidden_states)
        x = self.mixer(x, positions, position_embeddings, attn_metadata)
        hidden_states = Qwen4ExpGatedResidual.write(hidden_states, x, inject)
        x, inject = self.mlp_hc.read(hidden_states)
        x = self.mlp(x, is_prefill=not is_decode)
        return Qwen4ExpGatedResidual.write(hidden_states, x, inject)


class Qwen4ExpModel(nn.Module):
    """Embedding, decoder layers over the ``hc_count``-stream residual, final mixer read."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [Qwen4ExpDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self.rotary_emb = Q.Qwen3_5RotaryEmbedding(config)
        self.first_mixer_key = Q._mixer_key(config, 0)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata=None,
        rank=None,
        ple_embeds=None,
        return_multi: bool = False,
    ):
        """-> (mixer output [T, H], the pre-mixer residual [T, hc*H] if ``return_multi``)."""
        md0 = attn_metadata[self.first_mixer_key]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        position_embeddings = self.rotary_emb(positions, dtype=hidden_states.dtype)
        hidden_states = hidden_states.repeat(1, self.config.hc_count)  # [T, hc * H]
        # Padded decode rows (block table row -1) read the null block and can turn non-finite;
        # zero them after every layer so no kernel mixes them into real rows.
        bt0 = md0["block_table_tensor"]
        real = None
        if not is_prefill and bt0.shape[0] > 1:
            real = (
                (bt0[:, 0] >= 0)
                .repeat_interleave(hidden_states.shape[0] // bt0.shape[0])
                .view(-1, 1)
            )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                ple_embeds=ple_embeds,
                rank=rank,
            )
            if real is not None:
                hidden_states = torch.where(
                    real,
                    hidden_states,
                    torch.zeros(
                        (), dtype=hidden_states.dtype, device=hidden_states.device
                    ),
                )
        out = self.hyper_connection_mixer.read(hidden_states)  # no final norm
        if is_prefill and self.world_size > 1:
            out = self.tp_group.all_gather(out, dim=0)
        if not return_multi:
            return out, None
        # The MTP head drafts from the 4-stream residual before the final mixer.
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return out, hidden_states


# =============================================================================
# Section 5: Weight loaders
# =============================================================================


def _attach_hc_loaders(hc: Qwen4ExpGatedResidual) -> None:
    set_weight_loader(hc.norm_weight, _loader(lambda t, r: W.folded_norm(t[0])))
    set_weight_loader(
        hc.down_inject_weight,
        _loader(lambda t, r: torch.cat([W.linear_t(x) for x in t], dim=1)),
    )
    set_weight_loader(hc.up_weight, _loader(lambda t, r: W.linear_t(t[0])))


def _hc_weight_mappings(hc: Qwen4ExpGatedResidual, pre: str, ck: str) -> dict:
    down = [ck + "input_mix_weight_down.weight"]
    if hc.use_combine:
        down.append(ck + "block_inject_weight.weight")
    return {
        pre + "norm_weight": ck + "hc_norm.weight",
        pre + "down_inject_weight": down,
        pre + "up_weight": ck + "input_mix_weight_up.weight",
    }


def _attach_layer_loaders(layer: Qwen4ExpDecoderLayer, cfg, tp: int) -> None:
    for hc in (layer.attn_hc, layer.mlp_hc):
        _attach_hc_loaders(hc)
    attach_moe_loaders(layer.mlp, cfg, tp)
    fold = _loader(lambda t, r: W.folded_norm(t[0]))
    if layer.ple is not None:
        p = layer.ple
        set_weight_loader(
            p.key_value_weight,
            _loader(lambda t, r: torch.cat([W.linear_t(x) for x in t], dim=1)),
        )
        for n in ("norm_key_weight", "norm_query_weight", "norm_conv_weight"):
            set_weight_loader(getattr(p, n), fold)
        set_weight_loader(p.conv_weight, _loader(lambda t, r: W.depthwise_conv(t[0])))
    if layer.layer_type == FULL_ATTENTION:
        a = layer.self_attn
        set_weight_loader(
            a.qkv_proj_weight,
            _loader(lambda t, r: W.attention_qkv_device(t[0], t[1], t[2], cfg, r, tp)),
        )
        set_weight_loader(
            a.gate_weight,
            _loader(lambda t, r: W35.attention_gate_device(t[0], cfg, r, tp)),
        )
        set_weight_loader(
            a.o_proj_weight,
            _loader(lambda t, r: W35.attention_o_device(t[0], cfg, r, tp)),
        )
        qk = _loader(lambda t, r: W35.attention_qk_norm_device(t[0], cfg))
        set_weight_loader(a.q_norm.weight, qk)
        set_weight_loader(a.k_norm.weight, qk)
        set_weight_loader(a.index_qk_weight, _loader(lambda t, r: W.linear_t(t[0])))
        set_weight_loader(a.index_q_norm_weight, fold)
        set_weight_loader(a.index_k_norm_weight, fold)
    else:
        g = layer.linear_attn
        set_weight_loader(
            g.in_proj_qkv_weight,
            _loader(lambda t, r: W35.gdn_in_proj_qkv_device(t[0], cfg, r, tp)),
        )
        heads = _loader(lambda t, r: W35.gdn_in_proj_heads_device(t[0], r, tp))
        for wp in (g.in_proj_z_weight, g.in_proj_b_weight, g.in_proj_a_weight):
            set_weight_loader(wp, heads)
        set_weight_loader(
            g.out_proj_weight,
            _loader(lambda t, r: W35.gdn_out_proj_device(t[0], cfg, r, tp)),
        )
        set_weight_loader(
            g.conv_weight, _loader(lambda t, r: W35.gdn_conv_device(t[0], cfg, r, tp))
        )
        vec = _loader(lambda t, r: W35.gdn_head_vector_device(t[0], cfg, r, tp).float())
        set_weight_loader(g.A_log, vec)
        set_weight_loader(g.dt_bias, vec)


def _layer_weight_mappings(layer: Qwen4ExpDecoderLayer, pre: str, ck: str) -> dict:
    m = {}
    m.update(
        _hc_weight_mappings(
            layer.attn_hc, pre + "attn_hc.", ck + "attn_hyper_connection."
        )
    )
    m.update(
        _hc_weight_mappings(layer.mlp_hc, pre + "mlp_hc.", ck + "mlp_hyper_connection.")
    )
    m.update(moe_weight_mappings(layer.mlp, pre + "mlp.", ck + "mlp."))
    if layer.ple is not None:
        s, c = pre + "ple.", ck + "ple."
        m[s + "key_value_weight"] = [c + "key_proj.weight", c + "value_proj.weight"]
        m[s + "norm_key_weight"] = c + "norm_key.weight"
        m[s + "norm_query_weight"] = c + "norm_query.weight"
        m[s + "norm_conv_weight"] = c + "norm_conv.weight"
        m[s + "conv_weight"] = c + "conv1d.weight"
    if layer.layer_type == FULL_ATTENTION:
        s, c = pre + "self_attn.", ck + "self_attn."
        m[s + "qkv_proj_weight"] = [
            c + "q_proj.weight",
            c + "k_proj.weight",
            c + "v_proj.weight",
        ]
        m[s + "gate_weight"] = c + "q_proj.weight"
        m[s + "o_proj_weight"] = c + "o_proj.weight"
        m[s + "q_norm.weight"] = c + "q_norm.weight"
        m[s + "k_norm.weight"] = c + "k_norm.weight"
        m[s + "index_qk_weight"] = c + "indexer.index_qk_proj.weight"
        m[s + "index_q_norm_weight"] = c + "indexer.q_layernorm.weight"
        m[s + "index_k_norm_weight"] = c + "indexer.k_layernorm.weight"
    else:
        s, c = pre + "linear_attn.", ck + "linear_attn."
        for part in ("qkv", "z", "b", "a"):
            m[s + f"in_proj_{part}_weight"] = c + f"in_proj_{part}.weight"
        m[s + "conv_weight"] = c + "conv1d.weight"
        m[s + "A_log"] = c + "A_log"
        m[s + "dt_bias"] = c + "dt_bias"
        m[s + "norm_weight"] = c + "norm.weight"
        m[s + "out_proj_weight"] = c + "out_proj.weight"
    return m


def bind_attention_kv(attn: Qwen4ExpAttention, kv_caches: dict) -> None:
    """Bind ``kv_caches[attn.md_key]`` (K, V) to an attention layer (backbone and MTP)."""
    if attn.md_key not in kv_caches:
        raise KeyError(f"KV cache for layer {attn.md_key} not initialized")
    k_cache, v_cache = kv_caches[attn.md_key][:2]
    attn.bind_kv(k_cache, v_cache)


def load_rank_weights(
    module: nn.Module, checkpoint_path: str, device: torch.device, cache_dir: str | None
) -> SafetensorsCheckpoint:
    """Load this rank's shard of ``module._weight_mappings()`` into ``module`` (backbone and
    MTP), cast to each parameter's dtype (buffers: the model dtype)."""
    tp_group = module.tp_group
    checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
    rank_sharded = checkpoint.load_sharded_pipelined(
        tp_group.rank_in_group,
        tp_group.world_size,
        module,
        module._weight_mappings(),
        device,
    ).state_dict
    params = dict(module.named_parameters())
    for name, tensor in rank_sharded.items():
        target = params[name].dtype if name in params else module.config.torch_dtype
        if tensor.dtype != target:
            rank_sharded[name] = tensor.to(target)
    module.load_state_dict(rank_sharded, strict=False, assign=True)
    return checkpoint


# =============================================================================
# Section 6: Causal LM
# =============================================================================


@async_speculative_decoding
class Qwen4ExpForCausalLM(nn.Module):
    """Qwen3.8-Flash-Next text model + column-parallel LM head (untied)."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config
        self.model = Qwen4ExpModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )
        self._host_ple = None  # ngram.HostPLEInputs, set by load_weights
        self._mtp_hidden_output = False
        for layer in self.model.layers:
            _attach_layer_loaders(layer, config, self.world_size)
        _attach_hc_loaders(self.model.hyper_connection_mixer)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        ple_embeds=None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        md0 = attn_metadata[self.model.first_mixer_key]
        is_prefill = md0["max_query_len"] > md0["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prefill length ({T}) must be greater than and divisible by world_size "
                f"({self.world_size}) for sequence parallelism."
            )
        if ple_embeds is None and self.config.ple_layers:
            raise ValueError("Qwen3.8-Flash-Next needs ple_embeds (host n-gram rows)")
        hidden_states, mtp_hidden = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            ple_embeds=ple_embeds,
            return_multi=self._mtp_hidden_output,
        )
        if is_prefill:
            logits = self.lm_head(
                torch.index_select(hidden_states, dim=0, index=sampling_positions)
            )
        else:
            # Decode samples every row in order (see qwen3_5/model.py: index_select next to the
            # returned MTP hidden gave NaN rows on device).
            logits = self.lm_head(hidden_states)
        if self.on_device_sampling_config is None:
            return (logits, mtp_hidden) if mtp_hidden is not None else logits
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            sampled_tokens = rejection_sampler(spec_decode_metadata, sampled_tokens)
            if mtp_hidden is None:
                return sampled_tokens
        if mtp_hidden is not None:
            return sampled_tokens, mtp_hidden, gathered_logits
        return sampled_tokens, gathered_logits

    def set_mtp_hidden_output(self, value: bool) -> None:
        """Return the pre-mixer residual with the samples (speculative method ``mtp``)."""
        self._mtp_hidden_output = value

    @property
    def mtp_hidden_size(self) -> int:
        """Width of the hidden state the MTP head reads (the 4-stream residual)."""
        return self.config.hc_hidden_size

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(Qwen4ExpConfig.from_configs(hf_config, neuron_config))

    # ── Host inputs (n-gram embedding rows) ──────────────────────────────

    def host_input_specs(self) -> dict:
        """Per-token host inputs the runner allocates: name -> (trailing shape, dtype)."""
        if not self.config.ple_layers:
            return {}
        return {"ple_embeds": ((self.config.ple_embed_dim,), self.config.torch_dtype)}

    def build_host_inputs(self, input_ids, req_rows, positions, token_ids_cpu) -> dict:
        """The step's n-gram rows ``ple_embeds`` [T, ple_embed_dim] (``ngram.HostPLEInputs``;
        arguments as in ``ngram.context_tokens``)."""
        if not self.config.ple_layers:
            return {}
        if self._host_ple is None:
            raise RuntimeError("n-gram table not loaded (load_weights)")
        return {
            "ple_embeds": self._host_ple(input_ids, req_rows, positions, token_ids_cpu)
        }

    # ── Multimodal-rope protocol (text only) ─────────────────────────────

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return torch.arange(len(input_tokens), dtype=torch.long).expand(
            3, -1
        ).clone(), 0

    # ── KV cache / recurrent state ───────────────────────────────────────

    @property
    def kv_side_bytes_per_token_per_attn_layer(self) -> int:
        """Runner budget hook: bytes per cached token of every attention layer's side cache
        (QSA raw indexer keys, ``Qwen4ExpAttention.bind_index_cache``), allocated outside the
        KV pool in proportion to it."""
        return (
            self.config.indexer_head_dim
            * torch.empty((), dtype=self.config.torch_dtype).element_size()
        )

    @property
    def prefill_dense_context(self) -> int:
        """Runner opt-in (``NeuronModelRunner._prefill_dense_context``): a prefill step whose
        whole context fits ``block_topk * compress_ratio`` (2048) tokens gets its attention
        block tables trimmed to that span, where QSA selects every block and the dense Qwen3.5
        prefill kernels run (``Qwen4ExpAttention._qsa_dense``)."""
        return self.config.block_topk * self.config.indexer_compress_ratio

    def get_kv_spec(self):
        """K/V per attention layer; recurrent state per GDN layer and per PLE layer."""
        cfg, tp = self.config, self.world_size
        layers, state_layers = [], []
        conv_shape, ssm_shape = gdn_state_shapes(cfg, tp)
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type == FULL_ATTENTION:
                layers.append(layer.self_attn.kv_layer_spec())
            else:
                state_layers.append(
                    StateLayerSpec(
                        name=f"layers.{i}.linear_attn",
                        shapes=(conv_shape, ssm_shape),
                        dtypes=(cfg.mamba_ssm_dtype, cfg.mamba_ssm_dtype),
                    )
                )
            if layer.ple is not None:
                state_layers.append(
                    StateLayerSpec(
                        name=layer.ple.state_name,
                        shapes=((cfg.ple_conv_state_len, cfg.hc_hidden_size),),
                        dtypes=(torch.float32,),
                    )
                )
        return KVSpec(layers=layers, state_layers=state_layers)

    def bind_kv_cache(self, kv_caches):
        for layer in self.model.layers:
            if layer.layer_type == FULL_ATTENTION:
                bind_attention_kv(layer.self_attn, kv_caches)

    def set_runner_clears_kv_blocks(self, value: bool):
        for layer in self.model.layers:
            if layer.layer_type == FULL_ATTENTION:
                layer.self_attn.runner_clears_kv_blocks = value

    def bind_state_cache(self, state_caches):
        for i, layer in enumerate(self.model.layers):
            names = []
            if layer.layer_type == LINEAR_ATTENTION:
                names.append((f"layers.{i}.linear_attn", layer.linear_attn))
            if layer.ple is not None:
                names.append((layer.ple.state_name, layer.ple))
            for name, mod in names:
                if name not in state_caches:
                    raise KeyError(f"State cache for layer {name} not initialized")
                mod.bind_state(*state_caches[name])

    # ── Weight loading ───────────────────────────────────────────────────

    def _weight_mappings(self) -> dict:
        m = {
            "model.embed_tokens.weight": f"{CKPT}embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
        }
        m.update(
            _hc_weight_mappings(
                self.model.hyper_connection_mixer,
                "model.hyper_connection_mixer.",
                f"{CKPT}hyper_connection_mixer.",
            )
        )
        for i, layer in enumerate(self.model.layers):
            m.update(
                _layer_weight_mappings(
                    layer, f"model.layers.{i}.", f"{CKPT}layers.{i}."
                )
            )
        return m

    def _materialize_buffers(self) -> None:
        self.model.rotary_emb.inv_freq = W35.rope_inv_freq(
            self.config.rotary_dim, self.config.rope_theta
        )
        for layer in self.model.layers:
            if layer.layer_type == FULL_ATTENTION:
                layer.self_attn.materialize_buffers()

    def _load_host_ple(self, checkpoint_path: str) -> None:
        if not self.config.ple_layers:
            return
        from .ngram import HostPLEInputs

        self._host_ple = HostPLEInputs.from_checkpoint(checkpoint_path, self.config)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ):
        self._materialize_buffers()
        checkpoint = load_rank_weights(self, checkpoint_path, device, cache_dir)
        self._load_host_ple(getattr(checkpoint, "path", checkpoint_path))

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ):
        """CPU-compile path: no weights; non-persistent buffers need real data."""
        self._materialize_buffers()
