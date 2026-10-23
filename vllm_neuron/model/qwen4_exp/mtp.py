# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next multi-token-prediction (MTP) head as a speculative-decoding draft model
==========================================================================================

The checkpoint's ``mtp.*`` weights form one QSA + MoE decoder layer with its own gated-residual
read/write and final mixer (vLLM ``Qwen4ExpMTP``, "residual_linear_shared" fusion):

    e = fc_embedding(norm_e(embed(t[i+1])))                        # [T, H]
    h = fc_hidden(norm_h(target_multi[i]).view(T, hc, H))          # [T, hc, H], fc shared per stream
    x = decoder_layer(h + e)                                       # e written into every stream
    logits = lm_head(mixer(x)); x is the next draft step's multi-stream hidden

``target_multi`` is the backbone's 4-stream residual before its final mixer, [T, hc*H]
(``Qwen4ExpForCausalLM.set_mtp_hidden_output``); ``norm_h`` is one zero-centered RMSNorm over all
hc*H channels. The MTP layer has no PLE. Embedding and LM head are the backbone's. The draft
plugs into the plugin's EagleProposer exactly like ``qwen3_5/mtp.py`` (same interface: on-device
accepted-token extraction, unrolled greedy draft loop).
"""

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import model as Q
from vllm_neuron.model.qwen3_5 import weights as W35
from vllm_neuron.model.qwen3_5.mtp import _slot_mapping, extract_accepted_tokens
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from . import weights as W
from .config import FULL_ATTENTION, Qwen4ExpConfig
from .model import (
    CKPT,
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    _attach_hc_loaders,
    _attach_layer_loaders,
    _hc_weight_mappings,
    _layer_weight_mappings,
    _loader,
    bind_attention_kv,
    group_rmsnorm,
    load_rank_weights,
)

MTP = "mtp."


class Qwen4ExpMTPModel(nn.Module):
    """The MTP block: input fusion, one QSA + MoE decoder layer, final mixer read."""

    def __init__(self, config: Qwen4ExpConfig, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        H, hcH, dt = config.hidden_size, config.hc_hidden_size, config.torch_dtype
        self.H, self.hc = H, config.hc_count
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=H,
            dtype=dt,
            tp_group=self.tp_group.device_group,
        )
        # Replicated [in, out]: small decode matmuls, no collective.
        self.fc_embedding_weight = nn.Parameter(torch.empty(H, H, dtype=dt))
        self.fc_hidden_weight = nn.Parameter(torch.empty(H, H, dtype=dt))
        self.pre_fc_norm_embedding = Q.Qwen3_5RMSNorm(
            H, config.rms_norm_eps, torch.float32
        )
        self.pre_fc_norm_hidden_weight = nn.Parameter(
            torch.ones(hcH, dtype=torch.float32)
        )
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    config, start_layer_idx + i, layer_type=FULL_ATTENTION
                )
                for i in range(config.mtp_num_hidden_layers)
            ]
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self.rotary_emb = Q.Qwen3_5RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        target_hidden_states,
        attn_metadata,
        rank=None,
        step=0,
    ):
        """Returns (single [T, H] for the LM head, multi [T, hc*H] for the next draft step),
        gathered across ranks in prefill."""
        layer = self.layers[step % len(self.layers)]
        md = attn_metadata[layer.mixer_key]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        emb = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        hidden = target_hidden_states
        n = emb.shape[0]
        if (
            is_prefill and self.world_size > 1
        ):  # >>> PARALLELISM: SP rows of the gathered state <<<
            r = rank.reshape(()) if rank is not None else self.tp_group.rank_in_group
            idx = r * n + torch.arange(n, dtype=torch.int32, device=emb.device)
            hidden = hidden.index_select(0, idx)
        e = self.pre_fc_norm_embedding(emb).to(self.fc_embedding_weight.dtype)
        e = e @ self.fc_embedding_weight
        h = group_rmsnorm(
            hidden.to(e.dtype),
            self.pre_fc_norm_hidden_weight,
            hidden.shape[-1],
            self.config.rms_norm_eps,
        )
        h = h.view(n, self.hc, self.H) @ self.fc_hidden_weight
        x = (h + e.unsqueeze(1)).reshape(n, -1)
        pe = self.rotary_emb(positions, dtype=x.dtype)
        x = layer(
            x, positions=positions, position_embeddings=pe, attn_metadata=attn_metadata
        )
        single = self.hyper_connection_mixer.read(x)
        if is_prefill and self.world_size > 1:
            single = self.tp_group.all_gather(single, dim=0)
            x = self.tp_group.all_gather(x, dim=0)
        return single, x


class Qwen4ExpMTP(nn.Module):
    """Draft model for vLLM's ``mtp`` speculative method (architecture ``Qwen4ExpMTP``)."""

    def __init__(self, config: Qwen4ExpConfig, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.model = Qwen4ExpMTPModel(config, start_layer_idx)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=False,
            tp_group=self.tp_group.device_group,
            weight_layout="in_out",
        )
        self.num_speculative_tokens = 1  # set by the proposer
        self._setup_weight_loaders()

    @property
    def mtp_hidden_size(self) -> int:
        """The proposer sizes the target hidden state with this (4-stream residual)."""
        return self.config.hc_hidden_size

    # -- draft tokens -----------------------------------------------------------------
    def _greedy(self, hidden_states, rank):
        """Argmax over the full vocabulary from the per-rank logit shards."""
        logits = self.lm_head(hidden_states).float()  # [B, V/tp]
        val, idx = logits.max(dim=-1, keepdim=True)
        if self.world_size == 1:
            return idx.squeeze(-1).to(torch.int32)
        shard = logits.shape[-1]
        idx = idx + rank.to(idx.dtype) * shard
        vals = self.tp_group.all_gather(val, dim=1)  # [B, tp]
        idxs = self.tp_group.all_gather(idx.to(torch.float32), dim=1)
        best = vals.argmax(dim=1, keepdim=True)
        return idxs.gather(1, best).squeeze(1).to(torch.int32)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        initial_target_hidden_states,
        attn_metadata=None,
        sampling_positions=None,
        rank=None,
        raw_sampled_token_ids=None,
        prev_sampled_token_ids=None,
        prev_num_draft_tokens=None,
        req_indices_per_token=None,
    ):
        """EagleProposer interface (see qwen3_5/mtp.py). Returns (stacked_tokens [bs, 1+k] with
        the bonus token first when raw_sampled_token_ids is given, drafts_only [bs, k], None)."""
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )
        bonus = None
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )
        if rank is None:
            rank = torch.zeros((), dtype=torch.int32, device=input_ids.device)

        single, multi = self.model(
            input_ids, positions, initial_target_hidden_states, attn_metadata, rank
        )
        single, multi = single[sampling_positions], multi[sampling_positions]
        cur_positions = positions[sampling_positions]
        draft = self._greedy(single, rank)
        tokens = [bonus, draft] if bonus is not None else [draft]

        # Later draft steps: one token per request at the next position. The QSA indexer side
        # cache shares the attention layer's block table, so one metadata entry serves both.
        key = self.model.layers[0].mixer_key
        base = attn_metadata[key]
        for step in range(1, self.num_speculative_tokens):
            cur_positions = cur_positions + 1
            md = {
                key: {
                    "block_table_tensor": base["block_table_tensor"],
                    "slot_mapping": _slot_mapping(
                        cur_positions, base["block_table_tensor"], base["block_size"]
                    ),
                    "max_query_len": 1,
                    "block_size": base["block_size"],
                    "max_blocks_per_seq": base["max_blocks_per_seq"],
                    "decode_token_threshold": 1,
                }
            }
            if "cached_seq_len" in base:
                md[key]["cached_seq_len"] = base["cached_seq_len"]
            single, multi = self.model(draft, cur_positions, multi, md, rank, step=step)
            draft = self._greedy(single, rank)
            tokens.append(draft)

        stacked = torch.stack(tokens, dim=1)
        drafts_only = torch.stack(tokens[1:] if bonus is not None else tokens, dim=1)
        return stacked, drafts_only, None

    # -- KV cache -----------------------------------------------------------------------
    def get_kv_spec(self):
        """Attention K/V, declared as the backbone does (``Qwen4ExpAttention.kv_layer_spec``)."""
        return KVSpec(
            layers=[layer.self_attn.kv_layer_spec() for layer in self.model.layers]
        )

    def bind_kv_cache(self, kv_caches):
        for layer in self.model.layers:
            bind_attention_kv(layer.self_attn, kv_caches)

    def set_runner_clears_kv_blocks(self, value: bool):
        for layer in self.model.layers:
            layer.self_attn.runner_clears_kv_blocks = value

    # -- weights ------------------------------------------------------------------------
    def _setup_weight_loaders(self):
        cfg, tp, m = self.config, self.world_size, self.model
        for layer in m.layers:
            _attach_layer_loaders(layer, cfg, tp)
        _attach_hc_loaders(m.hyper_connection_mixer)
        fold = _loader(lambda t, r: W.folded_norm(t[0]))
        set_weight_loader(m.pre_fc_norm_embedding.weight, fold)
        set_weight_loader(m.pre_fc_norm_hidden_weight, fold)
        for p in (m.fc_embedding_weight, m.fc_hidden_weight):
            set_weight_loader(p, _loader(lambda t, r: W.linear_t(t[0])))
        set_weight_loader(
            m.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=m.embed_tokens.vocab_size_per_rank,
                num_shards=tp,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=cfg.vocab_size // tp,
                num_shards=tp,
                is_storage_transposed=True,
            ),
        )

    def _weight_mappings(self) -> dict:
        m = {
            "model.embed_tokens.weight": f"{CKPT}embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "model.fc_embedding_weight": MTP + "fc_embedding.weight",
            "model.fc_hidden_weight": MTP + "fc_hidden.weight",
            "model.pre_fc_norm_embedding.weight": MTP + "pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden_weight": MTP + "pre_fc_norm_hidden.weight",
        }
        m.update(
            _hc_weight_mappings(
                self.model.hyper_connection_mixer,
                "model.hyper_connection_mixer.",
                MTP + "hyper_connection_mixer.",
            )
        )
        for i, layer in enumerate(self.model.layers):
            m.update(
                _layer_weight_mappings(layer, f"model.layers.{i}.", f"{MTP}layers.{i}.")
            )
        return m

    def _materialize_buffers(self):
        self.model.rotary_emb.inv_freq = W35.rope_inv_freq(
            self.config.rotary_dim, self.config.rope_theta
        )
        for layer in self.model.layers:
            layer.self_attn.materialize_buffers()

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ):
        self._materialize_buffers()
        load_rank_weights(self, checkpoint_path, device, cache_dir)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ):
        self._materialize_buffers()

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ):
        return cls(Qwen4ExpConfig.from_configs(config, neuron_config), start_layer_idx)
