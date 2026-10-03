# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.8 multi-token-prediction (MTP) head as a speculative-decoding draft model
=========================================================================================

The checkpoint's ``mtp.*`` weights form one full-attention decoder layer that predicts the
token after next (as vLLM's ``Qwen3_5MultiTokenPredictor``):

    h = fc(cat(norm_e(embed(t[i+1])), norm_h(target_hidden[i])))   # [T, 2H] -> [T, H]
    h = norm(decoder_layer(h))                                     # predicts t[i+2]

``target_hidden`` is the backbone's final-normed hidden state. Embedding and LM head are the
backbone's (loaded again from the checkpoint). Further draft tokens reuse the layer with the
previous draft's hidden state. The draft is driven by the plugin's EagleProposer, whose
interface (on-device accepted-token extraction, unrolled draft loop) this class implements;
draft tokens are greedy (distributed argmax over the vocabulary shards).
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.eagle3_model import extract_accepted_tokens
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from . import weights as W
from .config import FULL_ATTENTION, Qwen3_5Config
from .model import (
    CKPT,
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5RotaryEmbedding,
    _attach_layer_loaders,
    _layer_weight_mappings,
    _loader,
)

MTP = "mtp."


def _slot_mapping(positions, block_table, block_size):
    """KV slot per position (as eagle3_model.compute_slot_mapping). Padded batch rows can
    carry negative corrected positions: their block index is clamped into the table and the
    -1 (unused) result is mapped to slot 0, the null block."""
    blocks = (positions // block_size).clamp(0, block_table.shape[1] - 1)
    ids = block_table.gather(1, blocks.view(-1, 1).long()).view(-1)
    slot = ids * block_size + positions % block_size
    return torch.where(ids < 0, torch.zeros_like(slot), slot)


class Qwen3_5MTPModel(nn.Module):
    def __init__(self, config: Qwen3_5Config, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        H = config.hidden_size
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=H, dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        # Replicated [2H, H] (x @ fc_weight): one small decode matmul, no collective.
        self.fc_weight = nn.Parameter(torch.empty(2 * H, H, dtype=config.torch_dtype))
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, config.rms_norm_eps, torch.float32)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, config.rms_norm_eps, torch.float32)
        self.norm = Qwen3_5RMSNorm(H, config.rms_norm_eps, torch.float32)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, start_layer_idx + i, layer_type=FULL_ATTENTION)
             for i in range(config.mtp_num_hidden_layers)])
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

    def forward(self, input_ids, positions, target_hidden_states, attn_metadata, rank=None, step: int = 0):
        """Returns the normed hidden state [T, H] (gathered across ranks in prefill)."""
        layer = self.layers[step % len(self.layers)]
        md = attn_metadata[layer.mixer_key]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        emb = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        hidden = target_hidden_states
        if is_prefill and self.world_size > 1:  # >>> PARALLELISM: SP slice of the gathered state <<<
            n = emb.shape[0]
            hidden = hidden[self.rank * n:(self.rank + 1) * n]
        x = torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden.to(emb.dtype))], dim=-1)
        x = (x.to(self.fc_weight.dtype) @ self.fc_weight)
        pe = self.rotary_emb(positions, device=x.device, dtype=x.dtype)
        x = layer(x, positions=positions, position_embeddings=pe, attn_metadata=attn_metadata)
        x = self.norm(x)
        if is_prefill and self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        return x


class Qwen3_5MTP(nn.Module):
    """Draft model for vLLM's ``mtp`` speculative method (architecture ``Qwen3_5MTP``)."""

    def __init__(self, config: Qwen3_5Config, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.model = Qwen3_5MTPModel(config, start_layer_idx)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype,
            gather_output=False, tp_group=self.tp_group.device_group)
        self.num_speculative_tokens = 1  # set by the proposer
        self._setup_weight_loaders()

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
    def forward(self, input_ids, positions, initial_target_hidden_states, attn_metadata=None,
                sampling_positions=None, rank=None, raw_sampled_token_ids=None,
                prev_sampled_token_ids=None, prev_num_draft_tokens=None, req_indices_per_token=None):
        """EagleProposer interface: draft-layer KV update over the target tokens, then
        num_speculative_tokens greedy drafts per request. Returns (stacked_tokens [bs, 1+k]
        with the bonus token first when raw_sampled_token_ids is given, drafts_only [bs, k],
        None)."""
        if prev_sampled_token_ids is not None and prev_num_draft_tokens is not None \
                and req_indices_per_token is not None:
            positions, attn_metadata = NF.correct_spec_decode_positions_and_slot_mapping(
                positions, attn_metadata, prev_sampled_token_ids, prev_num_draft_tokens,
                req_indices_per_token, self.config.vocab_size)
        bonus = None
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = extract_accepted_tokens(
                input_ids, sampling_positions, raw_sampled_token_ids, self.config.vocab_size,
                self.num_speculative_tokens)
        if rank is None:
            rank = torch.zeros((), dtype=torch.int32, device=input_ids.device)

        hidden = self.model(input_ids, positions, initial_target_hidden_states, attn_metadata, rank)
        hidden = hidden[sampling_positions]
        cur_positions = positions[sampling_positions]
        draft = self._greedy(hidden, rank)
        tokens = [bonus, draft] if bonus is not None else [draft]

        layer_name = self.model.layers[0].mixer_key
        base = attn_metadata[layer_name]
        for step in range(1, self.num_speculative_tokens):
            cur_positions = cur_positions + 1
            slot_mapping = _slot_mapping(cur_positions, base["block_table_tensor"], base["block_size"])
            md = {ln: {"block_table_tensor": base["block_table_tensor"], "slot_mapping": slot_mapping,
                       "max_query_len": 1, "block_size": base["block_size"],
                       "max_blocks_per_seq": base["max_blocks_per_seq"], "decode_token_threshold": 1}
                  for ln in [layer_name]}
            hidden = self.model(draft, cur_positions, hidden, md, rank, step=step)
            draft = self._greedy(hidden, rank)
            tokens.append(draft)

        stacked = torch.stack(tokens, dim=1)
        drafts_only = torch.stack(tokens[1:] if bonus is not None else tokens, dim=1)
        return stacked, drafts_only, None

    # -- KV cache -----------------------------------------------------------------------
    def get_kv_spec(self):
        layers = []
        for layer in self.model.layers:
            a = layer.self_attn
            layers.append(LayerSpec(name=layer.mixer_key, num_kv_heads=a.num_key_value_heads_per_rank,
                                    head_size=a.head_dim, dtype=a.dtype, sliding_window_size=None,
                                    chunk_size=None))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for layer in self.model.layers:
            a = layer.self_attn
            a.k_cache, a.v_cache = kv_caches[layer.mixer_key][0], kv_caches[layer.mixer_key][1]
            if a.kv_fp8:
                for name in ("k_scale", "v_scale"):
                    setattr(a, name, torch.ones(1, 1, dtype=torch.bfloat16, device=a.k_cache.device))

    def set_runner_clears_kv_blocks(self, value: bool):
        for layer in self.model.layers:
            layer.self_attn.runner_clears_kv_blocks = value

    # -- weights ------------------------------------------------------------------------
    def _setup_weight_loaders(self):
        cfg, tp = self.config, self.world_size
        for layer in self.model.layers:
            _attach_layer_loaders(layer, cfg, tp)
        fold = _loader(lambda t, r: W.fold_zero_centered_norm(t[0]).float())
        for norm in (self.model.pre_fc_norm_embedding, self.model.pre_fc_norm_hidden, self.model.norm):
            set_weight_loader(norm.weight, fold)
        set_weight_loader(self.model.fc_weight, _loader(lambda t, r: t[0].t().contiguous()))
        set_weight_loader(self.model.embed_tokens.weight, sharding_weight_loader(
            shard_dim=0, shard_size=self.model.embed_tokens.vocab_size_per_rank, num_shards=tp,
            is_storage_transposed=False, pad_shard=True))
        set_weight_loader(self.lm_head.weight, sharding_weight_loader(
            shard_dim=0, shard_size=cfg.vocab_size // tp, num_shards=tp, is_storage_transposed=False))

    def _weight_mappings(self) -> dict:
        m = {
            "model.embed_tokens.weight": f"{CKPT}embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "model.fc_weight": MTP + "fc.weight",
            "model.pre_fc_norm_embedding.weight": MTP + "pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden.weight": MTP + "pre_fc_norm_hidden.weight",
            "model.norm.weight": MTP + "norm.weight",
        }
        for i, layer in enumerate(self.model.layers):
            m.update(_layer_weight_mappings(layer, f"model.layers.{i}.", f"{MTP}layers.{i}."))
        return m

    def _materialize_buffers(self):
        self.model.rotary_emb.inv_freq = W.rope_inv_freq(self.config.rotary_dim, self.config.rope_theta)

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        self._materialize_buffers()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.tp_group.rank_in_group, self.world_size, self, self._weight_mappings(), device).state_dict
        params = dict(self.named_parameters())
        for name, tensor in rank_sharded.items():
            target = params[name].dtype if name in params else self.config.torch_dtype
            if tensor.dtype != target:
                rank_sharded[name] = tensor.to(target)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        self._materialize_buffers()

    @classmethod
    def from_configs(cls, config: PretrainedConfig, start_layer_idx: int,
                     neuron_config: NeuronConfig | None = None):
        return cls(Qwen3_5Config.from_configs(config, neuron_config), start_layer_idx)
