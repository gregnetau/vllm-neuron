# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next (``model_type=qwen4_exp``) text config
==========================================================

<-- MODEL-SPECIFIC: extends the Qwen3.5 hybrid config (Gated DeltaNet + gated attention) with:

  * QSA (Qwen Sparse Attention) on every attention layer: a 4-head x 128 indexer with one key
    head scores mean-pooled 4-token key blocks; each query attends to its top
    ``indexer_budget / indexer_compress_ratio`` blocks plus the open tail block.
  * Gated residual (hyper-connections): the residual stream is ``hc_count`` x hidden wide.
  * Sparse MoE MLP on every layer: ``num_experts`` routed (softmax, top-k, renormalised) plus a
    sigmoid-gated shared expert.
  * Per-layer n-gram embedding (PLE) on the one-indexed layers ``ple_layer_ids``.
  * GDN output gate ``output_gate_type`` (sigmoid for the published checkpoint).

HF names the attention layers ``full_attention`` (checkpoint), ``qwen_sparse_attention``
(transformers 5.17) or ``indexed_attention`` (transformers >= 5.18); all map to
``FULL_ATTENTION`` here so the shared Qwen3.5 code paths apply.
"""

from dataclasses import dataclass, field
from typing import Any

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen3_5Config,
)

__all__ = ["FULL_ATTENTION", "LINEAR_ATTENTION", "Qwen4ExpConfig"]

ATTENTION_ALIASES = ("full_attention", "qwen_sparse_attention", "indexed_attention")


@dataclass
class Qwen4ExpConfig(Qwen3_5Config):
    # <-- MODEL-SPECIFIC: Qwen3.8-Flash-Next defaults
    hidden_size: int = 2560
    intermediate_size: int = 0  # no dense MLP
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    eos_token_id: int = 248044

    # GDN
    output_gate_type: str = "sigmoid"

    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    # Gated residual
    hc_count: int = 4
    hc_lowrank: int = 320

    # PLE / n-gram embedding
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])  # one-indexed
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128

    # QSA indexer
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4

    def __post_init__(self):
        self.layer_types = [
            FULL_ATTENTION if t in ATTENTION_ALIASES else t for t in self.layer_types
        ]
        if isinstance(self.eos_token_id, (list, tuple)):
            self.eos_token_id = self.eos_token_id[0]
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        super().__post_init__()
        if self.indexer_kv_heads != 1:
            raise ValueError("QSA requires indexer_kv_heads == 1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        if self.ple_embed_dim % self.ngram_heads:
            raise ValueError("ple_embed_dim must be divisible by the n-gram head count")
        for lid in self.ple_layer_ids:
            if self.layer_types[lid - 1] != LINEAR_ATTENTION:
                raise ValueError(f"PLE layer {lid} must be a linear-attention layer")

    # -- derived ----------------------------------------------------------
    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def block_topk(self) -> int:
        """QSA blocks selected per query (512)."""
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads

    @property
    def ple_layers(self) -> list[int]:
        """Zero-based decoder layer indices with a PLE module."""
        return [lid - 1 for lid in self.ple_layer_ids]

    @property
    def ple_conv_state_len(self) -> int:
        """Inputs the dilated (dilation = ngram_size) causal PLE conv keeps per request."""
        return (self.ple_conv_kernel_size - 1) * self.ngram_size

    def fp8_quantized(self, module: str) -> bool:
        """``quantization="fp8"`` covers the routed experts only (per-channel weights, no
        calibration). Attention, GDN, shared experts, router and gated residual stay BF16."""
        return ".mlp.experts" in module and super().fp8_quantized(module)

    def kv_heads_per_rank(self, tp_size: int) -> int:
        """KV heads per rank; replicated (1 per rank) when tp_size > num_key_value_heads."""
        return max(1, self.num_key_value_heads // tp_size)

    def kv_head_for_rank(self, rank: int, tp_size: int) -> int:
        """First KV head of ``rank`` (replicated KV: rank r holds head r * kv // tp)."""
        if tp_size <= self.num_key_value_heads:
            return rank * (self.num_key_value_heads // tp_size)
        return rank * self.num_key_value_heads // tp_size

    def validate_tp(self, tp_size: int) -> None:
        """Sharded dims divide by tp; KV heads may be replicated (tp a multiple of them);
        experts are split across the TP ranks (expert parallel inside the TP group)."""
        for name, n in (
            ("num_attention_heads", self.num_attention_heads),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("num_experts", self.num_experts),
        ):
            if n % tp_size:
                raise ValueError(f"{name}={n} is not divisible by tp_size={tp_size}")
        kv = self.num_key_value_heads
        if kv % tp_size and tp_size % kv:
            raise ValueError(
                f"tp_size={tp_size} must divide or be a multiple of num_key_value_heads={kv}"
            )

    @classmethod
    def from_configs(cls, hf_config: Any, neuron_config: NeuronConfig | None = None):
        return super().from_configs(hf_config, neuron_config)
