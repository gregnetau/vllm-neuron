# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 / Qwen3.8 dense (hybrid Gated DeltaNet) config
======================================================

<-- MODEL-SPECIFIC: Qwen3.8-27B (``model_type=qwen3_5``, text tower only).
64 layers: 48 ``linear_attention`` (Gated DeltaNet) + 16 ``full_attention``
(every ``full_attention_interval``-th layer). Full-attention layers have an output
gate (q_proj emits 2*head_dim per head: [query | gate]), per-head QK-norm and
partial rotary (``partial_rotary_factor`` of head_dim) with interleaved mrope.
RMSNorm is zero-centered: ``x * (1 + weight)``.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_neuron.model.neuron_config import NeuronConfig

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


@dataclass
class Qwen3_5Config:
    # <-- MODEL-SPECIFIC: Qwen3.8-27B text architecture parameters
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Full-attention specifics
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True

    # Gated DeltaNet (linear attention) specifics
    layer_types: list[str] = field(default_factory=list)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    # Recurrent (temporal) state dtype; the conv state uses ``torch_dtype``.
    mamba_ssm_dtype: torch.dtype = torch.float32

    # Multi-token-prediction head (draft model for speculative decoding)
    mtp_num_hidden_layers: int = 1

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if not self.layer_types:
            # HF default: every 4th layer is full attention
            self.layer_types = [
                FULL_ATTENTION if (i + 1) % 4 == 0 else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(self.layer_types)} != "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        bad = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if bad:
            raise ValueError(f"Unsupported layer types: {sorted(bad)}")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "linear_num_value_heads must be a multiple of linear_num_key_heads"
            )

    # -- derived ----------------------------------------------------------
    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        """Channels of the depthwise conv: q | k | v."""
        return 2 * self.key_dim + self.value_dim

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION]

    @property
    def linear_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION]

    @property
    def fp8_enabled(self) -> bool:
        return self.neuron_config is not None and self.neuron_config.quantization == "fp8"

    def fp8_quantized(self, module: str) -> bool:
        """Whether HF-named ``module`` (e.g. "layers.3.mlp") runs in static FP8."""
        if not self.fp8_enabled:
            return False
        skip = self.neuron_config.modules_to_not_convert or []
        return not any(s in module for s in skip)

    def validate_tp(self, tp_size: int) -> None:
        """Every sharded dimension must divide evenly (KV heads are not replicated)."""
        for name, n in (
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("intermediate_size", self.intermediate_size),
        ):
            if n % tp_size:
                raise ValueError(f"{name}={n} is not divisible by tp_size={tp_size}")

    # -- construction -----------------------------------------------------
    @classmethod
    def from_configs(cls, hf_config: Any, neuron_config: NeuronConfig | None = None):
        """Build from a HF config (top-level or text-only), a dict, or a JSON path."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, dict):
            config_dict = dict(hf_config)
        else:
            config_dict = hf_config.to_dict()
            dtype = getattr(hf_config, "torch_dtype", None) or getattr(
                hf_config, "dtype", None
            )
            if dtype is not None:
                config_dict["torch_dtype"] = dtype

        # Top-level Qwen3_5ForConditionalGeneration config nests the text tower.
        text = config_dict.get("text_config")
        if text is not None:
            top_dtype = config_dict.get("torch_dtype") or config_dict.get("dtype")
            config_dict = dict(text)
            if top_dtype is not None:
                config_dict.setdefault("torch_dtype", top_dtype)

        rope = dict(config_dict.get("rope_parameters") or {})
        for src, dst in (
            ("rope_theta", "rope_theta"),
            ("partial_rotary_factor", "partial_rotary_factor"),
            ("mrope_section", "mrope_section"),
            ("mrope_interleaved", "mrope_interleaved"),
        ):
            if src in rope:
                config_dict[dst] = rope[src]

        # HF may serialize dtype under "dtype"; normalise both to torch dtypes.
        if "torch_dtype" not in config_dict and "dtype" in config_dict:
            config_dict["torch_dtype"] = config_dict["dtype"]
        for key in ("torch_dtype", "mamba_ssm_dtype"):
            val = config_dict.get(key)
            if isinstance(val, str):
                config_dict[key] = getattr(torch, val.replace("torch.", ""))

        field_names = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in config_dict.items() if k in field_names}
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config
        return cls(**filtered)
