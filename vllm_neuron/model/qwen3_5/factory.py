# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5 / Qwen3.8 dense hybrid text model (BF16, text only)."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForCausalLM(nn.Module):
    """Validates the config and builds the Neuron implementation.

    Registered under both ``Qwen3_5ForCausalLM`` and ``Qwen3_5ForConditionalGeneration``
    (the published Qwen3.8-27B checkpoint's architecture); the vision tower and MTP head are
    ignored, so both serve text only.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import Qwen3_5ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        from .config import Qwen3_5Config

        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3.5/3.8. "
                "Use None/'bf16', or 'fp8' (per-tensor static FP8 MLP, quantized at load "
                "from a BF16 checkpoint with calibrated activation scales)."
            )
        cfg = Qwen3_5Config.from_configs(hf_config)
        tp = get_tp_group().world_size
        cfg.validate_tp(tp)
        if cfg.num_key_value_heads // tp != 1:
            # Hybrid attention/state models split the KV manager's enlarged attention blocks
            # into kernel-sized blocks with a pure reshape, which needs 1 KV head per rank.
            raise NotImplementedError(
                f"tensor_parallel_size must equal num_key_value_heads "
                f"({cfg.num_key_value_heads}) for Qwen3.5/3.8; got {tp}."
            )
