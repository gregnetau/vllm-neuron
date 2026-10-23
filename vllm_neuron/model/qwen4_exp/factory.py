# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``), text only."""

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import (
    HasHostInputs,
    HasRecurrentState,
    ServesTextOnly,
)
from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen4ExpForCausalLM(nn.Module, HasRecurrentState, ServesTextOnly, HasHostInputs):
    """Validates the config and builds the Neuron implementation.

    Registered under ``Qwen4ExpForCausalLM`` and ``Qwen4ExpForConditionalGeneration`` (the
    published checkpoint's architecture); both serve the text tower. The n-gram embedding
    rows come from the host each step (``build_host_inputs``), so the runner must not run
    asynchronous scheduling for this model.
    """

    SEGMENTED_PREFILL_READS_UNPACKED_FP8_KV = True
    NATIVE_COMPILER_MODEL_TYPE = "generic"

    @classmethod
    def num_recurrent_state_layers(cls, hf_config: PretrainedConfig) -> int:
        from .config import LINEAR_ATTENTION, Qwen4ExpConfig

        cfg = Qwen4ExpConfig.from_configs(hf_config)
        return cfg.layer_types.count(LINEAR_ATTENTION) + len(cfg.ple_layers)

    @classmethod
    def serves_text_only(cls) -> bool:
        return True

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self.from_configs(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import Qwen4ExpForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_neuron.model.qwen3_5.factory import Qwen3_5ForCausalLM

        from .config import Qwen4ExpConfig

        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3.8-Flash-Next. "
                "Use None/'bf16', or 'fp8' (FP8 routed experts, quantized at load from the "
                "BF16 checkpoint)."
            )
        Qwen3_5ForCausalLM._validate_chunked_prefill(neuron_config)
        cfg = Qwen4ExpConfig.from_configs(hf_config)
        if cfg.tie_word_embeddings:
            raise ValueError("tie_word_embeddings=True is not supported")
        tp = get_tp_group().world_size
        cfg.validate_tp(tp)
        if cfg.kv_heads_per_rank(tp) != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next needs one KV head per rank: tensor_parallel_size must be a "
                f"multiple of num_key_value_heads ({cfg.num_key_value_heads}); got {tp}."
            )


class Qwen4ExpMTP(nn.Module):
    """Factory for the MTP draft model (vLLM's ``mtp`` speculative method)."""

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        from .mtp import Qwen4ExpMTP as Model

        return Model.from_configs(config, start_layer_idx, neuron_config)
