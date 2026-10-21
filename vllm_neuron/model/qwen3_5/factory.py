# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5 / Qwen3.8 dense hybrid text model (BF16 or static FP8, text only)."""

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.interfaces import HasRecurrentState, ServesTextOnly
from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForCausalLM(nn.Module, HasRecurrentState, ServesTextOnly):
    """Validates the config and builds the Neuron implementation.

    Registered under both ``Qwen3_5ForCausalLM`` and ``Qwen3_5ForConditionalGeneration``
    (the published Qwen3.8-27B checkpoint's architecture). Both serve the text tower; the
    vision tower is not loaded, and the MTP head is the separate ``Qwen3_5MTP`` draft model.
    """

    # Options the plugin core reads from the registered class (registry.get_model_cls):
    # Segmented prefill reads an unpacked FP8 K cache with the vendored kernel
    # (attention_segmented_cte.py), so the runner keeps segmented prefill on for FP8 KV.
    SEGMENTED_PREFILL_READS_UNPACKED_FP8_KV = True
    # Native compile route: --model-type=generic, as on the XLA route. The default
    # "transformer" partitions the prefill graph per layer and remats (about 2.5x slower).
    NATIVE_COMPILER_MODEL_TYPE = "generic"

    @classmethod
    def num_recurrent_state_layers(cls, hf_config: PretrainedConfig) -> int:
        from .config import LINEAR_ATTENTION, Qwen3_5Config

        return Qwen3_5Config.from_configs(hf_config).layer_types.count(LINEAR_ATTENTION)

    @classmethod
    def serves_text_only(cls) -> bool:
        return True

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
                "Use None/'bf16', or 'fp8' (per-tensor static FP8 MLP, attention and GDN "
                "projections, quantized at load from a BF16 checkpoint with calibrated "
                "activation scales)."
            )
        cls._validate_chunked_prefill(neuron_config)
        cfg = Qwen3_5Config.from_configs(hf_config)
        if cfg.tie_word_embeddings:
            # The model always loads a separate ``lm_head.weight`` (published checkpoints are
            # untied); the reference model in reference.py does handle tied embeddings.
            raise ValueError(
                "tie_word_embeddings=True is not supported for Qwen3.5/3.8 (the Neuron "
                "model loads a separate lm_head.weight)."
            )
        tp = get_tp_group().world_size
        cfg.validate_tp(tp)
        if cfg.num_key_value_heads // tp != 1:
            # Hybrid attention/state models split the KV manager's enlarged attention blocks
            # into kernel-sized blocks with a pure reshape, which needs 1 KV head per rank.
            raise NotImplementedError(
                f"tensor_parallel_size must equal num_key_value_heads "
                f"({cfg.num_key_value_heads}) for Qwen3.5/3.8; got {tp}."
            )

    @staticmethod
    def _validate_chunked_prefill(neuron_config: NeuronConfig | None) -> None:
        """Without kv_segment_size_buckets, prefill attends only within its own chunk, so a
        prompt split over several steps would ignore its cached prefix. Refuse that config."""
        from vllm.config import get_current_vllm_config

        if neuron_config is None or neuron_config.kv_segment_size_buckets is not None:
            return
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is None:
            return
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        if max_batched < max_model_len:
            raise ValueError(
                f"Qwen3.5/3.8 needs segmented prefill when a prompt can span several prefill "
                f"steps (max_num_batched_tokens={max_batched} < max_model_len={max_model_len}), "
                "but kv_segment_size_buckets resolved to None. Set kv_segment_size_buckets in "
                "neuron_config, or max_num_batched_tokens >= max_model_len."
            )


class Qwen3_5MTP(nn.Module):
    """Factory for the MTP draft model (vLLM's ``mtp`` speculative method)."""

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        from .mtp import Qwen3_5MTP as Model

        return Model.from_configs(config, start_layer_idx, neuron_config)
