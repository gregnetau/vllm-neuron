# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding pieces of the standalone V2 Neuron model runner.

V2 runs the draft model inside the target's compiled graph, as a submodule of
:class:`~vllm_neuron.vllm.worker.neuron_forward_adapter.NeuronForwardAdapter`.
:class:`V2DraftSpeculator` adapts a draft model with the fused draft forward that
EAGLE3 and MTP draft models share (``forward(input_ids, positions,
initial_target_hidden_states, attn_metadata, sampling_positions, rank,
raw_sampled_token_ids) -> (tokens [B, 1 + k], drafts [B, k], logits)``) and reads
the target output the draft needs. See
``docs/design/speculation/mrv2-spec-decode.md``.

This module must not import the V1 runner.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Speculative methods V2 serves, each with the target architectures it is validated
# for. Everything else stays on V1 (model_runner_v2_policy).
V2_SPEC_DECODE_METHODS: dict[str, frozenset[str]] = {
    "mtp": frozenset({"Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"}),
}

# Rejection methods the in-graph fixed-layout sampler implements.
V2_REJECTION_SAMPLE_METHODS = frozenset({"standard", "synthetic"})


def v2_spec_decode_unsupported_reason(
    method: str | None, architectures: list[str]
) -> str | None:
    """Return why V2 cannot serve a speculative method for a model, or None.

    Args:
        method: ``speculative_config.method``.
        architectures: The target model's architecture names.

    Returns:
        A short reason for the policy's unsupported list, or ``None`` when the
        combination is in the allowlist.
    """
    allowed = V2_SPEC_DECODE_METHODS.get(method or "")
    if allowed is None:
        return f"speculative method {method!r}"
    if not allowed.intersection(architectures):
        return f"speculative method {method!r} for {architectures}"
    return None


class V2DraftSpeculator(nn.Module):
    """Run a fused draft model inside the V2 forward-adapter graph.

    Args:
        draft_model: Draft model with the fused draft forward.
        num_speculative_tokens: Drafts per request and step, ``k``.
        layer_names: The draft model's attention layer names; its metadata is
            selected from the step's per-layer metadata.
        synthetic_conditional_rates: Conditional acceptance rates when
            ``rejection_sample_method == "synthetic"``, else ``None``.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        num_speculative_tokens: int,
        layer_names: list[str],
        synthetic_conditional_rates: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.layer_names = list(layer_names)
        self.synthetic_conditional_rates = synthetic_conditional_rates

    @staticmethod
    def target_hidden_states(result) -> torch.Tensor:
        """Return the target hidden states the draft reads (MTP: final hidden states).

        The target is configured with ``set_mtp_hidden_output(True)`` and returns
        ``(sampled, hidden_states, gathered_logits)``.
        """
        return result[1]

    @staticmethod
    def gathered_logits(result):
        """Return the target's gathered logits, or None."""
        return result[2] if len(result) > 2 else None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attn_metadata: dict,
        sampling_positions: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        """Propose ``k`` drafts per row.

        Args:
            input_ids: ``[T]`` the step's tokens shifted left by one (token ``i``
                holds token ``i + 1``).
            positions: ``[T]`` positions of the step's tokens.
            target_hidden_states: ``[T, H]`` target hidden states.
            attn_metadata: The step's per-layer metadata (target and draft layers).
            sampling_positions: ``[R]`` the last token of each row; the draft model
                moves it back to the last accepted token.
            raw_sampled_token_ids: ``[R, 1 + k]`` rejection output (verify) or
                ``[R, 1]`` next token (prefill).
            rank: Tensor-parallel rank tensor.

        Returns:
            ``[R, k]`` int32 draft tokens.
        """
        metadata = {name: attn_metadata[name] for name in self.layer_names}
        _, drafts, _ = self.draft_model(
            input_ids=input_ids,
            positions=positions,
            initial_target_hidden_states=target_hidden_states,
            attn_metadata=metadata,
            sampling_positions=sampling_positions,
            rank=rank,
            raw_sampled_token_ids=raw_sampled_token_ids,
        )
        return drafts


def draft_layer_names(draft_model: nn.Module) -> list[str]:
    """Return the attention layer names of a draft model's KV spec."""
    return [layer.name for layer in draft_model.get_kv_spec().layers]


def load_draft_model(
    vllm_config, device: torch.device, start_layer_idx: int
) -> nn.Module:
    """Build and load the draft model of a speculative configuration.

    Resolve the draft architecture in the plugin's model registry, build it on
    ``meta``, load its weights (compile-time constants only during CPU
    compilation) and move it to ``device``. Mirrors the model part of V1's
    ``EagleProposer.compile_and_load_draft_model`` for the methods V2 serves; the
    draft is not compiled on its own (it runs inside the target graph).

    Args:
        vllm_config: Engine configuration with a ``speculative_config``.
        device: Device for the weights.
        start_layer_idx: Index of the draft's first layer (the target's layer
            count), which names its KV layers.

    Returns:
        The loaded draft model.

    Raises:
        ValueError: For a method V2 does not serve or an unregistered architecture.
    """
    from vllm_neuron import envs
    from vllm_neuron.metrics import MODEL_LOAD_SIZE, MODEL_LOAD_TIME
    from vllm_neuron.model.neuron_config import NeuronConfig, OnDeviceSamplingConfig
    from vllm_neuron.model.registry import get_models

    spec_config = vllm_config.speculative_config
    if spec_config.method not in V2_SPEC_DECODE_METHODS:
        raise ValueError(
            f"V2 runner does not serve speculative method {spec_config.method!r}"
        )
    draft_model_config = spec_config.draft_model_config
    arch = draft_model_config.architecture
    models = dict(get_models())
    if arch not in models:
        raise ValueError(
            f"Draft model architecture {arch!r} is not in the vLLM Neuron model "
            f"registry ({sorted(models)})"
        )
    # The draft samples greedily; the target's rejection sampler keeps the target
    # distribution (same as V1).
    neuron_config = NeuronConfig(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    with torch.device("meta"):
        model = models[arch].from_configs(
            config=draft_model_config.hf_config,
            start_layer_idx=start_layer_idx,
            neuron_config=neuron_config,
        )
    model.num_speculative_tokens = spec_config.num_speculative_tokens
    if envs.VLLM_NEURON_CPU_COMPILE:
        if hasattr(model, "load_weights_lite"):
            model.load_weights_lite(
                spec_config.model,
                torch.device("cpu"),
                vllm_config.load_config.download_dir,
            )
        return model.to("meta")
    start = time.perf_counter()
    model.load_weights(spec_config.model, device, vllm_config.load_config.download_dir)
    model = model.to(device)
    MODEL_LOAD_TIME.labels(model_name=spec_config.model).set(
        time.perf_counter() - start
    )
    MODEL_LOAD_SIZE.labels(model_name=spec_config.model).set(
        sum(p.nbytes for p in model.parameters())
    )
    logger.info(
        "Loaded draft model %s (%s) in %.2f s",
        spec_config.model,
        arch,
        time.perf_counter() - start,
    )
    return model
