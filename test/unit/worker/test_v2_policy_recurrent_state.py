# SPDX-License-Identifier: Apache-2.0
"""Runner selection for recurrent-state (hybrid) models (model_runner_v2_policy), CPU.

Hybrid models stay on V1 by default and run on V2 when VLLM_USE_V2_MODEL_RUNNER=1, except
for the features only V1 serves. Under the Neuron policy an unsupported feature routes a
default config to V1 and makes a forced one fail validation (validate_neuron_model_runner).
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm_neuron.vllm import model_runner_v2_policy as policy

QWEN = "Qwen3_5ForConditionalGeneration"
CONFIG_JSON = (
    Path(__file__).parents[1] / "model" / "qwen3_5" / "qwen3_8_27b_config.json"
)


def _vllm_config(
    arch=QWEN,
    *,
    apc=False,
    spec=None,
    max_model_len=2048,
    max_batched=1024,
    cache_dtype="auto",
    neuron_config=None,
    is_hybrid=True,
    multimodal=True,
):
    # model_config mirrors what vLLM builds for the checkpoint: vLLM's own class for
    # Qwen3_5ForConditionalGeneration is hybrid and multimodal.
    model_config = SimpleNamespace(
        runner_type="generate",
        is_moe=False,
        is_hybrid=is_hybrid,
        is_attention_free=False,
        is_encoder_decoder=False,
        multimodal_config=object() if multimodal else None,
        enable_prompt_embeds=False,
        logits_processors=None,
        logprobs_mode="raw_logprobs",
        architectures=[arch],
        max_model_len=max_model_len,
        enforce_eager=False,
        get_sliding_window=lambda: None,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        cache_config=SimpleNamespace(
            enable_prefix_caching=apc, cache_dtype=cache_dtype, kv_offloading_size=None
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_batched, scheduler_cls=None
        ),
        speculative_config=spec,
        additional_config={"neuron_config": dict(neuron_config or {})},
    )


def _decide(monkeypatch, vllm_config, forced):
    """(runner decision, forced validation error or None)."""
    if forced:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    else:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    use_v2 = policy.resolve_neuron_model_runner(vllm_config)
    error = None
    if use_v2:
        try:
            policy.validate_neuron_model_runner(vllm_config)
        except ValueError as e:
            error = str(e)
    return use_v2, error


def test_hybrid_defaults_to_v1_and_is_honored_when_forced(monkeypatch):
    cfg = _vllm_config()
    result = policy.get_neuron_model_runner_v2_policy(cfg)
    # vLLM's is_hybrid / multimodal flags describe its own class, not the Neuron one.
    assert result == policy.NeuronModelRunnerPolicy(False, ())
    assert _decide(monkeypatch, cfg, forced=False) == (False, None)
    assert _decide(monkeypatch, cfg, forced=True) == (True, None)


@pytest.mark.parametrize(
    "kwargs, feature",
    [
        ({"apc": True}, "prefix caching for a recurrent-state model"),
        ({"spec": object()}, "speculative decoding"),
        # Single-shot prefill: no segmented prefill resolves.
        ({"max_batched": 2048}, "recurrent-state model without segmented prefill"),
        # max_num_batched_tokens not a supported segment size.
        ({"max_batched": 768}, "recurrent-state model without segmented prefill"),
        (
            {"neuron_config": {"kv_segment_size_buckets": None}},
            "recurrent-state model without segmented prefill",
        ),
    ],
)
def test_hybrid_features_that_need_v1(monkeypatch, kwargs, feature):
    cfg = _vllm_config(**kwargs)
    assert feature in policy.get_neuron_model_runner_v2_policy(cfg).unsupported_features
    assert _decide(monkeypatch, cfg, forced=False) == (False, None)
    use_v2, error = _decide(monkeypatch, cfg, forced=True)
    assert use_v2 and feature in error


def test_hybrid_shared_state_pool_needs_v1(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_SHARED_STATE_POOL", "1")
    cfg = _vllm_config()
    assert "shared recurrent-state pool" in (
        policy.get_neuron_model_runner_v2_policy(cfg).unsupported_features
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        # Explicit segment buckets.
        {"max_batched": 2048, "neuron_config": {"kv_segment_size_buckets": [512]}},
        # FP8 KV: Qwen3.5 reads the unpacked FP8 cache in segmented prefill.
        {"cache_dtype": "fp8"},
    ],
)
def test_hybrid_segmented_prefill_resolves(kwargs):
    cfg = _vllm_config(**kwargs)
    assert policy.get_neuron_model_runner_v2_policy(cfg).unsupported_features == ()


def test_dense_gates_unchanged():
    # A model without the Neuron capabilities keeps vLLM's hybrid / multimodal gates.
    cfg = _vllm_config(arch="LlamaForCausalLM")
    result = policy.get_neuron_model_runner_v2_policy(cfg)
    assert not result.default_enabled
    assert set(result.unsupported_features) == {"hybrid model", "multimodal model"}
    dense = _vllm_config(arch="LlamaForCausalLM", is_hybrid=False, multimodal=False)
    assert policy.get_neuron_model_runner_v2_policy(dense) == (
        policy.NeuronModelRunnerPolicy(True, ())
    )


def test_real_model_config_of_the_qwen3_8_checkpoint(tmp_path):
    # vLLM's ModelConfig for the published checkpoint architecture is hybrid and
    # multimodal (its own Qwen3_5ForConditionalGeneration class); the policy must not
    # let that block V2.
    from vllm.config import ModelConfig

    shutil.copy(CONFIG_JSON, tmp_path / "config.json")
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=2048
    )
    assert model_config.architectures == [QWEN]
    assert model_config.is_hybrid and model_config.multimodal_config is not None
    cfg = _vllm_config()
    cfg.model_config = model_config
    assert policy.get_neuron_model_runner_v2_policy(cfg) == (
        policy.NeuronModelRunnerPolicy(False, ())
    )
