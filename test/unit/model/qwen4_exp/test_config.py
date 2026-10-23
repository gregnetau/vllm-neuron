# SPDX-License-Identifier: Apache-2.0
import pytest

from qwen4_exp_testlib import real_text_config_dict


pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


def _cfg(d):
    from vllm_neuron.model.qwen4_exp.config import FULL_ATTENTION, Qwen4ExpConfig

    return Qwen4ExpConfig.from_configs(d), FULL_ATTENTION


def test_real_config_parses():
    cfg, FULL = _cfg({"text_config": real_text_config_dict(), "dtype": "bfloat16"})
    assert cfg.hidden_size == 2560 and cfg.num_hidden_layers == 48
    assert cfg.layer_types.count(FULL) == 12 and cfg.layer_types[3] == FULL
    assert cfg.ple_layers == [1] and cfg.ple_conv_state_len == 9
    assert cfg.ngram_heads == 16 and cfg.ngram_head_dim == 160
    assert cfg.block_topk == 512 and cfg.rotary_dim == 64
    assert cfg.hc_hidden_size == 10240 and cfg.output_gate_type == "sigmoid"
    assert cfg.num_experts == 512 and cfg.num_experts_per_tok == 10
    assert cfg.eos_token_id == 248044


@pytest.mark.parametrize(
    "alias", ["full_attention", "qwen_sparse_attention", "indexed_attention"]
)
def test_attention_aliases(alias):
    """Every HF name of the QSA attention layers maps to the shared FULL_ATTENTION type."""
    d = real_text_config_dict()
    d["layer_types"] = [alias if t == "full_attention" else t for t in d["layer_types"]]
    cfg, FULL = _cfg(d)
    assert cfg.layer_types.count(FULL) == 12
    assert set(cfg.layer_types) == {FULL, "linear_attention"}


def test_invalid_configs_rejected():
    d = real_text_config_dict()
    with pytest.raises(ValueError, match="indexer_kv_heads"):
        _cfg(dict(d, indexer_kv_heads=2))
    with pytest.raises(ValueError, match="indexer_budget"):
        _cfg(dict(d, indexer_budget=2050))
    with pytest.raises(ValueError, match="PLE layer"):
        _cfg(dict(d, ple_layer_ids=[4]))  # layer 4 (one-indexed) is an attention layer


def test_tp_layouts():
    cfg, _ = _cfg(real_text_config_dict())
    for tp in (1, 2, 4, 8):
        cfg.validate_tp(tp)
    assert cfg.kv_heads_per_rank(8) == 1
    assert [cfg.kv_head_for_rank(r, 8) for r in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [cfg.kv_head_for_rank(r, 2) for r in range(2)] == [0, 1]
    with pytest.raises(ValueError):
        cfg.validate_tp(16)  # 24 attention heads


def test_tiny_config(qwen4_config):
    assert qwen4_config.hidden_size == 256 and qwen4_config.block_topk == 4
    assert qwen4_config.ple_layers == [1] and qwen4_config.ngram_head_dim == 16


def test_fp8_covers_routed_experts_only():
    from vllm_neuron.model.neuron_config import NeuronConfig

    cfg, _ = _cfg(real_text_config_dict())
    cfg.neuron_config = NeuronConfig(quantization="fp8")
    assert cfg.fp8_quantized("layers.0.mlp.experts")
    assert not cfg.fp8_quantized("layers.0.mlp.shared_expert")
    assert not cfg.fp8_quantized("layers.3.self_attn")
