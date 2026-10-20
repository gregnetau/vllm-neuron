# SPDX-License-Identifier: Apache-2.0
import copy
import json
import pathlib

import pytest
import torch

FIXTURE = pathlib.Path(__file__).with_name("qwen3_8_27b_config.json")


@pytest.fixture(scope="module")
def raw():
    return json.loads(FIXTURE.read_text())


def test_config_from_real_checkpoint(qwen3_5_config, raw):
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    assert cfg.num_hidden_layers == 64 and cfg.hidden_size == 5120
    assert cfg.head_dim == 256 and cfg.num_attention_heads == 24 and cfg.num_key_value_heads == 4
    assert cfg.rotary_dim == 64 and cfg.rope_theta == 1e7
    assert cfg.mrope_section == [11, 11, 10] and cfg.mrope_interleaved
    assert (cfg.linear_num_key_heads, cfg.linear_num_value_heads) == (16, 48)
    assert cfg.conv_dim == 10240 and cfg.key_dim == 2048 and cfg.value_dim == 6144
    assert len(cfg.full_attention_layers) == 16 and len(cfg.linear_attention_layers) == 48
    assert cfg.full_attention_layers[:2] == [3, 7]
    assert cfg.torch_dtype == torch.bfloat16 and cfg.mamba_ssm_dtype == torch.float32
    assert cfg.attn_output_gate and not cfg.tie_word_embeddings


def test_config_from_text_only_dict_and_default_layer_types(qwen3_5_config, raw):
    text = copy.deepcopy(raw["text_config"])
    del text["layer_types"]
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(text)
    assert cfg.layer_types == qwen3_5_config.Qwen3_5Config.from_configs(raw).layer_types


def test_config_from_hf_object(qwen3_5_config, raw):
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    hf_cfg = Qwen3_5Config(**{k: v for k, v in raw.items() if k != "architectures"})
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(hf_cfg)
    ref = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    assert cfg == ref


def test_tp_validation(qwen3_5_config, raw):
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    for tp in (1, 2, 4):
        cfg.validate_tp(tp)
    with pytest.raises(ValueError):
        cfg.validate_tp(8)  # 4 KV heads: no replication supported
    with pytest.raises(ValueError):
        qwen3_5_config.Qwen3_5Config(num_hidden_layers=4, layer_types=["full_attention"])


def test_weight_names_match_real_index(qwen3_5_config, weights, raw):
    """Names we expect == text-tower names in the published safetensors index."""
    idx = pathlib.Path(__file__).with_name("qwen3_8_27b_weight_names.json")
    published = set(json.loads(idx.read_text()))
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    text = {n for n in published if weights.is_text_weight(n)}
    assert weights.expected_text_weight_names(cfg) == text



def test_fp8_helpers_roundtrip_and_trn2_range():
    from vllm_neuron.model.qwen3_5 import weights as W

    torch.manual_seed(0)
    w = torch.randn(256, 64) * 0.05
    w[3, 7] = 1.7  # outlier sets the per-tensor scale
    s = W.fp8_weight_scale(w)
    q = W.fp8_quantize(w, s)
    assert q.dtype == torch.float8_e4m3fn
    assert q.float().abs().max().item() <= W.FP8_MAX  # stays in trn2's legacy e4m3 range
    assert torch.allclose(q.float() * s, w, rtol=0.07, atol=s * 0.5)
    tile = W.fp8_scale_tile(s)
    assert tile.shape == (128, 1) and tile.dtype == torch.float32 and torch.all(tile == s)


def test_fp8_module_selection():
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    assert not Qwen3_5Config().fp8_quantized("layers.0.mlp")
    nc = NeuronConfig.from_dict({"quantization": "fp8", "modules_to_not_convert": ["layers.5.mlp"]})
    cfg = Qwen3_5Config(neuron_config=nc)
    assert cfg.fp8_enabled and cfg.fp8_quantized("layers.0.mlp")
    assert not cfg.fp8_quantized("layers.5.mlp")


def test_fp8_quantize_parts_uses_one_scale_per_block():
    from vllm_neuron.model.qwen3_5 import weights as W

    torch.manual_seed(1)
    q, k, v = torch.randn(64, 8) * 4.0, torch.randn(64, 4) * 0.5, torch.randn(64, 4) * 0.1
    scales = [W.fp8_weight_scale(t) for t in (q, k, v)]
    packed = W.fp8_quantize_parts(torch.cat([q, k, v], dim=1), scales, [8, 4, 4])
    for block, t, s in zip(torch.split(packed.float(), [8, 4, 4], dim=1), (q, k, v), scales):
        assert block.abs().max().item() <= W.FP8_MAX
        assert torch.allclose(block * s, t, rtol=0.07, atol=s * 0.5)
