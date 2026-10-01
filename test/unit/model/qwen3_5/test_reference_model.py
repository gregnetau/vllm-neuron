# SPDX-License-Identifier: Apache-2.0
"""Whole-model reference vs HF Qwen3_5ForCausalLM: chunked prefill + decode with carried state."""

import pytest
import torch

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402


@pytest.fixture(scope="module")
def reference_mod(qwen3_5_config):
    from conftest import ROOT, _load

    _load("vllm_neuron.model.qwen3_5.gdn_ops", ROOT / "model/qwen3_5/gdn_ops.py")
    return _load("vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py")


@pytest.mark.parametrize("tie", [False, True])
def test_reference_matches_hf_logits(qwen3_5_config, reference_mod, tie):
    torch.manual_seed(3)
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=96, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        vocab_size=128, partial_rotary_factor=0.25, tie_word_embeddings=tie,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2],
                         "mrope_interleaved": True},
    )
    model = hf.Qwen3_5ForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.1))
            elif "norm" in n and "linear_attn" not in n:
                p.copy_(torch.randn_like(p) * 0.1)  # zero-centered norms
            elif "linear_attn.norm" in n:
                p.copy_(torch.rand_like(p) + 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    ids = torch.randint(0, 128, (1, 75))
    with torch.no_grad():
        ref = model(ids).logits[0]

    cfg = qwen3_5_config.Qwen3_5Config.from_configs(hf_cfg.to_dict())
    assert cfg.layer_types.count("full_attention") == 2
    mine = reference_mod.Qwen3_5Reference(cfg, dict(model.state_dict()), prefix="model.")
    outs = [mine.forward(ids[0, :40]), mine.forward(ids[0, 40:69])]  # chunked prefill
    outs += [mine.forward(ids[0, t : t + 1], decode=True) for t in range(69, 75)]  # decode
    torch.testing.assert_close(torch.cat(outs), ref, rtol=2e-3, atol=2e-3)
