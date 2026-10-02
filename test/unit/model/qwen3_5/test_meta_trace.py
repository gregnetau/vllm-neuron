# SPDX-License-Identifier: Apache-2.0
"""Trace-safety: the model must run with every tensor on the `meta` device.

The Neuron compile traces with fake/meta tensors, so any CPU-only constructor or
data-dependent `.item()` (e.g. indexing with a 0-d tensor) breaks compilation on device.
Reuses the stubbed Neuron ops from test_model_smoke.
"""

import sys

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

import test_model_smoke as smoke  # noqa: E402


def test_prefill_and_gdn_decode_trace_on_meta():
    mod = smoke._install_stubs()
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=96, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=128, dtype="float32",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    model = mod.Qwen3_5ForCausalLM.from_configs(hf_cfg.to_dict(), None).to("meta")
    spec = model.get_kv_spec()
    model.bind_kv_cache({l.name: [torch.zeros(8, l.num_kv_heads, 8, l.head_size, device="meta")
                                  for _ in range(2)] for l in spec.layers})
    state_cache = sys.modules["vllm_neuron.model.state_cache"]
    halves = {}
    for sl in spec.state_layers:
        page_bytes = state_cache.page_layout(sl.shapes, sl.dtypes).page_bytes + 32
        raw = torch.zeros(10 * page_bytes, dtype=torch.int8, device="meta")
        halves[sl.name] = list(state_cache.paired_half_views(raw, page_bytes))
    model.bind_state_cache(halves)

    T = 32

    def md(decode):
        common = dict(max_query_len=1 if decode else T, decode_token_threshold=1, block_size=8,
                      max_blocks_per_seq=4, kv_segment_size=0,
                      cached_seq_len=torch.zeros(1, 1, dtype=torch.int32, device="meta"))
        out = {}
        for i, t in enumerate(model.config.layer_types):
            key = f"layers.{i}." + ("self_attn" if t == "full_attention" else "linear_attn")
            out[key] = dict(common, slot_mapping=torch.zeros(1 if decode else T, dtype=torch.long, device="meta"),
                            block_table_tensor=torch.zeros(1, 4, dtype=torch.int32, device="meta"))
        return out

    meta = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device="meta")
    logits = model(meta(T, dtype=torch.long), positions=torch.arange(T, device="meta"), attn_metadata=md(False),
                   sampling_positions=meta(1, dtype=torch.long))
    assert logits.device.type == "meta" and logits.shape == (1, 128)

    with torch.no_grad():
        out = model.model.layers[0](meta(1, 64), positions=meta(1, dtype=torch.int32),
                                    position_embeddings=(meta(1, 32), meta(1, 32)),
                                    attn_metadata=md(True))
    assert out.shape == (1, 64)


def test_buffers_are_real_after_meta_construction():
    """The runner builds the model under the meta device; non-persistent buffers must be re-created."""
    mod = smoke._install_stubs()
    hf_cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=96, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=128, dtype="float32",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    with torch.device("meta"):
        model = mod.Qwen3_5ForCausalLM.from_configs(hf_cfg.to_dict(), None)
    assert model.model.rotary_emb.inv_freq.device.type == "meta"
    model._materialize_buffers()
    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.model.rotary_emb.inv_freq.shape == (model.config.rotary_dim // 2,)
