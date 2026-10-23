# SPDX-License-Identifier: Apache-2.0
"""CPU test of the Qwen3.8-Flash-Next Neuron model against Hugging Face, dense QSA.

The plugin model runs at tp=1 (``qwen4_exp_testlib.patch_plugin``). It checks the weight
mappings and loaders (every parameter mapped, shapes), the KV / state specs and binding, a
padded-bucket prefill and decode steps (gated residual, PLE with its conv state, GDN with sigmoid
gate, gated attention, MoE) against the HF model's logits.

The QSA budget is raised so that every block is selected (QSA equals dense causal attention)
and the dense Qwen3.5 attention paths run; ``test_model_qsa.py`` covers the sparse paths.
"""

import torch

from qwen4_exp_testlib import load_from_hf, ple_rows, state_pages, tiny_text_config_dict

BLOCK, PAGE, PLE_PAGE = 8, 5, 3


pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


def _hf_model():
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(
        **tiny_text_config_dict(indexer_budget=4096, dtype="float32")
    )
    torch.manual_seed(0)
    m = Qwen4ExpForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.1))
            elif "linear_attn.norm" in n:
                p.copy_(torch.rand_like(p) + 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1)
    return m, cfg


def test_prefill_and_decode_match_hf(plugin):
    hf, hf_cfg = _hf_model()
    model = plugin.Qwen4ExpForCausalLM.from_configs(hf_cfg, None)
    cfg = model.config
    assert cfg.torch_dtype == torch.float32
    load_from_hf(model, hf)

    # ---- specs + caches
    spec = model.get_kv_spec()
    n_attn = cfg.layer_types.count("full_attention")
    assert len(spec.layers) == n_attn  # QSA indexer keys: side cache, not a KV layer
    assert len(spec.state_layers) == cfg.num_hidden_layers - n_attn + 1  # + PLE
    kv = {
        ls.name: [
            torch.zeros(8, ls.num_kv_heads, BLOCK, ls.head_size) for _ in range(2)
        ]
        for ls in spec.layers
    }
    model.bind_kv_cache(kv)
    attn = model.model.layers[3].self_attn
    assert attn.k_cache is kv["layers.3.self_attn"][0]
    assert attn.index_cache.shape == (8, 1, BLOCK, cfg.indexer_head_dim)
    model.bind_state_cache(state_pages(spec, 10, fill=7))  # dirty pages
    assert model.model.layers[1].ple.state_a.dim() == 3  # row layout (device path)

    n_real, T = 21, 32
    torch.manual_seed(1)
    ids_all = torch.randint(2, cfg.vocab_size, (n_real + 3,))
    ids_all[5] = cfg.eos_token_id  # exercise the n-gram EOS rule
    ple_all = ple_rows(hf, ids_all)
    blocks = torch.tensor([[2, 3, 4, 6]], dtype=torch.int32)

    def md_for(is_decode, slots, cached, B=1):
        common = {
            "max_query_len": 1 if is_decode else T,
            "decode_token_threshold": 1,
            "block_size": BLOCK,
            "max_blocks_per_seq": 4,
            "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32),
            "kv_segment_size": 0,
            "slot_mapping": slots,
        }
        md = {}
        for i, t in enumerate(cfg.layer_types):
            if t == "full_attention":
                md[f"layers.{i}.self_attn"] = dict(common, block_table_tensor=blocks)
            else:
                md[f"layers.{i}.linear_attn"] = dict(
                    common, block_table_tensor=torch.tensor([[PAGE]], dtype=torch.int32)
                )
        md["layers.1.ple"] = dict(
            common, block_table_tensor=torch.tensor([[PLE_PAGE]], dtype=torch.int32)
        )
        return md

    # ---- padded-bucket prefill
    padded = torch.cat([ids_all[:n_real], torch.zeros(T - n_real, dtype=torch.long)])
    pos = torch.arange(n_real)
    slot = torch.zeros(T, dtype=torch.long)
    slot[:n_real] = blocks[0][pos // BLOCK].long() * BLOCK + pos % BLOCK
    ple = torch.cat([ple_all[:n_real], torch.zeros(T - n_real, ple_all.shape[1])])
    with torch.no_grad():
        logits = model(
            padded,
            positions=torch.arange(T),
            attn_metadata=md_for(False, slot, 0),
            sampling_positions=torch.tensor([n_real - 1]),
            ple_embeds=ple,
        )
        ref = hf(ids_all[None]).logits[0]
    torch.testing.assert_close(logits[0], ref[n_real - 1], rtol=2e-3, atol=2e-3)

    # ---- three decode steps continue from the cached KV / GDN / PLE state
    for step in range(3):
        p = n_real + step
        s = blocks[0][p // BLOCK].long() * BLOCK + p % BLOCK
        with torch.no_grad():
            logits = model(
                ids_all[p : p + 1],
                positions=torch.tensor([p]),
                attn_metadata=md_for(True, s.view(1), p),
                ple_embeds=ple_all[p : p + 1],
            )
        torch.testing.assert_close(logits[0], ref[p], rtol=2e-3, atol=2e-3)


def test_host_inputs(plugin):
    """``host_input_specs`` / ``build_host_inputs``: the step's n-gram rows from the host table,
    equal to HF's for real rows and zero for padding rows."""
    import numpy as np
    import pytest

    from vllm_neuron.model.qwen4_exp.ngram import (
        HostNGramTable,
        HostPLEInputs,
        NGramHasher,
    )

    hf, hf_cfg = _hf_model()
    model = plugin.Qwen4ExpForCausalLM.from_configs(hf_cfg, None)
    cfg = model.config
    assert model.host_input_specs() == {
        "ple_embeds": ((cfg.ple_embed_dim,), torch.float32)
    }
    ids = torch.randint(
        2, cfg.vocab_size, (12,), generator=torch.Generator().manual_seed(0)
    )
    ids[4] = cfg.eos_token_id
    with pytest.raises(RuntimeError, match="n-gram table"):
        model.build_host_inputs(
            ids, torch.zeros(12), torch.arange(12), np.zeros((1, 16))
        )

    emb = hf.model.layers[1].ple.ple_embedding
    hasher = NGramHasher.from_buffers(
        emb.layer_multipliers,
        emb.ngram_heads_vocab_sizes,
        emb.ngram_heads_offsets,
        emb.ngram_size,
        emb.heads_per_ngram,
    )
    table = HostNGramTable.from_tensor(
        emb.ngram_embedding.weight.detach(), num_shards=2
    )
    model._host_ple = HostPLEInputs(hasher, table, cfg.eos_token_id)
    # Chunk of positions 4..11 (4 tokens already in the history), then 4 padding rows.
    hist = np.zeros((1, 16), np.int32)
    hist[0, :4] = ids[:4].numpy()
    x = torch.cat([ids[4:], torch.zeros(4, dtype=torch.long)])
    rows = torch.tensor([0] * 8 + [-1] * 4)
    pos = torch.cat([torch.arange(4, 12), torch.zeros(4, dtype=torch.long)])
    out = model.build_host_inputs(x, rows, pos, hist)["ple_embeds"]
    assert out.shape == (12, cfg.ple_embed_dim)
    torch.testing.assert_close(out[:8], ple_rows(hf, ids)[4:], rtol=0, atol=0)
    assert not out[8:].any()

    cfg.ple_layer_ids = []  # a config without PLE takes no host inputs
    assert (
        model.host_input_specs() == {}
        and model.build_host_inputs(x, rows, pos, hist) == {}
    )
