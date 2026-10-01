# SPDX-License-Identifier: Apache-2.0
"""CPU smoke test of the Neuron model class with the Neuron/vLLM pieces stubbed in torch.

Exercises: construction, weight mappings + loaders (names vs params), kv/state spec + binding,
a padded-bucket prefill vs the reference model (logits AND recurrent state pages), and GDN decode.
NKI-backed ops (NF.*) are replaced by plain-torch equivalents; the real kernels are not covered.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

BLOCK, PAGE = 8, 5  # attention block size; state page id


class _TP:
    world_size, rank_in_group, device_group = 1, 0, None


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    from conftest import ROOT, _load, _stub_packages

    _stub_packages()
    for pkg in ("vllm", "vllm.distributed"):
        _stub(pkg)
    _stub("vllm.distributed.parallel_state", get_tp_group=lambda: _TP())
    for name, path in (("vllm_neuron.utils", ROOT / "utils"), ("vllm_neuron.nn", ROOT / "nn")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

    def qkv_proj(hidden, qkv_weights, bias=None):
        return hidden @ qkv_weights

    def o_proj(active, weight, bias=None):
        if active.dim() == 4:  # [B, N, D, S]
            B, N, D, S = active.shape
            active = active.reshape(B, N * D, S).transpose(1, 2)
        return active @ weight

    def flash_attention(q, k, v, scale=None, tp_q=True, tp_out=False, **kw):
        # tp_q False: q [B, D, S]; k [B, D, S]; v [B, S, D]. Returns [B, D, S] for tp_out.
        S = q.shape[-1]
        scores = torch.einsum("bds,bdt->bst", q.float(), k.float()) * scale
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
        p = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        o = torch.einsum("bst,btd->bds", p, v.float()).to(q.dtype)
        return o

    def mlp(x, gate, up, down):
        return (F.silu(x @ gate) * (x @ up)) @ down

    _stub("vllm_neuron.functional", qkv_proj=qkv_proj, o_proj=o_proj, flash_attention=flash_attention,
          mlp=mlp, merge_prompt_embeds=lambda h, e, m: h)

    class Embed(torch.nn.Module):
        def __init__(self, vocab_size, embed_dim, dtype, tp_group=None):
            super().__init__()
            self.vocab_size_per_rank = vocab_size
            self.weight = torch.nn.Parameter(torch.empty(vocab_size, embed_dim, dtype=dtype))

        def forward(self, ids, scatter_tokens=False, rank=None):
            return F.embedding(ids, self.weight)

    class ColPar(torch.nn.Module):
        def __init__(self, i, o, bias=False, dtype=None, gather_output=True, tp_group=None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(o, i, dtype=dtype))

        def forward(self, x):
            return x @ self.weight.t()

    _stub("vllm_neuron.nn", ColumnParallelLinear=ColPar)
    _stub("vllm_neuron.nn.embedding", VocabDimShardedEmbedding=Embed)
    _stub("vllm_neuron.nn.sampler", Sampler=object)
    _stub("vllm_neuron.utils.checkpoints", SafetensorsCheckpoint=object)
    _stub("vllm_neuron.utils.dtype_utils", FP8_CLAMP_MAX=240.0)
    _load("vllm_neuron.utils.weight_loader", ROOT / "utils/weight_loader.py")
    _load("vllm_neuron.model.kv_cache", ROOT / "model/kv_cache.py")
    _load("vllm_neuron.model.neuron_config", ROOT / "model/neuron_config.py")
    _load("vllm_neuron.model.state_cache", ROOT / "model/state_cache.py")
    return _load("vllm_neuron.model.qwen3_5.model", ROOT / "model/qwen3_5/model.py")


class FakeSlice:
    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        return self.t[idx]


def test_model_smoke_prefill_padded_and_gdn_decode():
    torch.manual_seed(7)
    mod = _install_stubs()
    from conftest import ROOT, _load

    ref_mod = _load("vllm_neuron.model.qwen3_5.reference", ROOT / "model/qwen3_5/reference.py")
    state_cache = sys.modules["vllm_neuron.model.state_cache"]
    wl = sys.modules["vllm_neuron.utils.weight_loader"]

    hf_cfg = Qwen3_5TextConfig(
        hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=96, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=128, dtype="float32",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    hf_model = hf.Qwen3_5ForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        for n, p in hf_model.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.1))
            elif "linear_attn.norm" in n:
                p.copy_(torch.rand_like(p) + 0.5)
            elif "norm" in n:
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    sd = {(k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k): v
          for k, v in hf_model.state_dict().items()}

    model = mod.Qwen3_5ForCausalLM.from_configs(hf_cfg.to_dict(), None)
    cfg = model.config
    assert cfg.torch_dtype == torch.float32

    # ---- weight mappings + loaders: every param mapped, every key exists, shapes match
    mappings = model._weight_mappings()
    params = dict(model.named_parameters())
    assert set(mappings) == set(params), (set(mappings) ^ set(params))
    for name, keys in mappings.items():
        keys = [keys] if isinstance(keys, str) else keys
        loaded = wl.get_weight_loader(params[name]).load([FakeSlice(sd[k]) for k in keys], 0)
        assert loaded.shape == params[name].shape, (name, loaded.shape, params[name].shape)
        params[name].data = loaded.to(params[name].dtype)

    # ---- specs + caches
    spec = model.get_kv_spec()
    assert len(spec.layers) == 2 and len(spec.state_layers) == 6
    nblocks = 8
    kv = {}
    for l in spec.layers:
        kv[l.name] = [torch.zeros(nblocks, l.num_kv_heads, BLOCK, l.head_size) for _ in range(2)]
    model.bind_kv_cache(kv)
    states = {}
    for sl in spec.state_layers:
        layout = state_cache.page_layout(sl.shapes, sl.dtypes)
        page_bytes = layout.page_bytes + 32  # padded page, as vLLM does
        raw = torch.zeros(10 * page_bytes, dtype=torch.int8)
        states[sl.name] = list(state_cache.paired_half_views(raw, page_bytes))
        state_cache.state_half_elems(layout, page_bytes)
    model.bind_state_cache(states)
    page_idx = torch.tensor([PAGE])

    def read(i):
        conv, ssm = model.model.layers[i].linear_attn._read_state(page_idx)
        return conv[0], ssm[0]

    # ---- padded-bucket prefill (21 real tokens in a 32-token bucket)
    n_real, T = 21, 32
    ids = torch.randint(1, 128, (n_real,))
    padded = torch.cat([ids, torch.zeros(T - n_real, dtype=torch.long)])
    blocks = torch.tensor([[2, 3, 4, 6]], dtype=torch.int32)
    slot = torch.full((T,), -1, dtype=torch.long)
    pos = torch.arange(n_real)
    slot[:n_real] = blocks[0][pos // BLOCK].long() * BLOCK + pos % BLOCK

    def md_for(is_decode, cached=0, slots=None, table=None, T_=T):
        common = dict(max_query_len=1 if is_decode else T_, decode_token_threshold=1, block_size=BLOCK,
                      max_blocks_per_seq=4, cached_seq_len=torch.tensor([[cached]], dtype=torch.int32),
                      kv_segment_size=0)
        md = {}
        for i, t in enumerate(cfg.layer_types):
            if t == "full_attention":
                md[f"layers.{i}.self_attn"] = dict(common, slot_mapping=slot, block_table_tensor=blocks)
            else:
                tbl = torch.tensor([[PAGE]], dtype=torch.int32) if table is None else table
                md[f"layers.{i}.linear_attn"] = dict(common, slot_mapping=slot if slots is None else slots,
                                                     block_table_tensor=tbl)
        return md

    # dirty pages (recycled): new-request prefill must ignore them
    for halves in states.values():
        for h in halves:
            h[PAGE].fill_(123.0)

    logits = model(padded, torch.arange(T), attn_metadata=md_for(False),
                   sampling_positions=torch.tensor([n_real - 1]))
    ref = ref_mod.Qwen3_5Reference(cfg, sd, prefix="model.language_model.", lm_head="lm_head.weight")
    ref_logits = ref.forward(ids)
    torch.testing.assert_close(logits[0], ref_logits[-1], rtol=2e-3, atol=2e-3)

    # recurrent state pages hold the state after the last *valid* token
    for i, t in enumerate(cfg.layer_types):
        if t != "linear_attn" and t != "linear_attention":
            continue
        conv_ref, ssm_ref = ref.state[i]
        conv_page, ssm_page = read(i)
        torch.testing.assert_close(conv_page, conv_ref.float(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(ssm_page, ssm_ref, rtol=1e-3, atol=1e-4)

    # ---- GDN decode step on layer 0 vs the reference block, continuing from that state
    layer = model.model.layers[0].linear_attn
    lw = {n: sd[f"model.language_model.layers.0.linear_attn.{n}.weight"]
          for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "out_proj")}
    lw.update(A_log=sd["model.language_model.layers.0.linear_attn.A_log"],
              dt_bias=sd["model.language_model.layers.0.linear_attn.dt_bias"],
              norm=sd["model.language_model.layers.0.linear_attn.norm.weight"])
    x = torch.randn(1, cfg.hidden_size)
    conv_ref, ssm_ref = ref.state[0]
    expected, conv_new, ssm_new = ref_mod.gdn_ops.gdn_layer_reference(x, lw, cfg, conv_ref, ssm_ref, decode=True)
    md = {"layers.0.linear_attn": dict(max_query_len=1, decode_token_threshold=1,
                                       block_table_tensor=torch.tensor([[PAGE]], dtype=torch.int32))}
    with torch.no_grad():
        out = layer(x, None, None, md)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-4)
    conv_page, ssm_page = read(0)
    torch.testing.assert_close(conv_page, conv_new.float(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ssm_page, ssm_new, rtol=1e-3, atol=1e-4)


def test_state_row_layout_matches_flat_layout():
    """bind_state's [pages, rows, 128] view must read/write the same bytes as the flat [pages, half] path."""
    mod = _install_stubs()
    G = mod.Qwen3_5GatedDeltaNet
    g = types.SimpleNamespace(conv_dim=128, K=4, Hv=2, Dv=16, Dk=128)
    g.conv_numel, g.ssm_numel = g.conv_dim * (g.K - 1), g.Hv * g.Dv * g.Dk
    g.state_numel = g.conv_numel + g.ssm_numel
    for m in ("bind_state", "_read_state", "_write_state"):
        setattr(g, m, getattr(G, m).__get__(g))
    half = 18 * 128  # padded: 2*half = 4608 > state_numel = 4480
    slots = torch.tensor([3, 1])
    conv = torch.randn(2, g.conv_dim, g.K - 1)
    ssm = torch.randn(2, g.Hv, g.Dk, g.Dv)

    def run(row_width):
        g._STATE_ROW = row_width
        a, b = torch.zeros(5, half), torch.zeros(5, half)
        g.bind_state(a, b)
        assert g.state_a.dim() == (3 if row_width == 128 else 2)
        g._write_state(slots, conv, ssm)
        return a, b, g._read_state(slots)

    a_row, b_row, (c_row, s_row) = run(128)
    a_flat, b_flat, (c_flat, s_flat) = run(7)  # 7 does not divide -> flat path
    assert torch.equal(a_row, a_flat) and torch.equal(b_row, b_flat)
    assert torch.equal(c_row, conv) and torch.equal(s_row, ssm)
    assert torch.equal(c_flat, conv) and torch.equal(s_flat, ssm)


def test_attend_cached_torch_matches_causal_attention_and_ignores_stale_nan():
    """Segmented-prefill fallback (head_dim > 128): chunk queries over cached + new keys."""
    mod = _install_stubs()
    A = mod.Qwen3_5Attention
    torch.manual_seed(3)
    nkh, groups, Dh, bs, nblk = 1, 3, 16, 4, 6
    a = types.SimpleNamespace(num_key_value_heads_per_rank=nkh, num_key_value_groups=groups, scaling=Dh ** -0.5)
    a._attend_cached_torch = A._attend_cached_torch.__get__(a)
    prior, T, n_valid = 5, 6, 4                 # 5 cached tokens, 6-row bucket with 4 real tokens
    a.k_cache = torch.full((10, nkh, bs, Dh), float("nan"))
    a.v_cache = torch.full((10, nkh, bs, Dh), float("nan"))
    blocks = torch.tensor([7, 2, 9, 4, 0, 0])   # table padded with the null block
    L = prior + n_valid
    k_all, v_all = torch.randn(L, Dh), torch.randn(L, Dh)
    for pos in range(L):
        a.k_cache[blocks[pos // bs], 0, pos % bs] = k_all[pos]
        a.v_cache[blocks[pos // bs], 0, pos % bs] = v_all[pos]
    q = torch.randn(nkh * groups, T, Dh)
    slot = torch.zeros(T, dtype=torch.long); slot[:n_valid] = 1
    md = {"block_table_tensor": blocks.reshape(1, -1), "cached_seq_len": torch.tensor([prior]),
          "slot_mapping": slot}
    out = a._attend_cached_torch(q, md, bs)     # [Nh, Dh, T]
    assert torch.isfinite(out).all()
    for i in range(n_valid):
        s = (q[:, i] @ k_all[: prior + i + 1].t()) * a.scaling
        ref = torch.softmax(s, -1) @ v_all[: prior + i + 1]
        torch.testing.assert_close(out[:, :, i], ref, rtol=1e-4, atol=1e-5)
