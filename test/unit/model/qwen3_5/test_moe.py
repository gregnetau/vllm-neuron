# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 sparse MoE (qwen3_5_moe, e.g. Qwen3.5-35B-A3B) on CPU.

Config parsing and checkpoint key mapping against the published 35B-A3B config / safetensors
index (fixtures), expert device layouts, KV-head replication for TP > KV heads, TP / EP
partial sums of the MoE block, and a tiny random MoE model against HF
``Qwen3_5MoeForCausalLM``: padded prefill, then decode steps through the whole model.

The Neuron/vLLM pieces are stubbed as in test_model_smoke, except the MoE ops: the
plugin's own ``NF.router`` / ``build_blockwise_mapping`` / ``moe_cte`` / ``moe_block_tkg`` /
``moe_tkg`` run their PyTorch fallbacks, so the kernel arguments the model passes (routing
order, gate/up layout, affinity scaling, fused-norm gamma) are checked against HF.
"""

import copy
import json
import pathlib
import sys
import types

import pytest
import torch

hf = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
import test_model_smoke as smoke
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

HERE = pathlib.Path(__file__).parent
CONFIG = HERE / "qwen3_5_35b_a3b_config.json"
NAMES = HERE / "qwen3_5_35b_a3b_weight_names.json"
BLOCK, PAGE = 8, 5

_MOE_OPS = (
    "router",
    "build_blockwise_mapping",
    "moe_cte",
    "moe_block_tkg",
    "moe_tkg",
    "get_local_expert_affinities",
)


@pytest.fixture(scope="module")
def raw():
    return json.loads(CONFIG.read_text())


def _install_stubs_with_moe_ops():
    """test_model_smoke's stubs plus the plugin's real MoE ops (PyTorch fallbacks on CPU)."""
    nf = pytest.importorskip("vllm_neuron.functional")
    ops = {name: getattr(nf, name) for name in _MOE_OPS}
    mod = smoke._install_stubs()
    stub_nf = sys.modules["vllm_neuron.functional"]
    for name, fn in ops.items():
        setattr(stub_nf, name, fn)
    stub_nf.attention_decode = _attention_decode
    stub_nf.gen_attention_decode_mask = _gen_attention_decode_mask
    return mod


def _gen_attention_decode_mask(pos_ids, bs, s_active, **kw):
    return pos_ids.view(bs, s_active)  # the stubbed kernel below reads positions


def _attention_decode(
    X,
    W_qkv,
    rmsnorm_QK_pre_rope_eps,
    rmsnorm_QK_pre_rope_W_Q,
    rmsnorm_QK_pre_rope_W_K,
    cos,
    sin,
    active_blocks_table,
    K_cache,
    V_cache,
    attention_mask,
    softmax_scale,
    **kw,
):
    """Torch stand-in for NF.attention_decode with one KV head per rank, W_out=None:
    QKV projection, QK RMSNorm, full-width rotate_half RoPE, attention over the request's
    cached prefix + the new tokens. Returns (attn [B, Nh, Dh, S], K_new [B*S, Dh],
    V_new [B, 1, S, Dh])."""
    B, S, _ = X.shape
    Dh = K_cache.shape[-1]
    qkv = X.reshape(B * S, -1).float() @ W_qkv.float()
    nq = qkv.shape[1] // Dh - 2
    q = qkv[:, : nq * Dh].view(B, S, nq, Dh)
    k = qkv[:, nq * Dh : (nq + 1) * Dh].view(B, S, Dh)
    v = qkv[:, (nq + 1) * Dh :].view(B, S, Dh)

    def rms(x, w):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rmsnorm_QK_pre_rope_eps)
        return x * w.float().view(-1)

    def rope(x, c, s):
        h = x.shape[-1] // 2
        return x * c + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * s

    c = cos.float().permute(1, 2, 0)
    s = sin.float().permute(1, 2, 0)
    c, s = torch.cat([c, c], -1), torch.cat([s, s], -1)  # [B, S, Dh]
    q = rope(rms(q, rmsnorm_QK_pre_rope_W_Q), c[:, :, None], s[:, :, None])
    k = rope(rms(k, rmsnorm_QK_pre_rope_W_K), c, s)
    pos = attention_mask.long()
    out = torch.zeros(B, nq, Dh, S)
    for b in range(B):
        p0 = int(pos[b, 0])
        blocks = active_blocks_table[b].long()
        keys = torch.cat([K_cache[blocks].reshape(-1, Dh)[:p0].float(), k[b]])
        vals = torch.cat([V_cache[blocks].reshape(-1, Dh)[:p0].float(), v[b]])
        for i in range(S):
            n = p0 + i + 1
            p = torch.softmax(q[b, i] @ keys[:n].t() * softmax_scale, dim=-1)
            out[b, :, :, i] = p @ vals[:n]
    return out.to(X.dtype), k.reshape(B * S, Dh), v.view(B, 1, S, Dh)


def _tiny_hf_config(**over):
    kw = {
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "vocab_size": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 48,
        "dtype": "float32",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
    }
    kw.update(over)
    return Qwen3_5MoeTextConfig(**kw)


def _randomize(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.1))
            elif "linear_attn.norm" in n:
                p.copy_(torch.rand_like(p) + 0.5)
            elif "norm" in n:
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.15)


def _norm(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def _keys(mappings):
    out = set()
    for v in mappings.values():
        out |= {v} if isinstance(v, str) else set(v)
    return out


# ---------------------------------------------------------------------------
# Config, TP validation, checkpoint keys
# ---------------------------------------------------------------------------
def test_moe_config_from_real_checkpoint(qwen3_5_config, raw):
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    assert cfg.is_moe and cfg.num_hidden_layers == 40 and cfg.hidden_size == 2048
    assert (cfg.num_experts, cfg.num_experts_per_tok) == (256, 8)
    assert cfg.moe_intermediate_size == cfg.shared_expert_intermediate_size == 512
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (
        16,
        2,
        256,
    )
    assert (cfg.linear_num_key_heads, cfg.linear_num_value_heads) == (16, 32)
    assert len(cfg.full_attention_layers) == 10 and cfg.full_attention_layers[0] == 3
    assert cfg.mtp_num_hidden_layers == 1 and cfg.rotary_dim == 64
    assert cfg.torch_dtype == torch.bfloat16 and cfg.mamba_ssm_dtype == torch.float32
    assert not qwen3_5_config.Qwen3_5Config.from_configs(
        json.loads((HERE / "qwen3_8_27b_config.json").read_text())
    ).is_moe

    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )

    hf_cfg = Qwen3_5MoeConfig(**{k: v for k, v in raw.items() if k != "architectures"})
    assert qwen3_5_config.Qwen3_5Config.from_configs(hf_cfg) == cfg


def test_moe_tp_validation_and_kv_replication(qwen3_5_config, state, raw):
    cfg = qwen3_5_config.Qwen3_5Config.from_configs(raw)
    # TP=4 GDN state per layer and rank: conv [K-1, (2*16*128 + 32*128) / 4], 8 value heads
    assert state.gdn_state_shapes(cfg, 4) == ((3, 2048), (8, 128, 128))
    for tp in (1, 2, 4, 8):  # 2 KV heads: replicated at TP 4 and 8
        cfg.validate_tp(tp)
        assert cfg.num_kv_heads_per_rank(tp) == max(1, 2 // tp)
    with pytest.raises(ValueError):
        cfg.validate_tp(3)
    # TP=4: ranks 0-1 hold KV head 0 (query heads 0-7), ranks 2-3 KV head 1.
    assert [cfg.kv_head_shard(r, 4) for r in range(4)] == [
        (0, 2),
        (0, 2),
        (1, 2),
        (1, 2),
    ]
    assert [cfg.kv_head_shard(r, 2) for r in range(2)] == [(0, 2), (1, 2)]
    bad = copy.deepcopy(raw)
    bad["text_config"]["mlp_only_layers"] = [0]
    with pytest.raises(NotImplementedError):
        qwen3_5_config.Qwen3_5Config.from_configs(bad)
    with pytest.raises(ValueError):  # MoE requires the shared expert
        qwen3_5_config.Qwen3_5Config(num_experts=4, num_experts_per_tok=2)


def test_moe_fp8_only_applies_to_dense_mlp():
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    nc = NeuronConfig.from_dict({"quantization": "fp8"})
    cfg = Qwen3_5Config(
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        neuron_config=nc,
    )
    assert not cfg.fp8_quantized("layers.0.mlp")
    assert cfg.fp8_quantized("layers.0.self_attn")


def test_factory_rejects_fp8_and_unaligned_expert_shards(raw, monkeypatch):
    smoke._install_stubs()
    from conftest import ROOT, _load

    factory = _load(
        "vllm_neuron.model.qwen3_5.factory", ROOT / "model/qwen3_5/factory.py"
    )
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    cfg = Qwen3_5Config.from_configs(raw)
    validate = factory.Qwen3_5ForCausalLM._validate_moe
    validate(cfg, None)
    with pytest.raises(NotImplementedError, match="fp8"):
        validate(cfg, "fp8")
    monkeypatch.setattr(smoke._TP, "world_size", 8)  # 512 / 8 = 64 columns per rank
    with pytest.raises(ValueError, match="multiple of 128"):
        validate(cfg, None)


def test_moe_weight_mappings_match_real_checkpoint_index(raw):
    """Target and MTP draft map exactly the text-tower + ``mtp.*`` names of the published
    35B-A3B index (fused experts in the backbone, per-expert experts in the MTP layer)."""
    mod = smoke._install_stubs()
    from conftest import ROOT, _load

    mtp = _load("vllm_neuron.model.qwen3_5.mtp", ROOT / "model/qwen3_5/mtp.py")
    published = set(json.loads(NAMES.read_text()))
    with torch.device("meta"):
        model = mod.Qwen3_5ForCausalLM.from_configs(raw, None)
        draft = mtp.Qwen3_5MTP.from_configs(raw, model.config.num_hidden_layers)
    assert isinstance(model.model.layers[0].mlp, mod.Qwen3_5SparseMoeBlock)
    assert isinstance(draft.model.layers[0].mlp, mod.Qwen3_5SparseMoeBlock)
    target_keys, draft_keys = (
        _keys(model._weight_mappings()),
        _keys(draft._weight_mappings()),
    )
    assert target_keys == {
        n for n in published if n.startswith(mod.CKPT) or n == "lm_head.weight"
    }
    assert {k for k in draft_keys if k.startswith(mtp.MTP)} == {
        n for n in published if n.startswith(mtp.MTP)
    }
    assert draft_keys <= published
    params = dict(model.named_parameters())
    assert set(model._weight_mappings()) == set(params)
    assert set(draft._weight_mappings()) == set(dict(draft.named_parameters()))
    # TP=1 view of one layer's experts: [E, H, 2, I], [E, I, H]
    assert params["model.layers.0.mlp.experts_gate_up_weight"].shape == (
        256,
        2048,
        2,
        512,
    )
    assert params["model.layers.0.mlp.experts_down_weight"].shape == (256, 512, 2048)


# ---------------------------------------------------------------------------
# Device layouts and sharding
# ---------------------------------------------------------------------------
def test_expert_layouts_fused_per_expert_and_shards(weights):
    torch.manual_seed(0)
    E, H, inter, tp = 6, 16, 8, 2
    gate_up = torch.randn(E, 2 * inter, H)
    down = torch.randn(E, H, inter)
    full = weights.moe_gate_up_device(gate_up, 0, 1)
    assert full.shape == (E, H, 2, inter)
    for e in range(E):  # first I rows are gate, the next I are up (HF chunk(2, dim=-1))
        torch.testing.assert_close(full[e, :, 0], gate_up[e, :inter].t())
        torch.testing.assert_close(full[e, :, 1], gate_up[e, inter:].t())
    shards = [weights.moe_gate_up_device(gate_up, r, tp) for r in range(tp)]
    assert torch.equal(torch.cat(shards, dim=-1), full)
    down_full = weights.moe_down_device(down, 0, 1, chunk_experts=4)
    assert torch.equal(down_full, down.transpose(1, 2))
    down_shards = [
        weights.moe_down_device(down, r, tp, chunk_experts=4) for r in range(tp)
    ]
    assert torch.equal(torch.cat(down_shards, dim=1), down_full)

    # per-expert nn.Linear weights (MTP layer) give the same layout
    gates = [gate_up[e, :inter] for e in range(E)]
    ups = [gate_up[e, inter:] for e in range(E)]
    downs = [down[e] for e in range(E)]
    for r in range(tp):
        assert torch.equal(
            weights.moe_gate_up_device_per_expert(gates, ups, r, tp), shards[r]
        )
        assert torch.equal(
            weights.moe_down_device_per_expert(downs, r, tp), down_shards[r]
        )
    # checkpoint slices (get_shape + __getitem__) read only the shard
    assert torch.equal(
        weights.moe_gate_up_device(smoke.FakeSlice(gate_up), 1, tp), shards[1]
    )
    assert torch.equal(
        weights.moe_down_device(smoke.FakeSlice(down), 1, tp), down_shards[1]
    )


def test_replicated_kv_heads_attention_partial_sums(weights, qwen3_5_config):
    """TP=4 over 2 KV heads: each rank's K/V columns are its query heads' KV head, and the
    per-rank attention outputs through the o_proj shards sum to the unsharded layer."""
    torch.manual_seed(4)
    cfg = qwen3_5_config.Qwen3_5Config(
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        intermediate_size=64,
    )
    tp, H, hd, nh, nkv = 4, 32, 16, 8, 2
    q_proj = torch.randn(nh * 2 * hd, H) * 0.2
    k_proj, v_proj = torch.randn(nkv * hd, H) * 0.2, torch.randn(nkv * hd, H) * 0.2
    o_proj = torch.randn(H, nh * hd) * 0.2
    x = torch.randn(5, H)
    perm = weights.rotary_channel_permutation(hd, cfg.rotary_dim)

    def attend(
        q, k, v
    ):  # q [T, n, hd], k/v [T, hd]; causal, no rope (position-free check)
        s = torch.einsum("tnd,sd->nts", q, k) * hd**-0.5
        s = s.masked_fill(
            ~torch.tril(torch.ones(5, 5, dtype=torch.bool)), float("-inf")
        )
        return torch.einsum("nts,sd->tnd", s.softmax(-1), v)

    qg = (x @ q_proj.t()).view(5, nh, 2 * hd)
    k_all, v_all = (x @ k_proj.t()).view(5, nkv, hd), (x @ v_proj.t()).view(5, nkv, hd)
    ref = torch.cat(
        [
            attend(
                qg[:, g * 4 : (g + 1) * 4, :hd][..., perm],
                k_all[:, g][..., perm],
                v_all[:, g],
            )
            for g in range(nkv)
        ],
        dim=1,
    )  # [T, nh, hd], permuted q/k channels leave q.k unchanged
    ref = (ref * torch.sigmoid(qg[:, :, hd:])).reshape(5, -1) @ o_proj.t()

    parts = []
    for r in range(tp):
        qkv = x @ weights.attention_qkv_device(q_proj, k_proj, v_proj, cfg, r, tp)
        assert qkv.shape[1] == (nh // tp + 2) * hd  # one KV head per rank
        q = qkv[:, : 2 * hd].view(5, 2, hd)
        k, v = qkv[:, 2 * hd : 3 * hd], qkv[:, 3 * hd :]
        kv = r // 2
        torch.testing.assert_close(k, k_all[:, kv][..., perm])
        torch.testing.assert_close(v, v_all[:, kv])
        gate = x @ weights.attention_gate_device(q_proj, cfg, r, tp)
        o = attend(q, k, v).reshape(5, -1) * torch.sigmoid(gate)
        parts.append(o @ weights.attention_o_device(o_proj, cfg, r, tp))
    torch.testing.assert_close(sum(parts), ref, rtol=1e-4, atol=1e-5)


def _moe_block(mod, cfg):
    torch.manual_seed(11)
    m = mod.Qwen3_5SparseMoeBlock(cfg)
    hf_block = hf.Qwen3_5MoeSparseMoeBlock(
        _tiny_hf_config(
            hidden_size=cfg.hidden_size,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
        )
    ).eval()
    _randomize(hf_block)
    return m, hf_block


def _load_block(mod, m, hf_block, rank, tp, ep=1):
    """Load rank ``rank``'s shard (TP over the intermediate dims; with ``ep`` > 1 a contiguous
    expert range per EP partition, I sharded over the tp // ep ranks inside it) through the
    model's loaders, running the block as a one-rank module."""
    from conftest import ROOT, _load

    model_mod = _load(
        "vllm_neuron.model.qwen3_5.model", ROOT / "model/qwen3_5/model.py"
    )
    wl = sys.modules["vllm_neuron.utils.weight_loader"]
    ep_tp = tp // ep
    m.ep_degree, m.ep_rank = ep, rank // ep_tp
    m.num_local_experts = m.num_experts // ep
    m.tp_degree = ep_tp
    m.ep_tp_group = types.SimpleNamespace(world_size=ep_tp, rank_in_group=rank % ep_tp)
    m._ep_rank_tensor = lambda device: torch.tensor([m.ep_rank], dtype=torch.int32)
    model_mod._attach_moe_loaders(m, tp, fused_experts=True)
    sd = {f"mlp.{k}": v for k, v in hf_block.state_dict().items()}
    for name, keys in model_mod._moe_weight_mappings(m, "", "mlp.", True).items():
        keys = [keys] if isinstance(keys, str) else keys
        p = getattr(m, name)
        p.data = wl.get_weight_loader(p).load(
            [smoke.FakeSlice(sd[k]) for k in keys], rank
        )
    # One-rank forward: build the (identical) block mapping unsplit, without collectives.
    m.tp_degree = 1
    m.ep_tp_group = types.SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        rank_in_group_tensor=torch.zeros(1, dtype=torch.int32),
    )
    return m


@pytest.mark.parametrize("tp,ep", [(1, 1), (4, 1), (4, 2)])
def test_moe_block_shards_sum_to_hf(tp, ep):
    """Prefill (blockwise) and decode (fused tkg, and the >128-token blockwise fallback)
    partial outputs of every rank sum to HF Qwen3_5MoeSparseMoeBlock on the normed input."""
    mod = _install_stubs_with_moe_ops()
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    cfg = Qwen3_5Config.from_configs(_tiny_hf_config().to_dict())
    m, hf_block = _moe_block(mod, cfg)
    torch.manual_seed(2)
    x = torch.randn(12, cfg.hidden_size)
    ln_w = torch.rand(cfg.hidden_size) + 0.5
    normed = mod.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, torch.float32)
    normed.weight.data = ln_w
    with torch.no_grad():
        ref = hf_block(normed(x).unsqueeze(0))[0]
    mask = torch.ones(12, dtype=torch.bool)
    outs = {"prefill": [], "decode": [], "decode_big": []}
    for r in range(tp):
        blk = _load_block(mod, copy.deepcopy(m), hf_block, r, tp, ep)
        with torch.no_grad():
            outs["prefill"].append(blk(x, is_prefill=True, ln_w=ln_w, token_mask=mask))
            outs["decode"].append(blk(x, is_prefill=False, ln_w=ln_w))
            mod_moe = sys.modules["vllm_neuron.model.qwen3_5.moe"]
            big, mod_moe.MAX_TKG_TOKENS = mod_moe.MAX_TKG_TOKENS, 4
            try:
                outs["decode_big"].append(blk(x, is_prefill=False, ln_w=ln_w))
            finally:
                mod_moe.MAX_TKG_TOKENS = big
    for name, parts in outs.items():
        torch.testing.assert_close(sum(parts), ref, rtol=1e-4, atol=1e-5, msg=name)

    # padding tokens route to no expert: only the shared expert remains for them
    blk = _load_block(mod, copy.deepcopy(m), hf_block, 0, 1)
    mask[8:] = False
    with torch.no_grad():
        out = blk(x, is_prefill=True, ln_w=ln_w, token_mask=mask)
        shared = blk._shared_expert(normed(x))
    torch.testing.assert_close(out[:8], ref[:8], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out[8:], shared[8:], rtol=1e-4, atol=1e-5)


def test_moe_block_top_k_above_kernel_limit_routes_in_torch():
    """top-k > 8 (397B-A17B) decodes through the torch router + moe_tkg."""
    mod = _install_stubs_with_moe_ops()
    from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

    hf_cfg = _tiny_hf_config(num_experts=16, num_experts_per_tok=10)
    cfg = Qwen3_5Config.from_configs(hf_cfg.to_dict())
    m, hf_block = _moe_block(mod, cfg)
    blk = _load_block(mod, m, hf_block, 0, 1)
    x, ln_w = torch.randn(3, cfg.hidden_size), torch.ones(cfg.hidden_size)
    with torch.no_grad():
        ref = hf_block(_norm(x, ln_w, cfg.rms_norm_eps).unsqueeze(0))[0]
        out = blk(x, is_prefill=False, ln_w=ln_w)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Whole model vs HF: padded prefill + decode steps
# ---------------------------------------------------------------------------
def test_moe_model_prefill_and_decode_match_hf():
    torch.manual_seed(7)
    mod = _install_stubs_with_moe_ops()
    from conftest import ROOT, _load

    state_cache = _load("vllm_neuron.model.state_cache", ROOT / "model/state_cache.py")
    wl = sys.modules["vllm_neuron.utils.weight_loader"]

    hf_cfg = _tiny_hf_config()
    hf_model = hf.Qwen3_5MoeForCausalLM(hf_cfg).eval()
    _randomize(hf_model)
    sd = {
        (
            k.replace("model.", "model.language_model.", 1)
            if k.startswith("model.")
            else k
        ): v
        for k, v in hf_model.state_dict().items()
    }
    model = mod.Qwen3_5ForCausalLM.from_configs(hf_cfg.to_dict(), None)
    cfg = model.config
    assert cfg.is_moe and all(layer.is_moe for layer in model.model.layers)

    mappings = model._weight_mappings()
    params = dict(model.named_parameters())
    assert set(mappings) == set(params), set(mappings) ^ set(params)
    for name, keys in mappings.items():
        keys = [keys] if isinstance(keys, str) else keys
        loaded = wl.get_weight_loader(params[name]).load(
            [smoke.FakeSlice(sd[k]) for k in keys], 0
        )
        assert loaded.shape == params[name].shape, name
        params[name].data = loaded.to(params[name].dtype)

    spec = model.get_kv_spec()
    nblocks = 8
    model.bind_kv_cache(
        {
            l.name: [
                torch.zeros(nblocks, l.num_kv_heads, BLOCK, l.head_size)
                for _ in range(2)
            ]
            for l in spec.layers
        }
    )
    states = {}
    for sl in spec.state_layers:
        page_bytes = state_cache.page_layout(sl.shapes, sl.dtypes).page_bytes + 32
        raw = torch.zeros(10 * page_bytes, dtype=torch.int8)
        states[sl.name] = list(state_cache.paired_half_views(raw, page_bytes))
    model.bind_state_cache(states)

    blocks = torch.tensor([[2, 3, 4, 6]], dtype=torch.int32)
    page = torch.tensor([[PAGE]], dtype=torch.int32)

    def md_for(slot, cached, decode):
        common = {
            "max_query_len": 1 if decode else slot.shape[0],
            "decode_token_threshold": 1,
            "block_size": BLOCK,
            "max_blocks_per_seq": blocks.shape[1],
            "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32),
            "kv_segment_size": 0,
            "slot_mapping": slot,
        }
        return {
            mod._mixer_key(cfg, i): dict(
                common,
                block_table_tensor=blocks if t == "full_attention" else page,
            )
            for i, t in enumerate(cfg.layer_types)
        }

    def slots(pos):
        return blocks[0][pos // BLOCK].long() * BLOCK + pos % BLOCK

    n_real, T, n_decode = 13, 16, 4
    ids = torch.randint(1, 128, (n_real + n_decode,))
    with torch.no_grad():
        ref = hf_model(ids.unsqueeze(0)).logits[0]

    padded = torch.cat([ids[:n_real], torch.zeros(T - n_real, dtype=torch.long)])
    slot = torch.full((T,), -1, dtype=torch.long)
    slot[:n_real] = slots(torch.arange(n_real))
    logits = model(
        padded,
        positions=torch.arange(T),
        attn_metadata=md_for(slot, 0, False),
        sampling_positions=torch.tensor([n_real - 1]),
    )
    torch.testing.assert_close(logits[0], ref[n_real - 1], rtol=2e-3, atol=2e-3)

    for p in range(n_real, n_real + n_decode):
        pos = torch.tensor([p])
        logits = model(
            ids[p : p + 1],
            positions=pos,
            attn_metadata=md_for(slots(pos), p, True),
            sampling_positions=torch.tensor([0]),
        )
        torch.testing.assert_close(logits[0], ref[p], rtol=2e-3, atol=2e-3)


def test_mtp_layer_builds_moe_ffn_with_per_expert_loaders():
    """The MTP draft's decoder layer gets the MoE FFN; its per-expert checkpoint weights load
    to the same device layout as the backbone's fused experts."""
    mod = _install_stubs_with_moe_ops()
    from conftest import ROOT, _load

    mtp = _load("vllm_neuron.model.qwen3_5.mtp", ROOT / "model/qwen3_5/mtp.py")
    wl = sys.modules["vllm_neuron.utils.weight_loader"]
    hf_cfg = _tiny_hf_config()
    draft = mtp.Qwen3_5MTP.from_configs(hf_cfg.to_dict(), hf_cfg.num_hidden_layers)
    moe = draft.model.layers[0].mlp
    assert isinstance(moe, mod.Qwen3_5SparseMoeBlock)
    torch.manual_seed(5)
    E, inter, H = hf_cfg.num_experts, hf_cfg.moe_intermediate_size, hf_cfg.hidden_size
    gate_up, down = torch.randn(E, 2 * inter, H), torch.randn(E, H, inter)
    sd = {}
    for e in range(E):
        sd[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate_up[e, :inter]
        sd[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = gate_up[e, inter:]
        sd[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = down[e]
    mappings = draft._weight_mappings()
    W = sys.modules["vllm_neuron.model.qwen3_5.weights"]
    for name, expected in (
        ("experts_gate_up_weight", W.moe_gate_up_device(gate_up, 0, 1)),
        ("experts_down_weight", W.moe_down_device(down, 0, 1)),
    ):
        keys = mappings[f"model.layers.0.mlp.{name}"]
        p = getattr(moe, name)
        loaded = wl.get_weight_loader(p).load([smoke.FakeSlice(sd[k]) for k in keys], 0)
        assert torch.equal(loaded, expected)


def test_moe_model_traces_on_meta():
    """Trace safety (see test_meta_trace): MoE prefill and decode run on the meta device."""
    mod = _install_stubs_with_moe_ops()
    state_cache = sys.modules["vllm_neuron.model.state_cache"]
    model = mod.Qwen3_5ForCausalLM.from_configs(_tiny_hf_config().to_dict(), None).to(
        "meta"
    )
    spec = model.get_kv_spec()
    model.bind_kv_cache(
        {
            l.name: [
                torch.zeros(8, l.num_kv_heads, BLOCK, l.head_size, device="meta")
                for _ in range(2)
            ]
            for l in spec.layers
        }
    )
    halves = {}
    for sl in spec.state_layers:
        page_bytes = state_cache.page_layout(sl.shapes, sl.dtypes).page_bytes + 32
        raw = torch.zeros(10 * page_bytes, dtype=torch.int8, device="meta")
        halves[sl.name] = list(state_cache.paired_half_views(raw, page_bytes))
    model.bind_state_cache(halves)
    T = 16

    def md(decode):
        n = 1 if decode else T
        return {
            mod._mixer_key(model.config, i): {
                "max_query_len": n,
                "decode_token_threshold": 1,
                "block_size": BLOCK,
                "max_blocks_per_seq": 4,
                "kv_segment_size": 0,
                "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32, device="meta"),
                "slot_mapping": torch.zeros(n, dtype=torch.long, device="meta"),
                "block_table_tensor": torch.zeros(
                    1, 4, dtype=torch.int32, device="meta"
                ),
            }
            for i in range(model.config.num_hidden_layers)
        }

    meta = lambda *shape, dtype=torch.float32: torch.zeros(
        *shape, dtype=dtype, device="meta"
    )
    logits = model(
        meta(T, dtype=torch.long),
        positions=torch.arange(T, device="meta"),
        attn_metadata=md(False),
        sampling_positions=meta(1, dtype=torch.long),
    )
    assert logits.device.type == "meta" and logits.shape == (1, 128)
    with torch.no_grad():
        out = model.model.layers[0].mlp(meta(2, 64), is_prefill=False, ln_w=meta(64))
    assert out.shape == (2, 64)
