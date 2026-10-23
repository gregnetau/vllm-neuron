# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Qwen3.8-Flash-Next sparse MoE (``vllm_neuron/model/qwen4_exp/moe.py``).

The oracle is HF ``Qwen4ExpTextSparseMoeBlock`` from the seeded tiny model. Each test builds the
plugin module per simulated rank through its own weight loaders (checkpoint tensors stand in
for safetensors slices), runs the dense torch path (the kernels' CPU fallback), and sums the
per-rank partials, which is what the reduce-scatter / all-reduce does on device.
"""

import dataclasses
import os

import pytest
import torch

from vllm_neuron.model.qwen4_exp import moe as M

pytest_plugins = ["qwen4_exp_testlib"]  # fixtures

# Optional real checkpoint; the test that reads it skips when it is not available.
CKPT_DIR = os.environ.get("QWEN4_EXP_MODEL_PATH", "")


class _FakeTP:
    """Single-process stand-in for one rank of the TP group; collectives are recorded so the
    test can combine partials across simulated ranks."""

    def __init__(self, world_size: int, rank: int, gathered=None):
        self.world_size, self.rank_in_group = world_size, rank
        self.rank_in_group_tensor = torch.tensor([rank], dtype=torch.int32)
        self.gathered = gathered

    def all_gather(self, x, dim=0):
        assert self.gathered is not None and dim == 0
        return self.gathered

    def reduce_scatter(self, x, dim=0):
        return x  # partial over the full sequence; the test sums and slices

    def all_reduce(self, x):
        return x


class _RecordingSlice:
    """PySafeSlice stand-in over a tensor that records every index it is read with."""

    def __init__(self, t: torch.Tensor):
        self.t, self.reads = t, []

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, key):
        self.reads.append(key)
        return self.t[key].clone()


def _hf_block(tiny_hf_model, dtype):
    blk = tiny_hf_model.model.layers[0].mlp
    return blk.to(dtype)


def _hf_tensors(blk) -> dict:
    sd = {k: v.detach().clone() for k, v in blk.state_dict().items()}
    return {  # checkpoint key suffix -> tensor
        "gate.weight": sd["gate.weight"],
        "shared_expert_gate.weight": sd["shared_expert_gate.weight"],
        "shared_expert.gate_proj.weight": sd["shared_expert.gate_proj.weight"],
        "shared_expert.up_proj.weight": sd["shared_expert.up_proj.weight"],
        "shared_expert.down_proj.weight": sd["shared_expert.down_proj.weight"],
        "experts.gate_up_proj": sd["experts.gate_up_proj"],
        "experts.down_proj": sd["experts.down_proj"],
    }


def _cfg(qwen4_config, dtype, fp8=False):
    nc = None
    if fp8:
        from vllm_neuron.model.neuron_config import NeuronConfig

        nc = NeuronConfig(quantization="fp8")
    return dataclasses.replace(qwen4_config, torch_dtype=dtype, neuron_config=nc)


def _build(monkeypatch, cfg, tensors, tp, rank, gathered=None, slices=None):
    """One rank's module, loaded through attach_moe_loaders / moe_weight_mappings."""
    group = _FakeTP(tp, rank, gathered)
    monkeypatch.setattr(M, "get_tp_group", lambda: group)
    moe = M.Qwen4ExpSparseMoe(cfg, layer_idx=0)
    M.attach_moe_loaders(moe, cfg, tp)
    mapping = M.moe_weight_mappings(moe, "", "")
    params = dict(moe.named_parameters())
    assert set(mapping) == set(params)
    state = {}
    for name, keys in mapping.items():
        keys = [keys] if isinstance(keys, str) else keys
        src = [(slices or tensors)[k] for k in keys]
        state[name] = params[name].weight_loader.load(src, rank).to(params[name].dtype)
        assert state[name].shape == params[name].shape, name
    moe.load_state_dict(state, strict=True, assign=True)
    return moe


def _plugin_output(monkeypatch, cfg, tensors, x, tp, is_prefill):
    out = 0
    for rank in range(tp):
        moe = _build(monkeypatch, cfg, tensors, tp, rank, gathered=x)
        inp = (
            x[rank * x.shape[0] // tp : (rank + 1) * x.shape[0] // tp]
            if is_prefill
            else x
        )
        with torch.no_grad():
            out = out + moe(inp, is_prefill=is_prefill).float()
    return out


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_hf(monkeypatch, tiny_hf_model, qwen4_config, dtype, tp, is_prefill):
    torch.manual_seed(0)
    blk = _hf_block(tiny_hf_model, dtype)
    tensors = _hf_tensors(blk)
    cfg = _cfg(qwen4_config, dtype)
    x = torch.randn(24, cfg.hidden_size).to(dtype)
    with torch.no_grad():
        ref = blk(x.unsqueeze(0)).squeeze(0).float()
    out = _plugin_output(monkeypatch, cfg, tensors, x, tp, is_prefill)
    # Max error relative to the output scale (HF-init outputs are ~3e-2).
    err = (out - ref).abs().max() / ref.abs().max()
    if dtype == torch.float32:
        assert err < 1e-5, err
    else:  # per-rank bf16 partials summed in fp32 vs HF's bf16 index_add order
        assert err < 1e-2, err


def test_routing_matches_hf(monkeypatch, tiny_hf_model, qwen4_config):
    """Local dense affinities across ranks reassemble HF's (renormalised) top-k weights."""
    blk = _hf_block(tiny_hf_model, torch.float32)
    tensors = _hf_tensors(blk)
    cfg = _cfg(qwen4_config, torch.float32)
    x = torch.randn(16, cfg.hidden_size)
    with torch.no_grad():
        _, w, idx = blk.gate(x)
    ref = torch.zeros(16, cfg.num_experts).scatter(1, idx, w)
    tp = 4
    parts = []
    for rank in range(tp):
        moe = _build(monkeypatch, cfg, tensors, tp, rank)
        with torch.no_grad():
            aff, local, is_local = moe._route(x)
        assert is_local.sum(1).le(cfg.num_experts_per_tok).all()
        assert (local[is_local] >= 0).all() and (local[is_local] < 4).all()
        parts.append(aff)
    torch.testing.assert_close(torch.cat(parts, 1), ref, atol=1e-6, rtol=0)
    torch.testing.assert_close(ref.sum(1), torch.ones(16), atol=1e-5, rtol=0)


def test_selective_filler_index(monkeypatch, tiny_hf_model, qwen4_config):
    """The decode-selective remap: non-local slots point at a local expert with zero affinity,
    so the kernel's per-slot affinity gather reproduces the dense local affinities."""
    blk = _hf_block(tiny_hf_model, torch.float32)
    cfg = _cfg(qwen4_config, torch.float32)
    moe = _build(monkeypatch, cfg, _hf_tensors(blk), 2, 1)
    x = torch.randn(3, cfg.hidden_size)
    aff, local, is_local = moe._route(x)
    filler = torch.argmax((aff == 0).float(), -1, keepdim=True).to(torch.int32)
    index = torch.where(is_local, local, filler)
    assert (index >= 0).all() and (index < moe.num_local_experts).all()
    gathered = torch.gather(aff, 1, index.long())  # what the kernel scales each slot by
    recon = torch.zeros_like(aff).scatter_add(1, index.long(), gathered)
    # a filler expert gathers 0, a real one its affinity exactly once
    assert (gathered[~is_local] == 0).all()
    torch.testing.assert_close(recon, aff)


def test_fp8_round_trip_and_output(monkeypatch, tiny_hf_model, qwen4_config):
    blk = _hf_block(tiny_hf_model, torch.bfloat16)
    tensors = _hf_tensors(blk)
    # Per-output-channel quantization: per-element error <= half an e4m3 ulp (2^-4 relative)
    # above the subnormal range; whole-tensor relative error a few percent.
    w = tensors["experts.gate_up_proj"].float()
    q, s = M.fp8_quantize_per_channel(w, reduce_dim=2)
    assert q.dtype == torch.float8_e4m3fn and s.shape == w.shape[:2]
    assert q.float().abs().max() <= M.FP8_MAX
    deq = q.float() * s.unsqueeze(-1)
    normal = w.abs() >= s.unsqueeze(-1) * 2.0**-6
    rel = ((deq - w).abs() / w.abs().clamp(min=1e-30))[normal]
    assert rel.max() <= 2.0**-4 + 1e-6, rel.max()
    fro = (deq - w).norm() / w.norm()
    assert fro < 0.03, fro
    # amax channels are exact
    torch.testing.assert_close(deq.abs().amax(-1), w.abs().amax(-1), rtol=1e-6, atol=0)

    # MoE output: FP8 experts vs the BF16 module, through the loaders (tp=2).
    cfg8, cfg16 = (
        _cfg(qwen4_config, torch.bfloat16, fp8=True),
        _cfg(qwen4_config, torch.bfloat16),
    )
    moe8 = _build(monkeypatch, cfg8, tensors, 2, 1)
    assert moe8.fp8 and moe8.experts_gate_up_weight.dtype == torch.float8_e4m3fn
    assert moe8.experts_gate_up_scale.shape == (8, 2, cfg8.moe_intermediate_size)
    assert moe8.experts_down_scale.shape == (8, cfg8.hidden_size)
    assert (
        moe8.shared_gate_up_weight.dtype == torch.bfloat16
    )  # shared expert stays BF16
    x = torch.randn(24, cfg8.hidden_size).to(torch.bfloat16)
    out8 = _plugin_output(monkeypatch, cfg8, tensors, x, 2, False)
    out16 = _plugin_output(monkeypatch, cfg16, tensors, x, 2, False)
    err = (out8 - out16).norm() / out16.norm()
    assert err < 0.05, err


def test_loader_reads_only_local_experts(monkeypatch, tiny_hf_model, qwen4_config):
    blk = _hf_block(tiny_hf_model, torch.bfloat16)
    tensors = _hf_tensors(blk)
    cfg = _cfg(qwen4_config, torch.bfloat16, fp8=True)
    tp, rank, E_L = 4, 2, cfg.num_experts // 4
    slices = {k: _RecordingSlice(v) for k, v in tensors.items()}
    moe = _build(monkeypatch, cfg, tensors, tp, rank, slices=slices)
    for key in ("experts.gate_up_proj", "experts.down_proj"):
        reads = slices[key].reads
        # weight + scale share one read through the quantization cache
        assert reads == [slice(rank * E_L, (rank + 1) * E_L)], reads
    ref = M.experts_gate_up_device(
        tensors["experts.gate_up_proj"][rank * E_L : (rank + 1) * E_L]
    )
    deq = moe.experts_gate_up_weight.float() * moe.experts_gate_up_scale.unsqueeze(1)
    assert ((deq - ref.float()).norm() / ref.float().norm()) < 0.03


def _ckpt_file(key: str):
    """Checkpoint shard holding ``key``, or None if the checkpoint is not available."""
    import json

    index = os.path.join(CKPT_DIR, "model.safetensors.index.json")
    if not CKPT_DIR or not os.path.exists(index):
        return None
    with open(index) as f:
        path = os.path.join(CKPT_DIR, json.load(f)["weight_map"][key])
    return path if os.path.exists(path) else None


GATE_UP_KEY = "model.language_model.layers.0.mlp.experts.gate_up_proj"


@pytest.mark.skipif(
    _ckpt_file(GATE_UP_KEY) is None, reason="real checkpoint shard not available"
)
def test_real_checkpoint_slice_layout():
    """Real layer-0 expert slice at TP=8 rank 3 (lazy slice read): the layout transform matches
    direct indexing (gate rows first, then up rows)."""
    from safetensors import safe_open

    with safe_open(_ckpt_file(GATE_UP_KEY), "pt") as f:
        s = f.get_slice(GATE_UP_KEY)
        assert list(s.get_shape()) == [512, 1280, 2560]
        part = s[192:256]
    gu = M.experts_gate_up_device(part)
    assert gu.shape == (64, 2560, 2, 640)
    # expert 5 (global 197): gate row 7 / up row 7 = checkpoint rows 7 / 640 + 7
    torch.testing.assert_close(gu[5, :, 0, 7], part[5, 7])
    torch.testing.assert_close(gu[5, :, 1, 7], part[5, 640 + 7])
    q, sc = M.fp8_quantize_per_channel(gu, 1)
    err = (q.float() * sc.unsqueeze(1) - gu.float()).norm() / gu.float().norm()
    assert err < 0.03, err


# ----------------------------------------------------------------------------------------------
# Kernel glue: the NKI kernels replaced by torch emulations of their documented semantics, so the
# routing remaps, padding conventions and scale layouts fed to them are checked on CPU.
def _expert_mlp(x, gu, dn, gus, dns, e):
    """One expert in the kernel layout (gate_up [E, H, 2, I], down [E, I, H]); scales are the
    per-channel dequant multipliers in either kernel's layout."""
    g, u = gu[e, :, 0].float(), gu[e, :, 1].float()
    d = dn[e].float()
    if gus is not None:
        s = gus[e].reshape(2, -1)
        g, u, d = g * s[0], u * s[1], d * dns[e].reshape(1, -1)
    h = torch.nn.functional.silu(x.float() @ g) * (x.float() @ u)
    return h @ d


def _emulate_moe_cte(
    implementation,
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    conditions,
    block_size,
    gate_up_proj_scale=None,
    down_proj_scale=None,
    num_static_block=None,
    **kw,
):
    from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

    E = gate_up_proj_weight.shape[0]
    out = torch.zeros(hidden_states.shape, dtype=torch.float32)
    N = block_to_expert.numel()
    if implementation == MoECTEImplementation.shard_on_i_hybrid:
        assert num_static_block == 0
        n_run = int(
            conditions.sum()
        )  # the dynamic loop runs sum(conditions) blocks from 0
    else:
        n_run = N
    for b in range(n_run):
        e = int(block_to_expert.reshape(-1)[b])
        ids = token_position_to_id[b * block_size : (b + 1) * block_size].long()
        ids = ids[ids >= 0]  # skip_token
        y = _expert_mlp(
            hidden_states[ids],
            gate_up_proj_weight,
            down_proj_weight,
            gate_up_proj_scale,
            down_proj_scale,
            e,
        )
        out.index_add_(0, ids, y * expert_affinities_masked[ids * E + e].float())
    # every routed (token, local expert) pair must sit in a block that ran
    return out.to(hidden_states.dtype)


def _emulate_moe_tkg(
    hidden_input,
    expert_gate_up_weights,
    expert_down_weights,
    expert_affinities,
    expert_index,
    is_all_expert,
    expert_gate_up_weights_scale=None,
    expert_down_weights_scale=None,
    rank_id=None,
    **kw,
):
    assert expert_affinities.dtype == torch.float32
    T, E = hidden_input.shape[0], expert_gate_up_weights.shape[0]
    out = torch.zeros(T, hidden_input.shape[1])
    args = (
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
    )
    for t in range(T):
        experts = range(E) if is_all_expert else expert_index[t].tolist()
        for e in experts:
            out[t] += (
                expert_affinities[t, e]
                * _expert_mlp(hidden_input[t : t + 1], *args, e)[0]
            )
    return out.to(hidden_input.dtype)


def _emulate_moe_tkg_sparse(
    x, gu, dn, expert_list, num_iters, affinities, gus=None, dns=None, unroll=2
):
    """moe_tkg_sparse: the loop runs 2 * unroll entries per trip (one per core per unrolled
    expert); entry j adds affinities[j, t] * expert_list[j](x[t])."""
    assert affinities.dtype == torch.float32 and expert_list.dtype == torch.int32
    n_run = int(num_iters.reshape(-1)[0]) * 2 * unroll
    assert n_run <= expert_list.shape[0]
    out = torch.zeros(x.shape[0], x.shape[1])
    for j in range(n_run):
        e = int(expert_list[j, 0])
        out += affinities[j].unsqueeze(1) * _expert_mlp(x, gu, dn, gus, dns, e)
    return out


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "T, is_prefill, path",
    [
        (40, True, "hybrid"),
        (40, True, "block"),
        (2, False, "selective"),
        (5, False, "all"),
        (1, False, "sparse"),
        (5, False, "sparse"),
    ],
)
def test_kernel_glue(
    monkeypatch, tiny_hf_model, qwen4_config, fp8, T, is_prefill, path
):
    import vllm_neuron.functional as NF
    import vllm_neuron.utils.neuron_utils as NU
    from vllm_neuron.functional.moe import moe_tkg_sparse as S

    blk = _hf_block(tiny_hf_model, torch.bfloat16)
    tensors = _hf_tensors(blk)
    cfg = _cfg(qwen4_config, torch.bfloat16, fp8=fp8)
    tp = 2
    slots = 2 * cfg.num_experts_per_tok
    monkeypatch.setattr(M, "SELECTIVE_MAX_SLOTS", {"fp8": slots, "bf16": slots})
    monkeypatch.setattr(M, "CTE_IMPL", path if is_prefill else None)
    monkeypatch.setattr(M, "SPARSE_MAX_TOKENS", 16 if path == "sparse" else 0)
    x = torch.randn(T, cfg.hidden_size).to(torch.bfloat16)
    dense = _plugin_output(monkeypatch, cfg, tensors, x, tp, is_prefill)
    calls = []
    monkeypatch.setattr(NU, "can_run_kernel", lambda *a: True)
    monkeypatch.setattr(
        NF,
        "moe_cte",
        lambda **kw: calls.append(kw["implementation"]) or _emulate_moe_cte(**kw),
    )
    monkeypatch.setattr(
        NF,
        "moe_tkg",
        lambda **kw: calls.append(kw["is_all_expert"]) or _emulate_moe_tkg(**kw),
    )
    monkeypatch.setattr(
        S,
        "moe_tkg_sparse",
        lambda *a, **kw: calls.append("sparse") or _emulate_moe_tkg_sparse(*a, **kw),
    )
    kern = _plugin_output(monkeypatch, cfg, tensors, x, tp, is_prefill)
    assert len(calls) == tp
    if path == "sparse":
        assert calls[0] == "sparse"
    elif not is_prefill:
        assert calls[0] == (path == "all")
    else:
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation as CI

        assert calls[0] == (
            CI.shard_on_i_hybrid if path == "hybrid" else CI.shard_on_block
        )
    err = (kern - dense).abs().max() / dense.abs().max()
    assert err < 1e-2, err


# ----------------------------------------------------------------------------------------------
# moe_tkg_sparse: the torch-side plan, and a tile-level torch emulation of the kernel (its HBM
# views, SBUF layouts, per-core list split and scale transposes) against a dense reference.
def _sparse_kernel_tiles(
    x, gu, dn, expert_list, num_iters, aff, gus=None, dns=None, unroll=2
):
    """moe_tkg_sparse_kernel step by step in torch: returns the [2, T, H] per-core partials."""
    P = 128
    T, H = x.shape
    E, _, _, Ie = gu.shape
    I2 = 2 * Ie
    H1, NI, NI2 = H // P, Ie // P, I2 // P
    fp8 = gus is not None
    gu_view = gu.float().reshape(E, P, H1 * I2)
    dn_view = dn.float().reshape(E, NI, P, H)
    if fp8:
        sgu_view, sdn_view = gus.reshape(E, NI2, P), dns.reshape(E, H1, P)
    # x_sb[p, k, t] = x[t, p*H1 + k]
    x_sb = x.float().reshape(T, P, H1).permute(1, 2, 0)
    out = torch.zeros(2, T, H)
    for core in range(2):
        acc = torch.zeros(P, H1, T)
        slot = core * unroll
        for _ in range(int(num_iters.reshape(-1)[0])):
            for u in range(unroll):
                e = int(expert_list[slot + u, 0])
                w = aff[slot + u].expand(P, T)  # the same T-run on every partition
                w_gu = gu_view[e].reshape(P, H1, I2)  # row p*H1 + k on partition p
                gu_t = torch.zeros(P, NI2, T)
                for j in range(NI2):
                    bank = sum(
                        w_gu[:, k, j * P : (j + 1) * P].t() @ x_sb[:, k, :]
                        for k in range(H1)
                    )
                    gu_t[:, j, :] = (
                        bank * sgu_view[e].t()[:, j : j + 1] if fp8 else bank
                    )
                a = (
                    (torch.nn.functional.silu(gu_t[:, :NI]) * gu_t[:, NI:])
                    .to(x.dtype)
                    .float()
                )
                y = torch.zeros(P, H1, T)
                for n in range(H1):
                    bank = sum(
                        dn_view[e, r][:, n * P : (n + 1) * P].t() @ a[:, r, :]
                        for r in range(NI)
                    )
                    if fp8:
                        bank = bank * sdn_view[e].t()[:, n : n + 1]
                    y[:, n, :] = bank * w
                acc += y
            slot += 2 * unroll
        # out[t, n*128 + p] = acc[p, n, t]
        out[core] = acc.permute(2, 1, 0).reshape(T, H)
    return out


def _sparse_reference(x, gu, dn, affinities, gus=None, dns=None):
    """sum_e affinities[:, e] * expert_e(x) with dequantized weights, fp32."""
    out = torch.zeros(x.shape[0], x.shape[1])
    for e in range(gu.shape[0]):
        out += affinities[:, e : e + 1] * _expert_mlp(x, gu, dn, gus, dns, e)
    return out


def _sparse_affinities(T, E_L, top_k, seed):
    """Dense local affinities as _route produces them: top_k of 2 * E_L global experts, so
    about half of each token's experts are on this rank."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(T, 2 * E_L, generator=g).topk(top_k, dim=-1).indices
    w = torch.rand(T, top_k, generator=g) + 0.1
    dense = torch.zeros(T, 2 * E_L).scatter(1, idx, w / w.sum(-1, keepdim=True))
    return dense[:, :E_L].contiguous()


@pytest.mark.parametrize("unroll", [1, 2])
@pytest.mark.parametrize("T, max_experts", [(1, 4), (3, 12), (6, 4), (16, 16)])
def test_sparse_expert_plan(T, max_experts, unroll):
    from vllm_neuron.functional.moe import moe_tkg_sparse as S

    E_L, step = 16, 2 * unroll
    aff = _sparse_affinities(T, E_L, 4, seed=T)
    aff[:, 5] = 0  # at least one unselected expert
    expert_list, num_iters, plan_aff = S.sparse_expert_plan(aff, max_experts, unroll)
    n_slots = min(E_L, -(-max_experts // step) * step)
    assert expert_list.shape == (n_slots, 1) and expert_list.dtype == torch.int32
    assert num_iters.shape == (1, 1) and num_iters.dtype == torch.int32
    assert plan_aff.shape == (n_slots, T) and plan_aff.dtype == torch.float32

    used = (aff != 0).any(0)
    selected = torch.nonzero(used).flatten().tolist()
    rest = torch.nonzero(~used).flatten().tolist()
    # Selected experts first (ascending), then the others, each listed at most once.
    assert expert_list.flatten().tolist() == (selected + rest)[:n_slots]
    n_run = min(len(selected), n_slots)
    assert int(num_iters) == -(-n_run // step)
    # Row j holds every token's weight for expert_list[j] (0 for unselected experts).
    torch.testing.assert_close(
        plan_aff, aff.t()[expert_list.flatten().long()], atol=0, rtol=0
    )


def test_sparse_expert_plan_needs_whole_trips():
    from vllm_neuron.functional.moe import moe_tkg_sparse as S

    with pytest.raises(AssertionError):
        S.sparse_expert_plan(
            torch.ones(2, 6), 4, unroll=2
        )  # E_L = 6 is not a multiple of 4


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("unroll", [1, 2])
@pytest.mark.parametrize("T", [1, 5, 16])
def test_sparse_kernel_emulation_matches_reference(T, unroll, fp8):
    """Plan + tile-level kernel emulation vs the dense reference, with the module's weight
    layouts and per-channel FP8 scales ([E_L, 2, I] and [E_L, H], as attach_moe_loaders)."""
    from vllm_neuron.functional.moe import moe_tkg_sparse as S

    E_L, H, Ie, top_k = 8, 256, 256, 4
    g = torch.Generator().manual_seed(T)
    gu = (torch.randn(E_L, H, 2, Ie, generator=g) * H**-0.5).bfloat16()
    dn = (torch.randn(E_L, Ie, H, generator=g) * Ie**-0.5).bfloat16()
    # Distinct per-channel magnitudes, so a misplaced scale shows up in the output.
    gu = gu * torch.logspace(-1, 1, 2 * Ie).reshape(1, 1, 2, Ie).bfloat16()
    dn = dn * torch.logspace(-1, 1, H).reshape(1, 1, H).bfloat16()
    gus = dns = None
    if fp8:
        gu, gus = M.fp8_quantize_per_channel(gu, 1)
        dn, dns = M.fp8_quantize_per_channel(dn, 1)
        assert gus.shape == (E_L, 2, Ie) and dns.shape == (E_L, H)
    x = torch.randn(T, H, generator=g).bfloat16()
    aff = _sparse_affinities(T, E_L, top_k, seed=T + 1)

    expert_list, num_iters, plan_aff = S.sparse_expert_plan(
        aff, min(E_L, T * top_k), unroll
    )
    parts = _sparse_kernel_tiles(
        x, gu, dn, expert_list, num_iters, plan_aff, gus, dns, unroll
    )
    ref = _sparse_reference(x, gu, dn, aff, gus, dns)
    err = (parts.sum(0) - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, err  # the kernel rounds the activation to bf16
    # The semantic emulation used by test_kernel_glue agrees with the tile-level one.
    sem = _emulate_moe_tkg_sparse(
        x, gu, dn, expert_list, num_iters, plan_aff, gus, dns, unroll
    )
    torch.testing.assert_close(sem, ref, atol=1e-5 * ref.abs().max().item(), rtol=1e-4)
