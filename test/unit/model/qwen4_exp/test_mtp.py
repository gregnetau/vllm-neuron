# SPDX-License-Identifier: Apache-2.0
"""MTP draft model and multi-token (spec-verify) decode pieces, on CPU.

There is no Hugging Face MTP implementation to compare against; the draft's numerics are checked
on device through its acceptance rate. Here: every draft parameter maps to a real checkpoint
tensor of the right shape (after its loader), and the PLE layer's multi-token decode equals
decoding the same tokens one at a time.
"""

import json
import struct

import torch

from qwen4_exp_testlib import (
    FakeSlice,
    checkpoint_dir,
    real_text_config_dict,
    requires_checkpoint,
)


pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


def _checkpoint_shapes() -> dict:
    ckpt = checkpoint_dir()
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    shapes = {}
    for f in sorted(set(index.values())):
        with open(ckpt / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        shapes.update({k: v["shape"] for k, v in hdr.items() if k != "__metadata__"})
    return shapes


@requires_checkpoint
def test_mtp_mappings_cover_checkpoint(plugin_mtp):
    _, MTP = plugin_mtp
    from vllm_neuron.utils import weight_loader as wl

    with torch.device("meta"):
        draft = MTP.Qwen4ExpMTP.from_configs(real_text_config_dict(), 48)
    assert draft.mtp_hidden_size == 10240
    shapes = _checkpoint_shapes()
    mappings = draft._weight_mappings()
    params = dict(draft.named_parameters())
    assert set(mappings) == set(params), set(mappings) ^ set(params)
    used = set()
    for name, keys in mappings.items():
        keys = [keys] if isinstance(keys, str) else keys
        assert all(k in shapes for k in keys), (
            name,
            [k for k in keys if k not in shapes],
        )
        used.update(keys)
        loader = wl.get_weight_loader(params[name])
        if (
            loader is None or ".experts_" in name
        ):  # routed experts: tested in test_moe.py
            continue
        fakes = [FakeSlice(torch.empty(shapes[k], device="meta")) for k in keys]
        out = loader.load(fakes, 0)
        assert tuple(out.shape) == tuple(params[name].shape), name
    mtp_keys = {k for k in shapes if k.startswith("mtp.")}
    assert mtp_keys <= used, sorted(mtp_keys - used)[:5]


@torch.no_grad()
def test_ple_multi_token_decode_matches_sequential(plugin_mtp, tiny_text_config):
    from vllm_neuron.model import state_cache
    from vllm_neuron.model.qwen4_exp.config import Qwen4ExpConfig

    M, _ = plugin_mtp
    cfg = Qwen4ExpConfig.from_configs(tiny_text_config)
    cfg.torch_dtype = torch.float32
    torch.manual_seed(0)
    ple = M.Qwen4ExpPLE(cfg, 1)
    for p in ple.parameters():
        p.data.normal_(0, 0.2)
    shape = ((cfg.ple_conv_state_len, cfg.hc_hidden_size),)
    layout = state_cache.page_layout(shape, (torch.float32,))
    page_bytes = layout.page_bytes + 1024

    def fresh():
        raw = torch.zeros(12 * page_bytes, dtype=torch.int8)
        ple.bind_state(*state_cache.paired_half_views(raw, page_bytes))
        ple._write_state(
            torch.tensor([2, 5]), torch.randn(2, ple.L, ple.C, generator=g0())
        )

    def g0():
        return torch.Generator().manual_seed(3)

    B, S = 2, 3
    x = torch.randn(B * S, ple.C)
    emb = torch.randn(B * S, cfg.ple_embed_dim)
    # verify step: state starts from block 0 (pages 2 / 5); state after token j -> block j
    fresh()
    bt = torch.tensor([[2, 7, 8], [5, 9, 10]])
    md = {"block_table_tensor": bt, "state_in": torch.zeros(B, dtype=torch.int32)}
    out = ple._decode(x, emb, md)
    states = [ple._read_state(bt[:, j]) for j in range(S)]
    # reference: one token at a time from the same start state
    fresh()
    ref, slots = [], torch.tensor([2, 5])
    xs, es = x.view(B, S, -1), emb.view(B, S, -1)
    for j in range(S):
        ref.append(
            ple._decode(xs[:, j], es[:, j], {"block_table_tensor": slots.view(B, 1)})
        )
        torch.testing.assert_close(states[j], ple._read_state(slots))
    torch.testing.assert_close(out.view(B, S, -1), torch.stack(ref, dim=1))


def _mtp_state_dict(hf, cfg) -> dict:
    """Checkpoint tensors for the draft: the MTP layer takes the shapes (and, for the test,
    the values) of the backbone's first attention layer; the fusion weights are random."""
    from qwen4_exp_testlib import hf_checkpoint_state_dict

    sd = hf_checkpoint_state_dict(hf)
    src = "model.language_model.layers.3."
    out = {
        k: v for k, v in sd.items() if not k.startswith("model.language_model.layers.")
    }
    out.update(
        {"mtp.layers.0." + k[len(src) :]: v for k, v in sd.items() if k.startswith(src)}
    )
    mixer = "model.language_model.hyper_connection_mixer."
    out.update(
        {
            "mtp." + k[len("model.language_model.") :]: v
            for k, v in sd.items()
            if k.startswith(mixer)
        }
    )
    g = torch.Generator().manual_seed(0)
    H, hcH = cfg.hidden_size, cfg.hc_count * cfg.hidden_size
    out["mtp.fc_embedding.weight"] = torch.randn(H, H, generator=g) * 0.05
    out["mtp.fc_hidden.weight"] = torch.randn(H, H, generator=g) * 0.05
    out["mtp.pre_fc_norm_embedding.weight"] = torch.randn(H, generator=g) * 0.1
    out["mtp.pre_fc_norm_hidden.weight"] = torch.randn(hcH, generator=g) * 0.1
    return out


@torch.no_grad()
def test_mtp_tiny_load_and_draft(plugin_mtp, tiny_hf_model, tiny_text_config):
    """Mappings / loaders / KV binding of the draft on the tiny config, and a two-token greedy
    draft whose second step reads the first step's multi-stream hidden and KV."""
    from qwen4_exp_testlib import load_state

    _, MTP = plugin_mtp
    cfg = tiny_text_config
    start = cfg.num_hidden_layers
    draft = MTP.Qwen4ExpMTP.from_configs(cfg, start)
    assert draft.mtp_hidden_size == cfg.hc_count * cfg.hidden_size
    load_state(draft, _mtp_state_dict(tiny_hf_model, cfg))
    m = draft._weight_mappings()
    assert m["model.layers.0.self_attn.index_qk_weight"] == (
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight"
    )
    spec = draft.get_kv_spec()
    key = f"layers.{start}.self_attn"
    assert [ls.name for ls in spec.layers] == [key] and not spec.state_layers
    BS, nblk = 8, 4
    ls = spec.layers[0]
    kv = {key: [torch.zeros(16, ls.num_kv_heads, BS, ls.head_size) for _ in range(2)]}
    draft.bind_kv_cache(kv)
    attn = draft.model.layers[0].self_attn
    assert attn.k_cache is kv[key][0] and attn.index_cache.shape[:3] == (16, 1, BS)

    B, k = 2, 2
    draft.num_speculative_tokens = k
    bt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    pos = torch.tensor([5, 9])
    md = {
        key: {
            "block_table_tensor": bt,
            "slot_mapping": (bt[torch.arange(B), pos // BS].long() * BS + pos % BS),
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "block_size": BS,
            "max_blocks_per_seq": nblk,
        }
    }
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, cfg.vocab_size, (B,), generator=g)
    hidden = torch.randn(B, draft.mtp_hidden_size, generator=g)
    stacked, drafts, _ = draft(ids, pos, hidden, md, sampling_positions=torch.arange(B))
    assert stacked.shape == drafts.shape == (B, k)
    assert ((drafts >= 0) & (drafts < cfg.vocab_size)).all()
    # The second draft token wrote K/V and raw keys at position + 1 of each request.
    for b in range(B):
        p = int(pos[b]) + 1
        blk, off = int(bt[b, p // BS]), p % BS
        assert attn.k_cache[blk, :, off].abs().sum() > 0
        assert attn.index_cache[blk, :, off].abs().sum() > 0
