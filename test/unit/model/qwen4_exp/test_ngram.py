# SPDX-License-Identifier: Apache-2.0
"""Host-side n-gram (PLE) embedding vs HF ``Qwen4ExpTextNGramEmbedding`` (CPU only).

The tiny-config tests compare the full host pipeline (context tokens with the EOS rule, hash,
sharded gather) bit-exactly with HF, including HF's cached (chunked prefill + decode) path.
The real-checkpoint tests run only when ``QWEN4_EXP_MODEL_PATH`` names a checkpoint with the
PLE layer's safetensors files.
"""

import json
import os

import numpy as np
import pytest
import torch

from qwen4_exp_testlib import checkpoint_dir, real_text_config_dict
from vllm_neuron.model.qwen4_exp.ngram import (
    HostNGramTable,
    HostPLEInputs,
    NGramHasher,
    context_tokens,
    load_hasher_buffers,
)

pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


EOS = 1  # tiny config
GARBAGE = (
    7  # fills token_ids_cpu beyond the known history: reading it would change the hash
)


def _hf_emb(model):
    return model.model.layers[1].ple.ple_embedding


def _host(model, num_shards=4):
    emb = _hf_emb(model)
    hasher = NGramHasher.from_buffers(
        emb.layer_multipliers,
        emb.ngram_heads_vocab_sizes,
        emb.ngram_heads_offsets,
        emb.ngram_size,
        emb.heads_per_ngram,
    )
    table = HostNGramTable.from_tensor(
        emb.ngram_embedding.weight, num_shards=num_shards, num_threads=2
    )
    return HostPLEInputs(hasher, table, EOS)


def _seq(n, eos_at=(), seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 1024, (n,), generator=g)
    ids[list(eos_at)] = EOS
    return ids


def _history(rows, length=128):
    """token_ids_cpu: one row per request, prefix ``rows[r]`` known, garbage after."""
    hist = np.full((len(rows), length), GARBAGE, dtype=np.int32)
    for r, known in enumerate(rows):
        hist[r, : len(known)] = known.numpy()
    return hist


@torch.no_grad()
def _hf_full(model, ids):
    return _hf_emb(model)(ids[None], None)[0]


def test_from_config_matches_hf_buffers(tiny_hf_model, tiny_text_config, qwen4_config):
    emb = _hf_emb(tiny_hf_model)
    for cfg in (tiny_text_config, qwen4_config):
        h = NGramHasher.from_config(cfg, ple_layer_index=0)
        assert torch.equal(h.layer_multipliers, emb.layer_multipliers)
        assert torch.equal(h.head_vocab_sizes, emb.ngram_heads_vocab_sizes)
        assert torch.equal(h.head_offsets, emb.ngram_heads_offsets)
    assert h.total_vocab_size == emb.total_vocab_size


def test_context_tokens_eos_rule():
    # positions 0..7, EOS at 2 and 5: n-grams never reach back across an EOS.
    t = torch.tensor([10, 11, EOS, 13, 14, EOS, 16, 17])
    s = context_tokens(
        t,
        torch.zeros(8, dtype=torch.long),
        torch.arange(8),
        np.zeros((1, 8), np.int32),
        EOS,
    )
    E = EOS
    assert s[0].tolist() == t.tolist()
    assert s[1].tolist() == [E, 10, 11, E, 13, 14, E, 16]
    assert s[2].tolist() == [E, E, 10, E, E, 13, E, E]


@pytest.mark.parametrize("eos_at", [(), (0, 5, 6, 20, 31), (1, 2, 3, 30)])
def test_prefill_one_call_matches_hf(tiny_hf_model, eos_at):
    ple = _host(tiny_hf_model)
    ids = _seq(32, eos_at)
    out = ple(
        ids, torch.zeros(32, dtype=torch.long), torch.arange(32), _history([ids[:0]])
    )
    assert out.dtype == torch.float32 and out.shape == (
        32,
        256,
    )  # tiny table is fp32: 16 heads x 16
    torch.testing.assert_close(out, _hf_full(tiny_hf_model, ids), rtol=0, atol=0)


@torch.no_grad()
def test_chunked_prefill_and_decode_match_hf_cached(tiny_hf_model):
    ple = _host(tiny_hf_model)
    n, cuts = (
        40,
        [0, 1, 13, 26, 30],
    )  # 1-token first chunk, then chunks, then token-by-token decode
    ids = _seq(n, eos_at=(12, 24, 25, 33))  # EOS at chunk boundaries and inside decode
    steps = [(a, b) for a, b in zip(cuts, cuts[1:])] + [
        (i, i + 1) for i in range(cuts[-1], n)
    ]

    captured = []
    hook = _hf_emb(tiny_hf_model).register_forward_hook(
        lambda m, i, o: captured.append(o[0])
    )
    try:
        pkv = None
        for a, b in steps:
            out = tiny_hf_model(ids[None, a:b], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
    finally:
        hook.remove()
    hf_cached = torch.cat(captured)
    torch.testing.assert_close(hf_cached, _hf_full(tiny_hf_model, ids), rtol=0, atol=0)

    ours = []
    for a, b in steps:
        hist = _history([ids[:a]])  # the runner has only the tokens before this step
        ours.append(
            ple(
                ids[a:b], torch.zeros(b - a, dtype=torch.long), torch.arange(a, b), hist
            )
        )
    torch.testing.assert_close(torch.cat(ours), hf_cached, rtol=0, atol=0)


def test_batch_of_requests_with_padding(tiny_hf_model):
    ple = _host(tiny_hf_model)
    seqs = [
        _seq(24, (3, 15), seed=1),
        _seq(30, (28,), seed=2),
        _seq(40, (0, 37), seed=3),
        _seq(8, (), seed=4),
    ]
    ref = [_hf_full(tiny_hf_model, s) for s in seqs]
    # (request row, positions in this step, tokens already in token_ids_cpu)
    plan = [
        (0, range(10, 18), 10),  # prefill chunk
        (1, range(29, 30), 29),  # decode, previous token in the history
        (
            2,
            range(36, 40),
            36,
        ),  # spec-decode verify rows: drafts not in the history yet
        (3, range(0, 8), 0),  # first prefill chunk (EOS context)
    ]
    hist = _history([s[:known] for s, (_, _, known) in zip(seqs, plan)])
    rows, pos, toks, expect = [], [], [], []
    for r, rng, _ in plan:
        for p in rng:
            (
                rows.append(r),
                pos.append(p),
                toks.append(int(seqs[r][p])),
                expect.append(ref[r][p]),
            )
        (
            rows.append(-1),
            pos.append(0),
            toks.append(0),
            expect.append(torch.zeros(256)),
        )  # interleaved padding
    rows += [-1] * 5
    pos += [0] * 5
    toks += [0] * 5
    expect += [torch.zeros(256)] * 5
    # Shuffle the batch: lookups are by (row, position), not by order.
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))
    args = [torch.tensor(x)[perm] for x in (toks, rows, pos)]
    out = ple(*args, hist)
    torch.testing.assert_close(out, torch.stack(expect)[perm], rtol=0, atol=0)


def _write_sharded_checkpoint(path, emb, cfg, layer_idx=1):
    """Same layout as the real checkpoint: shards spread over files, buffers in the first/last file."""
    from safetensors.torch import save_file

    prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding."
    shards = torch.chunk(
        emb.ngram_embedding.weight.detach().to(torch.bfloat16), cfg.split_ngram_parts
    )
    files = {
        "model-00001-of-00002.safetensors": {
            prefix + "layer_multipliers": emb.layer_multipliers,
            prefix + "ngram_embedding.shard_0.weight": shards[0],
            "model.language_model.layers.0.dummy": torch.zeros(
                3
            ),  # unaligned neighbour
            prefix + "ngram_embedding.shard_1.weight": shards[1],
        },
        "model-00002-of-00002.safetensors": {
            **{
                prefix + f"ngram_embedding.shard_{i}.weight": shards[i]
                for i in range(2, len(shards))
            },
            prefix + "ngram_heads_vocab_sizes": emb.ngram_heads_vocab_sizes,
            prefix + "ngram_heads_offsets": emb.ngram_heads_offsets,
        },
    }
    weight_map = {}
    for fname, tensors in files.items():
        save_file(
            {k: v.contiguous() for k, v in tensors.items()}, os.path.join(path, fname)
        )
        weight_map.update({k: fname for k in tensors})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return torch.cat(shards)


@pytest.mark.parametrize("with_index", [True, False])
def test_from_checkpoint_sharded_layout(
    tmp_path, tiny_hf_model, qwen4_config, with_index
):
    emb = _hf_emb(tiny_hf_model)
    weight = _write_sharded_checkpoint(tmp_path, emb, qwen4_config)
    if not with_index:
        os.remove(tmp_path / "model.safetensors.index.json")
    ple = HostPLEInputs.from_checkpoint(str(tmp_path), qwen4_config, prefetch=False)
    assert (
        len(ple.table.shards) == qwen4_config.split_ngram_parts
        and ple.table.num_rows == weight.shape[0]
    )
    ple.table.prefetch(block=True)
    ids = _seq(32, (4, 9, 10))
    out = ple(
        ids, torch.zeros(32, dtype=torch.long), torch.arange(32), _history([ids[:0]])
    )
    assert out.dtype == torch.bfloat16
    ref = HostPLEInputs(ple.hasher, HostNGramTable.from_tensor(weight.float()), EOS)
    ref_out = ref(
        ids, torch.zeros(32, dtype=torch.long), torch.arange(32), _history([ids[:0]])
    )
    hf = _hf_full(tiny_hf_model, ids).to(
        torch.bfloat16
    )  # rounding commutes with the gather
    torch.testing.assert_close(ref_out.to(torch.bfloat16), hf, rtol=0, atol=0)
    torch.testing.assert_close(out, hf, rtol=0, atol=0)
    # random ids, incl. negative (zero rows), against direct indexing
    g = torch.Generator().manual_seed(0)
    rid = torch.randint(-1, weight.shape[0], (50, 16), generator=g)
    expect = torch.where((rid >= 0)[..., None], weight[rid.clamp_min(0)], 0).flatten(1)
    torch.testing.assert_close(ple.table.gather(rid), expect, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Real checkpoint (skipped unless QWEN4_EXP_MODEL_PATH has the PLE layer's files)
# ---------------------------------------------------------------------------
def _real_checkpoint() -> str | None:
    ckpt = checkpoint_dir()
    if ckpt is None:
        return None
    with open(ckpt / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    need = {v for k, v in wm.items() if ".layers.1.ple.ple_embedding." in k}
    if need and all((ckpt / f).exists() for f in need):
        return str(ckpt)
    return None


CKPT = _real_checkpoint()
real = pytest.mark.skipif(
    CKPT is None, reason="set QWEN4_EXP_MODEL_PATH to a checkpoint with the PLE shards"
)


@pytest.fixture(scope="module")
def real_cfg():
    from vllm_neuron.model.qwen4_exp.config import Qwen4ExpConfig

    return Qwen4ExpConfig.from_configs(
        {"text_config": real_text_config_dict(), "dtype": "bfloat16"}
    )


class _IdsOut(torch.nn.Module):
    """Stands in for HF's nn.Embedding so the HF module returns its int64 ids."""

    weight = torch.empty(0)

    def forward(self, ids):
        return ids.unsqueeze(-1)


@torch.no_grad()
def test_real_size_hash_matches_hf(tiny_text_config, real_cfg):
    """Real multipliers/primes (large products wrap negative): numpy hash == HF torch hash."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextNGramEmbedding,
    )

    h = NGramHasher.from_config(real_cfg)
    emb = Qwen4ExpTextNGramEmbedding(tiny_text_config, 256, layer_idx=1)
    emb.layer_multipliers, emb.ngram_heads_vocab_sizes, emb.ngram_heads_offsets = (
        h.layer_multipliers,
        h.head_vocab_sizes,
        h.head_offsets,
    )
    emb.eos_token_id, emb.ngram_embedding = real_cfg.eos_token_id, _IdsOut()
    toks = torch.randint(
        0, real_cfg.vocab_size, (300,), generator=torch.Generator().manual_seed(0)
    )
    toks[[0, 7, 8, 100]] = real_cfg.eos_token_id
    ref = emb(toks[None], None)[0]
    shifted = context_tokens(
        toks,
        torch.zeros(300, dtype=torch.long),
        torch.arange(300),
        np.zeros((1, 300), np.int32),
        real_cfg.eos_token_id,
    )
    ours = h.ids(shifted)
    assert torch.equal(ours, ref)


@real
def test_real_from_config_equals_stored_buffers(real_cfg):
    b = load_hasher_buffers(CKPT, 1)
    h = NGramHasher.from_config(real_cfg, ple_layer_index=0)
    assert torch.equal(h.layer_multipliers, b["layer_multipliers"])
    assert torch.equal(h.head_vocab_sizes, b["ngram_heads_vocab_sizes"])
    assert torch.equal(h.head_offsets, b["ngram_heads_offsets"])
    assert h.num_heads == 16 and h.total_vocab_size <= 128 * 2_500_012


@real
def test_real_gather_equals_safetensors_reads(real_cfg):
    from safetensors import safe_open

    ple = HostPLEInputs.from_checkpoint(CKPT, real_cfg, prefetch=False)
    table = ple.table
    assert (
        len(table.shards) == 128
        and table.dim == 160
        and table.num_rows == 128 * 2_500_012
    )
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, table.num_rows, (8, 16), generator=g)
    ids[0, :3] = torch.tensor(
        [0, table.num_rows - 1, 2_500_012]
    )  # first / last row, shard edge
    out = table.gather(ids).view(8, 16, 160)
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    for (t, h), i in np.ndenumerate(ids.numpy()):
        name = f"{prefix}shard_{i // 2_500_012}.weight"
        with safe_open(os.path.join(CKPT, wm[name]), framework="pt") as f:
            row = f.get_slice(name)[i % 2_500_012 : i % 2_500_012 + 1]
        assert torch.equal(out[t, h], row[0]), (t, h, i)
    # End to end on a short prompt: the hash ids are the stored buffers' ids.
    toks = torch.tensor(
        [real_cfg.eos_token_id, 9707, 11, 1879, real_cfg.eos_token_id, 13]
    )
    emb = ple(
        toks,
        torch.zeros(6, dtype=torch.long),
        torch.arange(6),
        np.zeros((1, 8), np.int32),
    )
    assert emb.shape == (6, 2560) and emb.dtype == torch.bfloat16


@pytest.mark.parametrize("num_rows", [64, 5000])
def test_gather_threaded_matches_serial(num_rows):
    # Small gathers stay on the calling thread, large ones use the pool; both give the same rows.
    weight = torch.randn(10000, 16).to(torch.bfloat16)
    serial = HostNGramTable.from_tensor(weight, num_shards=4, num_threads=1)
    threaded = HostNGramTable.from_tensor(weight, num_shards=4, num_threads=4)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(-1, 10000, (num_rows, 1), generator=g)
    expected = torch.where(ids >= 0, ids, 0)
    ref = weight[expected.view(-1)] * (ids >= 0)
    assert torch.equal(serial.gather(ids), ref)
    assert torch.equal(threaded.gather(ids), ref)
