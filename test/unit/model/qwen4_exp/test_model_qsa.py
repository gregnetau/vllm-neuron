# SPDX-License-Identifier: Apache-2.0
"""CPU test of the full plugin model with sparse QSA against Hugging Face.

Same harness as ``test_model_dense.py`` (tp=1, ``qwen4_exp_testlib.patch_plugin``), but with the tiny
config's real ``indexer_budget = 16`` (4 blocks), so a 64-slot context bucket takes the QSA
paths: the masked torch prefill (``_prefill_qsa``) and the masked decode (``_decode_torch``
with the token mask). Covers a padded prefill, a chunked prefill (cached prefix) and several
decode steps, including MTP-style multi-token verify rows (S_q = 2) and a batch of 2 requests.
"""

import pytest
import torch

from qwen4_exp_testlib import load_from_hf, ple_rows, state_pages, tiny_text_config_dict

BLOCK, PAGE, PLE_PAGE = 8, 30, 31


pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


def _hf_model():
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(**tiny_text_config_dict(dtype="float32"))
    assert cfg.indexer_budget == 16
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


def _load(plugin, hf, hf_cfg):
    model = plugin.Qwen4ExpForCausalLM.from_configs(hf_cfg, None)
    load_from_hf(model, hf)
    return model


class _Paged:
    """Caches + metadata for B requests; NaN-filled K/V and raw-key pools (stale pages)."""

    def __init__(self, model, B, n_blk, nan=True):
        cfg = model.config
        self.cfg, self.B, self.n_blk = cfg, B, n_blk
        spec = model.get_kv_spec()
        fill = float("nan") if nan else 0.0
        kv = {
            ls.name: [
                torch.full((64, ls.num_kv_heads, BLOCK, ls.head_size), fill)
                for _ in range(2)
            ]
            for ls in spec.layers
        }
        model.bind_kv_cache(kv)
        for layer in model.model.layers:  # stale side-cache slots must be masked out
            if layer.layer_type == "full_attention":
                assert layer.self_attn.index_cache.shape == (
                    64,
                    1,
                    BLOCK,
                    cfg.indexer_head_dim,
                )
                layer.self_attn.index_cache.fill_(fill)
        model.bind_state_cache(state_pages(spec, 40))
        g = torch.Generator().manual_seed(3)
        perm = torch.randperm(63, generator=g) + 1
        self.bt = perm[: B * n_blk].view(B, n_blk).int()

    def slots(self, bt, rows, pos):
        return bt[rows, pos // BLOCK].long() * BLOCK + pos % BLOCK

    def md(self, rows, pos, is_decode, q_len, cached, valid=None):
        valid = torch.ones_like(pos, dtype=torch.bool) if valid is None else valid
        neg = torch.full_like(pos, -1)
        common = {
            "max_query_len": q_len,
            "decode_token_threshold": q_len if is_decode else 1,
            "block_size": BLOCK,
            "max_blocks_per_seq": self.n_blk,
            "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32),
            "kv_segment_size": 0,
        }
        md = {}
        for i, t in enumerate(self.cfg.layer_types):
            if t == "full_attention":
                md[f"layers.{i}.self_attn"] = dict(
                    common,
                    block_table_tensor=self.bt,
                    slot_mapping=torch.where(
                        valid, self.slots(self.bt, rows, pos), neg
                    ),
                )
            else:
                md[f"layers.{i}.linear_attn"] = dict(
                    common,
                    block_table_tensor=(PAGE + torch.arange(self.B)).view(-1, 1).int(),
                    slot_mapping=torch.where(
                        valid, self.slots(self.bt, rows, pos), neg
                    ),
                )
        md["layers.1.ple"] = dict(
            common,
            block_table_tensor=(PLE_PAGE + 2 * torch.arange(self.B)).view(-1, 1).int(),
            slot_mapping=torch.where(valid, self.slots(self.bt, rows, pos), neg),
        )
        return md


def _trim_dense(md, blocks):
    """The runner's dense prefill variant: attention block tables cut to ``blocks``."""
    out = dict(md)
    for k, v in md.items():
        if k.endswith("self_attn"):
            out[k] = dict(
                v,
                block_table_tensor=v["block_table_tensor"][:, :blocks].contiguous(),
                max_blocks_per_seq=blocks,
            )
    return out


@pytest.mark.parametrize(
    "chunks, row_block",
    [([45], None), ([24, 21], None), ([24, 21], 8), ([24, 21], -8), ([13, 16, 16], 8)],
)
def test_qsa_prefill_and_decode_match_hf(plugin, monkeypatch, chunks, row_block):
    """Sparse prefill (one or several chunks, optionally in 8-row query blocks) and decode.
    A chunk whose whole context fits ``prefill_dense_context`` (16 tokens here) runs the
    runner's dense variant (tables trimmed to 2 blocks) and takes the dense paths."""
    hf, hf_cfg = _hf_model()
    model = _load(plugin, hf, hf_cfg)
    if row_block:
        attn_cls = type(model.model.layers[3].self_attn)
        if row_block > 0:  # selection and attention both in row_block rows
            monkeypatch.setattr(attn_cls, "QSA_SELECT_BYTES", 1)
        # (negative: selection over the whole chunk, attention in sub-blocks of |row_block|)
        monkeypatch.setattr(attn_cls, "QSA_PREFILL_SCORE_BYTES", 1)
        monkeypatch.setattr(attn_cls, "QSA_PREFILL_MIN_ROWS", abs(row_block))
    dense_ctx = model.prefill_dense_context
    assert dense_ctx == 16
    n_real, steps = sum(chunks), 5
    torch.manual_seed(1)
    ids = torch.randint(2, model.config.vocab_size, (n_real + steps,))
    ple_all = ple_rows(hf, ids)
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    pg = _Paged(model, B=1, n_blk=8)  # S = 64 slots > 4 * 4 blocks: sparse
    assert not model.model.layers[3].self_attn._qsa_dense(
        pg.md(
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            True,
            1,
            0,
        )
    )
    start = 0
    for n in chunks:
        T = (n + 7) // 8 * 8  # padded bucket
        pos = torch.arange(start, start + T)
        valid = pos < start + n
        x = torch.where(
            valid,
            ids[pos.clamp(max=ids.shape[0] - 1)],
            torch.zeros((), dtype=torch.long),
        )
        ple = torch.where(
            valid[:, None], ple_all[pos.clamp(max=ids.shape[0] - 1)], torch.zeros(())
        )
        md = pg.md(torch.zeros(T, dtype=torch.long), pos, False, T, start, valid)
        if start + n <= dense_ctx:
            md = _trim_dense(md, dense_ctx // BLOCK)
            assert model.model.layers[3].self_attn._qsa_dense(md)
        with torch.no_grad():
            logits = model(
                x,
                positions=pos,
                attn_metadata=md,
                sampling_positions=torch.tensor([n - 1]),
                ple_embeds=ple,
            )
        torch.testing.assert_close(logits[0], ref[start + n - 1], rtol=2e-3, atol=2e-3)
        start += n
    for step in range(steps - 1):
        p = n_real + step
        pos = torch.tensor([p])
        with torch.no_grad():
            logits = model(
                ids[p : p + 1],
                positions=pos,
                attn_metadata=pg.md(torch.zeros(1, dtype=torch.long), pos, True, 1, p),
                ple_embeds=ple_all[p : p + 1],
            )
        torch.testing.assert_close(logits[0], ref[p], rtol=2e-3, atol=2e-3)


def test_qsa_batched_and_multitoken_decode_match_hf(plugin):
    """B = 2 requests at different lengths; decode steps of S_q = 2 tokens (MTP verify rows,
    teacher-forced) after one S_q = 1 step."""
    hf, hf_cfg = _hf_model()
    model = _load(plugin, hf, hf_cfg)
    lens, extra = [37, 46], 5
    torch.manual_seed(2)
    seqs = [torch.randint(2, model.config.vocab_size, (n + extra,)) for n in lens]
    with torch.no_grad():
        refs = [hf(s[None]).logits[0] for s in seqs]
    ples = [ple_rows(hf, s) for s in seqs]
    pg = _Paged(model, B=2, n_blk=8)
    # prefill each request separately (one request per prefill step)
    for r, (s, n) in enumerate(zip(seqs, lens)):
        T = (n + 7) // 8 * 8
        pos = torch.arange(T)
        valid = pos < n
        one = _Paged.__new__(_Paged)
        one.__dict__.update(pg.__dict__)
        one.B, one.bt = 1, pg.bt[r : r + 1]
        md = one.md(torch.zeros(T, dtype=torch.long), pos, False, T, 0, valid)
        for k in md:  # recurrent-state pages of request r
            if k.endswith("linear_attn"):
                md[k]["block_table_tensor"] = torch.tensor(
                    [[PAGE + r]], dtype=torch.int32
                )
            elif k.endswith("ple"):
                md[k]["block_table_tensor"] = torch.tensor(
                    [[PLE_PAGE + 2 * r]], dtype=torch.int32
                )
        x = torch.where(valid, s[:T], torch.zeros((), dtype=torch.long))
        ple = torch.where(valid[:, None], ples[r][:T], torch.zeros(()))
        with torch.no_grad():
            lg = model(
                x,
                positions=pos,
                attn_metadata=md,
                sampling_positions=torch.tensor([n - 1]),
                ple_embeds=ple,
            )
        torch.testing.assert_close(lg[0], refs[r][n - 1], rtol=2e-3, atol=2e-3)
    cur = list(lens)
    for _ in range(3):
        rows = torch.arange(2)
        pos = torch.tensor(cur)
        x = torch.stack([seqs[r][c] for r, c in enumerate(cur)])
        ple = torch.stack([ples[r][c] for r, c in enumerate(cur)])
        with torch.no_grad():
            lg = model(
                x,
                positions=pos,
                attn_metadata=pg.md(rows, pos, True, 1, max(cur)),
                ple_embeds=ple,
            )
        ref = torch.stack([refs[r][c] for r, c in enumerate(cur)])
        torch.testing.assert_close(lg, ref, rtol=2e-3, atol=2e-3)
        cur = [c + 1 for c in cur]

    # MTP verify rows: one attention-layer decode with S_q = 2 tokens per request equals two
    # S_q = 1 steps (which match HF above). GDN has no multi-token decode on CPU, so this runs
    # the attention layer alone, on hidden states that make each request's selection sparse.
    attn = model.model.layers[3].self_attn
    H = model.config.hidden_size
    caches = [attn.k_cache, attn.v_cache, attn.index_cache]
    snap = [c.clone() for c in caches]
    x = torch.randn(4, H, generator=torch.Generator().manual_seed(4))
    rows = torch.arange(2).repeat_interleave(2)
    pos = torch.cat([torch.arange(c, c + 2) for c in cur])
    md = pg.md(rows, pos, True, 2, max(cur) + 1)
    rope = model.model.rotary_emb(pos.int(), dtype=x.dtype)
    with torch.no_grad():
        out2 = attn.forward_decode(x, pos.int(), rope, md)
        for c, t in zip(caches, snap):
            c.copy_(t)
        outs = []
        for i in range(2):
            sel = torch.tensor([i, 2 + i])
            p1 = pos[sel]
            md1 = pg.md(torch.arange(2), p1, True, 1, max(cur) + i)
            outs.append(
                attn.forward_decode(
                    x[sel],
                    p1.int(),
                    model.model.rotary_emb(p1.int(), dtype=x.dtype),
                    md1,
                )
            )
    out1 = torch.stack(outs, 1).reshape(4, H)
    torch.testing.assert_close(out2, out1, rtol=1e-5, atol=1e-5)
