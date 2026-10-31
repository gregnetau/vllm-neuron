# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the QSA indexer / token selection (``vllm_neuron.model.qwen4_exp.qsa``)
against HF ``Qwen4ExpTextQSAIndexer`` and ``Qwen4ExpTextAttention``.

The tiny config has ``indexer_budget = 16`` (4 blocks of 4), so 40-80 tokens already take the
sparse path. Raw keys go through a simulated paged key-only cache (shuffled block table,
unused slots NaN) so that visibility masking is exercised.

Tie-breaking: HF picks ``torch.topk`` per query, and CPU topk returns an arbitrary subset of
equal scores; ``qsa`` takes the lowest block indices. Exact set equality is checked against
HF with ``Tensor.topk`` patched to a stable lowest-index-first top-k; against unpatched HF the
check is tie-aware (same count, same blocks above the k-th score, selected scores equal).
"""

import pytest
import torch

from vllm_neuron.model.qwen3_5.weights import rope_inv_freq
from vllm_neuron.model.qwen4_exp import qsa

pytest_plugins = ["qwen4_exp_testlib"]  # fixtures


BS = 8  # cache block size of the simulated paged caches


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attn_layer_idx(model):
    return next(
        i for i, t in enumerate(model.config.layer_types) if t != "linear_attention"
    )


def _rotary(model):
    cfg = model.config
    R = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 0.25))
    return rope_inv_freq(R, float(cfg.rope_parameters["rope_theta"]))


def _idx_params(attn):
    ix = attn.indexer
    return dict(
        w_qk=ix.index_qk_proj.weight.detach().T.contiguous(),
        q_gamma=1.0 + ix.q_layernorm.weight.detach(),
        k_gamma=1.0 + ix.k_layernorm.weight.detach(),
        eps=ix.q_layernorm.eps,
        n_heads=ix.index_n_heads,
        budget_blocks=ix.block_topk,
        ratio=ix.compress_ratio,
    )


def _hf_rope(model, T, offset=0):
    pos = torch.arange(offset, offset + T).view(1, 1, -1).expand(3, 1, -1)
    x = torch.zeros(1, T, model.config.hidden_size)
    return model.model.rotary_emb(x, pos)


class _PagedKeys:
    """Key-only paged cache of one request, NaN-initialised, with a shuffled block table."""

    def __init__(self, n_blk, D, seed=0, num_blocks=64):
        g = torch.Generator().manual_seed(seed)
        self.cache = torch.full((num_blocks, 1, BS, D), float("nan"))
        self.bt = (torch.randperm(num_blocks - 1, generator=g)[:n_blk] + 1).view(1, -1)

    def slots(self, positions):
        return self.bt[0, positions // BS] * BS + positions % BS


def _our_selection(model, hidden, S, positions=None, cache=None):
    """Token mask [T, S] for hidden [T, H] at positions (default arange)."""
    attn = model.model.layers[_attn_layer_idx(model)].self_attn
    p = _idx_params(attn)
    T = hidden.shape[0]
    positions = torch.arange(T) if positions is None else positions
    cos, sin = qsa.qsa_rope_cos_sin(positions, _rotary(model), hidden.dtype)
    q, k_raw = qsa.index_qk(
        hidden, p["w_qk"], p["q_gamma"], cos, sin, n_heads=p["n_heads"], eps=p["eps"]
    )
    cache = cache or _PagedKeys(S // BS, k_raw.shape[-1])
    qsa.write_raw_keys(cache.cache, k_raw, cache.slots(positions))
    kctx = qsa.gather_raw_keys(cache.cache, cache.bt)
    kc = qsa.compressed_keys(
        kctx, p["k_gamma"], _rotary(model), ratio=p["ratio"], eps=p["eps"]
    )
    q_pos = positions.view(1, -1)
    sel = qsa.select_blocks(q.unsqueeze(0), kc, q_pos, p["budget_blocks"], p["ratio"])
    tm = qsa.token_mask(sel, q_pos, S, p["ratio"])
    full = qsa.qsa_token_mask(
        q.unsqueeze(0),
        kctx,
        q_pos,
        p["k_gamma"],
        _rotary(model),
        budget_blocks=p["budget_blocks"],
        ratio=p["ratio"],
        eps=p["eps"],
    )
    assert torch.equal(tm, full)
    scores = qsa.index_scores(q.unsqueeze(0), kc, q_pos, p["ratio"])
    return tm[0], sel[0], scores[0]


def _hf_selection(model, hidden):
    """HF indexer selected-token mask [T, T] for one prompt (no cache)."""
    attn = model.model.layers[_attn_layer_idx(model)].self_attn
    T = hidden.shape[0]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
    with torch.no_grad():
        m = attn.indexer(hidden.unsqueeze(0), _hf_rope(model, T), causal, None)
    return m[0, 0]


def _stable_topk(self, k, dim=-1, largest=True, sorted=True):
    vals, idx = torch.sort(self, dim=dim, descending=largest, stable=True)
    return torch.return_types.topk((vals.narrow(dim, 0, k), idx.narrow(dim, 0, k)))


def _hidden(model, T, kind, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(T, model.config.hidden_size, generator=g)
    if kind == "zero_blocks":
        # Only blocks 1 and 5 carry signal: the other blocks' raw keys are 0, so their
        # compressed keys and scores are exactly 0 -> ties at the k-th value. Queries on
        # zero rows have q = 0 -> every visible block ties.
        keep = torch.zeros(T, dtype=torch.bool)
        keep[4:8] = True
        keep[20:24] = True
        h = torch.where(keep[:, None], h, torch.zeros(()))
    return h


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------


def test_rope_matches_hf(tiny_hf_model):
    cos_hf, sin_hf = _hf_rope(tiny_hf_model, 50, offset=7)
    cos, sin = qsa.qsa_rope_cos_sin(
        torch.arange(7, 57), _rotary(tiny_hf_model), torch.float32
    )
    torch.testing.assert_close(cos, cos_hf[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(sin, sin_hf[0], atol=1e-6, rtol=0)


@pytest.mark.parametrize("int_bits", [False, True], ids=["fp32-thresholds", "int-bits"])
@pytest.mark.parametrize("radix_bits", [1, 2, 5])
@pytest.mark.parametrize("N", [1, 2, 7, 64, 1000])
def test_topk_block_mask_exact(N, radix_bits, int_bits):
    g = torch.Generator().manual_seed(N)
    # heavy ties: many exact zeros and repeated values, -1 = ineligible
    s = torch.randint(0, 4, (6, N), generator=g).float() * 0.5
    s = torch.where(torch.rand(6, N, generator=g) < 0.4, torch.zeros(()), s)
    s[0] = 0.0
    s[1, : N // 2] = -1.0
    s[2, 0] = -0.0
    eligible = (s >= 0).sum(-1)
    for budget in (0, 1, 3, 512):
        k = eligible.clamp(max=budget)
        m = qsa.topk_block_mask(s, k, radix_bits, int_bits=int_bits)
        for r in range(6):
            kk = int(k[r])
            ref = torch.zeros(N, dtype=torch.bool)
            if kk:
                ref[_stable_topk(s[r], kk).indices] = True
            assert torch.equal(m[r], ref), (N, budget, r)


@pytest.mark.parametrize("N", [64, 1024, 16384])
def test_topk_block_mask_int_bits_equals_fp32_thresholds(N):
    """The int32 bit-pattern comparison (native route) selects exactly what the fp32
    thresholds select (XLA route), over a wide range of magnitudes, heavy ties, zeros, -0.0
    and ineligible (-1) entries, at every k."""
    g = torch.Generator().manual_seed(N + 1)
    rows = 16
    mag = torch.exp(torch.empty(rows, N).uniform_(-80, 80, generator=g))  # 1e-35 .. 1e35
    s = mag * torch.randint(1, 5, (rows, N), generator=g).float()
    s[1] = torch.round(s[1].clamp(max=8.0))  # ties
    s[2] = 0.0
    s[3, ::3] = -0.0
    s[4, : N // 2] = -1.0
    s[5] = torch.where(torch.rand(N, generator=g) < 0.5, s[5], torch.zeros(()))
    eligible = (s >= 0).sum(-1)
    for k in sorted({0, 1, 2, 7, 128, 2048, N // 2, N}):
        kk = eligible.clamp(max=k).to(torch.int32)
        a = qsa.topk_block_mask(s, kk, int_bits=False)
        b = qsa.topk_block_mask(s, kk, int_bits=True)
        assert torch.equal(a, b), (N, k)
        assert torch.equal(b.sum(-1), kk.to(torch.int64)), (N, k)


def test_dense_shortcut():
    assert qsa.qsa_is_dense(2048, 512) and not qsa.qsa_is_dense(2048 + 128, 512)


# ---------------------------------------------------------------------------
# selection vs HF indexer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [17, 40, 64, 80])
@pytest.mark.parametrize("kind", ["random", "zero_blocks", "zero_q"])
def test_selection_matches_hf(tiny_hf_model, monkeypatch, T, kind):
    model = tiny_hf_model
    if kind == "zero_q":  # q_layernorm gamma = 0 -> q = 0 -> every score is 0
        ix = model.model.layers[_attn_layer_idx(model)].self_attn.indexer
        with torch.no_grad():
            ix.q_layernorm.weight.fill_(-1.0)
    hidden = _hidden(model, T, "zero_blocks" if kind == "zero_blocks" else "random")
    S = (T + 2 * BS) // BS * BS  # bucket larger than the context
    with torch.no_grad():
        ours, sel, scores = _our_selection(model, hidden, S)
        hf_any = _hf_selection(model, hidden)
        monkeypatch.setattr(torch.Tensor, "topk", _stable_topk)
        hf_stable = _hf_selection(model, hidden)
        monkeypatch.undo()
    assert not ours[:, T:].any()
    # exact vs HF with the lowest-index tie rule
    assert torch.equal(ours[:, :T], hf_stable)
    # tie-aware vs unpatched HF: same blocks above the k-th score, same selected scores
    budget = model.config.indexer_budget // model.config.indexer_compress_ratio
    nb = S // 4
    hf_blocks = hf_any[:, : T // 4 * 4].reshape(T, -1, 4).any(-1)
    hf_blocks = torch.cat(
        [hf_blocks, hf_blocks.new_zeros(T, nb - hf_blocks.shape[1])], -1
    )
    for t in range(T):
        vis = (t + 1) // 4
        hf_b = hf_blocks[t] & (torch.arange(nb) < vis)
        assert hf_b.sum() == sel[t].sum() == min(budget, vis)
        assert torch.equal(
            scores[t][hf_b].sort().values, scores[t][sel[t]].sort().values
        )
        tail = torch.arange(T) >= vis * 4
        assert torch.equal(hf_any[t] & tail, ours[t, :T] & tail)
    if kind != "random" and T >= 40:
        # the construction really produces ties that straddle the cut-off
        straddle = 0
        for t in range(T):
            if sel[t].any():
                kth = scores[t][sel[t]].min()
                straddle += int(((scores[t] == kth) & ~sel[t]).any())
        assert straddle > 0


# ---------------------------------------------------------------------------
# attention outputs vs HF Qwen4ExpTextAttention
# ---------------------------------------------------------------------------


class _Capture:
    """Records (hidden_states, output) of every call of the first attention layer."""

    def __init__(self, model):
        self.calls = []
        attn = model.model.layers[_attn_layer_idx(model)].self_attn
        self.h = attn.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs, out):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        self.calls.append((hs[0].detach().clone(), out[0][0].detach().clone()))

    def close(self):
        self.h.remove()


class _PagedQSAAttention:
    """Reference QSA attention over paged caches built from the qsa functions (fp32, torch).

    Mirrors qwen3_5 ``_attend_cached_torch`` with the QSA token mask ANDed into ``allowed``.
    One instance holds the caches of B requests (each with its own block table).
    """

    def __init__(self, model, n_blk, B=1, num_blocks=128):
        self.model = model
        self.attn = model.model.layers[_attn_layer_idx(model)].self_attn
        cfg = model.config
        self.Nh, self.Nkv, self.Dh = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        self.p = _idx_params(self.attn)
        self.inv_freq = _rotary(model)
        nan = float("nan")
        self.k_cache = torch.full((num_blocks, self.Nkv, BS, self.Dh), nan)
        self.v_cache = torch.full((num_blocks, self.Nkv, BS, self.Dh), nan)
        self.ix_cache = torch.full((num_blocks, 1, BS, cfg.indexer_head_dim), nan)
        g = torch.Generator().manual_seed(1)
        self.bt = (torch.randperm(num_blocks - 1, generator=g)[: B * n_blk] + 1).view(
            B, n_blk
        )
        self.S = n_blk * BS

    def _proj(self, h, positions):
        a, T = self.attn, h.shape[0]
        qg = (h @ a.q_proj.weight.T).view(T, self.Nh, 2 * self.Dh)
        q, gate = qg[..., : self.Dh], qg[..., self.Dh :].reshape(T, -1)
        q = qsa.zero_centered_rms_norm(q, 1 + a.q_norm.weight, a.q_norm.eps)
        k = (h @ a.k_proj.weight.T).view(T, self.Nkv, self.Dh)
        k = qsa.zero_centered_rms_norm(k, 1 + a.k_norm.weight, a.k_norm.eps)
        v = (h @ a.v_proj.weight.T).view(T, self.Nkv, self.Dh)
        cos, sin = qsa.qsa_rope_cos_sin(positions, self.inv_freq, h.dtype)
        q, k = qsa.apply_qsa_rope(q, cos, sin), qsa.apply_qsa_rope(k, cos, sin)
        qi, k_raw = qsa.index_qk(
            h,
            self.p["w_qk"],
            self.p["q_gamma"],
            cos,
            sin,
            n_heads=self.p["n_heads"],
            eps=self.p["eps"],
        )
        return q, k, v, gate, qi, k_raw

    def _write(self, rows, positions, k, v, k_raw):
        slots = self.bt[rows, positions // BS] * BS + positions % BS
        blk, off = slots // BS, slots % BS
        self.k_cache[blk, :, off] = k
        self.v_cache[blk, :, off] = v
        qsa.write_raw_keys(self.ix_cache, k_raw, slots)

    def _gather(self, cache):
        B, n = self.bt.shape
        return (
            cache[self.bt.reshape(-1)]
            .view(B, n, self.Nkv, BS, self.Dh)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, self.Nkv, self.S, self.Dh)
        )

    def token_mask(self, qi, q_pos):
        """qi [B, Tq, nh, D], q_pos [B, Tq] -> bool [B, Tq, S]."""
        kctx = qsa.gather_raw_keys(self.ix_cache, self.bt)
        return qsa.qsa_token_mask(
            qi,
            kctx,
            q_pos,
            self.p["k_gamma"],
            self.inv_freq,
            budget_blocks=self.p["budget_blocks"],
            ratio=self.p["ratio"],
            eps=self.p["eps"],
        )

    def _attend(self, q, gate, allowed, n_valid):
        """q [B, Tq, Nh, Dh], allowed [B, Tq, S], n_valid [B] -> o_proj output [B, Tq, H]."""
        B, Tq = q.shape[:2]
        kc, vc = (
            self._gather(self.k_cache),
            self._gather(self.v_cache),
        )  # [B, Nkv, S, Dh]
        key_ok = torch.arange(self.S).view(1, -1) < n_valid.view(-1, 1)  # [B, S]
        allowed = allowed & key_ok[:, None, :]
        vc = torch.where(key_ok[:, None, :, None], vc, torch.zeros(()))
        rep = self.Nh // self.Nkv
        kc, vc = kc.repeat_interleave(rep, 1), vc.repeat_interleave(rep, 1)
        s = torch.einsum("bthd,bhsd->bhts", q, kc) * self.attn.scaling
        s = torch.where(allowed[:, None], s, torch.full((), -1e30))
        o = torch.einsum("bhts,bhsd->bthd", torch.softmax(s, -1), vc).reshape(B, Tq, -1)
        o = o * torch.sigmoid(gate.view(B, Tq, -1))
        return o @ self.attn.o_proj.weight.T

    def prefill(self, h, start):
        T = h.shape[0]
        positions = torch.arange(start, start + T)
        q, k, v, gate, qi, k_raw = self._proj(h, positions)
        self._write(torch.zeros(T, dtype=torch.long), positions, k, v, k_raw)
        assert self.bt.shape[0] == 1, "prefill reference is single-request"
        q_pos = positions.view(1, -1)
        tm = self.token_mask(qi.unsqueeze(0), q_pos)
        allowed = tm & (torch.arange(self.S).view(1, 1, -1) <= q_pos.unsqueeze(-1))
        return self._attend(q.unsqueeze(0), gate, allowed, torch.tensor([start + T]))[0]

    def decode_inputs(self, h, positions):
        """One token per request: write caches, return (q [B,1,Nh,Dh], k, v, gate, tok_mask)."""
        q, k, v, gate, qi, k_raw = self._proj(h, positions)
        self._write(torch.arange(h.shape[0]), positions, k, v, k_raw)
        tm = self.token_mask(qi.unsqueeze(1), positions.view(-1, 1))
        return q.unsqueeze(1), k, v, gate, tm

    def decode(self, h, positions):
        q, _, _, gate, tm = self.decode_inputs(h, positions)
        allowed = tm & (torch.arange(self.S).view(1, 1, -1) <= positions.view(-1, 1, 1))
        return self._attend(q, gate, allowed, positions + 1)[:, 0]


def _run_hf(model, ids, chunks):
    """Run HF over ``ids`` in the given chunk lengths with a cache; return captured calls."""
    from transformers import DynamicCache

    cap = _Capture(model)
    try:
        with torch.no_grad():
            pkv = DynamicCache(config=model.config)
            s = 0
            for n in chunks:
                model(ids[:, s : s + n], past_key_values=pkv, use_cache=True)
                s += n
    finally:
        cap.close()
    return cap.calls


@pytest.mark.parametrize("chunks", [[72], [40, 32], [24, 24, 25]])
def test_prefill_attention_matches_hf(tiny_hf_model, chunks):
    model = tiny_hf_model
    ids = torch.randint(
        2,
        model.config.vocab_size,
        (1, sum(chunks)),
        generator=torch.Generator().manual_seed(3),
    )
    calls = _run_hf(model, ids, chunks)
    ref = _PagedQSAAttention(model, n_blk=(sum(chunks) + BS) // BS + 2)
    start = 0
    with torch.no_grad():
        for (h, out), n in zip(calls, chunks):
            ours = ref.prefill(h, start)
            torch.testing.assert_close(ours, out, atol=2e-5, rtol=1e-4)
            start += n


def _decode_setup(model, prompts, steps):
    """HF incremental decode of each prompt; returns per-request prefill call + per-step calls."""
    per_req = []
    for ids in prompts:
        calls = _run_hf(model, ids, [ids.shape[1] - steps] + [1] * steps)
        per_req.append(calls)
    return per_req


def test_decode_attention_matches_hf(tiny_hf_model):
    model = tiny_hf_model
    g = torch.Generator().manual_seed(5)
    steps = 24
    prompts = [torch.randint(2, 1024, (1, n + steps), generator=g) for n in (37, 58)]
    per_req = _decode_setup(model, prompts, steps)
    n_blk = 12
    ref = _PagedQSAAttention(model, n_blk=n_blk, B=2)
    with torch.no_grad():
        # prefill each request into its own block-table row (reference prefill is per request)
        for r, calls in enumerate(per_req):
            h, out = calls[0]
            single = _PagedQSAAttention.__new__(_PagedQSAAttention)
            single.__dict__.update(ref.__dict__)
            single.bt = ref.bt[r : r + 1]
            torch.testing.assert_close(single.prefill(h, 0), out, atol=2e-5, rtol=1e-4)
        lens = torch.tensor([c[0][0].shape[0] for c in per_req])
        for i in range(steps):
            h = torch.cat([per_req[r][1 + i][0] for r in range(2)], 0)
            out = torch.cat([per_req[r][1 + i][1] for r in range(2)], 0)
            ours = ref.decode(h, lens + i)
            torch.testing.assert_close(ours, out, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("compact", [False, True])
def test_decode_kernel_mask_through_attention_decode(tiny_hf_model, compact):
    """QSA token mask -> ``decode_mask_from_token_mask`` -> ``NF.attention_decode`` (torch
    fallback on CPU, which reads the mask in the kernel's layout) matches HF decode. With
    ``compact`` the kernel instead reads only the selected tokens: the caches in pages of one
    QSA block and the block table / mask of ``compact_decode_selection``."""
    import vllm_neuron.functional as NF

    model = tiny_hf_model
    g = torch.Generator().manual_seed(9)
    steps = 6
    prompts = [
        torch.randint(2, 1024, (1, n + steps), generator=g) for n in (45, 70, 52)
    ]
    per_req = _decode_setup(model, prompts, steps)
    B = len(prompts)
    n_blk = 128 // BS  # S_ctx must be a multiple of 128
    ref = _PagedQSAAttention(model, n_blk=n_blk, B=B, num_blocks=64)
    with torch.no_grad():
        for r, calls in enumerate(per_req):
            single = _PagedQSAAttention.__new__(_PagedQSAAttention)
            single.__dict__.update(ref.__dict__)
            single.bt = ref.bt[r : r + 1]
            single.prefill(calls[0][0], 0)
        lens = torch.tensor([c[0][0].shape[0] for c in per_req])
        rep = ref.Nh // ref.Nkv
        Dh = ref.Dh
        for i in range(steps):
            h = torch.cat([per_req[r][1 + i][0] for r in range(B)], 0)
            out_hf = torch.cat([per_req[r][1 + i][1] for r in range(B)], 0)
            pos = lens + i
            # snapshot caches before this step's write: the kernel takes the new K/V as inputs
            k_prev, v_prev = ref.k_cache.clone(), ref.v_cache.clone()
            q, k, v, gate, tm = ref.decode_inputs(h, pos)
            if compact:
                # the block selection behind tm: its complete blocks with all tokens allowed
                R = ref.p["ratio"]
                nb = ref.S // R
                sel = tm.view(B, 1, nb, R).all(-1) & (
                    torch.arange(nb) < (pos.view(B, 1, 1) + 1) // R
                )
                n_slots = qsa.compact_slots(ref.p["budget_blocks"], R)
                # as an index list in shuffled order, padded with invalid entries
                key = torch.rand(B, nb, generator=g) + sel[:, 0].float()
                vals, idx = key.topk(ref.p["budget_blocks"], -1)
                valid = vals >= 1
                table, seq = qsa.compact_decode_selection(
                    idx, valid, pos.view(B, 1), ref.bt, BS, n_slots, R
                )
                assert seq.sum() == tm.sum()  # the same tokens, compacted
                base = NF.gen_attention_decode_mask(
                    pos_ids=torch.full((1, B), n_slots - 1.0),
                    bs=B,
                    q_head=rep,
                    s_active=1,
                    s_prior=n_slots,
                    start_pos=None,
                    block_len=R,
                )
                mask = qsa.compact_decode_mask(seq, pos.view(B, 1), base, R)
                bt, page = table, R
            else:
                base = NF.gen_attention_decode_mask(
                    pos_ids=pos.view(1, B).float(),
                    bs=B,
                    q_head=rep,
                    s_active=1,
                    s_prior=ref.S,
                    start_pos=None,
                    block_len=BS,
                )
                mask = qsa.decode_mask_from_token_mask(tm, pos.view(B, 1), base, BS)
                bt, page = ref.bt.to(torch.int32), BS
            outs = []
            for kv in range(ref.Nkv):  # one KV head per "rank", like TP deployment
                qr = q[:, 0, kv * rep : (kv + 1) * rep].reshape(B, 1, rep * Dh)
                X = torch.cat(
                    [qr, k[:, kv].view(B, 1, Dh), v[:, kv].view(B, 1, Dh)], -1
                )
                o, _, _ = NF.attention_decode(
                    X=X,
                    W_qkv=torch.eye(X.shape[-1]),
                    active_blocks_table=bt,
                    K_cache=torch.nan_to_num(k_prev[:, kv]).reshape(-1, page, Dh),
                    V_cache=torch.nan_to_num(v_prev[:, kv]).reshape(-1, page, Dh),
                    attention_mask=mask,
                    softmax_scale=ref.attn.scaling,
                    update_cache=False,
                )  # [B, rep, Dh, 1]
                outs.append(o[..., 0])
            o = torch.cat(outs, 1).reshape(B, -1) * torch.sigmoid(gate)
            torch.testing.assert_close(
                o @ ref.attn.o_proj.weight.T, out_hf, atol=2e-5, rtol=1e-4
            )


def test_topk_block_ids_matches_topk_block_mask():
    """The index-list top-k selects the same blocks as the mask top-k (no ties here), with
    the invisible blocks of short requests marked invalid."""
    g = torch.Generator().manual_seed(1)
    nb, k = 600, 64
    q_pos = torch.tensor([[2399], [100], [255]])  # visible 600, 25, 64 blocks
    vis = torch.arange(nb) < (q_pos + 1) // 4
    scores = torch.where(vis, torch.rand(3, nb, generator=g), -1.0).unsqueeze(1)
    mask = qsa.topk_block_mask(scores, ((q_pos + 1) // 4).clamp(max=k))[:, 0]
    idx, valid = qsa.topk_block_ids(scores, k)
    got = torch.zeros_like(mask)
    for b in range(3):
        got[b, idx[b][valid[b]].long()] = True
    assert torch.equal(got, mask)
    assert valid.sum(-1).tolist() == [64, 25, 64]


@pytest.mark.parametrize("S_tkg", [1, 2, 3])
@pytest.mark.parametrize("S_ctx,block_len", [(128, 16), (512, 32), (1024, 128)])
def test_decode_mask_layout_roundtrip(S_tkg, S_ctx, block_len):
    """The kernel-layout mask, read back with the torch fallback's inverse layout map, is
    (causal base mask) AND (token selection at the right cache / active slots)."""
    import vllm_neuron.functional as NF
    from vllm_neuron.functional.attention.attention_decode_mask import (
        P_MAX,
        _resize_block_len,
    )

    B, H = 3, 3
    g = torch.Generator().manual_seed(S_ctx + S_tkg)
    p0 = torch.tensor([5, S_ctx // 2, S_ctx - S_tkg - 1])
    q_pos = p0.view(B, 1) + torch.arange(S_tkg).view(1, -1)
    tm = torch.rand(B, S_tkg, S_ctx, generator=g) < 0.5
    base = NF.gen_attention_decode_mask(
        pos_ids=q_pos.reshape(1, -1).float(),
        bs=B,
        q_head=H,
        s_active=S_tkg,
        s_prior=S_ctx,
        start_pos=None,
        block_len=block_len,
    )
    out = qsa.decode_mask_from_token_mask(tm, q_pos, base, block_len)

    def to_seq(m):  # same map as attention_decode._torch_attention_decode_impl
        bl = _resize_block_len(block_len, B, H, S_tkg, S_ctx)
        return (
            m.permute(1, 2, 3, 0)
            .reshape(-1, bl, P_MAX)
            .swapaxes(-1, -2)
            .reshape(B, H, S_tkg, S_ctx)
        )

    seq_base, seq_out = to_seq(base), to_seq(out)
    expect = tm.clone()
    for b in range(B):
        for i in range(S_tkg):
            expect[b, :, S_ctx - S_tkg + i] = tm[b, :, int(p0[b]) + i]
    assert torch.equal(seq_out, seq_base * expect[:, None].float())
    # the base mask is the causal pattern: prior slots < p0, active staircase at the end
    j = torch.arange(S_ctx)
    for b in range(B):
        for t in range(S_tkg):
            row = (j < p0[b]) | ((j >= S_ctx - S_tkg) & (j <= S_ctx - S_tkg + t))
            assert torch.equal(seq_base[b, 0, t] > 0, row)


@pytest.mark.parametrize(
    "B,H,S_tkg", [(1, 3, 1), (8, 3, 1), (8, 3, 2), (4, 6, 1), (64, 3, 1)]
)
def test_nki_mask_block_len_matches_nkilib(B, H, S_tkg):
    from nkilib.core.attention.attention_tkg_utils import (
        resize_cache_block_len_for_attention_tkg_kernel,
    )

    for S in (512, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072):
        for bl in (16, 32, 64, 128, 256):
            for lnc in (1, 2):
                ref, _ = resize_cache_block_len_for_attention_tkg_kernel(
                    S // bl, bl, lnc, 128, B, H, S_tkg
                )
                assert qsa.nki_mask_block_len(bl, B, H, S_tkg, S, lnc) == ref, (
                    S,
                    bl,
                    lnc,
                )
