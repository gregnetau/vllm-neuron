# SPDX-License-Identifier: Apache-2.0
"""On-device greedy rejection sampler with fewer drafts than max_spec_len (CPU).

Near max_model_len the scheduler stops drafting for some requests while others keep
drafting, so one verify step mixes requests with 2 drafts and with none. A request
without drafts must still emit its bonus token: an empty row leaves the scheduler
waiting forever for the token it reserved.
"""

import torch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_neuron.nn.rejection_sampler import rejection_sampler


def _metadata(num_draft_tokens, draft_token_ids):
    cu = torch.tensor(num_draft_tokens).cumsum(0)
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(
            [t for d in draft_token_ids for t in d], dtype=torch.int32
        ),
        num_draft_tokens=list(num_draft_tokens),
        cu_num_draft_tokens=cu,
        cu_num_sampled_tokens=cu + torch.arange(1, len(num_draft_tokens) + 1),
        target_logits_indices=torch.zeros(int(cu[-1]), dtype=torch.long),
        bonus_logits_indices=torch.zeros(len(num_draft_tokens), dtype=torch.long),
        logits_indices=torch.zeros(
            int(cu[-1]) + len(num_draft_tokens), dtype=torch.long
        ),
    )


def test_mixed_draft_counts_emit_bonus_tokens():
    # Request 0: 2 drafts, both accepted. Request 1: 2 drafts, second rejected.
    # Request 2: no drafts. Request 3: 1 draft, accepted.
    md = _metadata([2, 2, 0, 1], [[5, 6], [8, 9], [], [12]])
    target = torch.tensor(
        [5, 6, 7, 8, 1, 2, 11, 12, 13], dtype=torch.int32
    )  # [targets_r, bonus_r] per request
    out = rejection_sampler(md, target)
    assert out.tolist() == [
        [5, 6, 7],
        [8, 1, -1],
        [11, -1, -1],
        [12, 13, -1],
    ]


def test_full_draft_rows_unchanged():
    md = _metadata([2, 2], [[1, 2], [3, 4]])
    target = torch.tensor([1, 2, 9, 3, 7, 8], dtype=torch.int32)
    assert rejection_sampler(md, target).tolist() == [[1, 2, 9], [3, 7, -1]]


# --- Fixed [batch, 1 + k] layout (Model Runner V2) ---------------------------------


def test_fixed_layout_mixed_draft_counts():
    from vllm_neuron.nn.rejection_sampler import rejection_sample_fixed

    # Same requests as test_mixed_draft_counts_emit_bonus_tokens, each in a row of
    # 1 + k = 3 columns; padding columns hold arbitrary target samples.
    target = torch.tensor(
        [[5, 6, 7], [8, 1, 2], [11, 99, 98], [12, 13, 97]], dtype=torch.int32
    )
    drafts = torch.tensor([[5, 6], [8, 9], [42, 43], [12, 44]], dtype=torch.int32)
    out = rejection_sample_fixed(target, drafts, torch.tensor([2, 2, 0, 1]))
    assert out.tolist() == [
        [5, 6, 7],
        [8, 1, -1],
        [11, -1, -1],
        [12, 13, -1],
    ]


def test_fixed_layout_matches_the_packed_sampler():
    from vllm_neuron.nn.rejection_sampler import rejection_sample_fixed

    torch.manual_seed(0)
    k = 3
    for _ in range(20):
        counts = [k] + torch.randint(0, k + 1, (4,)).tolist()  # packed: >= 1 draft
        target = torch.randint(0, 4, (5, k + 1), dtype=torch.int32)
        drafts = torch.randint(0, 4, (5, k), dtype=torch.int32)
        packed_target = torch.cat([target[b, : counts[b] + 1] for b in range(5)]).to(
            torch.int32
        )
        md = _metadata(counts, [drafts[b, : counts[b]].tolist() for b in range(5)])
        md.max_spec_len = k  # the packed sampler's width follows max_spec_len
        expected = rejection_sampler(md, packed_target)
        out = rejection_sample_fixed(target, drafts, torch.tensor(counts))
        assert out.tolist() == expected.tolist(), (counts, target, drafts)


def test_fixed_layout_synthetic_rates():
    from vllm_neuron.nn.rejection_sampler import rejection_sample_fixed

    target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    drafts = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    counts = torch.tensor([2, 1])
    accept_all = rejection_sample_fixed(target, drafts, counts, [1.0, 1.0])
    assert accept_all.tolist() == [[7, 8, 3], [9, 5, -1]]
    reject_all = rejection_sample_fixed(target, drafts, counts, [0.0, 0.0])
    assert reject_all.tolist() == [[1, -1, -1], [4, -1, -1]]
