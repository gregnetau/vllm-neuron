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
