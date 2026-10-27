# SPDX-License-Identifier: Apache-2.0
"""NeuronAsyncScheduler: recurrent-state models with spec decode end a request a few
tokens before max_model_len, so it never has to stop drafting (CPU).

A request that stops drafting near max_model_len makes its steps differ from the other
requests' steps (mixed draft counts, or batch-wide spec / non-spec switches), which
corrupted the other requests' tokens under load.
"""

import types

import pytest

pytest.importorskip("vllm")


def _scheduler(group_specs, num_spec=2, max_model_len=4096):
    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    s = object.__new__(NeuronAsyncScheduler)
    s.num_spec_tokens = num_spec
    s.max_model_len = max_model_len
    s.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=g) for g in group_specs]
    )
    return s


def _request(num_prompt, max_tokens):
    return types.SimpleNamespace(
        request_id="r", num_prompt_tokens=num_prompt, max_tokens=max_tokens
    )


@pytest.fixture
def specs():
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    state = MambaSpec(block_size=4096, shapes=((4, 4),), dtypes=(torch.float32,))
    return attn, state


def _add(s, req, monkeypatch):
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    monkeypatch.setattr(NeuronScheduler, "add_request", lambda self, r: None)
    s.add_request(req)
    return req.max_tokens


def test_hybrid_spec_caps_max_tokens(specs, monkeypatch):
    attn, state = specs
    s = _scheduler([attn, state])
    assert _add(s, _request(3000, 2000), monkeypatch) == 4096 - 12 - 3000
    assert _add(s, _request(3000, 100), monkeypatch) == 100  # already short
    assert _add(s, _request(4090, 6), monkeypatch) == 1  # prompt inside the margin


def test_attention_only_and_no_spec_unchanged(specs, monkeypatch):
    attn, state = specs
    assert _add(_scheduler([attn]), _request(3000, 2000), monkeypatch) == 2000
    s = _scheduler([attn, state], num_spec=0)
    assert _add(s, _request(3000, 2000), monkeypatch) == 2000


def test_v2_runner_unchanged(specs, monkeypatch):
    attn, state = specs
    s = _scheduler([attn, state])
    s.use_v2_model_runner = True
    assert _add(s, _request(3000, 2000), monkeypatch) == 2000
