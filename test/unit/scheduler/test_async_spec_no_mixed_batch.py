# SPDX-License-Identifier: Apache-2.0
"""NeuronAsyncScheduler: no step mixes requests with and without drafts (CPU).

Near max_model_len a request stops drafting (sticky). The verify graph lays a spec batch
out as [requests, 1 + num_spec] tokens, so a one-token request among drafting ones
shifts the later requests' tokens. While any running request has drafting disabled, no
request gets draft placeholders.
"""

import types

import pytest

pytest.importorskip("vllm")


def _req(rid, num_computed):
    return types.SimpleNamespace(
        request_id=rid,
        num_computed_tokens=num_computed,
        is_prefill_chunk=False,
        use_structured_output=False,
        num_output_placeholders=0,
        spec_token_ids=[],
    )


def _scheduler(reqs, max_model_len=4096, num_spec=2):
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    s = object.__new__(NeuronAsyncScheduler)
    s.requests = {r.request_id: r for r in reqs}
    s.running = list(reqs)
    s.max_model_len = max_model_len
    s.num_spec_tokens = num_spec
    s._spec_token_placeholders = [-1] * num_spec
    return s, Scheduler


def _output(scheduled):
    return types.SimpleNamespace(
        num_scheduled_tokens={r.request_id: 3 for r in scheduled},
        scheduled_spec_decode_tokens={r.request_id: [-1, -1] for r in scheduled},
        pending_structured_output_tokens=False,
    )


def test_one_request_near_max_len_stops_drafting_for_all(monkeypatch):
    far, near, held = _req("far", 100), _req("near", 4090), _req("held", 200)
    s, base = _scheduler([far, near, held])
    monkeypatch.setattr(base, "_update_after_schedule", lambda self, out: None)
    s._update_after_schedule(_output([far, near]))  # "held" sat this step out
    assert near._async_spec_disabled
    assert far.spec_token_ids == [] and near.spec_token_ids == []
    assert held.spec_token_ids == []
    assert not getattr(far, "_async_spec_disabled", False)  # not sticky for the others


def test_drafting_resumes_once_the_disabled_request_is_gone(monkeypatch):
    far = _req("far", 100)
    s, base = _scheduler([far])
    monkeypatch.setattr(base, "_update_after_schedule", lambda self, out: None)
    s._update_after_schedule(_output([far]))
    assert far.spec_token_ids == [-1, -1]
