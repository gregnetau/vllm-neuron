# SPDX-License-Identifier: Apache-2.0
"""interfaces.HasHostInputs: the V1 runner's per-step host inputs and their warmup zero-fill (CPU).

A fake model implements the protocol and records the arguments the runner passes. The runner is
built without ``__init__`` (``object.__new__``) and given only the attributes these paths read.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")
interfaces = pytest.importorskip("vllm_neuron.model.interfaces")

HasHostInputs = interfaces.HasHostInputs
NeuronModelRunner = runner_mod.NeuronModelRunner
host_input_req_rows = runner_mod.host_input_req_rows
host_input_zero_kwargs = runner_mod.host_input_zero_kwargs
build_step_host_inputs = runner_mod.build_step_host_inputs

D = 8
SPECS = {"ple_embeds": ((D,), torch.bfloat16)}


class FakeHostInputModel(torch.nn.Module):
    """Returns, per row, [req_row, position, input_id, previous token, 0...] so the test can
    read the arguments back from the output; records the last call."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def host_input_specs(self):
        return dict(SPECS)

    def build_host_inputs(self, input_ids, req_rows, positions, token_ids_cpu):
        self.calls.append((input_ids.copy(), req_rows.copy(), positions.copy()))
        out = torch.zeros(input_ids.shape[0], D, dtype=torch.bfloat16)
        for t in range(input_ids.shape[0]):
            r = int(req_rows[t])
            if r < 0:
                continue
            # Previous token: an earlier row of the same request first, then the history.
            if t > 0 and req_rows[t - 1] == r:
                prev = int(input_ids[t - 1])
            else:
                prev = (
                    int(token_ids_cpu[r, positions[t] - 1]) if positions[t] > 0 else -1
                )
            out[t, :4] = torch.tensor(
                [r, positions[t], input_ids[t], prev], dtype=torch.float32
            )
        return {"ple_embeds": out}

    def forward(self, **kwargs):
        return kwargs


class PlainModel(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs


class CompiledWrapper:
    """Stands in for torch.compile's OptimizedModule (``_orig_mod``)."""

    def __init__(self, mod):
        self._orig_mod = mod


def make_runner(model, *, async_scheduling=False, token_ids_cpu=None):
    runner = object.__new__(NeuronModelRunner)
    runner.model = model
    runner.use_async_scheduling = async_scheduling
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        token_ids_cpu=token_ids_cpu
        if token_ids_cpu is not None
        else np.zeros((4, 64), dtype=np.int32)
    )
    return runner


def test_protocol_runtime_checkable():
    assert isinstance(FakeHostInputModel(), HasHostInputs)
    assert not isinstance(PlainModel(), HasHostInputs)


# --- req_rows layout -------------------------------------------------------------------------


def test_req_rows_prefill_chunk_with_bucket_padding():
    # One request, a 5-token chunk padded to an 8-token prefill bucket.
    rows = host_input_req_rows(np.array([5]), np.array([3]), 8)
    assert rows.tolist() == [0, 0, 0, 0, 0, -1, -1, -1]
    assert rows.dtype == np.int64


def test_req_rows_padded_decode_batch():
    # Three decode requests in a 4-request bucket (the 4th row is decode batch padding).
    rows = host_input_req_rows(np.array([1, 1, 1]), np.zeros(3, np.int64), 4)
    assert rows.tolist() == [0, 1, 2, -1]


def test_req_rows_spec_verify_and_mixed_batch():
    # Spec verify, 1 + 2 drafts per request, 2 requests in a 3-request bucket.
    rows = host_input_req_rows(np.array([3, 3]), np.zeros(2, np.int64), 9)
    assert rows.tolist() == [0, 0, 0, 1, 1, 1, -1, -1, -1]
    # Mixed spec batch (DI): the first request has no drafts; padding goes to the end.
    rows = host_input_req_rows(np.array([1, 3]), np.zeros(2, np.int64), 6)
    assert rows.tolist() == [0, 1, 1, 1, -1, -1]


def test_req_rows_too_many_rows_raises():
    with pytest.raises(ValueError, match="more than"):
        host_input_req_rows(np.array([4]), np.array([2]), 5)


# --- zero fill -------------------------------------------------------------------------------


def test_zero_kwargs_without_capability_is_empty():
    assert host_input_zero_kwargs(None, 16, "cpu") == {}
    runner = make_runner(PlainModel())
    assert runner._host_input_warmup_kwargs(16) == {}
    assert runner._host_input_specs() is None


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_warmup_zero_fill(device):
    runner = make_runner(CompiledWrapper(FakeHostInputModel()))
    kw = runner._host_input_warmup_kwargs(12, torch.device(device))
    assert set(kw) == {"ple_embeds"}
    t = kw["ple_embeds"]
    assert t.shape == (12, D) and t.dtype == torch.bfloat16 and t.device.type == device
    if device == "cpu":
        assert not t.any()


def test_prefill_synthetic_inputs_carry_zero_host_inputs():
    runner = make_runner(FakeHostInputModel())
    runner.uses_mrope = False
    runner.rank_tensor = torch.zeros(1, dtype=torch.int32)
    runner.supports_mm_inputs = False
    runner.encoder_cache = None
    runner.enable_prompt_embeds = False
    runner._build_warmup_attention_metadata = lambda **kw: {}
    runner._build_noop_logit_mask = lambda *a, **kw: None
    kw = runner._build_prefill_synthetic_inputs(32, 0)
    assert kw["ple_embeds"].shape == (32, D)
    assert kw["ple_embeds"].dtype == torch.bfloat16
    # A model without the capability gets no host-input kwargs.
    runner.model = PlainModel()
    runner._host_input_specs_resolved = False
    assert "ple_embeds" not in runner._build_prefill_synthetic_inputs(32, 0)


def test_decode_synthetic_inputs_carry_zero_host_inputs():
    runner = make_runner(FakeHostInputModel())
    runner.uses_mrope = False
    runner.rank_tensor = torch.zeros(1, dtype=torch.int32)
    runner.speculative_config = None
    runner.enable_prompt_embeds = False
    runner.encoder_neuron_config = None
    runner.on_device_sampling = True
    runner._build_warmup_attention_metadata = lambda **kw: {}
    runner._build_noop_logit_mask = lambda *a, **kw: None
    runner._use_async_reorder = lambda: False
    kw = runner._build_decode_synthetic_inputs(4, compiled_graph_input=True)
    assert kw["ple_embeds"].shape == (4, D)


# --- real step -------------------------------------------------------------------------------


def test_step_prefill_chunk_arguments():
    # Request in input_batch row 1, prompt [10..19]; this chunk is tokens 4..8 (5 tokens) padded
    # to an 8-token bucket. Padding rows repeat the last position (as _create_padded_inputs does).
    token_ids_cpu = np.zeros((4, 64), dtype=np.int32)
    token_ids_cpu[1, :10] = np.arange(10, 20)
    model = FakeHostInputModel()
    runner = make_runner(model, token_ids_cpu=token_ids_cpu)
    # Row 0 is another request with nothing scheduled this step.
    input_ids = torch.tensor([14, 15, 16, 17, 18, 0, 0, 0], dtype=torch.int32)
    positions = torch.tensor([4, 5, 6, 7, 8, 8, 8, 8], dtype=torch.long)
    req_rows = host_input_req_rows(np.array([0, 5]), np.array([0, 3]), 8)
    runner._set_step_host_inputs(input_ids, req_rows, positions)

    ids, rows, pos = model.calls[-1]
    assert ids.dtype == np.int64 and ids.tolist() == input_ids.tolist()
    assert rows.tolist() == [1, 1, 1, 1, 1, -1, -1, -1]
    assert pos.tolist() == [4, 5, 6, 7, 8, 0, 0, 0]  # padding rows -> 0
    out = runner._current_host_inputs["ple_embeds"].float()
    # First row resolves its previous token from the history, the rest from earlier rows.
    assert out[:5, 3].tolist() == [13, 14, 15, 16, 17]
    assert not out[5:].any()


def test_step_padded_decode_and_spec_verify_arguments():
    token_ids_cpu = np.zeros((4, 64), dtype=np.int32)
    token_ids_cpu[0, :6] = [1, 2, 3, 4, 5, 6]
    token_ids_cpu[1, :3] = [7, 8, 9]
    model = FakeHostInputModel()
    runner = make_runner(CompiledWrapper(model), token_ids_cpu=token_ids_cpu)

    # Plain decode: 2 requests (computed 5 and 2 tokens) in a 4-request bucket.
    input_ids = torch.tensor([6, 9, 0, 0], dtype=torch.int32)
    positions = torch.tensor([5, 2, 0, 0], dtype=torch.long)
    rows = host_input_req_rows(np.array([1, 1]), np.zeros(2, np.int64), 4)
    runner._set_step_host_inputs(input_ids, rows, positions)
    _, r, p = model.calls[-1]
    assert r.tolist() == [0, 1, -1, -1] and p.tolist() == [5, 2, 0, 0]
    out = runner._current_host_inputs["ple_embeds"].float()
    assert out[:2, 3].tolist() == [5, 8]

    # Spec verify, 1 + 2 drafts: drafts (40, 41 / 50, 51) are rows of the same request at
    # positions num_computed + j; the model resolves their previous token from earlier rows.
    input_ids = torch.tensor([6, 40, 41, 9, 50, 51, 0, 0, 0], dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 2, 3, 4, 0, 0, 0], dtype=torch.long)
    rows = host_input_req_rows(np.array([3, 3]), np.zeros(2, np.int64), 9)
    runner._set_step_host_inputs(input_ids, rows, positions)
    _, r, p = model.calls[-1]
    assert r.tolist() == [0, 0, 0, 1, 1, 1, -1, -1, -1]
    assert p.tolist() == [5, 6, 7, 2, 3, 4, 0, 0, 0]
    out = runner._current_host_inputs["ple_embeds"].float()
    assert out[:6, 3].tolist() == [5, 6, 40, 8, 9, 50]


def test_step_without_capability_builds_nothing():
    runner = make_runner(PlainModel())
    runner._current_host_inputs = {"stale": torch.zeros(1)}
    runner._set_step_host_inputs(
        torch.zeros(4, dtype=torch.int32),
        np.full(4, -1),
        torch.zeros(4, dtype=torch.long),
    )
    assert runner._current_host_inputs is None


def test_async_spec_decode_is_rejected():
    runner = make_runner(FakeHostInputModel(), async_scheduling=True)
    runner.speculative_config = SimpleNamespace(method="mtp")
    with pytest.raises(RuntimeError, match="async scheduling"):
        runner._host_input_specs()


def test_async_without_spec_decode_uses_the_join():
    runner = make_runner(FakeHostInputModel(), async_scheduling=True)
    runner.speculative_config = None
    assert runner._host_input_specs() == SPECS
    assert runner._host_inputs_async()
    assert not make_runner(FakeHostInputModel())._host_inputs_async()
    assert not make_runner(PlainModel(), async_scheduling=True)._host_inputs_async()


@pytest.mark.parametrize(
    "bad",
    [
        {"other": torch.zeros(4, D, dtype=torch.bfloat16)},
        {"ple_embeds": torch.zeros(4, D, dtype=torch.float32)},
        {"ple_embeds": torch.zeros(3, D, dtype=torch.bfloat16)},
    ],
)
def test_build_step_host_inputs_validates(bad):
    model = FakeHostInputModel()
    model.build_host_inputs = lambda *a: bad
    with pytest.raises(ValueError):
        build_step_host_inputs(
            model,
            SPECS,
            np.zeros(4, np.int32),
            np.zeros(4, np.int64),
            np.zeros(4, np.int64),
            np.zeros((1, 8), np.int32),
        )


# --- platform --------------------------------------------------------------------------------


class _HostInputFactory(torch.nn.Module, HasHostInputs):
    """A registered factory class that declares the capability."""


@pytest.mark.parametrize(
    "model_cls, spec, env, v2, expect_async",
    [
        (_HostInputFactory, None, None, False, True),  # async with the host-token join
        (_HostInputFactory, SimpleNamespace(method="mtp"), None, False, False),
        (_HostInputFactory, None, "0", False, False),
        (PlainModel, SimpleNamespace(method="mtp"), "0", False, True),
        # Model Runner V2's join also covers verify steps.
        (_HostInputFactory, SimpleNamespace(method="mtp"), None, True, True),
        (_HostInputFactory, SimpleNamespace(method="mtp"), "0", True, False),
    ],
)
def test_platform_async_scheduling_choice(
    monkeypatch, model_cls, spec, env, v2, expect_async
):
    platform_mod = pytest.importorskip("vllm_neuron.vllm.platform")
    registry = pytest.importorskip("vllm_neuron.model.registry")
    monkeypatch.setattr(registry, "get_model_cls", lambda archs: model_cls)
    if env is None:
        monkeypatch.delenv("VLLM_NEURON_HOST_INPUTS_ASYNC", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_HOST_INPUTS_ASYNC", env)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["FakeArch"]),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=spec,
        use_v2_model_runner=v2,  # the cached runner decision
    )
    platform_mod.NeuronPlatform._configure_host_input_models(cfg)
    assert cfg.scheduler_config.async_scheduling is expect_async


# --- async host-token join -------------------------------------------------------------------


def make_async_runner(prompts):
    """A runner under async scheduling with real InputBatch-like arrays and request states.

    ``prompts`` maps req_id -> prompt token list; row i holds the i-th request.
    """
    token_ids_cpu = np.zeros((4, 64), dtype=np.int32)
    model = FakeHostInputModel()
    runner = make_runner(
        CompiledWrapper(model), async_scheduling=True, token_ids_cpu=token_ids_cpu
    )
    runner.speculative_config = None
    runner.vocab_size = 1000
    runner._input_batch_lock = runner_mod.threading.Lock()
    req_ids = list(prompts)
    runner.input_batch.req_ids = req_ids
    runner.input_batch.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
    runner.input_batch.token_ids_cpu_tensor = torch.from_numpy(token_ids_cpu)
    runner.input_batch.num_tokens_no_spec = np.zeros(4, dtype=np.int32)
    runner.requests = {}
    for i, (req_id, prompt) in enumerate(prompts.items()):
        token_ids_cpu[i, : len(prompt)] = prompt
        runner.input_batch.num_tokens_no_spec[i] = len(prompt)
        runner.requests[req_id] = SimpleNamespace(output_token_ids=[])
    return runner, model


def async_output(runner, req_ids, sampled):
    """An async output whose sampled tokens are still a (CPU stand-in for a device) tensor."""
    mro = SimpleNamespace(
        req_ids=list(req_ids), sampled_token_ids=torch.tensor(sampled)
    )
    return runner_mod.AsyncNeuronModelRunnerOutput(mro, runner)


def test_join_drains_pending_steps_in_order_and_uses_true_tokens():
    # Non-output TP rank: nobody materialized the last two steps. Step N-1 sampled (7, 8),
    # step N sampled (9, 10); step N+1 decodes positions 4 and 3 in a 4-request bucket.
    runner, model = make_async_runner({"a": [1, 2, 3], "b": [4, 5]})
    runner._track_host_join_output(async_output(runner, ["a", "b"], [7, 8]))
    runner._track_host_join_output(async_output(runner, ["a", "b"], [9, 10]))
    stale = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    rows = host_input_req_rows(np.array([1, 1]), np.zeros(2, np.int64), 4)
    runner._set_step_host_inputs(stale, rows, torch.tensor([4, 3, 0, 0]))
    assert (
        runner._current_host_inputs is None and not model.calls
    )  # deferred to the join

    ids, host = runner._host_input_join(stale)
    assert ids.dtype == torch.int32 and ids.tolist() == [9, 10, 0, 0]
    assert runner.input_batch.token_ids_cpu[0, :5].tolist() == [1, 2, 3, 7, 9]
    assert runner.input_batch.token_ids_cpu[1, :4].tolist() == [4, 5, 8, 10]
    assert runner.input_batch.num_tokens_no_spec[:2].tolist() == [5, 4]
    assert runner.requests["a"].output_token_ids == [7, 9]
    out = host["ple_embeds"].float()
    assert out[:2, 2].tolist() == [9, 10]  # current token
    assert out[:2, 3].tolist() == [7, 8]  # previous token, from the drained history
    assert not out[2:].any()
    assert (
        runner._pending_host_input_rows is None
        and not runner._host_join_pending_outputs
    )


def test_join_after_output_rank_thread_materialized_applies_once():
    # Output rank: the async-output thread already materialized the step; the join must not
    # apply it a second time.
    runner, _ = make_async_runner({"a": [1, 2, 3]})
    out = async_output(runner, ["a"], [7])
    runner._track_host_join_output(out)
    out.get_output()  # the async-output thread
    rows = host_input_req_rows(np.array([1]), np.zeros(1, np.int64), 1)
    runner._set_step_host_inputs(
        torch.tensor([0], dtype=torch.int32), rows, torch.tensor([3])
    )
    ids, _ = runner._host_input_join(torch.tensor([0], dtype=torch.int32))
    assert ids.tolist() == [7]
    assert runner.input_batch.num_tokens_no_spec[0] == 4
    assert runner.requests["a"].output_token_ids == [7]


def test_join_rows_follow_the_request_after_reorder():
    # The batch was condensed after sampling: "b" moved to row 0. Tokens follow the request id.
    runner, _ = make_async_runner({"b": [4, 5], "a": [1, 2, 3]})
    runner._track_host_join_output(async_output(runner, ["a", "b"], [7, 8]))
    rows = host_input_req_rows(np.array([1, 1]), np.zeros(2, np.int64), 2)
    stale = torch.tensor([0, 0], dtype=torch.int32)
    runner._set_step_host_inputs(stale, rows, torch.tensor([2, 3]))
    ids, _ = runner._host_input_join(stale)
    assert ids.tolist() == [8, 7]


def test_join_prefill_rows_keep_prompt_tokens():
    # A new request's prefill chunk (row 1) next to nothing pending: the join is a no-op wait.
    runner, model = make_async_runner({"a": [1, 2, 3], "c": [20, 21, 22, 23, 24]})
    ids_in = torch.tensor([22, 23, 24, 0], dtype=torch.int32)
    rows = host_input_req_rows(np.array([0, 3]), np.array([0, 1]), 4)
    runner._set_step_host_inputs(ids_in, rows, torch.tensor([2, 3, 4, 4]))
    ids, host = runner._host_input_join(ids_in)
    assert ids.tolist() == [22, 23, 24, 0]
    assert host["ple_embeds"].float()[:3, 3].tolist() == [21, 22, 23]


def test_join_raises_on_a_token_the_host_has_not_written():
    runner, _ = make_async_runner({"a": [1, 2, 3]})
    rows = host_input_req_rows(np.array([1]), np.zeros(1, np.int64), 1)
    runner._set_step_host_inputs(
        torch.tensor([0], dtype=torch.int32), rows, torch.tensor([3])
    )
    with pytest.raises(RuntimeError, match="has not written"):
        runner._host_input_join(torch.tensor([0], dtype=torch.int32))


def test_join_without_recorded_layout_raises():
    runner, _ = make_async_runner({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="host-input layout"):
        runner._host_input_join(torch.tensor([0], dtype=torch.int32))


def test_materialize_pending_drains_every_queued_step():
    # The composition-change guard materializes before _update_states; with host inputs it
    # must apply every queued step, oldest first, not only the latest.
    runner, _ = make_async_runner({"a": [1, 2, 3]})
    runner.async_execution_buffer = {}
    assert runner._materialize_pending_async_output("test") is False
    runner._track_host_join_output(async_output(runner, ["a"], [7]))
    runner._track_host_join_output(async_output(runner, ["a"], [9]))
    assert runner._materialize_pending_async_output("test") is True
    assert runner.input_batch.token_ids_cpu[0, :5].tolist() == [1, 2, 3, 7, 9]


def sched(scheduled, finished=(), resumed=(), new=()):
    return SimpleNamespace(
        num_scheduled_tokens={r: 1 for r in scheduled},
        finished_req_ids=set(finished),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
    )


def test_early_drain_only_when_a_live_request_leaves_or_resumes():
    runner, _ = make_async_runner({"a": [1, 2, 3], "b": [4, 5]})
    assert not runner._host_join_needs_early_drain(
        sched(["a"], finished=["b"])
    )  # nothing queued
    runner._track_host_join_output(async_output(runner, ["a", "b"], [7, 8]))
    # A finished request or a zero-token step: the join applies the queue after _update_states.
    assert not runner._host_join_needs_early_drain(sched(["a"], finished=["b"]))
    assert not runner._host_join_needs_early_drain(sched([]))
    assert not runner._host_join_needs_early_drain(sched(["a", "b", "c"], new=["c"]))
    # A live request leaves the batch (unscheduled), a request resumes, an id is reused,
    # or a zero-token step that finishes a request (it drops the whole batch).
    assert runner._host_join_needs_early_drain(sched(["a"]))
    assert runner._host_join_needs_early_drain(sched(["a", "b", "c"], resumed=["c"]))
    assert runner._host_join_needs_early_drain(
        sched(["a", "b"], finished=["b"], new=["b"])
    )
    assert runner._host_join_needs_early_drain(sched([], finished=["b"]))


def test_join_keeps_uncommitted_continuation_rows():
    # A request's later rows past its write cursor (jump-forward or draft tokens) are
    # host-side tokens: the join keeps them and refreshes only the committed first row.
    runner, model = make_async_runner({"a": [1, 2, 3]})
    runner._track_host_join_output(async_output(runner, ["a"], [7]))
    ids_in = torch.tensor(
        [0, 40, 41, 0], dtype=torch.int32
    )  # stale decode row + 2 JF tokens
    rows = host_input_req_rows(np.array([3]), np.zeros(1, np.int64), 4)
    runner._set_step_host_inputs(ids_in, rows, torch.tensor([3, 4, 5, 0]))
    ids, host = runner._host_input_join(ids_in)
    assert ids.tolist() == [7, 40, 41, 0]
    assert host["ple_embeds"].float()[:3, 3].tolist() == [3, 7, 40]
