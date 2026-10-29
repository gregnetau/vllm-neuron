# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 host inputs (``interfaces.HasHostInputs``) end to end on CPU.

V2 builds ``input_ids`` and positions inside the graph from device state, so the runner
rebuilds that layout on the host, from a host copy of every slot's token history, to build
the host inputs (docs/design/host-inputs.md). The tiny Qwen3.5 model of
test_v2_spec_decode_e2e gets two host inputs that echo, per token row, the input id and the
position the host expects; its forward checks them against the ones the graph built. At
every build the host token history and computed counts must equal the device's.

Runs through the real Neuron schedulers (sync and async with the engine core's batch queue),
plain decode and MTP (oracle drafts with partial acceptance, and the real head), chunked
prefills held back or interleaved with decode steps, a request that runs to max_model_len,
and preemption. Greedy outputs must equal the HF model's.
"""

import json
from collections import deque

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("transformers.models.qwen3_5")

import test_v2_spec_decode_e2e as e2e

BLOCK = e2e.BLOCK
CHUNK = e2e.CHUNK


class _EchoHostInputs:
    """Mixin: host inputs that echo the host's input id and position of each row (plus
    one, so padding is 0), checked against the graph's in ``forward``."""

    runner = None  # set by the test
    checked_rows = 0

    def host_input_specs(self):
        return {
            "echo_ids": ((1,), torch.float32),
            "echo_pos": ((1,), torch.float32),
        }

    def build_host_inputs(self, input_ids, req_rows, positions, token_ids_cpu):
        runner = self.runner
        rs = runner.request_state
        assert token_ids_cpu is runner._host_token_ids
        # After the join every earlier graph has completed (eager on CPU: always), so
        # the host copy equals the device state of every occupied slot.
        for slot in rs.req_id_to_index.values():
            n = int(rs.total_len[slot])
            assert int(runner._host_total_len[slot]) == n, slot
            assert int(runner._host_num_computed[slot]) == int(
                rs.num_computed_tokens[slot]
            ), slot
            assert token_ids_cpu[slot, :n].tolist() == rs.token_ids[slot, :n].tolist()
        real = req_rows >= 0
        ids = np.where(real, input_ids + 1, 0).astype(np.float32)
        pos = np.where(real, positions + 1, 0).astype(np.float32)
        return {
            "echo_ids": torch.from_numpy(ids).view(-1, 1),
            "echo_pos": torch.from_numpy(pos).view(-1, 1),
        }

    def forward(self, input_ids, positions, echo_ids=None, echo_pos=None, **kw):
        assert echo_ids is not None and echo_pos is not None, "host inputs not passed"
        assert echo_ids.shape == (input_ids.shape[0], 1)
        real = echo_ids[:, 0] > 0
        assert torch.equal(echo_ids[real, 0].long() - 1, input_ids[real].long())
        assert torch.equal(echo_pos[real, 0].long() - 1, positions[real].long())
        type(self).checked_rows += int(real.sum())
        self.real_rows_per_call.append(int(real.sum()))
        return super().forward(input_ids=input_ids, positions=positions, **kw)


def _with_echo(model):
    """The model, retyped to also take the echo host inputs."""
    model.__class__ = type(
        "EchoHost" + type(model).__name__, (_EchoHostInputs, type(model)), {}
    )
    model.real_rows_per_call = []
    return model


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(3)
    hf_model, model, mtp = e2e._build_models()
    return hf_model, _with_echo(model), mtp


@pytest.fixture(scope="module")
def scenario(models):
    return e2e._scenario_expected(models)


def _vllm_config(
    tmp_path, *, spec, async_scheduling, decode_steps, max_model_len=e2e.MAX_MODEL_LEN
):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    (tmp_path / "config.json").write_text(json.dumps(e2e._TINY_QWEN3_5))
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=max_model_len,
    )
    parallel_config = ParallelConfig()
    speculative_config = None
    if spec:
        speculative_config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=e2e.K,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(
            block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=False
        ),
        speculative_config=speculative_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=e2e.MAX_REQS,
            max_num_batched_tokens=CHUNK,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [CHUNK],
                "kv_segment_size_buckets": [512],
                "decode_steps_between_prefill_chunks": decode_steps,
            }
        },
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    return vllm_config


class _Engine(e2e._Engine):
    """test_v2_spec_decode_e2e's engine loop around a V2 runner whose model has host
    inputs; ``num_blocks`` sizes the KV cache (small: requests are preempted)."""

    def __init__(self, vllm_config, model, draft, *, oracle=False, num_blocks=160):
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.structured_output import StructuredOutputManager

        from vllm_neuron.vllm.core.scheduler import (
            NeuronAsyncScheduler,
            NeuronScheduler,
        )
        from vllm_neuron.vllm.worker.neuron_forward_adapter import (
            NeuronForwardAdapter,
        )
        from vllm_neuron.vllm.worker.neuron_model_runner_v2 import NeuronModelRunnerV2

        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
        runner = NeuronModelRunnerV2(vllm_config, torch.device("cpu"))
        runner._rank_tensor_cache = torch.zeros(1, dtype=torch.int32)
        runner.model = model
        model.runner = runner
        model.set_mtp_hidden_output(draft is not None)
        if draft is not None:
            runner.draft_model = draft(runner) if oracle else draft
        runner._init_host_inputs()
        assert runner._host_input_specs is not None
        specs = runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(vllm_config, specs)
        attn_page = 2 * BLOCK * 2 * 32 * 4
        num_attn_layers = max(
            len(g.layer_names)
            for g in groups
            if type(g.kv_cache_spec).__name__ != "MambaSpec"
        )
        config = get_kv_cache_config_from_groups(
            vllm_config, groups, num_blocks * attn_page * num_attn_layers
        )
        runner.initialize_kv_cache(config)
        speculator = runner._build_speculator()
        if oracle:
            speculator = e2e._FullMetadataSpeculator(speculator)
        target = e2e._GreedyTarget(model) if draft is not None else _GreedyOnly(model)
        runner._forward_adapter = NeuronForwardAdapter(
            target,
            emit_gathered_logits=runner._logprobs_enabled,
            speculator=speculator,
        )
        self.runner = runner
        vllm_config.cache_config.num_gpu_blocks = config.num_blocks
        block_size, hash_block_size = resolve_kv_cache_block_sizes(config, vllm_config)
        cls = NeuronAsyncScheduler if self.async_scheduling else NeuronScheduler
        self.scheduler = cls(
            vllm_config=vllm_config,
            kv_cache_config=config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
            hash_block_size=hash_block_size,
        )
        self.queue = deque()
        self.depth = 2 if self.async_scheduling else 1
        self.outputs = {}
        self.logprobs = {}
        self.steps = []
        self.num_draft = self.num_accepted = 0
        self.drafts = []

    def _record_drafts(self, scheduler_output):
        if self.runner.request_state.draft_tokens is not None:
            super()._record_drafts(scheduler_output)

    def step(self):
        before = len(self.steps)
        super().step()
        if len(self.steps) > before:
            # Every real token of the step had a host-input row.
            scheduled = sum(self.steps[-1][0].values())
            if scheduled:
                assert self.runner.model.real_rows_per_call[-1] == scheduled


class _GreedyOnly(torch.nn.Module):
    """On-device-sampling contract without MTP: greedy tokens and the logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        logits = self.model(**kwargs)
        return logits.argmax(dim=-1).to(torch.int32), logits


def _oracle_corruption():
    corrupt = set()
    for p in range(e2e.MAX_MODEL_LEN):
        if p % 3 == 0:
            corrupt.add(("a", p, 1))
        if p % 4 == 0:
            corrupt |= {("b", p, 0), ("b", p, 1)}
    return corrupt


@pytest.mark.parametrize(
    "spec,async_scheduling,decode_steps",
    [
        ("none", False, 0),
        ("none", True, 0),
        ("none", True, 1),
        ("oracle", False, 1),
        ("oracle", True, 0),
        ("oracle", True, 1),
        ("mtp", True, 1),
    ],
    ids=[
        "plain-sync",
        "plain-async-holdback",
        "plain-async-interleave",
        "oracle-sync",
        "oracle-async-holdback",
        "oracle-async-interleave",
        "mtp-async",
    ],
)
def test_host_inputs_match_graph_and_hf(
    tmp_path, models, scenario, spec, async_scheduling, decode_steps
):
    _, model, mtp = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path,
        spec=spec != "none",
        async_scheduling=async_scheduling,
        decode_steps=decode_steps,
    )
    if spec == "oracle":
        corrupt = _oracle_corruption()
        engine = _Engine(
            vllm_config,
            model,
            lambda runner: e2e._OracleDraft(runner, expected, corrupt),
            oracle=True,
        )
    else:
        engine = _Engine(vllm_config, model, mtp if spec == "mtp" else None)
    rows_before = type(model).checked_rows
    engine.run(e2e._arrivals(prompts))
    e2e._check(engine, expected, prompts)
    # Every step's real tokens (prefill chunks, decode and verify rows) were echoed.
    total = sum(sum(s[0].values()) for s in engine.steps)
    assert type(model).checked_rows - rows_before == total
    if spec == "oracle":
        assert 0 < engine.num_accepted < engine.num_draft
    assert not engine.runner._host_join_queue or async_scheduling


PREEMPT = {f"p{i}": (40 + 7 * i, 64) for i in range(4)}  # prompt length, max_tokens


@pytest.fixture(scope="module")
def preempt_scenario(models):
    hf_model = models[0]
    prompts = dict(zip(PREEMPT, e2e._prompts(21, [n for n, _ in PREEMPT.values()])))
    expected = {r: e2e._hf_greedy(hf_model, prompts[r], PREEMPT[r][1]) for r in PREEMPT}
    return prompts, expected


@pytest.mark.parametrize("spec", ["none", "oracle"])
def test_host_inputs_through_preemption(
    tmp_path, monkeypatch, models, preempt_scenario, spec
):
    """Four requests admitted past the worst-case KV cap (VLLM_NEURON_MAX_KV_CONCURRENT)
    outgrow a small KV cache while decoding: the scheduler preempts and recomputes, and
    resumed requests restart the host history from their full tokens."""
    monkeypatch.setenv("VLLM_NEURON_MAX_KV_CONCURRENT", str(e2e.MAX_REQS))
    _, model, _ = models
    prompts, expected = preempt_scenario
    vllm_config = _vllm_config(
        tmp_path, spec=spec != "none", async_scheduling=True, decode_steps=1
    )

    def draft(runner):
        return e2e._OracleDraft(runner, expected, set())

    # 4 x (prompt + 64) tokens need 27-29 attention blocks of 16, and vLLM also takes
    # 1 + k state blocks per request and state group from the shared pool: 20 blocks
    # (58 with MTP) admit all four but do not hold them to the end.
    engine = _Engine(
        vllm_config,
        model,
        draft if spec == "oracle" else None,
        oracle=spec == "oracle",
        num_blocks=58 if spec == "oracle" else 20,
    )
    preemptions = []
    schedule = engine.scheduler.schedule

    def tracking_schedule():
        out = schedule()
        if out.preempted_req_ids:
            preemptions.append(set(out.preempted_req_ids))
        return out

    engine.scheduler.schedule = tracking_schedule
    engine.run({0: [(r, prompts[r], PREEMPT[r][1]) for r in PREEMPT]}, max_steps=2000)
    e2e._check(engine, expected, prompts)
    assert preemptions, "the small KV cache did not preempt"


@pytest.mark.parametrize("spec", [False, True], ids=["plain", "mtp"])
def test_synthetic_inputs_zero_fill_host_inputs(tmp_path, models, spec):
    """Warmup, capture and dummy inputs carry zero host inputs of the graph's width
    (a verify graph's B * (1 + k) tokens with MTP)."""
    _, model, mtp = models
    vllm_config = _vllm_config(
        tmp_path, spec=spec, async_scheduling=True, decode_steps=0
    )
    engine = _Engine(vllm_config, model, mtp if spec else None)
    runner = engine.runner
    inputs = [
        runner._build_prefill_synthetic_inputs(),
        runner._build_decode_synthetic_inputs(2),
    ]
    if not spec:
        inputs.append(
            runner._build_dp_dummy_inputs(
                is_prefill=False, token_bucket=2, row_bucket=2
            )
        )
    for model_input in inputs:
        host = model_input.host_inputs
        assert set(host) == {"echo_ids", "echo_pos"}
        for tensor in host.values():
            assert tensor.shape == (model_input.padded_num_tokens, 1)
            assert not tensor.any()
        kwargs = runner._build_forward_adapter_kwargs(model_input)
        assert kwargs["host_inputs"] is host
    assert inputs[1].padded_num_tokens == 2 * (1 + (e2e.K if spec else 0))
    meta = runner._build_prefill_synthetic_inputs(device=torch.device("meta"))
    assert all(t.device.type == "meta" for t in meta.host_inputs.values())


def test_dense_prefill_variant_and_side_cache_ratio(tmp_path, monkeypatch, models):
    """``prefill_dense_context``: prefill chunks whose cached prefix plus padded chunk fit
    it run with the attention block tables trimmed to it (the dense graph variant), the
    others with the whole table; warmup builds both. ``kv_side_cache_ratio`` follows
    ``kv_side_bytes_per_token_per_attn_layer``."""
    hf_model, model, _ = models
    dense_ctx, max_model_len = 512, 768
    prompts = dict(zip(["short", "long"], e2e._prompts(31, [300, 600])))
    expected = {r: e2e._hf_greedy(hf_model, p, 6) for r, p in prompts.items()}
    vllm_config = _vllm_config(
        tmp_path,
        spec=False,
        async_scheduling=True,
        decode_steps=0,
        max_model_len=max_model_len,
    )
    model.prefill_dense_context = dense_ctx
    model.kv_side_bytes_per_token_per_attn_layer = 128
    try:
        engine = _Engine(vllm_config, model, None)
        runner = engine.runner
        attn = next(
            g for g in range(len(runner.block_sizes)) if g not in runner._state_groups
        )
        # K and V, 2 KV heads of 32 float32 per token: 512 bytes.
        assert runner.kv_side_cache_ratio() == 128 / 512
        assert runner.prefill_ctx_variants(512, CHUNK) == [False, True]
        assert runner.prefill_ctx_variants(1024, CHUNK) == [False]
        synthetic = runner._build_prefill_synthetic_inputs(CHUNK, 512, dense_ctx=True)
        assert synthetic.dense_prefill
        assert synthetic.block_tables[attn].shape[1] == dense_ctx // BLOCK
        assert (
            runner._build_prefill_synthetic_inputs(CHUNK, 512)
            .block_tables[attn]
            .shape[1]
            == max_model_len // BLOCK
        )
        seen = []
        original = runner._build_forward_adapter_kwargs

        def record(model_input, **kw):
            if model_input.is_prefill:
                seen.append(
                    (
                        model_input.cached_seq_len,
                        model_input.dense_prefill,
                        model_input.block_tables[attn].shape[1],
                    )
                )
            return original(model_input, **kw)

        runner._build_forward_adapter_kwargs = record
        engine.run({0: [("short", prompts["short"], 6), ("long", prompts["long"], 6)]})
        e2e._check(engine, expected, prompts)
    finally:
        del model.prefill_dense_context
        del model.kv_side_bytes_per_token_per_attn_layer
    for cached, dense, width in seen:
        assert dense == (cached + CHUNK <= dense_ctx), (cached, dense)
        assert width == (dense_ctx if dense else max_model_len) // BLOCK
    assert {dense for _, dense, _ in seen} == {False, True}
    # With MTP (the drafter in the prefill graph) V2 builds no dense variant.
    spec_config = _vllm_config(
        tmp_path,
        spec=True,
        async_scheduling=True,
        decode_steps=0,
        max_model_len=max_model_len,
    )
    model.prefill_dense_context = dense_ctx
    try:
        spec_runner = _Engine(spec_config, model, models[2]).runner
        assert spec_runner.prefill_ctx_variants(512, CHUNK) == [False]
        assert not spec_runner._use_prefill_dense(CHUNK, 512)
        monkeypatch.setenv("VLLM_NEURON_V2_DENSE_PREFILL_WITH_SPEC", "1")
        assert spec_runner.prefill_ctx_variants(512, CHUNK) == [False, True]
    finally:
        del model.prefill_dense_context
