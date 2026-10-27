# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 speculative decoding (MTP) end to end against the HF model (CPU).

Real pieces: vLLM's config (a tiny local Qwen3.5 checkpoint config with an ``mtp``
speculative config), KV cache grouping and the Neuron schedulers (NeuronScheduler /
NeuronAsyncScheduler with the engine core's batch queue of depth 2), the V2 runner
(constructor, KV-cache init, update_states, input building, forward adapter, output
formatting, take_draft_token_ids) and the Qwen3.5 module with the NKI ops stubbed in torch
(test_model_smoke). The forward adapter runs eagerly.

The draft model is either the real Qwen3_5MTP head (random weights: drafts are mostly
rejected) or an oracle that proposes the HF greedy continuation with deliberate errors, so
steps accept all, some or none of their drafts. Greedy outputs must equal the HF model's
greedy continuation in every configuration.

Every test runs for the dense model and for the sparse MoE one (``qwen3_5_moe``: every MLP,
also the MTP head's, is a Qwen3_5SparseMoeBlock on the plugin's MoE ops' PyTorch fallbacks).

Scenarios: requests arriving together, a long chunked prefill arriving while others decode
(held back, or interleaved with decode steps), partial acceptance, and a request that runs to
max_model_len (drafts trimmed by the scheduler near the end).
"""

import inspect
import itertools
import json
import os
import sys
from collections import deque
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("transformers.models.qwen3_5")

K = 2
VOCAB = 128
MAX_MODEL_LEN = 320
CHUNK = 128  # max_num_batched_tokens: prompts above it prefill in chunks
BLOCK = 16
MAX_REQS = 4
ATTN_LAYERS = (3, 7)
NUM_LAYERS = 8

_TINY_QWEN3_5 = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 96,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "vocab_size": VOCAB,
    "dtype": "float32",
    "max_position_embeddings": 1024,
    "mtp_num_hidden_layers": 1,
    "full_attention_interval": 4,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 2],
        "mrope_interleaved": True,
    },
}

# The sparse MoE checkpoint: same shapes, every MLP replaced by routed experts.
_TINY_QWEN3_5_MOE = {
    **{k: v for k, v in _TINY_QWEN3_5.items() if k != "intermediate_size"},
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe_text",
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
}
# The sparse MoE Qwen3.5 (qwen3_5_moe) is not on this branch: dense only.
CHECKPOINTS = {"dense": _TINY_QWEN3_5}


# ---------------------------------------------------------------- models (stubbed NKI)


def _restore_stubbed_modules_now(before):
    """Undo test_model_smoke's sys.modules stubs (as the conftest fixture does at
    teardown), keeping the models built against them. The runner and the schedulers
    import vLLM lazily and need the real modules back. ``before``: sys.modules names
    before the stubs; qwen3_5 modules imported against the stubs since (moe.py, imported
    by model.py) are dropped too, so later imports and Dynamo's module guards see the
    real ones."""
    import conftest

    for name in [
        n
        for n in sys.modules
        if n not in before and n.startswith("vllm_neuron.model.qwen3_5")
    ]:
        del sys.modules[name]

    for name, prev in conftest._REPLACED.items():
        if prev is conftest._MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev
    conftest._REPLACED.clear()
    for (parent, attr), prev in conftest._PARENT_ATTRS.items():
        if prev is conftest._MISSING:
            if hasattr(parent, attr):
                delattr(parent, attr)
        else:
            setattr(parent, attr, prev)
    conftest._PARENT_ATTRS.clear()


def _torch_attention(attn):
    """NF.segmented_attention / attention_decode are not stubbed: run each decode row
    (1 token, or the 1 + k tokens of a verify row) as a chunk of the torch prefill path."""
    attn._segmented_kernel_ok = lambda block_size: False
    name = f"layers.{attn.layer_idx}.self_attn"

    def forward_decode(hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[name]
        cos, sin = position_embeddings
        B = md["block_table_tensor"].shape[0]
        S = hidden_states.shape[0] // B
        outs = []
        for b in range(B):
            rows = slice(b * S, (b + 1) * S)
            row_md = dict(
                md,
                block_table_tensor=md["block_table_tensor"][b : b + 1],
                slot_mapping=md["slot_mapping"][rows],
                cached_seq_len=positions[b * S].view(1, 1).to(torch.int32),
                kv_segment_size=1,
            )
            outs.append(
                attn.forward_prefill(
                    hidden_states[rows],
                    positions[rows],
                    (cos[rows], sin[rows]),
                    {name: row_md},
                )
            )
        return torch.cat(outs)

    attn.forward_decode = forward_decode


class _GreedyTarget(torch.nn.Module):
    """On-device-sampling contract for the logits-returning test model: greedy tokens,
    the MTP hidden states, and the logits as the gathered logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        logits, hidden = self.model(**kwargs)
        return logits.argmax(dim=-1).to(torch.int32), hidden, logits


def _build_models(moe=False):
    """HF reference, Neuron target and MTP head (same random weights), built against
    test_model_smoke's stubs, which are then removed."""
    before = set(sys.modules)
    import test_model_smoke as smoke
    import test_v2_runner_inputs as base
    from conftest import ROOT, _load

    hf_model, model = base._hf_and_model(moe)
    for i in ATTN_LAYERS:
        _torch_attention(model.model.layers[i].self_attn)
    mtp_mod = _load("vllm_neuron.model.qwen3_5.mtp", ROOT / "model/qwen3_5/mtp.py")
    hf_cfg = dict(hf_model.config.to_dict(), mtp_num_hidden_layers=1)
    mtp = mtp_mod.Qwen3_5MTP.from_configs(hf_cfg, NUM_LAYERS, None)
    assert mtp.model.layers[0].is_moe == moe
    mtp.num_speculative_tokens = K  # set by load_draft_model when serving
    torch.manual_seed(11)
    with torch.no_grad():
        for name, p in mtp.named_parameters():
            if "norm" in name and p.dim() == 1:
                p.data = torch.randn(p.shape) * 0.1
            else:
                p.data = torch.randn(p.shape) * 0.15
        mtp.model.embed_tokens.weight.data = (
            model.model.embed_tokens.weight.data.clone()
        )
    mtp._materialize_buffers()
    _torch_attention(mtp.model.layers[0].self_attn)
    del smoke
    _restore_stubbed_modules_now(before)
    return hf_model, model, mtp


def _hf_greedy(hf_model, prompt, num_tokens):
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            logits = hf_model(torch.tensor([tokens])).logits[0, -1]
            tokens.append(int(logits.argmax()))
    return tokens


# ---------------------------------------------------------------- oracle draft


class _OracleDraft(torch.nn.Module):
    """Fused-draft contract; proposes the HF continuation after each row's next token,
    with the drafts listed in ``corrupt`` replaced, to force partial acceptance. A row is
    mapped to its request through its first attention block."""

    def __init__(self, runner, expected, corrupt):
        super().__init__()
        self.runner, self.expected, self.corrupt = runner, expected, corrupt
        self.proposed = []  # (req_id, next_position, drafts)

    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches):
        pass

    def forward(
        self,
        input_ids,
        positions,
        initial_target_hidden_states,
        attn_metadata,
        sampling_positions,
        rank,
        raw_sampled_token_ids,
    ):
        rs = self.runner.request_state
        attn_group = next(
            g
            for g in range(len(self.runner.block_sizes))
            if g not in self.runner._state_groups
        )
        req_by_block = {
            int(rs.block_table(attn_group)[slot, 0]): req
            for req, slot in rs.req_id_to_index.items()
        }
        table = attn_metadata[f"layers.{ATTN_LAYERS[0]}.self_attn"][
            "block_table_tensor"
        ]
        rows = raw_sampled_token_ids.shape[0]
        width = raw_sampled_token_ids.shape[1]
        drafts = torch.zeros(rows, K, dtype=torch.int32)
        for r in range(rows):
            req = req_by_block.get(int(table[r, 0]))
            if req is None:
                continue
            if width == 1:  # prefill: the token after the chunk's last token
                next_pos = int(positions[int(sampling_positions[r])]) + 1
            else:
                accepted = int((raw_sampled_token_ids[r] >= 0).sum())
                next_pos = int(positions[r * width]) + accepted
            seq = self.expected[req]
            for j in range(K):
                p = next_pos + 1 + j
                tok = seq[p] if p < len(seq) else 0
                if self.corrupt == "all" or (req, next_pos, j) in self.corrupt:
                    tok = (tok + 1) % VOCAB
                drafts[r, j] = tok
            self.proposed.append((req, next_pos, drafts[r].tolist()))
        return None, drafts, None


class _FullMetadataSpeculator(torch.nn.Module):
    """V2DraftSpeculator that passes the whole step metadata (the oracle reads the
    attention block table)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.num_speculative_tokens = base.num_speculative_tokens
        self.synthetic_conditional_rates = None
        self.target_hidden_states = base.target_hidden_states
        self.gathered_logits = base.gathered_logits

    def forward(self, *, attn_metadata, target_hidden_states, **kw):
        _, drafts, _ = self.base.draft_model(
            attn_metadata=attn_metadata,
            initial_target_hidden_states=target_hidden_states,
            **kw,
        )
        return drafts


# ---------------------------------------------------------------- engine harness


def _vllm_config(
    tmp_path, checkpoint, *, async_scheduling, decode_steps, decode_ctx_buckets=None
):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    (tmp_path / "config.json").write_text(json.dumps(CHECKPOINTS[checkpoint]))
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=MAX_MODEL_LEN,
    )
    parallel_config = ParallelConfig()
    cache_config = CacheConfig(
        block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=False
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        speculative_config=SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=K,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=MAX_REQS,
            max_num_batched_tokens=CHUNK,
            max_model_len=MAX_MODEL_LEN,
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
    if decode_ctx_buckets is not None:
        neuron_config = vllm_config.additional_config["neuron_config"]
        neuron_config["decode_context_length_buckets"] = decode_ctx_buckets
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    return vllm_config


class _Engine:
    """The engine core's loop around one V2 runner and a Neuron scheduler."""

    def __init__(self, vllm_config, model, draft, *, oracle=False):
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
        model.set_mtp_hidden_output(True)
        runner.draft_model = draft(runner) if oracle else draft
        specs = runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(vllm_config, specs)
        attn_page = 2 * BLOCK * 2 * 32 * 4
        num_attn_layers = max(
            len(g.layer_names)
            for g in groups
            if type(g.kv_cache_spec).__name__ != "MambaSpec"
        )
        config = get_kv_cache_config_from_groups(
            vllm_config, groups, 160 * attn_page * num_attn_layers
        )
        runner.initialize_kv_cache(config)
        speculator = runner._build_speculator()
        if oracle:
            speculator = _FullMetadataSpeculator(speculator)
        runner._forward_adapter = NeuronForwardAdapter(
            _GreedyTarget(model),
            emit_gathered_logits=runner._logprobs_enabled,
            speculator=speculator,
        )
        if os.environ.get("SPEC_DYNAMO"):
            # Opt-in (slow): trace the adapter with Dynamo, fullgraph, as on device.
            torch._dynamo.config.cache_size_limit = 1 << 20
            torch._dynamo.config.recompile_limit = 1 << 20
            runner._forward_adapter = torch.compile(
                runner._forward_adapter, backend="eager", fullgraph=True
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
        self.outputs: dict[str, list[int]] = {}
        self.logprobs: dict[str, list] = {}
        self.steps = []  # (num_scheduled_tokens, scheduled_spec_decode_tokens)
        self.num_draft = self.num_accepted = 0
        self.drafts = []  # (req_id, committed tokens, drafts after the step)

    def add(self, req_id, prompt, max_tokens, logprobs=None):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, logprobs=logprobs, ignore_eos=True
        )
        self.scheduler.add_request(Request(req_id, prompt, params, None))
        self.outputs[req_id] = []

    def _process(self, scheduler_output, output):
        if hasattr(output, "get_output"):
            output = output.get_output()
        for req_id, toks in zip(output.req_ids, output.sampled_token_ids):
            spec = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if spec and toks:
                self.num_draft += len(spec)
                self.num_accepted += len(toks) - 1
        results = self.scheduler.update_from_output(scheduler_output, output)
        for outs in results.values():
            for out in outs.outputs:
                self.outputs[out.request_id] += list(out.new_token_ids)
                if out.new_logprobs is not None:
                    self.logprobs.setdefault(out.request_id, []).append(
                        out.new_logprobs
                    )
        if not self.async_scheduling:
            drafts = self.runner.take_draft_token_ids()
            if drafts is not None:
                self.scheduler.update_draft_token_ids(drafts)

    def _record_drafts(self, scheduler_output):
        """Remember each scheduled request's committed tokens and its new drafts (the
        eager CPU graph has already run)."""
        rs = self.runner.request_state
        for req_id in scheduler_output.num_scheduled_tokens:
            slot = rs.req_id_to_index[req_id]
            n = int(rs.total_len[slot])
            if n <= rs.prefill_len[slot]:
                continue  # non-final prefill chunk: no sampled token yet
            self.drafts.append(
                (
                    req_id,
                    rs.token_ids[slot, :n].tolist(),
                    rs.draft_tokens[slot].tolist(),
                )
            )

    def step(self):
        """One EngineCore.step_with_batch_queue iteration."""
        if self.scheduler.has_requests() and len(self.queue) < self.depth:
            so = self.scheduler.schedule()
            self.steps.append(
                (dict(so.num_scheduled_tokens), dict(so.scheduled_spec_decode_tokens))
            )
            out = self.runner.execute_model(so)
            if so.total_num_scheduled_tokens > 0:
                out = self.runner.sample_tokens(None)
                self._record_drafts(so)
            self.queue.append((so, out))
            if len(self.queue) < self.depth and self.scheduler.has_requests():
                return
        if self.queue:
            self._process(*self.queue.popleft())

    def run(self, arrivals=None, max_steps=600):
        arrivals = dict(arrivals or {})
        for n in range(max_steps):
            for args in arrivals.pop(n, ()):
                self.add(*args)
            if not arrivals and not self.scheduler.has_requests() and not self.queue:
                return
            self.step()
        raise AssertionError("engine did not finish")


@pytest.fixture(scope="module", params=list(CHECKPOINTS))
def checkpoint(request):
    return request.param


@pytest.fixture(scope="module")
def models(checkpoint):
    torch.manual_seed(3)
    return _build_models(moe=checkpoint == "moe")


def _prompts(seed, lengths):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=g).tolist() for n in lengths]


def _check(engine, expected, prompts):
    for req_id, prompt in prompts.items():
        got = prompt + engine.outputs[req_id]
        assert got == expected[req_id][: len(got)], req_id
        assert len(engine.outputs[req_id]) == len(expected[req_id]) - len(prompt), (
            req_id
        )


SCENARIO = {
    # (request id, prompt length, max_tokens, arrival step)
    "a": (20, 40, 0),
    "b": (33, 30, 0),
    "c": (9, 24, 0),
    # Three chunks while a, b, c decode; with every draft accepted its sixth token
    # is at position 317, so the scheduler trims that step to one draft (sync).
    "end": (MAX_MODEL_LEN - 9, 40, 2),
    "long": (200, 10, 9),  # two chunks while the others decode
}


def _scenario_expected(models, seed=5):
    hf_model = models[0]
    lengths = [v[0] for v in SCENARIO.values()]
    prompts = dict(zip(SCENARIO, _prompts(seed, lengths)))
    expected = {}
    for req_id, (n, max_tokens, _) in SCENARIO.items():
        # max_model_len caps the output: prompt + output <= max_model_len.
        num = min(max_tokens, MAX_MODEL_LEN - n)
        expected[req_id] = _hf_greedy(hf_model, prompts[req_id], num)
    return prompts, expected


@pytest.fixture(scope="module")
def scenario(models):
    return _scenario_expected(models)


def _arrivals(prompts):
    arrivals = {}
    for req_id, (_, max_tokens, step) in SCENARIO.items():
        arrivals.setdefault(step, []).append((req_id, prompts[req_id], max_tokens))
    return arrivals


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("decode_steps", [0, 1], ids=["holdback", "interleave"])
def test_oracle_drafts_partial_acceptance_match_hf(
    tmp_path, checkpoint, models, scenario, async_scheduling, decode_steps
):
    _, model, _ = models
    prompts, expected = scenario
    # Reject the second draft of every third step of "a", both drafts of every fourth
    # step of "b"; "c", "end" and "long" accept everything.
    corrupt = set()
    for p in range(MAX_MODEL_LEN):
        if p % 3 == 0:
            corrupt.add(("a", p, 1))
        if p % 4 == 0:
            corrupt |= {("b", p, 0), ("b", p, 1)}

    vllm_config = _vllm_config(
        tmp_path,
        checkpoint,
        async_scheduling=async_scheduling,
        decode_steps=decode_steps,
    )
    engine = _Engine(
        vllm_config,
        model,
        lambda runner: _OracleDraft(runner, expected, corrupt),
        oracle=True,
    )
    engine.run(_arrivals(prompts))
    _check(engine, expected, prompts)
    # Drafts were verified, and both full and partial acceptance happened.
    assert engine.num_draft > 0
    assert 0 < engine.num_accepted < engine.num_draft
    spec_steps = [s for s in engine.steps if s[1]]
    assert spec_steps, "no verify step"
    # Some step mixed rows with fewer drafts than k (near max_model_len, sync) or
    # skipped drafting entirely (async near max_model_len), and the long prefill ran in
    # chunks while other requests were running.
    long_chunks = [s[0]["long"] for s in engine.steps if s[0].get("long", 0) > 1]
    assert long_chunks[:2] == [CHUNK, 200 - CHUNK]
    # Near max_model_len: sync scheduling trims one row's drafts inside a multi-row
    # verify step; NeuronAsyncScheduler stops drafting for the request near the limit
    # while the others keep drafting. Both are mixed verify steps, padded per row.
    multi_row = [
        s for s in engine.steps if len(s[0]) > 1 and max(s[0].values()) <= 1 + K
    ]
    if async_scheduling:
        assert any(s[1] and len(s[1]) < len(s[0]) for s in multi_row)
    else:
        assert any(
            any(0 < len(d) < K for d in s[1].values()) and len(s[1]) > 1
            for s in multi_row
        )


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_real_mtp_head_matches_hf(
    tmp_path, checkpoint, models, scenario, async_scheduling
):
    _, model, mtp = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path, checkpoint, async_scheduling=async_scheduling, decode_steps=1
    )
    engine = _Engine(vllm_config, model, mtp)
    engine.run(_arrivals(prompts))
    _check(engine, expected, prompts)
    assert engine.num_draft > 0
    # The drafts the head proposed inside the verify graphs equal a from-scratch run
    # of the head over each committed sequence (fresh KV, HF hidden states): the
    # shifted inputs, positions, hidden-state handoff and the head's KV history across
    # verify steps with rejected drafts are right.
    hf_model = models[0]
    records = engine.drafts
    assert len(records) > 20
    for req_id, tokens, drafts in records[::3]:
        assert drafts == _reference_mtp_drafts(mtp, hf_model, tokens), (req_id, tokens)


def _reference_mtp_drafts(mtp, hf_model, tokens):
    """Drafts of the MTP head for the token after ``tokens``: one prefill of the head
    over the whole sequence on a fresh cache, then its recurrent draft steps."""
    import copy

    head = copy.deepcopy(mtp)
    _torch_attention(head.model.layers[0].self_attn)
    n = len(tokens)
    num_blocks = (n + 2 * K) // BLOCK + 2
    caches = [torch.zeros(num_blocks, 2, BLOCK, 32) for _ in range(2)]
    head.bind_kv_cache({f"layers.{NUM_LAYERS}.self_attn": caches})
    with torch.no_grad():
        hidden = hf_model.model(torch.tensor([tokens[:-1]])).last_hidden_state[0]
    T = n - 1
    positions = torch.arange(T, dtype=torch.int32)
    table = torch.arange(1, num_blocks, dtype=torch.int32).view(1, -1)
    slots = table[0][positions.long() // BLOCK].long() * BLOCK + positions % BLOCK
    md = {
        f"layers.{NUM_LAYERS}.self_attn": {
            "block_table_tensor": table,
            "slot_mapping": slots.long(),
            "max_query_len": T,
            "block_size": BLOCK,
            "max_blocks_per_seq": table.shape[1],
            "decode_token_threshold": 1,
            "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32),
            "kv_segment_size": 1,
        }
    }
    shifted = torch.tensor(tokens[1:], dtype=torch.int32)
    with torch.no_grad():
        _, drafts, _ = head(
            input_ids=shifted,
            positions=positions,
            initial_target_hidden_states=hidden,
            attn_metadata=md,
            sampling_positions=torch.tensor([T - 1]),
            rank=torch.zeros(1, dtype=torch.int32),
            raw_sampled_token_ids=torch.tensor([[tokens[-1]]], dtype=torch.int32),
        )
    return drafts[0].tolist()


def test_all_drafts_rejected_and_logprobs(tmp_path, checkpoint, models):
    hf_model, model, _ = models
    prompts = dict(zip(["x", "y"], _prompts(9, [17, 25])))
    expected = {r: _hf_greedy(hf_model, p, 9) for r, p in prompts.items()}
    vllm_config = _vllm_config(
        tmp_path, checkpoint, async_scheduling=True, decode_steps=0
    )
    engine = _Engine(
        vllm_config,
        model,
        lambda runner: _OracleDraft(runner, expected, "all"),
        oracle=True,
    )
    engine.run({0: [("x", prompts["x"], 9, 2), ("y", prompts["y"], 9)]})
    _check(engine, expected, prompts)
    assert engine.num_draft > 0 and engine.num_accepted == 0
    # One logprobs entry per generated token, matching the HF log-softmax.
    entries = [lp for lists in engine.logprobs["x"] for lp in _logprob_rows(lists)]
    assert len(entries) == 9
    seq = expected["x"]
    with torch.no_grad():
        for i, (token_ids, logprobs) in enumerate(entries):
            pos = len(prompts["x"]) + i
            ref = torch.log_softmax(
                hf_model(torch.tensor([seq[:pos]])).logits[0, -1], dim=-1
            )
            assert token_ids[0] == seq[pos]
            torch.testing.assert_close(
                torch.tensor(logprobs[0]), ref[seq[pos]], atol=1e-3, rtol=1e-3
            )


def _logprob_rows(lists):
    return [
        (lists.logprob_token_ids[i].tolist(), lists.logprobs[i].tolist())
        for i in range(len(lists.logprobs))
    ]


def test_runner_state_after_scenario(tmp_path, checkpoint, models, scenario):
    """Pool pages are 1 + k per slot and state_in follows the accepted count."""
    _, model, _ = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path, checkpoint, async_scheduling=False, decode_steps=0
    )
    engine = _Engine(
        vllm_config,
        model,
        lambda runner: _OracleDraft(runner, expected, {("a", 23, 0)}),
        oracle=True,
    )
    runner = engine.runner
    assert runner._state_pool_slot_pages == 1 + K
    for halves in runner.state_caches.values():
        assert halves[0].shape[0] == 1 + MAX_REQS * (1 + K)
    seen = []
    original = runner._build_forward_adapter_kwargs

    def record(model_input, **kw):
        kwargs = original(model_input, **kw)
        if model_input.num_draft_by_row is not None:
            seen.append(
                (
                    kwargs["attn_metadata"]["layers.0.linear_attn"][
                        "block_table_tensor"
                    ].clone(),
                    kwargs["num_accepted"].clone(),
                    model_input.idx_mapping_np.copy(),
                )
            )
        return kwargs

    runner._build_forward_adapter_kwargs = record
    engine.run({0: [("a", prompts["a"], SCENARIO["a"][1])]})
    _check(engine, {"a": expected["a"]}, {"a": prompts["a"]})
    slot = 0
    for table, num_accepted, idx in seen:
        assert idx.tolist() == [slot]
        assert table[0].tolist() == [1 + slot * (1 + K) + j for j in range(1 + K)]
        assert 1 <= int(num_accepted[slot]) <= 1 + K
    # The step after the rejected draft resumes from block 0 (one token accepted).
    assert sorted({int(a[slot]) for _, a, _ in seen}) == [1, 3]


def test_policy_and_runner_gate_spec_decode(tmp_path, checkpoint, models):
    from vllm_neuron.vllm import model_runner_v2_policy as policy

    vllm_config = _vllm_config(
        tmp_path, checkpoint, async_scheduling=True, decode_steps=0
    )
    result = policy.get_neuron_model_runner_v2_policy(vllm_config)
    assert result.default_enabled is False  # V1 stays the default
    assert not any("speculative" in f for f in result.unsupported_features)
    # vLLM's MTP draft config for the checkpoint, as load_draft_model resolves it: the
    # MoE target's head is Qwen3_5MoeMTP, built with the MoE FFN from that config.
    draft_config = vllm_config.speculative_config.draft_model_config
    assert draft_config.architecture == (
        "Qwen3_5MoeMTP" if checkpoint == "moe" else "Qwen3_5MTP"
    )
    mtp = models[2]
    with torch.device("meta"):
        draft = type(mtp).from_configs(draft_config.hf_config, NUM_LAYERS)
    assert type(draft.model.layers[0].mlp) is type(mtp.model.layers[0].mlp)
    vllm_config.speculative_config.method = "eagle3"
    result = policy.get_neuron_model_runner_v2_policy(vllm_config)
    assert any("eagle3" in f for f in result.unsupported_features)
    from vllm_neuron.vllm.worker.neuron_model_runner_v2 import NeuronModelRunnerV2

    with pytest.raises(NotImplementedError, match="eagle3"):
        NeuronModelRunnerV2(vllm_config, torch.device("cpu"))
    vllm_config.speculative_config.method = "mtp"
    vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    with pytest.raises(NotImplementedError, match="KV transfer"):
        NeuronModelRunnerV2(vllm_config, torch.device("cpu"))
    vllm_config.kv_transfer_config = None
    vllm_config.additional_config["neuron_config"]["num_batched_tokens_buckets"] = [
        3,
        CHUNK,
    ]
    with pytest.raises(ValueError, match=r"must exceed 1 \+ num_speculative_tokens"):
        NeuronModelRunnerV2(vllm_config, torch.device("cpu"))


# ------------------------------------------------- decode context-length buckets

CTX_BUCKETS = [128, 256]
# The in-graph position checks read tensor values: eager runs only (not SPEC_DYNAMO).
_EAGER_GRAPH_CHECKS = not os.environ.get("SPEC_DYNAMO")
CTX_SCENARIO = {
    # (request id, prompt length, max_tokens, arrival step)
    "short": (20, 40, 0),
    "x": (110, 40, 0),  # decodes across the 128 boundary next to "short"
    # Later: two chunks while the others decode, then across 256 into the max_model_len
    # graph next to "c", which outlives it (the bucket drops back to 128).
    "y": (240, 30, 30),
    "c": (9, 48, 30),
}


@pytest.fixture(scope="module")
def ctx_scenario(models):
    hf_model = models[0]
    lengths = [v[0] for v in CTX_SCENARIO.values()]
    prompts = dict(zip(CTX_SCENARIO, _prompts(13, lengths)))
    expected = {
        r: _hf_greedy(hf_model, prompts[r], min(n, MAX_MODEL_LEN - len(prompts[r])))
        for r, (_, n, _) in CTX_SCENARIO.items()
    }
    return prompts, expected


def _record_ctx_steps(monkeypatch, runner):
    """Record every verify step's bucket, scheduler counts and per-layer table widths, and
    check inside the graph that no position of the step's target tokens falls past its
    attention table."""
    from vllm_neuron.vllm.worker import neuron_forward_adapter as adapter_mod

    steps = []
    original = runner._build_forward_adapter_kwargs

    def record(model_input, **kw):
        kwargs = original(model_input, **kw)
        if model_input.num_draft_by_row is not None:
            rs = runner.request_state
            slots = model_input.idx_mapping_np.astype(int)
            steps.append(
                {
                    "bucket": model_input.decode_ctx_bucket,
                    "num_computed": rs.scheduler_num_computed_tokens[slots].tolist(),
                    "widths": {
                        name: meta["block_table_tensor"].shape[1]
                        for name, meta in kwargs["attn_metadata"].items()
                    },
                }
            )
        return kwargs

    monkeypatch.setattr(runner, "_build_forward_adapter_kwargs", record)

    build_verify_inputs = adapter_mod._build_verify_inputs

    def checked_verify_inputs(**kw):
        out = build_verify_inputs(**kw)
        positions, metadata = out[2], out[3]
        for meta in metadata.values():
            if "state_in" not in meta:
                limit = meta["block_table_tensor"].shape[1] * meta["block_size"]
                assert int(positions.max()) < limit, (int(positions.max()), limit)
        return out

    if _EAGER_GRAPH_CHECKS:
        monkeypatch.setattr(adapter_mod, "_build_verify_inputs", checked_verify_inputs)
    return steps


def _check_ctx_steps(runner, steps):
    from vllm_neuron.vllm.worker.neuron_input_build import select_decode_ctx_bucket

    groups = runner._kv_cache_config.kv_cache_groups
    full = {
        name: runner.request_state.block_table(g).shape[1]
        for g, group in enumerate(groups)
        for name in group.layer_names
    }
    state_layers = {
        name for g in runner._state_groups for name in groups[g].layer_names
    }
    assert steps
    for step in steps:
        bucket = step["bucket"]
        assert bucket == select_decode_ctx_bucket(
            max(step["num_computed"]), 2 * K, CTX_BUCKETS, MAX_MODEL_LEN
        )
        attn_width = full[f"layers.{ATTN_LAYERS[0]}.self_attn"]
        if bucket < MAX_MODEL_LEN:
            attn_width = -(-bucket // BLOCK)
        for name, width in step["widths"].items():
            # State groups (1 + k pages) are never trimmed; every attention group,
            # the MTP head's layer included, is trimmed alike.
            assert width == (full[name] if name in state_layers else attn_width), name
    buckets = [s["bucket"] for s in steps]
    assert set(buckets) == {*CTX_BUCKETS, MAX_MODEL_LEN}
    # A verify step whose rows all sit below a boundary but whose verify / draft
    # positions straddle it runs in the next bucket.
    for boundary in CTX_BUCKETS:
        assert any(
            max(s["num_computed"]) < boundary < max(s["num_computed"]) + 2 * K
            and s["bucket"] > boundary
            for s in steps
        ), boundary
    # Batches mix short and long contexts: the longest row picks the bucket.
    assert any(
        min(s["num_computed"]) < CTX_BUCKETS[0] and s["bucket"] > CTX_BUCKETS[0]
        for s in steps
    )
    # The bucket follows the batch down again when a long request finishes.
    assert any(a > b for a, b in itertools.pairwise(buckets))


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("draft", ["oracle", "mtp"])
def test_decode_ctx_buckets_match_hf(
    tmp_path, monkeypatch, checkpoint, models, ctx_scenario, async_scheduling, draft
):
    """Requests cross the 128 and 256 buckets (and reach the max_model_len graph) in
    the middle of their generation, with MTP verify steps straddling each boundary and
    batches mixing short and long contexts; greedy outputs equal HF's."""
    _, model, mtp = models
    prompts, expected = ctx_scenario
    vllm_config = _vllm_config(
        tmp_path,
        checkpoint,
        async_scheduling=async_scheduling,
        decode_steps=1,
        decode_ctx_buckets=CTX_BUCKETS,
    )
    if draft == "oracle":
        # Partial acceptance around both boundaries.
        corrupt = {("x", p, 1) for p in range(MAX_MODEL_LEN) if p % 3 == 0}
        corrupt |= {("y", p, 0) for p in range(MAX_MODEL_LEN) if p % 4 == 1}
        engine = _Engine(
            vllm_config,
            model,
            lambda runner: _OracleDraft(runner, expected, corrupt),
            oracle=True,
        )
    else:
        engine = _Engine(vllm_config, model, mtp)
        # The MTP head's draft steps build their slot mapping from the trimmed table:
        # their positions (up to num_computed + 2k - 1) stay inside it.
        head_globals = inspect.unwrap(type(mtp).forward).__globals__
        slot_mapping = head_globals["_slot_mapping"]
        draft_tables = []

        def checked_slot_mapping(positions, block_table, block_size):
            limit = block_table.shape[1] * block_size
            assert int(positions.max()) < limit, (int(positions.max()), limit)
            draft_tables.append(block_table.shape[1])
            return slot_mapping(positions, block_table, block_size)

        if _EAGER_GRAPH_CHECKS:
            monkeypatch.setitem(head_globals, "_slot_mapping", checked_slot_mapping)
    runner = engine.runner
    assert runner.decode_ctx_buckets == CTX_BUCKETS
    assert runner.decode_ctx_headroom == 2 * K
    steps = _record_ctx_steps(monkeypatch, runner)
    arrivals = {}
    for req_id, (_, max_tokens, step) in CTX_SCENARIO.items():
        arrivals.setdefault(step, []).append((req_id, prompts[req_id], max_tokens))
    engine.run(arrivals)
    _check(engine, expected, prompts)
    assert engine.num_draft > 0
    if draft == "oracle":
        assert 0 < engine.num_accepted < engine.num_draft
    else:
        # The head's in-graph drafts (its attention reads the trimmed table) equal a
        # from-scratch run of the head over each committed sequence.
        for req_id, tokens, drafts in engine.drafts[::5]:
            assert drafts == _reference_mtp_drafts(mtp, models[0], tokens), req_id
        # Its draft steps ran on the trimmed tables of every bucket.
        assert set(draft_tables) >= {8, 16} or not _EAGER_GRAPH_CHECKS
    _check_ctx_steps(runner, steps)


def test_decode_ctx_bucket_config_and_warmup_inputs(tmp_path, checkpoint, models):
    from vllm_neuron.vllm import model_runner_v2_policy as policy
    from vllm_neuron.vllm.worker.neuron_model_runner_v2 import NeuronModelRunnerV2

    _, model, mtp = models
    vllm_config = _vllm_config(
        tmp_path,
        checkpoint,
        async_scheduling=True,
        decode_steps=0,
        decode_ctx_buckets=CTX_BUCKETS,
    )
    # The policy admits the buckets for a recurrent-state model with MTP.
    result = policy.get_neuron_model_runner_v2_policy(vllm_config)
    assert not any("context-length" in f for f in result.unsupported_features)
    engine = _Engine(vllm_config, model, mtp)
    runner = engine.runner
    assert runner.neuron_config.decode_context_length_buckets == CTX_BUCKETS
    groups = runner._kv_cache_config.kv_cache_groups
    attn = [g for g in range(len(groups)) if g not in runner._state_groups]
    # One verify graph per (row bucket, ctx bucket); each fixes the attention tables'
    # width, the state tables keep theirs.
    for ctx_bucket, width in ((128, 8), (256, 16), (MAX_MODEL_LEN, 24), (None, 8)):
        for rows in runner.num_seqs_buckets:
            model_input = runner._build_decode_synthetic_inputs(
                rows, ctx_bucket=ctx_bucket
            )
            assert model_input.spec_width == 1 + K
            assert model_input.decode_ctx_bucket == (ctx_bucket or 128)
            for g, table in enumerate(model_input.block_tables):
                state_width = runner.request_state.block_table(g).shape[1]
                assert table.shape == (rows, width if g in attn else state_width), (
                    g,
                    ctx_bucket,
                )
    assert runner._decode_graph_name(4, 256) == f"decode_b4_s{1 + K}_ctx256"
    with pytest.raises(ValueError, match="not configured"):
        runner._build_decode_synthetic_inputs(1, ctx_bucket=192)
    # Prefill graphs keep the whole table.
    prefill = runner._build_prefill_synthetic_inputs(CHUNK)
    assert prefill.decode_ctx_bucket is None
    for g in attn:
        assert prefill.block_tables[g].shape[1] == 24
    # Invalid bucket lists fail at construction (multiples of 128 below max_model_len).
    neuron_config = vllm_config.additional_config["neuron_config"]
    for bad in ([100], [256, 128], [MAX_MODEL_LEN]):
        neuron_config["decode_context_length_buckets"] = bad
        with pytest.raises(ValueError, match="decode_context_length_buckets"):
            NeuronModelRunnerV2(vllm_config, torch.device("cpu"))
