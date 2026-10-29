# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next on Model Runner V2, end to end on CPU against Hugging Face.

The tiny plugin model (tp=1, ``qwen4_exp_testlib``) with sparse QSA runs through the real V2
runner and Neuron schedulers: its n-gram rows are built on the host from the HF table every
step (``HasHostInputs``, through the runner's host token history and join), its PLE conv state
is a recurrent-state layer in the dedicated pool next to the GDN layers, and its QSA raw and
compressed key side caches are addressed by the attention block tables. Greedy outputs must
equal the HF model's, with plain decode (sync and async scheduling), with MTP oracle drafts
(partial acceptance: verify rows and their drafts' n-gram rows) and with the real MTP head.
"""

import json
from collections import deque

import pytest
import torch
from qwen4_exp_testlib import load_from_hf, load_state, tiny_text_config_dict

pytest_plugins = ["qwen4_exp_testlib"]  # fixtures

K = 2
BLOCK = 16
CHUNK = 128
MAX_MODEL_LEN = 256
MAX_REQS = 4
ATTN_LAYER = "layers.3.self_attn"


def _hf_model():
    from test_model_qsa import _hf_model as qsa_hf_model

    return qsa_hf_model()  # sparse QSA (budget 16 tokens)


def _hf_greedy(hf, prompt, num_tokens):
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(num_tokens):
            tokens.append(int(hf(torch.tensor([tokens])).logits[0, -1].argmax()))
    return tokens


def _host_ple(model, hf):
    """The model's host n-gram inputs from the HF model's table (as test_host_inputs)."""
    from vllm_neuron.model.qwen4_exp.ngram import (
        HostNGramTable,
        HostPLEInputs,
        NGramHasher,
    )

    emb = hf.model.layers[1].ple.ple_embedding
    hasher = NGramHasher.from_buffers(
        emb.layer_multipliers,
        emb.ngram_heads_vocab_sizes,
        emb.ngram_heads_offsets,
        emb.ngram_size,
        emb.heads_per_ngram,
    )
    table = HostNGramTable.from_tensor(emb.ngram_embedding.weight.detach(), 2)
    model._host_ple = HostPLEInputs(hasher, table, model.config.eos_token_id)


class _Greedy(torch.nn.Module):
    """On-device-sampling contract for the logits-returning tp=1 model."""

    def __init__(self, model, mtp):
        super().__init__()
        self.model, self.mtp = model, mtp

    def forward(self, **kwargs):
        out = self.model(**kwargs)
        if self.mtp:
            logits, hidden = out
            return logits.argmax(-1).to(torch.int32), hidden, logits
        return out.argmax(-1).to(torch.int32), out


class _Oracle(torch.nn.Module):
    """Fused-draft contract: the HF continuation after each row's next token, every
    third draft corrupted. A row's request is found by its first attention block."""

    def __init__(self, runner, expected):
        super().__init__()
        self.runner, self.expected = runner, expected

    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches):
        pass

    def forward(self, input_ids, positions, initial_target_hidden_states, attn_metadata,
                sampling_positions, rank, raw_sampled_token_ids):  # fmt: skip
        rs = self.runner.request_state
        g = next(
            g for g in range(len(self.runner.block_sizes))
            if g not in self.runner._state_groups
        )  # fmt: skip
        by_block = {
            int(rs.block_table(g)[s, 0]): r for r, s in rs.req_id_to_index.items()
        }
        table = attn_metadata[ATTN_LAYER]["block_table_tensor"]
        rows, width = raw_sampled_token_ids.shape
        drafts = torch.zeros(rows, K, dtype=torch.int32)
        for r in range(rows):
            req = by_block.get(int(table[r, 0]))
            if req is None:
                continue
            if width == 1:
                nxt = int(positions[int(sampling_positions[r])]) + 1
            else:
                nxt = int(positions[r * width]) + int(
                    (raw_sampled_token_ids[r] >= 0).sum()
                )
            seq = self.expected[req]
            for j in range(K):
                p = nxt + 1 + j
                tok = seq[p] if p < len(seq) else 0
                drafts[r, j] = (tok + 1) % 1024 if (p % 3 == 0) else tok
        return None, drafts, None


class _Speculator(torch.nn.Module):
    """V2DraftSpeculator passing the whole step metadata (the oracle reads it)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.num_speculative_tokens = base.num_speculative_tokens
        self.synthetic_conditional_rates = None
        self.target_hidden_states = base.target_hidden_states
        self.gathered_logits = base.gathered_logits

    def forward(self, *, attn_metadata, target_hidden_states, **kw):
        _, drafts, _ = self.base.draft_model(
            attn_metadata=attn_metadata, initial_target_hidden_states=target_hidden_states, **kw
        )  # fmt: skip
        return drafts


def _vllm_config(
    tmp_path,
    hf_cfg,
    *,
    spec,
    async_scheduling,
    apc=False,
    cache_dtype="auto",
    max_model_len=MAX_MODEL_LEN,
):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    # vLLM's config only shapes the runner and the scheduler; the model under test is
    # built from ``hf_cfg`` (sparse QSA budget 16), which vLLM's config check refuses.
    cfg = dict(
        tiny_text_config_dict(dtype="float32"),
        architectures=["Qwen4ExpForCausalLM"],
        model_type="qwen4_exp_text",
        mtp_num_hidden_layers=1,
        indexer_budget=2048,
    )
    assert hf_cfg.indexer_budget == 16
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=max_model_len,
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(
            block_size=BLOCK, cache_dtype=cache_dtype, enable_prefix_caching=apc
        ),
        speculative_config=(
            SpeculativeConfig(
                method="mtp",
                num_speculative_tokens=K,
                target_model_config=model_config,
                target_parallel_config=parallel_config,
            )
            if spec
            else None
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=MAX_REQS,
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
                "decode_steps_between_prefill_chunks": 1,
            }
        },
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    return vllm_config


class _Engine:
    """The engine core's loop (batch queue of depth 2 with async scheduling) around one
    V2 runner and a Neuron scheduler."""

    def __init__(self, vllm_config, model, draft):
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec
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
        model.set_mtp_hidden_output(draft is not None)
        if draft is not None:
            # A draft model, or a factory of one (the oracle reads the runner).
            is_model = isinstance(draft, torch.nn.Module)
            runner.draft_model = draft if is_model else draft(runner)
        runner._init_host_inputs()
        specs = runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(vllm_config, specs)
        page = next(
            s.page_size_bytes
            for s in specs.values()
            if isinstance(s, FullAttentionSpec)
        )
        num_attn = sum(isinstance(s, FullAttentionSpec) for s in specs.values())
        if vllm_config.cache_config.enable_prefix_caching:
            # Mamba align mode: every group's block is the state page.
            page = max(s.page_size_bytes for s in specs.values())
            num_attn = max(len(g.layer_names) for g in groups)
        config = get_kv_cache_config_from_groups(
            vllm_config, groups, 120 * page * num_attn
        )
        runner.initialize_kv_cache(config)
        speculator = runner._build_speculator()
        if speculator is not None and isinstance(runner.draft_model, _Oracle):
            speculator = _Speculator(speculator)
        runner._forward_adapter = NeuronForwardAdapter(
            _Greedy(model, draft is not None),
            emit_gathered_logits=runner._logprobs_enabled,
            speculator=speculator,
        )
        self.runner = runner
        vllm_config.cache_config.num_gpu_blocks = config.num_blocks
        # EngineCore._initialize_kv_caches: the scheduler sees the manager block.
        vllm_config.cache_config.block_size = min(
            g.kv_cache_spec.block_size for g in config.kv_cache_groups
        )
        block_size, hash_block_size = resolve_kv_cache_block_sizes(config, vllm_config)
        self.block_hasher = None
        if vllm_config.cache_config.enable_prefix_caching:
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core.kv_cache_utils import (
                get_request_block_hasher,
                init_none_hash,
            )

            hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(hash_fn)
            self.block_hasher = get_request_block_hasher(hash_block_size, hash_fn)
        self.admitted = {}  # req_id -> cached tokens at admission
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
        self.num_draft = self.num_accepted = 0

    def add(self, req_id, prompt, max_tokens):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        self.scheduler.add_request(
            Request(req_id, prompt, params, None, block_hasher=self.block_hasher)
        )
        self.outputs[req_id] = []

    def _process(self, scheduler_output, output):
        if hasattr(output, "get_output"):
            output = output.get_output()
        for req_id, toks in zip(output.req_ids, output.sampled_token_ids):
            if scheduler_output.scheduled_spec_decode_tokens.get(req_id) and toks:
                self.num_draft += len(
                    scheduler_output.scheduled_spec_decode_tokens[req_id]
                )
                self.num_accepted += len(toks) - 1
        for outs in self.scheduler.update_from_output(
            scheduler_output, output
        ).values():
            for out in outs.outputs:
                self.outputs[out.request_id] += list(out.new_token_ids)
        if not self.async_scheduling:
            drafts = self.runner.take_draft_token_ids()
            if drafts is not None:
                self.scheduler.update_draft_token_ids(drafts)

    def run(self, arrivals, max_steps=800):
        for n in range(max_steps):
            for args in arrivals.pop(n, ()):
                self.add(*args)
            if not arrivals and not self.scheduler.has_requests() and not self.queue:
                return
            if self.scheduler.has_requests() and len(self.queue) < self.depth:
                so = self.scheduler.schedule()
                for r in so.scheduled_new_reqs:
                    self.admitted.setdefault(r.req_id, r.num_computed_tokens)
                out = self.runner.execute_model(so)
                if so.total_num_scheduled_tokens > 0:
                    out = self.runner.sample_tokens(None)
                self.queue.append((so, out))
                if len(self.queue) < self.depth and self.scheduler.has_requests():
                    continue
            if self.queue:
                self._process(*self.queue.popleft())
        raise AssertionError("engine did not finish")


# (request id, prompt length, max_tokens, arrival step): "long" prefills in two chunks
# while the others decode.
SCENARIO = {"a": (20, 24, 0), "b": (33, 20, 0), "c": (9, 16, 0), "long": (150, 8, 4)}


@pytest.fixture(scope="module")
def scenario():
    hf, hf_cfg = _hf_model()
    g = torch.Generator().manual_seed(5)
    prompts = {
        r: torch.randint(2, hf_cfg.vocab_size, (n,), generator=g).tolist()
        for r, (n, _, _) in SCENARIO.items()
    }
    prompts["a"][7] = hf_cfg.eos_token_id  # the n-gram EOS rule inside a prompt
    expected = {r: _hf_greedy(hf, prompts[r], SCENARIO[r][1]) for r in SCENARIO}
    return hf, hf_cfg, prompts, expected


def _model(plugin_mtp, hf, hf_cfg):
    plugin, _ = plugin_mtp
    model = plugin.Qwen4ExpForCausalLM.from_configs(hf_cfg, None)
    load_from_hf(model, hf)
    _host_ple(model, hf)
    return model


@pytest.mark.parametrize(
    "draft,async_scheduling",
    [(None, False), (None, True), ("oracle", True), ("oracle", False), ("mtp", True)],
    ids=["plain-sync", "plain-async", "oracle-async", "oracle-sync", "mtp-async"],
)
def test_qwen4_exp_on_v2_matches_hf(
    tmp_path, plugin_mtp, scenario, draft, async_scheduling
):
    hf, hf_cfg, prompts, expected = scenario
    model = _model(plugin_mtp, hf, hf_cfg)
    vllm_config = _vllm_config(
        tmp_path, hf_cfg, spec=draft is not None, async_scheduling=async_scheduling
    )
    if draft == "oracle":
        draft_model = lambda runner: _Oracle(runner, expected)
    elif draft == "mtp":
        from test_mtp import _mtp_state_dict

        _, MTP = plugin_mtp
        draft_model = MTP.Qwen4ExpMTP.from_configs(hf_cfg, hf_cfg.num_hidden_layers)
        load_state(draft_model, _mtp_state_dict(hf, hf_cfg))
        draft_model.num_speculative_tokens = K
    else:
        draft_model = None
    engine = _Engine(vllm_config, model, draft_model)
    runner = engine.runner
    # The PLE conv state is its own recurrent-state layer in the dedicated pool.
    assert "layers.1.ple" in runner.state_caches
    assert runner._host_input_specs == {
        "ple_embeds": ((hf_cfg.ple_embed_dim,), torch.float32)
    }
    arrivals = {}
    for r, (_, max_tokens, step) in SCENARIO.items():
        arrivals.setdefault(step, []).append((r, prompts[r], max_tokens))
    engine.run(arrivals)
    for r, prompt in prompts.items():
        assert prompt + engine.outputs[r] == expected[r], r
    if draft == "oracle":
        assert 0 < engine.num_accepted < engine.num_draft


@pytest.mark.parametrize(
    "draft,async_scheduling",
    [(None, False), (None, True), ("oracle", True)],
    ids=["plain-sync", "plain-async", "oracle-async"],
)
def test_qwen4_exp_prefix_caching_on_v2_matches_hf(
    tmp_path, plugin_mtp, draft, async_scheduling
):
    """Prefix caching (Mamba align mode): the GDN and PLE state checkpoints at block
    boundaries, the QSA side caches of cached blocks and the host token history of a
    request admitted with a hit. Two requests share a multi-block prefix (the second
    arrives once the first is cached), and a third continues the first one's prompt and
    answer. Greedy outputs equal HF's; the later requests hit."""
    hf, hf_cfg = _hf_model()
    model = _model(plugin_mtp, hf, hf_cfg)
    g = torch.Generator().manual_seed(7)
    shared = torch.randint(2, hf_cfg.vocab_size, (150,), generator=g).tolist()
    tail = torch.randint(2, hf_cfg.vocab_size, (30,), generator=g).tolist()
    prompts = {"p": shared + tail[:10], "q": shared + tail[10:]}
    expected = {r: _hf_greedy(hf, p, 12) for r, p in prompts.items()}
    prompts["r"] = expected["p"] + tail[20:]
    expected["r"] = _hf_greedy(hf, prompts["r"], 8)
    vllm_config = _vllm_config(
        tmp_path,
        hf_cfg,
        spec=draft is not None,
        async_scheduling=async_scheduling,
        apc=True,
    )
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    draft_model = (lambda runner: _Oracle(runner, expected)) if draft else None
    engine = _Engine(vllm_config, model, draft_model)
    runner = engine.runner
    assert runner._ckpt is not None
    # vLLM groups layers of equal specs, as many as the attention layers per group: the
    # GDN and PLE state layers must look alike to it (one placeholder shape), else every
    # layer gets its own group and each attention layer's KV view spans the whole pool.
    spec = model.get_kv_spec()
    n_attn, n_state = len(spec.layers), len(spec.state_layers)
    groups = runner._kv_cache_config.kv_cache_groups
    assert len(groups) == 1 + -(-n_state // n_attn), [
        len(g.layer_names) for g in groups
    ]
    engine.run(
        {
            0: [("p", prompts["p"], 12)],
            12: [("q", prompts["q"], 12)],
            40: [("r", prompts["r"], 8)],
        }
    )
    for r, prompt in prompts.items():
        assert prompt + engine.outputs[r] == expected[r], r
    block = engine.runner._ckpt_block_size
    assert engine.admitted["p"] == 0
    assert engine.admitted["r"] >= block, (engine.admitted, block)
    if draft is None:
        assert engine.admitted["q"] >= block, (engine.admitted, block)
    # With MTP vLLM drops the last matched block of a hit (EAGLE-style drafter): q's
    # shared prefix has a checkpoint only where p's first chunk ended, the last block.


@pytest.mark.parametrize("draft", [None, "oracle"], ids=["plain", "oracle"])
def test_qwen4_exp_fp8_kv_prefix_caching_matches_no_apc(tmp_path, plugin_mtp, draft):
    """FP8 KV cache (unit scales; the QSA side caches stay in the model dtype), async
    scheduling, with and without prefix caching: the same prompts give the same greedy
    outputs (both runs quantize the KV cache alike; FP8 makes HF a loose reference), and
    the later request hits the cache. The FP8 attention page is half the BF16 one, so
    the prefix-cache block is twice as long (256 tokens here)."""
    hf, hf_cfg = _hf_model()
    g = torch.Generator().manual_seed(11)
    shared = torch.randint(2, hf_cfg.vocab_size, (300,), generator=g).tolist()
    tails = [
        torch.randint(2, hf_cfg.vocab_size, (n,), generator=g).tolist() for n in (9, 21)
    ]
    prompts = {"p": shared + tails[0], "q": shared + tails[1]}
    expected = {r: _hf_greedy(hf, p, 10) for r, p in prompts.items()}
    outputs, admitted = {}, {}
    for apc in (False, True):
        model = _model(plugin_mtp, hf, hf_cfg)
        vllm_config = _vllm_config(
            tmp_path,
            hf_cfg,
            spec=draft is not None,
            async_scheduling=True,
            apc=apc,
            cache_dtype="fp8",
            max_model_len=640,
        )
        draft_model = (lambda runner: _Oracle(runner, expected)) if draft else None
        engine = _Engine(vllm_config, model, draft_model)
        k_cache = next(iter(engine.runner.kv_caches.values()))[0]
        assert k_cache.dtype == torch.float8_e4m3fn
        if apc:
            assert engine.runner._ckpt_block_size == 256
        engine.run({0: [("p", prompts["p"], 10)], 12: [("q", prompts["q"], 10)]})
        outputs[apc], admitted[apc] = dict(engine.outputs), dict(engine.admitted)
    assert outputs[True] == outputs[False]
    assert all(len(v) == 10 for v in outputs[True].values())
    assert admitted[False]["q"] == 0
    if draft is None:
        assert admitted[True]["q"] == 256, admitted


def test_compiler_args_policy(plugin_mtp):
    """The dense prefill graph with the MTP head inside it compiles at -O2 (neuronx-cc 2.28
    fails it at -O1 with NCC_INLA001); every other graph keeps the base arguments."""
    _, hf_cfg = _hf_model()
    plugin, _ = plugin_mtp
    model = plugin.Qwen4ExpForCausalLM.from_configs(hf_cfg, None)
    base = ["--auto-cast=none", "-O1", "--logical-nc-config=2"]
    assert model.compiler_args_for("prefill_dense_spec", None, base) == [
        "--auto-cast=none",
        "-O2",
        "--logical-nc-config=2",
    ]
    for graph_type in (None, "prefill_dense", "decode"):
        assert model.compiler_args_for(graph_type, None, base) == base
    assert base == ["--auto-cast=none", "-O1", "--logical-nc-config=2"]
