# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 prefix caching (vLLM Mamba "align" mode) for Qwen3.5, against HF (CPU).

Real pieces, as in test_v2_spec_decode_e2e: vLLM's config (enable_prefix_caching, which the
Neuron platform turns into Mamba cache mode "align"), KV cache grouping, vLLM's KV cache
manager and block pool with prefix caching and the engine core's request block hasher, the
Neuron schedulers (sync, and async with the engine core's batch queue of depth 2), the V2
runner and the Qwen3.5 module (NKI ops stubbed in torch), dense and sparse MoE, without
and with MTP (k = 2: oracle drafts with partial acceptance, and the real MTP head).

Geometry: kernel block 8, state page = 2 attention pages, so every KV group's block (the
prefix-cache block) is 16 tokens, and each attention manager block is split into 2 kernel
blocks. The running state stays in the dedicated pool; vLLM's state blocks in the shared
pool hold the block-boundary checkpoints (hybrid_state.StateCheckpointer).

Scenario: requests sharing a prompt prefix with a running request (concurrent sharers),
with the generated tokens of a finished request (its blocks were freed and are reused from
the cache; the checkpoints were saved by decode / verify steps, which straddle block
boundaries with partial acceptance), a chunked prefill crossing block boundaries and a
request hitting its chunks, and hits of one and of several blocks. Checks: greedy outputs
equal HF's (and so the run without prefix caching, which is checked too); hits happened
(num_computed_tokens > 0 at admission); every state block vLLM cached was written by a
save that reached exactly its boundary; every restored page equals its checkpoint and HF's
GDN state after the hit's tokens; KV-block zeroing never touches a hit block.
"""

import json
from collections import deque

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("transformers.models.qwen3_5")

import test_v2_spec_decode_e2e as spec

K = 2
VOCAB = spec.VOCAB
BLOCK = 8  # attention kernel block (--block-size)
CKPT_BLOCK = 16  # every group's KV-manager block: the prefix-cache block
MAX_MODEL_LEN = 320
CHUNK = 128
MAX_REQS = 4
NUM_BLOCKS = 800


# ---------------------------------------------------------------- engine harness


def _vllm_config(
    tmp_path,
    checkpoint,
    *,
    apc,
    spec_k,
    async_scheduling,
    decode_steps=0,
    retention=None,
    cache_dtype="auto",
):
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    (tmp_path / "config.json").write_text(json.dumps(spec.CHECKPOINTS[checkpoint]))
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=MAX_MODEL_LEN,
    )
    parallel_config = ParallelConfig()
    speculative_config = None
    if spec_k:
        speculative_config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=spec_k,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(
            block_size=BLOCK,
            cache_dtype=cache_dtype,
            enable_prefix_caching=apc,
            # None: vLLM retains a state checkpoint at every block boundary (decode
            # boundaries too); its default 0 keeps only the prompt's last boundary
            # and shared-prefix junctions.
            prefix_cache_retention_interval=retention,
        ),
        speculative_config=speculative_config,
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
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    return vllm_config


class _GreedyPlainTarget(torch.nn.Module):
    """On-device-sampling contract without speculative decoding: greedy tokens and the
    logits as the gathered logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        logits = self.model(**kwargs)
        return logits.argmax(dim=-1).to(torch.int32), logits


class _OracleDraft(spec._OracleDraft):
    """spec._OracleDraft, mapping a row to its request through the attention block of
    its current position: with prefix caching, sharers have the same first block."""

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
        attn = next(
            g
            for g in range(len(self.runner.block_sizes))
            if g not in self.runner._state_groups
        )
        table = attn_metadata[f"layers.{spec.ATTN_LAYERS[0]}.self_attn"][
            "block_table_tensor"
        ]
        rows, width = raw_sampled_token_ids.shape
        req_by_block = {}
        for req, slot in rs.req_id_to_index.items():
            for col in range(rs.num_blocks(attn, slot)):
                req_by_block.setdefault(
                    int(rs.block_table(attn)[slot, col]), set()
                ).add(req)
        drafts = torch.zeros(rows, K, dtype=torch.int32)
        for r in range(rows):
            if width == 1:  # prefill: the token after the chunk's last token
                pos = int(positions[int(sampling_positions[r])])
                next_pos = pos + 1
            else:
                pos = int(positions[r * width])
                accepted = int((raw_sampled_token_ids[r] >= 0).sum())
                next_pos = pos + accepted
            reqs = req_by_block.get(int(table[r, pos // BLOCK]), set())
            if len(reqs) != 1:
                continue
            (req,) = reqs
            seq = self.expected[req]
            for j in range(K):
                p = next_pos + 1 + j
                tok = seq[p] if p < len(seq) else 0
                if self.corrupt == "all" or (req, next_pos, j) in self.corrupt:
                    tok = (tok + 1) % VOCAB
                drafts[r, j] = tok
            self.proposed.append((req, next_pos, drafts[r].tolist()))
        return None, drafts, None


class _Engine(spec._Engine):
    """spec._Engine with prefix caching, an optional draft, the engine core's request
    block hasher, and its block-size update (cache_config.block_size = the smallest
    group block, which the scheduler's align-mode chunking reads). Records every
    admission's cached tokens, the blocks vLLM asked to zero, the state blocks vLLM
    cached, and the checkpoint copies."""

    def __init__(
        self, vllm_config, model, draft=None, *, oracle=False, num_blocks=NUM_BLOCKS
    ):
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
            get_request_block_hasher,
            init_none_hash,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.structured_output import StructuredOutputManager

        from vllm_neuron.vllm.core.scheduler import (
            NeuronAsyncScheduler,
            NeuronScheduler,
        )
        from vllm_neuron.vllm.worker import hybrid_state
        from vllm_neuron.vllm.worker.neuron_forward_adapter import (
            NeuronForwardAdapter,
        )
        from vllm_neuron.vllm.worker.neuron_model_runner_v2 import NeuronModelRunnerV2

        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
        runner = NeuronModelRunnerV2(vllm_config, torch.device("cpu"))
        runner._rank_tensor_cache = torch.zeros(1, dtype=torch.int32)
        runner.model = model
        model.set_mtp_hidden_output(draft is not None)
        model.set_runner_clears_kv_blocks(False)
        if draft is not None:
            runner.draft_model = draft(runner) if oracle else draft
        specs = runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(vllm_config, specs)
        page = max(s.page_size_bytes for s in specs.values())
        tensors_per_block = max(len(g.layer_names) for g in groups)
        config = get_kv_cache_config_from_groups(
            vllm_config, groups, num_blocks * page * tensors_per_block
        )
        runner.initialize_kv_cache(config)
        self.zeroed = []
        if runner._ckpt is not None:
            # The device runner clears new KV blocks itself (CPU runs leave it to the
            # model): run the real clearer here.
            caches = {id(c): c for kv in runner.kv_caches.values() for c in kv}

            def zero(caches, kernel_blocks):
                self.zeroed.append(kernel_blocks.tolist())
                return hybrid_state.zero_kv_blocks(caches, kernel_blocks)

            runner._kv_clearer = hybrid_state.KVBlockClearer(
                list(caches.values()),
                runner._ckpt_block_size // BLOCK,
                runner.device,
                zero_fn=zero,
            )
            model.set_runner_clears_kv_blocks(True)
            if hasattr(runner.draft_model, "set_runner_clears_kv_blocks"):
                runner.draft_model.set_runner_clears_kv_blocks(True)
            self._record_checkpoints(runner)
        speculator = runner._build_speculator()
        if oracle:
            speculator = spec._FullMetadataSpeculator(speculator)
        target = (
            spec._GreedyTarget(model)
            if draft is not None
            else _GreedyPlainTarget(model)
        )
        runner._forward_adapter = NeuronForwardAdapter(
            target,
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
        cls = NeuronAsyncScheduler if self.async_scheduling else NeuronScheduler
        self.scheduler = cls(
            vllm_config=vllm_config,
            kv_cache_config=config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
            hash_block_size=hash_block_size,
        )
        self.block_hasher = None
        if vllm_config.cache_config.enable_prefix_caching:
            hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(hash_fn)
            self.block_hasher = get_request_block_hasher(hash_block_size, hash_fn)
        self.queue = deque()
        self.depth = 2 if self.async_scheduling else 1
        self.outputs: dict[str, list[int]] = {}
        self.logprobs: dict[str, list] = {}
        self.steps = []
        self.num_draft = self.num_accepted = 0
        self.drafts = []
        self.prompts: dict[str, list[int]] = {}
        self.admissions: dict[str, list[int]] = {}  # req -> num_computed per admission
        self.preempted: set[str] = set()
        self.zero_requests = []  # (new_block_ids_to_zero, blocks hit by admissions)
        self.cached_state_blocks = set()  # (state group, block, boundary)
        self.cached_boundaries = set()  # (req, boundary)
        self._step_output = None
        self.decode_tables = []  # (slots of the rows, GDN table, slot mappings) per step
        build_kwargs = runner._build_forward_adapter_kwargs

        def record_kwargs(model_input, **kw):
            kwargs = build_kwargs(model_input, **kw)
            if not model_input.is_prefill and runner._state_groups:
                md = kwargs["attn_metadata"]
                self.decode_tables.append(
                    (
                        model_input.idx_mapping_np.tolist(),
                        md["layers.0.linear_attn"]["block_table_tensor"].clone(),
                        md["layers.0.linear_attn"]["slot_mapping"].clone(),
                        md[f"layers.{spec.ATTN_LAYERS[0]}.self_attn"][
                            "slot_mapping"
                        ].clone(),
                    )
                )
            return kwargs

        runner._build_forward_adapter_kwargs = record_kwargs

    def add(self, req_id, prompt, max_tokens, logprobs=None):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, logprobs=logprobs, ignore_eos=True
        )
        self.scheduler.add_request(
            Request(req_id, prompt, params, None, block_hasher=self.block_hasher)
        )
        self.outputs[req_id] = []
        self.prompts[req_id] = list(prompt)

    def _record_checkpoints(self, runner):
        """Wrap the checkpoint copies (eager on CPU): record each save candidate and
        whether the device counts reached it, and each restore."""
        ckpt = runner._ckpt
        # (state group, block, boundary) -> (request, verify column, pages) of the
        # saves that reached their boundary (the last one per key).
        self.saved = {}
        self.save_columns = []  # verify-step column of each reached save
        self.unreached = 0  # candidates the step did not reach
        self.restores = []  # (request, hit tokens, restored page, [blocks])
        save_fn, restore_fn = ckpt._save_fn, ckpt._restore_fn

        def save(pool, ckpts, groups, row_slot, boundary, dst, valid, nc, na, pps):
            n_after = nc.long()[row_slot]
            n_before = n_after - na.long()[row_slot]
            reached = valid & (boundary > n_before) & (boundary <= n_after)
            out = save_fn(
                pool, ckpts, groups, row_slot, boundary, dst, valid, nc, na, pps
            )
            assert int(out) == int(reached.sum())
            for r in range(row_slot.shape[0]):
                if not bool(valid[r]):
                    continue
                if not bool(reached[r]):
                    self.unreached += 1
                    continue
                j = int(boundary[r] - n_before[r] - 1)
                self.save_columns.append(j)
                slot = int(row_slot[r])
                page = 1 + slot * pps + j
                req_id = runner.request_state.index_to_req_id[slot]
                for g in range(dst.shape[0]):
                    block = int(dst[g, r])
                    assert block > 0
                    pages = []
                    for c, p, gi in zip(ckpts, pool, groups):
                        if gi == g:
                            assert torch.equal(c[block], p[page])
                            pages.append(c[block].clone())
                    self.saved[(g, block, int(boundary[r]))] = (req_id, j, pages)
            return out

        def restore(ckpts, pool, groups, src, dst_page, valid):
            out = restore_fn(ckpts, pool, groups, src, dst_page, valid)
            for r in range(dst_page.shape[0]):
                if not bool(valid[r]):
                    continue
                page = int(dst_page[r])
                blocks = [int(src[g, r]) for g in range(src.shape[0])]
                for c, p, g in zip(ckpts, pool, groups):
                    assert torch.equal(p[page], c[blocks[g]])
                (req,) = [
                    (n.req_id, n.num_computed_tokens)
                    for n in self._step_output.scheduled_new_reqs
                    if n.num_computed_tokens > 0
                    and 1
                    + runner.request_state.req_id_to_index[n.req_id]
                    * ckpt.pages_per_slot
                    == page
                ]
                # The restored page of every layer, in the checkpointer's order.
                pages = [p[page].clone() for p in pool]
                self.restores.append((*req, pages, blocks))
            return out

        ckpt._save_fn, ckpt._restore_fn = save, restore

    def _record_drafts(self, scheduler_output):
        if self.runner.request_state.draft_tokens is not None:
            super()._record_drafts(scheduler_output)

    def _record_cached_state_blocks(self):
        """The state-group blocks the KV manager has hashed, with the boundary each ends
        on (blocks are hashed at schedule time, before the step computes them)."""
        if self.runner._ckpt is None:
            return
        coordinator = self.scheduler.kv_cache_manager.coordinator
        group_index = {g: k for k, g in enumerate(self.runner._ckpt.group_ids)}
        for gi in self.runner._state_groups:
            manager = coordinator.single_type_managers[gi]
            for req_id, blocks in manager.req_to_blocks.items():
                for col, block in enumerate(blocks):
                    if block.is_null or block.block_hash is None:
                        continue
                    boundary = (col + 1) * self.runner._ckpt_block_size
                    self.cached_state_blocks.add(
                        (group_index[gi], block.block_id, boundary)
                    )
                    self.cached_boundaries.add((req_id, boundary))

    def step(self):
        if self.scheduler.has_requests() and len(self.queue) < self.depth:
            so = self.scheduler.schedule()
            self.preempted |= set(so.preempted_req_ids or ())
            hit_blocks = set()
            for req in so.scheduled_new_reqs:
                self.admissions.setdefault(req.req_id, []).append(
                    req.num_computed_tokens
                )
                n = req.num_computed_tokens // (self.runner._ckpt_block_size or 1)
                for ids in req.block_ids:
                    hit_blocks.update(ids[:n])
            hit_blocks.discard(0)
            self.zero_requests.append(
                (list(so.new_block_ids_to_zero or ()), hit_blocks)
            )
            self._record_cached_state_blocks()
            self.steps.append(
                (dict(so.num_scheduled_tokens), dict(so.scheduled_spec_decode_tokens))
            )
            self._step_output = so
            out = self.runner.execute_model(so)
            if so.total_num_scheduled_tokens > 0:
                out = self.runner.sample_tokens(None)
                self._record_drafts(so)
            self.queue.append((so, out))
            if len(self.queue) < self.depth and self.scheduler.has_requests():
                return
        if self.queue:
            self._process(*self.queue.popleft())


# ---------------------------------------------------------------- scenario


@pytest.fixture(scope="module", params=list(spec.CHECKPOINTS))
def checkpoint(request):
    return request.param


@pytest.fixture(scope="module")
def models(checkpoint):
    torch.manual_seed(3)
    return spec._build_models(moe=checkpoint == "moe")


def _rand(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (n,), generator=g).tolist()


# request: (max_tokens, arrival step). "a" runs first; "b" and "c" share its prompt
# while it runs, "e" shares c's, "d" continues a's output after a finished, "long"
# prefills in chunks across block boundaries, and "long2" and "long3" hit its chunks.
ARRIVALS = {
    "a": (40, 0),
    "b": (12, 4),
    "c": (12, 6),
    "e": (10, 30),
    "long": (12, 34),
    "d": (12, 80),
    "long2": (10, 90),
    "long3": (8, 100),
    "f": (8, 50),
}


@pytest.fixture(scope="module")
def scenario(models):
    hf_model = models[0]
    base, long = _rand(1, 40), _rand(2, 200)
    expected_a = spec._hf_greedy(hf_model, base, ARRIVALS["a"][0])
    prompts = {
        "a": base,
        "b": base[:36] + _rand(3, 14),
        "c": base[:20] + _rand(4, 10),
        "e": base[:18] + _rand(5, 15),
        "d": expected_a[:70] + _rand(6, 5),
        "long": long,
        "long2": long[:150] + _rand(7, 20),
        # A hit, then two chunks across several block boundaries.
        "long3": long[:150] + _rand(8, 150),
        # One token past a cached block: the decode graph computes the last prompt
        # token from the restored state.
        "f": base[:32] + _rand(9, 1),
    }
    expected = {"a": expected_a}
    for req_id, prompt in prompts.items():
        if req_id != "a":
            expected[req_id] = spec._hf_greedy(hf_model, prompt, ARRIVALS[req_id][0])
    return prompts, expected


def _arrivals(prompts):
    arrivals = {}
    for req_id, (max_tokens, step) in ARRIVALS.items():
        arrivals.setdefault(step, []).append((req_id, prompts[req_id], max_tokens))
    return arrivals


def _hf_gdn_states(hf_model, tokens):
    """{layer index: (conv [C, K - 1], ssm [Hv, Dk, Dv])} of HF's GDN layers after
    ``tokens``."""
    with torch.no_grad():
        cache = hf_model(torch.tensor([tokens]), use_cache=True).past_key_values
    states = {}
    for i, layer in enumerate(cache.layers):
        conv = getattr(layer, "conv_states", None)
        if conv is not None:
            states[i] = (
                conv[0][0, :, 1:].float(),
                layer.recurrent_states[0][0].float(),
            )
    return states


def _decode_page(gdn, half_a, half_b):
    """(conv [C, K - 1], ssm [Hv, Dk, Dv]) of one state page, in
    Qwen3_5GatedDeltaNet._read_state's layout."""
    rows = gdn._STATE_ROW
    page = torch.cat([half_a.reshape(-1, rows), half_b.reshape(-1, rows)], dim=0)
    cr, sr = gdn.conv_numel // rows, gdn.ssm_numel // rows
    conv = page[sr : sr + cr].reshape(gdn.K - 1, gdn.conv_dim).transpose(0, 1)
    return conv, page[:sr].reshape(gdn.Hv, gdn.Dk, gdn.Dv)


def _check_outputs(engine, expected, prompts):
    for req_id, prompt in prompts.items():
        assert prompt + engine.outputs[req_id] == expected[req_id], req_id


def _check_prefix_caching(engine, models, batched=True, hf=True):
    """Checkpoints and restores of a prefix-caching run (see the module docstring).
    ``batched``: some decode steps had several rows, padding and rows out of slot
    order. ``hf``: compare checkpoint contents with HF's state (not with an FP8 KV
    cache). Returns the cached checkpoints saved from a verify step's draft column."""
    hf_model, model = models[0], models[1]
    runner = engine.runner
    # Every state block vLLM cached was written by a save that reached its boundary
    # (the step computing it saved the state after exactly that many tokens).
    assert engine.cached_state_blocks
    missing = engine.cached_state_blocks - set(engine.saved)
    assert not missing, sorted(missing)[:5]
    # The checkpoints a verify step saved from a draft column (j > 0: the boundary
    # fell after an accepted draft) hold HF's state after the boundary.
    groups = runner._kv_cache_config.kv_cache_groups
    straddled = [
        (key, info)
        for key, info in engine.saved.items()
        if key in engine.cached_state_blocks and info[1] > 0
    ]
    for (k, _, boundary), (req_id, _, pages) in straddled[: 4 if hf else 0]:
        seq = engine.prompts[req_id] + engine.outputs[req_id]
        ref = _hf_gdn_states(hf_model, seq[:boundary])
        names = groups[runner._ckpt.group_ids[k]].layer_names
        for name, half_a, half_b in zip(names, pages[0::2], pages[1::2]):
            layer = int(name.split(".")[1])
            conv, ssm = _decode_page(
                model.model.layers[layer].linear_attn, half_a, half_b
            )
            torch.testing.assert_close(conv, ref[layer][0], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(ssm, ref[layer][1], atol=1e-4, rtol=1e-4)
    # Every restore copied a cached block and matches HF's state after the hit.
    assert engine.restores
    assert runner._ckpt.num_restores == len(engine.restores)
    hits = [(r, n) for r, ns in engine.admissions.items() for n in ns if n]
    assert sorted((r, n) for r, n, _, _ in engine.restores) == sorted(hits)
    names = [n for g in runner._ckpt.group_ids for n in groups[g].layer_names]
    for req_id, hit, pages, blocks in engine.restores:
        for k in range(len(runner._ckpt.group_ids)):
            assert (k, blocks[k], hit) in engine.cached_state_blocks
    for req_id, hit, pages, blocks in engine.restores[: 6 if hf else 0]:
        ref = _hf_gdn_states(hf_model, engine.prompts[req_id][:hit])
        for name, half_a, half_b in zip(names, pages[0::2], pages[1::2]):
            layer = int(name.split(".")[1])
            gdn = model.model.layers[layer].linear_attn
            conv, ssm = _decode_page(gdn, half_a, half_b)
            ref_conv, ref_ssm = ref[layer]
            torch.testing.assert_close(conv, ref_conv, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(ssm, ref_ssm, atol=1e-4, rtol=1e-4)
    # KV-block zeroing never touched a block an admission hit (new_block_ids_to_zero
    # leaves prefix hits out; the clearer zeroes each manager block's kernel blocks).
    ratio = runner.request_state.kernel_ratios[_attn_group(runner)]
    for zero_ids, hit_blocks in engine.zero_requests:
        assert not set(zero_ids) & hit_blocks
    assert engine.zeroed
    for chunk in engine.zeroed:
        for i in range(0, len(chunk), ratio):
            first = chunk[i]
            assert first % ratio == 0 and chunk[i : i + ratio] == list(
                range(first, first + ratio)
            )
    _check_decode_tables(engine, batched)
    return straddled


def _attn_group(runner):
    return next(
        g for g in range(len(runner.block_sizes)) if g not in runner._state_groups
    )


def _check_decode_tables(engine, batched):
    """Decode / verify steps: the GDN table holds each row's slot pages in row order and
    -1 for padded rows, and the state groups' valid tokens are the attention group's.
    Some steps ran several rows with padding, and rows in an order other than their
    slots'."""
    pages = engine.runner._state_pool_slot_pages
    assert engine.decode_tables
    for slots, table, gdn_slots, attn_slots in engine.decode_tables:
        expected = [[1 + s * pages + j for j in range(pages)] for s in slots]
        expected += [[-1] * pages] * (table.shape[0] - len(slots))
        assert table.tolist() == expected
        assert torch.equal(gdn_slots, attn_slots)
    if not batched:
        return
    assert any(1 < len(s) < t.shape[0] for s, t, _, _ in engine.decode_tables)
    assert any(s != sorted(s) for s, _, _, _ in engine.decode_tables)


def _hits(engine):
    hits = {r: max(v) for r, v in engine.admissions.items()}
    block = engine.runner._ckpt_block_size or 1
    assert all(h % block == 0 for h in hits.values())
    return hits


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("apc", [True, False], ids=["apc", "no_apc"])
def test_prefix_caching_matches_hf(
    tmp_path, checkpoint, models, scenario, async_scheduling, apc
):
    _, model, _ = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path, checkpoint, apc=apc, spec_k=0, async_scheduling=async_scheduling
    )
    engine = _Engine(vllm_config, model)
    runner = engine.runner
    engine.run(_arrivals(prompts))
    _check_outputs(engine, expected, prompts)
    hits = _hits(engine)
    if not apc:
        assert runner._ckpt is None and not any(hits.values())
        assert vllm_config.cache_config.mamba_cache_mode == "none"
        return
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert runner._ckpt_block_size == CKPT_BLOCK
    assert runner.block_sizes == [
        CKPT_BLOCK if g in runner._state_groups else BLOCK
        for g in range(len(runner.block_sizes))
    ]
    assert hits == {
        "a": 0,
        # A concurrent sharer: a's first chunk stopped at its last full block.
        "b": 32,
        # Shares one block with a, whose first chunk did not stop there: a miss,
        # but its own first chunk stops at the junction ...
        "c": 0,
        # ... so this one hits one block.
        "e": 16,
        "long": 0,
        # a's decode checkpoints, after a finished.
        "d": 64,
        # long's chunk boundary (long's own chunks stop at 128 and 192).
        "long2": 128,
        # long2's chunk stopped at the junction it shares with long (144).
        "long3": 144,
        "f": 32,
    }
    _check_prefix_caching(engine, models)


def _corrupt():
    """Oracle draft errors: a's second draft every third position, long's first and d's
    second every fourth, so verify steps accept all, some or none of their drafts
    around the block boundaries."""
    corrupt = set()
    for p in range(MAX_MODEL_LEN):
        if p % 3 == 0:
            corrupt.add(("a", p, 1))
        if p % 4 == 1:
            corrupt |= {("long", p, 0), ("d", p, 1)}
    return corrupt


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("draft", ["oracle", "mtp"])
def test_prefix_caching_with_mtp_matches_hf(
    tmp_path, checkpoint, models, scenario, async_scheduling, draft
):
    _, model, mtp = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path, checkpoint, apc=True, spec_k=K, async_scheduling=async_scheduling
    )
    if draft == "oracle":
        engine = _Engine(
            vllm_config,
            model,
            lambda runner: _OracleDraft(runner, expected, _corrupt()),
            oracle=True,
        )
    else:
        engine = _Engine(vllm_config, model, mtp)
    runner = engine.runner
    assert runner._state_pool_slot_pages == 1 + K
    engine.run(_arrivals(prompts))
    _check_outputs(engine, expected, prompts)
    hits = _hits(engine)
    # With MTP (an EAGLE-style drafter) vLLM matches one block past each hit and drops
    # it, and ends a prompt's cached chunks one block earlier.
    assert hits["b"] == 16 and hits["long2"] == 128, hits
    assert hits["d"] >= 32, hits  # a's (or b's) checkpoints, after a finished
    assert engine.num_draft > 0
    straddled = _check_prefix_caching(engine, models)
    if draft == "oracle":
        assert 0 < engine.num_accepted < engine.num_draft
        # Verify steps reached block boundaries after accepted drafts, and vLLM cached
        # those checkpoints (checked against HF above).
        assert straddled
    else:
        # Mostly rejected drafts: boundaries among the rejected drafts are not saved.
        assert engine.unreached > 0
        # The head's drafts after a prefix hit (its KV from the sharer's blocks) equal a
        # from-scratch run of the head over each committed sequence.
        records = [r for r in engine.drafts if hits.get(r[0])]
        assert records
        for req_id, tokens, drafts in records[::4]:
            assert drafts == spec._reference_mtp_drafts(mtp, models[0], tokens), req_id


@pytest.mark.parametrize("spec_k", [0, K], ids=["plain", "mtp"])
def test_prefix_caching_with_evictions_matches_hf(
    tmp_path, checkpoint, models, scenario, spec_k
):
    """A pool of 48 blocks: vLLM evicts cached checkpoints (the hits get shorter) and
    hands their blocks out again, as attention or state blocks, while decode steps run
    between prefill chunks (async scheduling)."""
    _, model, mtp = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path,
        checkpoint,
        apc=True,
        spec_k=spec_k,
        async_scheduling=True,
        decode_steps=1,
    )
    engine = _Engine(vllm_config, model, mtp if spec_k else None, num_blocks=48)
    engine.run(_arrivals(prompts))
    _check_outputs(engine, expected, prompts)
    hits = _hits(engine)
    # a's checkpoints were partly evicted before d arrived.
    assert 0 < hits["d"] < (64 if not spec_k else 48), hits
    # Blocks that held a checkpoint were allocated again (and zeroed then).
    ckpt_blocks = {block for _, block, _ in engine.cached_state_blocks}
    reused = ckpt_blocks & {b for ids, _ in engine.zero_requests for b in ids}
    assert reused
    _check_prefix_caching(engine, models, batched=False)


@pytest.mark.parametrize("spec_k", [0, K], ids=["plain", "mtp"])
def test_prefix_caching_default_retention_matches_hf(
    tmp_path, checkpoint, models, scenario, spec_k
):
    """vLLM's default retention keeps only the prompt's last boundary and shared-prefix
    junctions: decode steps still checkpoint, but vLLM caches none of those blocks."""
    _, model, mtp = models
    prompts, expected = scenario
    vllm_config = _vllm_config(
        tmp_path,
        checkpoint,
        apc=True,
        spec_k=spec_k,
        async_scheduling=False,
        retention=0,
    )
    engine = _Engine(vllm_config, model, mtp if spec_k else None)
    engine.run(_arrivals(prompts))
    _check_outputs(engine, expected, prompts)
    hits = _hits(engine)
    # d hits only a prompt boundary of a or b, not a's decode checkpoints, and long2
    # misses long's first chunk boundary (not long's last prompt boundary).
    assert 0 < hits["d"] <= 32 and hits["long2"] == 0, hits
    _check_prefix_caching(engine, models)


@pytest.fixture(scope="module")
def fp8_prompts():
    base = _rand(11, 150)
    return {
        "a": base,
        "b": base[:140] + _rand(12, 20),  # concurrent with a: hits 128
        "c": _rand(13, 70),  # concurrent, no shared prefix
        "d": base[:135] + _rand(14, 30),  # after a and b finished
    }


FP8_ARRIVALS = {"a": (40, 0), "b": (30, 3), "c": (30, 5), "d": (12, 120)}


@pytest.mark.parametrize("spec_k", [0, K], ids=["plain", "mtp"])
def test_prefix_caching_fp8_kv_batched_decode_matches_no_apc(
    tmp_path, checkpoint, models, fp8_prompts, spec_k
):
    """FP8 KV cache (64-token prefix-cache blocks split into 8 kernel blocks of 8):
    batched decode / verify steps with padded rows and rows out of slot order, shared
    and unshared prompts, async scheduling. Greedy outputs equal the run without
    prefix caching (the FP8 cache makes HF a loose reference)."""
    _, model, mtp = models
    arrivals = {}
    for req_id, (max_tokens, step) in FP8_ARRIVALS.items():
        arrivals.setdefault(step, []).append((req_id, fp8_prompts[req_id], max_tokens))
    outputs = {}
    for apc in (False, True):
        vllm_config = _vllm_config(
            tmp_path,
            checkpoint,
            apc=apc,
            spec_k=spec_k,
            async_scheduling=True,
            cache_dtype="fp8",
        )
        engine = _Engine(vllm_config, model, mtp if spec_k else None)
        k_cache = next(iter(engine.runner.kv_caches.values()))[0]
        assert k_cache.dtype == torch.float8_e4m3fn
        engine.run(dict(arrivals))
        outputs[apc] = dict(engine.outputs)
    runner = engine.runner
    assert runner._ckpt_block_size == 64 and k_cache.shape[2] == BLOCK
    assert outputs[True] == outputs[False]
    hits = _hits(engine)
    assert hits["b"] == (128 if not spec_k else 64) and hits["c"] == 0, hits
    assert hits["d"] >= 64, hits
    _check_prefix_caching(engine, models, hf=False)
