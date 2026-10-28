# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2 with recurrent-state (hybrid) models, on CPU.

The runner is a bare ``object.__new__(NeuronModelRunnerV2)`` with only the attributes the
tested methods read; the model is a stand-in that records its bindings. The KV cache
config comes from vLLM's own grouping (get_kv_cache_groups + get_kv_cache_config_from_groups),
as the engine core builds it, so the tests see the real group order, page sizes and
co-tenant layout. The request state, input building and attention metadata are the real
ones.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, StateLayerSpec
from vllm_neuron.vllm.worker import hybrid_state
from vllm_neuron.vllm.worker.neuron_forward_adapter import build_attention_metadata
from vllm_neuron.vllm.worker.neuron_input_build import (
    PAD_SLOT_ID,
    build_state_pool_rows,
)

BLOCK = 32
MAX_MODEL_LEN = 2048
MAX_REQS = 4
NUM_BLOCKS = 64
ATTN_PAGE = 2 * BLOCK * 128 * 2  # K + V, one head of 128, bf16
# conv [3, 64] + ssm [2, 128, 128] float32 -> whole 256-token blocks -> 16 pages (2^4).
REAL_PAGE = 16 * ATTN_PAGE


def _v2():
    # Imported at test time, not at collection (see test_mamba_align_state._nmr).
    from vllm_neuron.vllm.worker import neuron_model_runner_v2

    return neuron_model_runner_v2


class _HybridModel:
    """Two (state, state, state, attention) blocks; records the runner's bindings."""

    def __init__(self, extra_state_layers=()):
        layers, state_layers = [], []
        for i in range(8):
            if i % 4 == 3:
                layers.append(
                    LayerSpec(
                        name=f"layers.{i}.self_attn",
                        num_kv_heads=1,
                        head_size=128,
                        dtype=torch.bfloat16,
                    )
                )
            else:
                state_layers.append(self._state_layer(f"layers.{i}.linear_attn"))
        state_layers += [self._state_layer(n) for n in extra_state_layers]
        self.kv_spec = KVSpec(layers=layers, state_layers=state_layers)
        self.kv_caches = self.state_caches = self.runner_clears = None

    @staticmethod
    def _state_layer(name):
        return StateLayerSpec(
            name=name,
            shapes=((3, 64), (2, 128, 128)),
            dtypes=(torch.float32, torch.float32),
        )

    def get_kv_spec(self):
        return self.kv_spec

    def bind_kv_cache(self, kv_caches):
        self.kv_caches = kv_caches

    def bind_state_cache(self, state_caches):
        self.state_caches = state_caches

    def set_runner_clears_kv_blocks(self, value):
        self.runner_clears = value


def _runner(model=None, *, has_state=True, kv_segment_size=1024):
    v2 = _v2()
    from vllm_neuron.vllm.worker.neuron_request_state import NeuronRequestState

    runner = object.__new__(v2.NeuronModelRunnerV2)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=False
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, architectures=["Qwen3_5ForConditionalGeneration"]
        ),
        speculative_config=None,
    )
    runner.model = _HybridModel() if model is None else model
    runner.max_num_reqs = MAX_REQS
    runner.max_model_len = MAX_MODEL_LEN
    runner.neuron_config = SimpleNamespace(
        fp8_packed_kv=False,
        num_batched_tokens_buckets=[512],
        num_seqs_buckets=[1, 4],
        kv_segment_size_buckets=[kv_segment_size] if kv_segment_size else None,
    )
    runner.num_seqs_buckets = [1, 4]
    runner.kv_segment_size = kv_segment_size
    runner._has_state_layers = has_state
    runner._state_real_pages = {}
    runner._state_groups = frozenset()
    runner._state_pool_slot_pages = 1
    runner.state_caches = {}
    runner._kv_clearer = None
    runner._kv_cache_config = None
    runner.kv_caches = {}
    runner.block_sizes = [BLOCK]
    runner._swa_num_blocks = [None]
    runner._inflight_graphs = []
    runner._quarantined_slots = set()
    runner._dp_sync_shape = lambda tokens, rows: (tokens, rows)
    runner._rank_tensor_cache = torch.zeros(1, dtype=torch.int32)
    runner.request_state = NeuronRequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_MODEL_LEN,
        vocab_size=1000,
        device=runner.device,
        block_sizes=[BLOCK],
    )
    runner._next_token_by_slot = torch.zeros(MAX_REQS, 1, dtype=torch.int32)
    return runner


def _kv_cache_config(specs):
    vllm_config = VllmConfig()
    vllm_config.cache_config.kv_cache_layout = "LBNHC"  # Neuron's only layout
    groups = get_kv_cache_groups(vllm_config, specs)
    num_layers = len(groups[0].layer_names)
    return get_kv_cache_config_from_groups(
        vllm_config, groups, NUM_BLOCKS * ATTN_PAGE * num_layers
    )


def _initialized_runner(**kwargs):
    runner = _runner(**kwargs)
    config = _kv_cache_config(runner.get_kv_cache_spec())
    runner.initialize_kv_cache(config)
    return runner


def _state_and_attn_groups(runner):
    groups = runner._kv_cache_config.kv_cache_groups
    state = [
        g for g, grp in enumerate(groups) if isinstance(grp.kv_cache_spec, MambaSpec)
    ]
    attn = [g for g in range(len(groups)) if g not in state]
    return state, attn


def _block_ids(runner, state_block, attn_blocks):
    """One block-id list per KV group, as vLLM hands them to the runner."""
    groups = runner._kv_cache_config.kv_cache_groups
    return tuple(
        [state_block + g]
        if isinstance(grp.kv_cache_spec, MambaSpec)
        else list(attn_blocks)
        for g, grp in enumerate(groups)
    )


def _new_req(runner, req_id, prompt_len, state_block, attn_blocks):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        prefill_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SimpleNamespace(max_tokens=16),
        num_computed_tokens=0,
        block_ids=_block_ids(runner, state_block, attn_blocks),
    )


def _step(new=(), cached=(), scheduled=None, finished=(), padded=None, zero=None):
    """SchedulerOutput stand-in. cached: (req_id, num_computed_tokens) pairs."""
    scheduled = dict(scheduled or {})
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _ in cached],
            num_computed_tokens=[n for _, n in cached],
            new_block_ids=[None for _ in cached],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens=scheduled,
        num_scheduled_tokens_padded=dict(padded or scheduled),
        total_num_scheduled_tokens=sum(scheduled.values()),
        finished_req_ids=set(finished),
        preempted_req_ids=None,
        new_block_ids_to_zero=zero,
    )


# ----------------------------------------------------------------- KV cache specs


def test_spec_state_layers_get_one_page_placeholders():
    runner = _runner()
    specs = runner.get_kv_cache_spec()
    state = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    attn = {n: s for n, s in specs.items() if isinstance(s, FullAttentionSpec)}
    assert len(state) == 6 and len(attn) == 2
    for name, spec in state.items():
        assert spec.page_size_bytes == ATTN_PAGE
        assert spec.mamba_cache_mode == "none" and spec.num_speculative_blocks == 0
        assert spec.block_size == MAX_MODEL_LEN
        assert runner._state_real_pages[name][1] == REAL_PAGE


def test_spec_groups_keep_attention_block_size():
    config = _kv_cache_config(_runner().get_kv_cache_spec())
    kinds = sorted(
        (type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size, len(g.layer_names))
        for g in config.kv_cache_groups
    )
    # No enlargement of the attention block, so no manager -> kernel block split.
    assert (
        kinds
        == [("FullAttentionSpec", BLOCK, 2)] + [("MambaSpec", MAX_MODEL_LEN, 2)] * 3
    )


def test_spec_without_state_layers_is_unchanged():
    model = _HybridModel()
    model.kv_spec = KVSpec(layers=model.kv_spec.layers)
    runner = _runner(model, has_state=False)
    specs = runner.get_kv_cache_spec()
    assert all(isinstance(s, FullAttentionSpec) for s in specs.values())
    assert runner._state_real_pages == {}


def test_spec_state_layers_need_the_capability():
    with pytest.raises(RuntimeError, match="HasRecurrentState"):
        _runner(has_state=False).get_kv_cache_spec()


def test_state_pool_bytes():
    assert _runner().state_pool_bytes() == 6 * (1 + MAX_REQS) * REAL_PAGE
    model = _HybridModel()
    model.kv_spec = KVSpec(layers=model.kv_spec.layers)
    assert _runner(model, has_state=False).state_pool_bytes() == 0


# ------------------------------------------------------------ initialize_kv_cache


def test_initialize_binds_attention_kv_and_dedicated_state_pools():
    runner = _initialized_runner()
    model = runner.model
    state_groups, attn_groups = _state_and_attn_groups(runner)
    assert runner._state_groups == frozenset(state_groups) and len(state_groups) == 3
    # Attention layers only in the KV binding; each gets its own (2, ...) buffer views.
    assert sorted(model.kv_caches) == ["layers.3.self_attn", "layers.7.self_attn"]
    k, _v = model.kv_caches["layers.3.self_attn"]
    assert k.shape == (NUM_BLOCKS, 1, BLOCK, 128) and k.dtype == torch.bfloat16
    # One pool per state layer: two zeroed float32 halves of 1 + max_num_reqs pages.
    assert sorted(model.state_caches) == sorted(
        n for n, _ in runner._state_real_pages.items()
    )
    assert len(model.state_caches) == 6
    for half_a, half_b in model.state_caches.values():
        assert half_a.shape == half_b.shape == (1 + MAX_REQS, REAL_PAGE // 8)
        assert half_a.dtype == torch.float32 and not half_a.any()
        # Not views of the shared KV buffer.
        assert half_a.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
    assert runner.state_caches is model.state_caches
    # Block rows: the state groups' width is one block (a block spans max_model_len).
    for g in state_groups:
        assert runner.request_state.block_table(g).shape == (MAX_REQS, 1)
    for g in attn_groups:
        assert runner.block_sizes[g] == BLOCK
    # CPU keeps the model's own clearing (as V1 does).
    assert runner._kv_clearer is None and model.runner_clears is None


def test_initialize_rebuilds_real_pages_when_specs_were_not_requested():
    runner = _runner()
    config = _kv_cache_config(runner.get_kv_cache_spec())
    runner._state_real_pages = {}
    runner.initialize_kv_cache(config)
    assert len(runner.model.state_caches) == 6


def test_initialize_rejects_state_layer_the_model_does_not_declare():
    runner = _runner(_HybridModel(extra_state_layers=["layers.99.linear_attn"]))
    specs = runner.get_kv_cache_spec()
    model = _HybridModel()  # the loaded model lacks layers.99
    runner.model = model
    with pytest.raises(ValueError, match="layer-name mismatch"):
        runner.initialize_kv_cache(_kv_cache_config(specs))
    assert runner._kv_cache_config is None and model.state_caches is None


@pytest.mark.parametrize(
    "change, match",
    [
        ({"mamba_cache_mode": "align"}, "Mamba cache mode"),
        ({"num_speculative_blocks": 1}, "speculative state blocks"),
    ],
)
def test_initialize_rejects_unsupported_state_specs(change, match):
    import dataclasses

    runner = _runner()
    specs = {
        n: dataclasses.replace(s, **change) if isinstance(s, MambaSpec) else s
        for n, s in runner.get_kv_cache_spec().items()
    }
    with pytest.raises(NotImplementedError, match=match):
        runner.initialize_kv_cache(_kv_cache_config(specs))
    assert runner._kv_cache_config is None


def test_kv_block_clearer_covers_every_attention_cache(monkeypatch):
    monkeypatch.setattr(
        hybrid_state, "compile_zero_kv_blocks", lambda: hybrid_state.zero_kv_blocks
    )
    runner = _initialized_runner()
    runner.device = SimpleNamespace(type="neuron")
    clearer = runner._build_kv_block_clearer(runner.kv_caches)
    assert clearer.ratio == 1
    assert len(clearer.caches) == 4  # K and V of two attention layers
    runner.device = torch.device("cpu")
    clearer.device = runner.device
    k = runner.kv_caches["layers.7.self_attn"][0]
    k.fill_(1)
    clearer.clear([5])
    assert not k[5].any() and bool((k[4] == 1).all())


# --------------------------------------------------------- state-pool block rows


def test_build_state_pool_rows():
    rows = build_state_pool_rows(
        np.array([2, 0], dtype=np.int32), 4, 1, 1, output_device=torch.device("cpu")
    )
    assert rows.dtype == torch.int32
    assert rows.tolist() == [[3], [1], [PAD_SLOT_ID], [PAD_SLOT_ID]]
    # Wider table and 1 + k pages per slot: extra columns are the null page.
    rows = build_state_pool_rows(
        np.array([1], dtype=np.int32), 2, 2, 4, output_device=torch.device("cpu")
    )
    assert rows.tolist() == [[3, 4, 0, 0], [PAD_SLOT_ID] * 4]
    with pytest.raises(AssertionError):
        build_state_pool_rows(
            np.array([0, 1], dtype=np.int32), 1, 1, 1, output_device=torch.device("cpu")
        )


def _admit(runner, *reqs, chunk=512):
    """Admit requests (prefill chunk of ``chunk`` tokens each, one per step)."""
    for req in reqs:
        runner.update_states(
            _step(
                new=[req], scheduled={req.req_id: min(chunk, len(req.prompt_token_ids))}
            )
        )


def _state_rows(model_input, runner):
    state_groups, _ = _state_and_attn_groups(runner)
    tables = [model_input.block_tables[g] for g in state_groups]
    for t in tables[1:]:
        assert torch.equal(t, tables[0])
    return tables[0]


def test_prefill_uses_slot_page_and_raw_state_slot_mapping():
    runner = _initialized_runner()
    req = _new_req(runner, "a", 300, state_block=10, attn_blocks=range(20, 30))
    step = _step(new=[req], scheduled={"a": 300}, padded={"a": 512})
    runner.update_states(step)
    model_input = runner._prepare_model_input(step)
    assert model_input.is_prefill and model_input.kv_segment_size == 1024
    assert _state_rows(model_input, runner).tolist() == [[1]]  # slot 0 -> page 1
    state_groups, attn_groups = _state_and_attn_groups(runner)
    for g in state_groups:
        # Raw vLLM state block ids: valid tokens > 0, padding at the null block.
        sm = model_input.slot_mappings[g]
        assert bool((sm[:300] > 0).all()) and not sm[300:].any()
        assert int(sm[0]) == (10 + g) * MAX_MODEL_LEN
    attn_rows = model_input.block_tables[attn_groups[0]]
    assert attn_rows[0, :10].tolist() == list(range(20, 30))
    assert model_input.cached_seq_len == 0


def test_cached_seq_len_on_every_hybrid_prefill_and_only_there():
    runner = _initialized_runner()
    req = _new_req(runner, "a", 1000, state_block=10, attn_blocks=range(20, 52))
    first = _step(new=[req], scheduled={"a": 512})
    runner.update_states(first)
    kwargs = runner._build_forward_adapter_kwargs(runner._prepare_model_input(first))
    md = kwargs["attn_metadata"]
    assert set(md) == {
        n for g in runner._kv_cache_config.kv_cache_groups for n in g.layer_names
    }
    for layer_md in md.values():
        assert layer_md["cached_seq_len"].tolist() == [[0]]
        assert layer_md["kv_segment_size"] == 1024
    second = _step(cached=[("a", 512)], scheduled={"a": 488}, padded={"a": 512})
    runner.update_states(second)
    model_input = runner._prepare_model_input(second)
    assert model_input.cached_seq_len == 512
    md = runner._build_forward_adapter_kwargs(model_input)["attn_metadata"]
    assert md["layers.0.linear_attn"]["cached_seq_len"].tolist() == [[512]]
    # Decode: no cached_seq_len (the decode graph signature is unchanged).
    decode = _step(cached=[("a", 1000)], scheduled={"a": 1})
    runner.update_states(decode)
    md = runner._build_forward_adapter_kwargs(runner._prepare_model_input(decode))[
        "attn_metadata"
    ]
    assert all("cached_seq_len" not in m for m in md.values())


def test_cached_seq_len_without_segmentation_only_for_hybrid():
    # Single-shot prefill (no KV segment): a hybrid model still gets cached_seq_len,
    # other models' prefill metadata is unchanged.
    groups = [
        SimpleNamespace(
            layer_names=["l0"], kv_cache_spec=SimpleNamespace(block_size=32)
        )
    ]
    args = {
        "kv_cache_groups": groups,
        "block_tables": [torch.zeros(1, 4, dtype=torch.int32)],
        "slot_mapping": torch.zeros(128, dtype=torch.int64),
        "max_query_len": 128,
        "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32),
    }
    dense = build_attention_metadata(**args)["l0"]
    assert "cached_seq_len" not in dense and "kv_segment_size" not in dense
    hybrid = build_attention_metadata(**args, always_cached_seq_len=True)["l0"]
    assert "cached_seq_len" in hybrid and "kv_segment_size" not in hybrid
    # Key order of the segmented metadata is unchanged (graph signatures).
    segmented = build_attention_metadata(**args, kv_segment_size=512)["l0"]
    assert list(segmented)[-2:] == ["cached_seq_len", "kv_segment_size"]

    runner = _initialized_runner(kv_segment_size=0)
    req = _new_req(runner, "a", 100, state_block=10, attn_blocks=range(20, 24))
    step = _step(new=[req], scheduled={"a": 100}, padded={"a": 512})
    runner.update_states(step)
    md = runner._build_forward_adapter_kwargs(runner._prepare_model_input(step))[
        "attn_metadata"
    ]
    assert md["layers.0.linear_attn"]["cached_seq_len"].tolist() == [[0]]
    assert "kv_segment_size" not in md["layers.0.linear_attn"]
    runner._has_state_layers = False
    md = runner._build_forward_adapter_kwargs(runner._prepare_model_input(step))[
        "attn_metadata"
    ]
    assert "cached_seq_len" not in md["layers.0.linear_attn"]


def test_held_back_request_keeps_its_page():
    # Decode steps between prefill chunks: a request absent from a step keeps its slot,
    # so its state page; the decode rows of the others use their own slots' pages.
    runner = _initialized_runner()
    reqs = [
        _new_req(runner, r, 100, state_block=10 + 4 * i, attn_blocks=[20 + 4 * i])
        for i, r in enumerate("abc")
    ]
    _admit(runner, *reqs)
    assert runner.request_state.req_id_to_index == {"a": 0, "b": 1, "c": 2}
    decode = _step(cached=[("c", 100), ("a", 100)], scheduled={"c": 1, "a": 1})
    runner.update_states(decode)
    model_input = runner._prepare_model_input(decode)
    assert model_input.padded_num_reqs == 4
    # Rows in scheduler order; padded rows -1 (the model skips their state).
    assert _state_rows(model_input, runner)[:, 0].tolist() == [3, 1, -1, -1]
    back = _step(
        cached=[("b", 100), ("c", 101), ("a", 101)], scheduled={"b": 1, "c": 1, "a": 1}
    )
    runner.update_states(back)
    rows = _state_rows(runner._prepare_model_input(back), runner)
    assert rows[:, 0].tolist() == [2, 3, 1, -1]


def test_recycled_slot_waits_for_in_flight_graph():
    # A slot freed while a submitted graph may still write its page is quarantined; the
    # next request gets another slot (and so another page).
    runner = _initialized_runner()
    a, b = (
        _new_req(runner, r, 100, state_block=10 + 4 * i, attn_blocks=[20 + 4 * i])
        for i, r in enumerate("ab")
    )
    _admit(runner, a, b)
    done = {"done": False}
    runner.note_graph_submitted(lambda: done["done"], {0, 1})
    runner.reap_quarantined()
    runner.update_states(_step(finished={"a"}))
    c = _new_req(runner, "c", 100, state_block=30, attn_blocks=[40])
    step = _step(new=[c], scheduled={"c": 100}, padded={"c": 512})
    runner.update_states(step)
    assert runner.request_state.req_id_to_index["c"] == 2
    assert _state_rows(runner._prepare_model_input(step), runner).tolist() == [[3]]
    done["done"] = True
    runner.reap_quarantined()
    runner.update_states(_step(finished={"c"}))
    d = _new_req(runner, "d", 100, state_block=50, attn_blocks=[60])
    step = _step(new=[d], scheduled={"d": 100}, padded={"d": 512})
    runner.update_states(step)
    assert runner.request_state.req_id_to_index["d"] == 0
    assert _state_rows(runner._prepare_model_input(step), runner).tolist() == [[1]]


def test_synthetic_warmup_inputs_use_pool_pages():
    runner = _initialized_runner()
    prefill = runner._build_prefill_synthetic_inputs(512, 1024)
    assert _state_rows(prefill, runner).tolist() == [[1]]
    assert prefill.cached_seq_len == 0
    md = runner._build_forward_adapter_kwargs(prefill)["attn_metadata"]
    assert md["layers.0.linear_attn"]["cached_seq_len"].tolist() == [[0]]
    decode = runner._build_decode_synthetic_inputs(4)
    assert _state_rows(decode, runner)[:, 0].tolist() == [1, 2, 3, 4]


# ---------------------------------------------------------- execute_model clearing


def test_execute_model_clears_new_blocks_before_a_zero_token_return():
    runner = _initialized_runner()
    calls = []
    runner._kv_clearer = SimpleNamespace(clear=lambda ids: calls.append(ids))
    runner._forward_adapter = object()
    runner._drain_async_output_for_new_request_join = lambda so: None
    runner.kv_connector = SimpleNamespace(no_forward=lambda so: "no-forward")
    assert runner.execute_model(_step(zero=[7, 9])) == "no-forward"
    assert calls == [[7, 9]]


def test_warm_up_hybrid_graphs():
    runner = _runner()
    runner.warm_up_hybrid_graphs()  # no clearer: no-op
    calls = []
    runner._kv_clearer = SimpleNamespace(warm_up=lambda: calls.append(1))
    runner.warm_up_hybrid_graphs()
    assert calls == [1]


# ------------------------------------------------------------- startup validation


def _validation_runner(
    monkeypatch, *, apc=False, spec=None, shared=False, buckets=(512,)
):
    runner = _runner()
    runner.vllm_config.cache_config.enable_prefix_caching = apc
    runner.vllm_config.speculative_config = spec
    runner.neuron_config.num_batched_tokens_buckets = list(buckets)
    monkeypatch.setattr(
        _v2().envs, "VLLM_NEURON_SHARED_STATE_POOL", shared, raising=False
    )
    return runner


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"apc": True}, NotImplementedError, "prefix caching"),
        ({"spec": object()}, NotImplementedError, "speculative decoding"),
        ({"shared": True}, NotImplementedError, "VLLM_NEURON_SHARED_STATE_POOL"),
        ({"buckets": (64, 512)}, ValueError, r"multiples of 128.*\[64\]"),
    ],
)
def test_validation_rejects(monkeypatch, kwargs, error, match):
    runner = _validation_runner(monkeypatch, **kwargs)
    with pytest.raises(error, match=match):
        runner._validate_recurrent_state_config()


def test_validation_accepts_supported_config(monkeypatch):
    _validation_runner(
        monkeypatch, buckets=(128, 512)
    )._validate_recurrent_state_config()
    runner = _validation_runner(monkeypatch, buckets=(64, 512))
    runner.kv_segment_size = 0  # single-shot prefill: always from zero state
    runner._validate_recurrent_state_config()


@pytest.mark.parametrize(
    "arch, expected",
    [("Qwen3_5ForConditionalGeneration", [1024]), ("LlamaForCausalLM", None)],
)
def test_fp8_kv_keeps_segmented_prefill_for_models_that_read_it(arch, expected):
    # Qwen3.5 declares SEGMENTED_PREFILL_READS_UNPACKED_FP8_KV; other models fall back to
    # full prefill with a non-packed FP8 KV cache.
    runner = object.__new__(_v2().NeuronModelRunnerV2)
    runner.vllm_config = SimpleNamespace(
        additional_config={"neuron_config": {}},
        cache_config=SimpleNamespace(cache_dtype="fp8", enable_prefix_caching=False),
        model_config=SimpleNamespace(architectures=[arch], max_logprobs=0),
    )
    runner.max_num_batched_tokens = 1024
    runner.max_model_len = 4096
    runner.max_num_reqs = 4
    assert runner._build_neuron_config().kv_segment_size_buckets == expected


def test_kv_cache_manager_blocks_feed_the_runner():
    # vLLM's KV cache manager on the V2 config: one state block per state group and
    # request, attention blocks of --block-size; the runner admits them as is.
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

    runner = _initialized_runner()
    manager = KVCacheManager(
        runner._kv_cache_config,
        max_model_len=MAX_MODEL_LEN,
        # resolve_kv_cache_block_sizes: the LCM of the group block sizes.
        scheduler_block_size=MAX_MODEL_LEN,
        hash_block_size=MAX_MODEL_LEN,
        enable_caching=False,
    )
    state_groups, attn_groups = _state_and_attn_groups(runner)
    for slot, req_id in enumerate("ab"):
        request = Request(
            req_id, list(range(1, 701)), SamplingParams(max_tokens=8), None
        )
        block_ids = manager.allocate_slots(request, 512).get_block_ids()
        for g in state_groups:
            assert len(block_ids[g]) == 1
        assert len(block_ids[attn_groups[0]]) == 512 // BLOCK
        # Newly allocated attention blocks are listed for clearing; state blocks are not.
        to_zero = set(manager.take_new_block_ids())
        assert to_zero == set(block_ids[attn_groups[0]])
        new_req = SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=request.prompt_token_ids,
            prefill_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            num_computed_tokens=0,
            block_ids=block_ids,
        )
        step = _step(new=[new_req], scheduled={req_id: 512}, zero=sorted(to_zero))
        runner.update_states(step)
        model_input = runner._prepare_model_input(step)
        assert _state_rows(model_input, runner).tolist() == [[1 + slot]]
        for g in state_groups:
            sm = model_input.slot_mappings[g]
            assert int(sm[0]) == block_ids[g][0] * MAX_MODEL_LEN
            assert int(sm[511]) == block_ids[g][0] * MAX_MODEL_LEN + 511
