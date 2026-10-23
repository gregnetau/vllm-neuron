# SPDX-License-Identifier: Apache-2.0
"""Optional model hooks read by the V1 runner, the worker and the scheduler (CPU).

- ``prefill_dense_context``: the runtime pick of the trimmed-table prefill variant;
- ``kv_side_bytes_per_token_per_attn_layer``: the side-cache share of the KV budget;
- ``mtp_hidden_size``: the target hidden width a draft head reads;
- ``VLLM_NEURON_MAX_KV_CONCURRENT``: the scheduler's KV concurrency cap override.

Runners and workers are built without ``__init__`` (``object.__new__``) and given only the
attributes these paths read.
"""

from types import SimpleNamespace

import pytest
import torch

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")
kv_iface = pytest.importorskip("vllm.v1.kv_cache_interface")
NeuronModelRunner = runner_mod.NeuronModelRunner
FullAttentionSpec = kv_iface.FullAttentionSpec
MLAAttentionSpec = kv_iface.MLAAttentionSpec
MambaSpec = kv_iface.MambaSpec


def make_runner(model, *, segment_buckets=None, cp_world_size=1):
    runner = object.__new__(NeuronModelRunner)
    runner.model = model
    runner.cp_world_size = cp_world_size
    runner.neuron_config = SimpleNamespace(kv_segment_size_buckets=segment_buckets)
    return runner


# --- dense prefill variant -------------------------------------------------------------------


def test_dense_prefill_selected_when_the_context_fits():
    runner = make_runner(SimpleNamespace(prefill_dense_context=2048))
    assert runner._select_prefill_dense(2048, 0) == (True, 0)
    assert runner._select_prefill_dense(1024, 1024) == (True, 1024)
    assert runner._select_prefill_dense(2049, 0) == (False, 0)
    assert runner._select_prefill_dense(4096, 4096) == (False, 4096)


def test_dense_prefill_moves_a_larger_segment_down_to_a_fitting_bucket():
    runner = make_runner(
        SimpleNamespace(prefill_dense_context=2048), segment_buckets=[512, 1024, 4096]
    )
    assert runner._select_prefill_dense(1536, 4096) == (True, 1024)
    # No segment bucket fits the dense context: the step keeps the regular variant.
    runner.neuron_config.kv_segment_size_buckets = [4096]
    assert runner._select_prefill_dense(1536, 4096) == (False, 4096)


def test_dense_prefill_off_without_opt_in_or_with_context_parallelism():
    assert make_runner(SimpleNamespace())._select_prefill_dense(16, 0) == (False, 0)
    runner = make_runner(SimpleNamespace(prefill_dense_context=2048), cp_world_size=2)
    assert runner._select_prefill_dense(16, 0) == (False, 0)
    assert runner.prefill_ctx_variants(512, 512) == [False]


def test_dense_prefill_reads_through_the_compiled_wrapper():
    wrapped = SimpleNamespace(_orig_mod=SimpleNamespace(prefill_dense_context=2048))
    runner = make_runner(wrapped)
    assert runner._prefill_dense_context() == 2048
    assert runner.prefill_ctx_variants(1024, 1024) == [False, True]


# --- KV side-cache budget --------------------------------------------------------------------


def spec_runner(model, specs):
    runner = make_runner(model)
    runner.get_kv_cache_spec = lambda: specs
    return runner


def attention_spec(head_size=128, num_kv_heads=2):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def test_side_cache_ratio_over_attention_kv_bytes():
    # K and V: 2 * 2 heads * 128 * 2 bytes = 1024 bytes per token of one layer.
    model = SimpleNamespace(kv_side_bytes_per_token_per_attn_layer=256)
    runner = spec_runner(model, {"layers.0.attn": attention_spec()})
    assert runner.kv_side_cache_ratio() == pytest.approx(0.25)


def test_side_cache_ratio_skips_non_attention_specs():
    mla = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    mamba = MambaSpec(block_size=16, shapes=((4,),), dtypes=(torch.float32,))
    model = SimpleNamespace(
        _orig_mod=SimpleNamespace(kv_side_bytes_per_token_per_attn_layer=512)
    )
    runner = spec_runner(model, {"mla": mla, "gdn": mamba})
    assert runner.kv_side_cache_ratio() == 0.0
    runner = spec_runner(model, {"mla": mla, "gdn": mamba, "attn": attention_spec()})
    assert runner.kv_side_cache_ratio() == pytest.approx(0.5)


def test_side_cache_ratio_zero_without_the_hook():
    runner = spec_runner(SimpleNamespace(), {"attn": attention_spec()})
    assert runner.kv_side_cache_ratio() == 0.0


@pytest.mark.parametrize("ratio, expected", [(0.0, 900), (0.5, 600)])
def test_worker_budget_leaves_room_for_side_caches(monkeypatch, ratio, expected):
    worker_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_worker")
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    worker = object.__new__(worker_mod.NeuronWorker)
    worker._determine_available_memory_cpu = lambda: 1000
    worker.model_runner = SimpleNamespace(
        state_pool_bytes=lambda: 100, kv_side_cache_ratio=lambda: ratio
    )
    assert worker.determine_available_memory() == expected


# --- MTP hidden size -------------------------------------------------------------------------


def test_target_hidden_size_prefers_mtp_hidden_size():
    eagle = pytest.importorskip("vllm_neuron.vllm.spec_decode.eagle")
    config = SimpleNamespace(hidden_size=64)
    assert eagle.target_hidden_size_of(SimpleNamespace(config=config)) == 64
    model = SimpleNamespace(config=config, mtp_hidden_size=256)
    assert eagle.target_hidden_size_of(model) == 256


# --- scheduler KV concurrency override -------------------------------------------------------


@pytest.mark.parametrize(
    "env, worst_case, expected",
    [(None, 12, 12), ("0", 12, 12), ("-3", 12, 12), ("48", 12, 48), ("48", 0, 48)],
)
def test_kv_concurrency_override(monkeypatch, env, worst_case, expected):
    scheduler = pytest.importorskip("vllm_neuron.vllm.core.scheduler")
    if env is None:
        monkeypatch.delenv("VLLM_NEURON_MAX_KV_CONCURRENT", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_MAX_KV_CONCURRENT", env)
    assert scheduler._kv_concurrency_override(worst_case) == expected


def test_runner_dense_prefill_variant_helpers():
    """Runner side of ``prefill_dense_context``: trimmed widths, variants and the warmup trim
    (attention-type groups trimmed, recurrent-state groups untouched)."""
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner as R

    model = SimpleNamespace(prefill_dense_context=2048)
    attn = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    mamba = MambaSpec(block_size=65536, shapes=((1,),), dtypes=(torch.float32,))
    groups = [
        SimpleNamespace(kv_cache_spec=attn, layer_names=["a"]),
        SimpleNamespace(kv_cache_spec=mamba, layer_names=["m"]),
    ]
    r = SimpleNamespace(
        model=model,
        cp_world_size=1,
        _dcp_size=1,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
    )
    for name in (
        "_prefill_dense_context",
        "_prefill_dense_ctx_blocks",
        "prefill_ctx_variants",
    ):
        setattr(r, name, getattr(R, name).__get__(r))
    assert (
        r._prefill_dense_ctx_blocks(128) == 16 and r._prefill_dense_ctx_blocks(32) == 64
    )
    assert r.prefill_ctx_variants(1024, 1024) == [False, True]
    assert r.prefill_ctx_variants(4096, 1024) == [False]
    assert r.prefill_ctx_variants(1024, 4096) == [
        False
    ]  # padded chunk exceeds the span
    md = {
        "a": {
            "block_size": 128,
            "block_table_tensor": torch.arange(512).view(1, 512),
            "max_blocks_per_seq": 512,
        },
        "m": {
            "block_size": 65536,
            "block_table_tensor": torch.tensor([[3]]),
            "max_blocks_per_seq": 1,
        },
    }
    out = R._trim_prefill_metadata_dense(r, md)
    assert (
        out["a"]["block_table_tensor"].shape == (1, 16)
        and out["a"]["max_blocks_per_seq"] == 16
    )
    assert out["m"] is md["m"]
    r.model = SimpleNamespace()
    assert r.prefill_ctx_variants(0) == [False]
