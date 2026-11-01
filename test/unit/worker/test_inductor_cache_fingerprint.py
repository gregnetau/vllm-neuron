# SPDX-License-Identifier: Apache-2.0
"""The native-route compile-cache fingerprint must change with the KV cache size.

torch keeps every graph that a function compiled in one precompile package, and a
warm start restores (and loads into HBM) all of them. Two configurations that
share a fingerprint therefore share a package, and each restores the graphs of
the other. vLLM's compute_hash() leaves out the KV sizing knobs, because the KV
cache size is a runtime value on GPU, while on Neuron it is a static graph shape.
"""

from types import SimpleNamespace

import pytest

from vllm_neuron.vllm import inductor_cache


class _Config(SimpleNamespace):
    """Stand-in VllmConfig: compute_hash() ignores the KV sizing, as vLLM does."""

    def compute_hash(self) -> str:
        return "same-graph-hash"


def _vllm_config(**cache_fields):
    cache = dict(
        gpu_memory_utilization=0.82,
        kv_cache_memory_bytes=None,
        num_gpu_blocks_override=None,
        enable_prefix_caching=False,
    )
    cache.update(cache_fields)
    return _Config(cache_config=SimpleNamespace(**cache), optimization_level=None)


def _neuron_config(**buckets):
    fields = dict(num_batched_tokens_buckets=[1024, 4096], num_seqs_buckets=[1, 8])
    fields.update(buckets)
    return SimpleNamespace(**fields)


@pytest.fixture(autouse=True)
def _no_runtime(monkeypatch):
    # The platform, version and NEFF-directory probes read the Neuron runtime.
    monkeypatch.setattr(inductor_cache, "_platform_target", lambda: "trn2")
    monkeypatch.setattr(inductor_cache, "_versions", lambda: {"torch": "x"})
    monkeypatch.setattr(inductor_cache, "_neff_cache_dir", lambda: "/neff")
    monkeypatch.setattr(inductor_cache, "_compiler_options", lambda cfg: {"o": 1})
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    monkeypatch.delenv("VLLM_NEURON_RESERVED_HEADROOM_GIB", raising=False)


def _fingerprint(vllm_config, neuron_config):
    inputs = inductor_cache.build_fingerprint_inputs(vllm_config, neuron_config)
    return inductor_cache.compute_fingerprint(inputs)


def test_same_configuration_keeps_its_fingerprint():
    # A warm restart of one configuration must find its own package.
    assert _fingerprint(_vllm_config(), _neuron_config()) == _fingerprint(
        _vllm_config(), _neuron_config()
    )


@pytest.mark.parametrize(
    "cache_fields",
    [
        {"gpu_memory_utilization": 0.80},
        {"kv_cache_memory_bytes": 2 << 30},
        {"num_gpu_blocks_override": 512},
        {"enable_prefix_caching": True},
    ],
)
def test_kv_sizing_changes_the_fingerprint(cache_fields):
    base = _fingerprint(_vllm_config(), _neuron_config())
    assert _fingerprint(_vllm_config(**cache_fields), _neuron_config()) != base


@pytest.mark.parametrize(
    "name,value",
    [
        ("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.4"),
        ("VLLM_NEURON_RESERVED_HEADROOM_GIB", "2"),
    ],
)
def test_neuron_kv_sizing_variables_change_the_fingerprint(monkeypatch, name, value):
    base = _fingerprint(_vllm_config(), _neuron_config())
    monkeypatch.setenv(name, value)
    assert _fingerprint(_vllm_config(), _neuron_config()) != base


def test_bucket_change_changes_the_fingerprint():
    base = _fingerprint(_vllm_config(), _neuron_config())
    other = _neuron_config(decode_context_length_buckets=[4096])
    assert _fingerprint(_vllm_config(), other) != base


def test_fingerprint_inputs_name_the_kv_sizing():
    inputs = inductor_cache.build_fingerprint_inputs(_vllm_config(), _neuron_config())
    assert inputs["kv_cache_sizing"]["gpu_memory_utilization"] == 0.82
    assert set(inputs) == {
        "config_hash",
        "buckets",
        "kv_cache_sizing",
        "platform_target",
        "compiler_options",
        "versions",
        "neff_cache_dir",
    }
