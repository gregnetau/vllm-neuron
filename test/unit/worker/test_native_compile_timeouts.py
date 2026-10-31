# SPDX-License-Identifier: Apache-2.0
"""Native-route compile timeouts: the barrier after graph extraction (which compiles every
NEFF in-rank on the native route) and the NEFF cache-lock timeout seeded at import."""

import importlib
import os
from datetime import timedelta

import pytest

from vllm_neuron.vllm.worker import neuron_worker


def test_barrier_timeout_after_native_extraction(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_BARRIER_TIMEOUT", raising=False)
    monkeypatch.delenv("VLLM_NEURON_NATIVE_COMPILE_BARRIER_TIMEOUT", raising=False)
    f = neuron_worker._graph_extraction_barrier_timeout
    assert f(False) is None  # XLA route: the default barrier timeout
    assert f(True) == timedelta(seconds=21600)
    monkeypatch.setenv("VLLM_NEURON_NATIVE_COMPILE_BARRIER_TIMEOUT", "7200")
    assert f(True) == timedelta(seconds=7200)
    monkeypatch.setenv("VLLM_NEURON_BARRIER_TIMEOUT", "900")  # an explicit value wins
    assert f(True) is None


@pytest.mark.parametrize(
    "backend,explicit,expected",
    [
        ("neuron_native", None, "21600"),
        ("neuron_native", "600", "600"),
        ("vllm_neuron", None, None),
    ],
)
def test_neff_cache_lock_timeout_seeded_on_native_route(
    monkeypatch, backend, explicit, expected
):
    import vllm_neuron

    monkeypatch.setenv("VLLM_NEURON_BACKEND", backend)
    if explicit is None:
        monkeypatch.delenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", explicit)
    importlib.reload(vllm_neuron)
    assert os.environ.get("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT") == expected
