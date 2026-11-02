# SPDX-License-Identifier: Apache-2.0
"""Native-route compile timeouts: the barrier after graph extraction (which compiles every
NEFF in-rank on the native route) and the NEFF cache-lock timeout seeded at import."""

import os
import subprocess
import sys
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
    monkeypatch.setenv("VLLM_NEURON_BARRIER_TIMEOUT", "")  # empty: not set
    assert f(True) == timedelta(seconds=7200)
    monkeypatch.setenv("VLLM_NEURON_BARRIER_TIMEOUT", "900")  # an explicit value wins
    assert f(True) is None


@pytest.mark.parametrize(
    "backend,explicit,expected",
    [
        ("neuron_native", None, "21600"),
        ("neuron_native", "600", "600"),
        ("vllm_neuron", None, "None"),
    ],
)
def test_neff_cache_lock_timeout_seeded_on_native_route(backend, explicit, expected):
    # A fresh interpreter: importing vllm_neuron initializes the backend (torch.neuron,
    # torch.accelerator hooks), which a reload inside the test process would redo.
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT"
    }
    env["VLLM_NEURON_BACKEND"] = backend
    if explicit is not None:
        env["TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT"] = explicit
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, vllm_neuron; "
                "print(os.environ.get('TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT'))"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == expected
