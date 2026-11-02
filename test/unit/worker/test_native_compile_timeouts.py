# SPDX-License-Identifier: Apache-2.0
"""Native-route compile timeouts: the barrier after graph extraction (which compiles every
NEFF in-rank on the native route) and the NEFF cache-lock timeout seeded at import."""

import os
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from vllm_neuron.parallel import neuron_parallel_state
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


class _StubCpuGroup:
    """A gloo-like cpu_group: ``set_timeout`` updates the backend options' timeout."""

    def __init__(self, timeout, readable=True, fail=False):
        self.options = SimpleNamespace(_timeout=timeout)
        self.readable, self.fail = readable, fail
        self.calls = []

    def _get_backend(self, device):
        if not self.readable:
            raise RuntimeError("no backend")
        return self

    def set_timeout(self, timeout):
        self.calls.append(("set_timeout", timeout))
        self.options._timeout = timeout

    def barrier(self):
        group = self

        class _Work:
            def wait(self, timeout):
                group.calls.append(("wait", timeout))
                if group.fail:
                    raise RuntimeError("barrier timed out")

        return _Work()


def _patch_tp_group(monkeypatch, cpu_group, world_size=2):
    import vllm.distributed.parallel_state as vps

    group = SimpleNamespace(world_size=world_size, cpu_group=cpu_group)
    monkeypatch.setattr(vps, "get_tp_group", lambda: group)


@pytest.mark.parametrize("readable", [True, False], ids=["getter", "no_getter"])
@pytest.mark.parametrize("fail", [False, True], ids=["ok", "raises"])
def test_tp_barrier_restores_timeout_after_an_explicit_one(monkeypatch, readable, fail):
    monkeypatch.setenv("VLLM_NEURON_BARRIER_TIMEOUT", "900")
    before = timedelta(seconds=1800)
    six_h = timedelta(hours=6)
    g = _StubCpuGroup(before, readable=readable, fail=fail)
    _patch_tp_group(monkeypatch, g)
    if fail:
        with pytest.raises(RuntimeError):
            neuron_parallel_state.tp_barrier(six_h)
    else:
        neuron_parallel_state.tp_barrier(six_h)
    # Unreadable: back to the default barrier timeout.
    restored = before if readable else timedelta(seconds=900)
    assert g.calls == [
        ("set_timeout", six_h),
        ("wait", six_h),
        ("set_timeout", restored),
    ]
    assert g.options._timeout == restored


def test_tp_barrier_default_timeout_stays_set(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_BARRIER_TIMEOUT", "900")
    g = _StubCpuGroup(timedelta(seconds=1800))
    _patch_tp_group(monkeypatch, g)
    neuron_parallel_state.tp_barrier()
    default = timedelta(seconds=900)
    assert g.calls == [("set_timeout", default), ("wait", default)]


def test_tp_barrier_single_rank_is_a_no_op(monkeypatch):
    g = _StubCpuGroup(timedelta(seconds=1800))
    _patch_tp_group(monkeypatch, g, world_size=1)
    neuron_parallel_state.tp_barrier(timedelta(hours=6))
    assert g.calls == []


def test_cpu_group_timeout_reads_a_real_gloo_group():
    # The getter reads a private attribute: check it against torch's gloo ProcessGroup,
    # the type of vLLM's cpu_group, and that set_timeout moves it.
    import torch
    import torch.distributed as dist

    store = dist.HashStore()
    gloo = dist.ProcessGroupGloo(store, 0, 1, timedelta(seconds=55))
    pg = dist.ProcessGroup(store, 0, 1)
    pg._set_default_backend(dist.ProcessGroup.BackendType.GLOO)
    pg._register_backend(torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO, gloo)
    assert neuron_parallel_state._cpu_group_timeout(pg) == timedelta(seconds=55)
    pg.set_timeout(timedelta(seconds=3))
    assert neuron_parallel_state._cpu_group_timeout(pg) == timedelta(seconds=3)
    assert neuron_parallel_state._cpu_group_timeout(object()) is None
