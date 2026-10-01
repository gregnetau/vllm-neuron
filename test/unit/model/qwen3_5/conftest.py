# SPDX-License-Identifier: Apache-2.0
"""Load qwen3_5 modules by file path so tests run without vllm / Neuron installed.

``import vllm_neuron`` pulls in vLLM, so for the pure-torch modules we register a bare
``vllm_neuron`` package stub (only if the real one is unavailable) and import the
files under ``vllm_neuron.model.qwen3_5`` directly.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4] / "vllm_neuron"


def _stub_packages():
    try:
        import vllm_neuron  # noqa: F401

        return
    except ImportError:
        pass
    for name, path in (
        ("vllm_neuron", ROOT),
        ("vllm_neuron.model", ROOT / "model"),
        ("vllm_neuron.model.qwen3_5", ROOT / "model" / "qwen3_5"),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg


def _load(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def gdn_ops():
    _stub_packages()
    return _load(
        "vllm_neuron.model.qwen3_5.gdn_ops", ROOT / "model/qwen3_5/gdn_ops.py"
    )


@pytest.fixture(scope="session")
def qwen3_5_config():
    _stub_packages()
    _load("vllm_neuron.model.neuron_config", ROOT / "model/neuron_config.py")
    return _load("vllm_neuron.model.qwen3_5.config", ROOT / "model/qwen3_5/config.py")


@pytest.fixture(scope="session")
def weights(qwen3_5_config):
    return _load(
        "vllm_neuron.model.qwen3_5.weights", ROOT / "model/qwen3_5/weights.py"
    )


@pytest.fixture(scope="session")
def state_cache(qwen3_5_config):
    return _load("vllm_neuron.model.state_cache", ROOT / "model/state_cache.py")


@pytest.fixture(scope="session")
def state(state_cache):
    return _load("vllm_neuron.model.qwen3_5.state", ROOT / "model/qwen3_5/state.py")
