# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next state kept on the host or copied between state pages (CPU):
- the host n-gram inputs live outside the module, which torch.compile pickles on the
  native route;
- MTP verify's state-page copies, on both routes' code paths, with padded rows."""

import copy
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model import route_caps
from vllm_neuron.model.qwen4_exp import model as M


def test_host_ple_inputs_stay_off_the_module():
    m = torch.nn.Module.__new__(M.Qwen4ExpForCausalLM)
    torch.nn.Module.__init__(m)
    table = threading.Lock()  # unpicklable, like the memory-mapped n-gram table
    m._host_ple = table
    assert m._host_ple is table
    assert "_host_ple" not in vars(m)
    copy.deepcopy(m)  # the module's own state stays copyable
    m._host_ple = None
    assert m._host_ple is None and m not in M._HOST_PLE


@pytest.mark.parametrize("native", [False, True], ids=["index_copy", "index_put"])
def test_copy_state_pages_copies_rows_and_zeroes_padding(monkeypatch, native):
    monkeypatch.setattr(route_caps, "index_copy_lowers_to_scatter", lambda: native)
    g = torch.Generator().manual_seed(0)
    a0, b0 = torch.randn(8, 4, 3, generator=g), torch.randn(8, 4, 3, generator=g)
    layer = SimpleNamespace(state_a=a0.clone(), state_b=b0.clone())
    src = torch.tensor([2, 5, 7, 1], dtype=torch.int32)
    dst = torch.tensor([3, 6, -1, -1], dtype=torch.int32)  # -1: padded rows
    M.Qwen4ExpGatedDeltaNet._copy_state_pages(layer, src, dst)
    for new, old in ((layer.state_a, a0), (layer.state_b, b0)):
        exp = old.clone()
        exp[3], exp[6], exp[0] = old[2], old[5], 0.0  # padded rows zero the null page
        torch.testing.assert_close(new, exp, rtol=0, atol=0)
