# SPDX-License-Identifier: Apache-2.0
"""Device test: the multi-request GDN decode kernel path of Qwen3.8-Flash-Next
(``Qwen4ExpGatedDeltaNet._decode`` -> ``gdn_kernels.gdn_decode(own_conv_rows=True)``) against
the PyTorch decode path (``_decode_torch``) and a float32 CPU run of it.

Per-rank TP=8 dims (Hk=2, Hv=6, Dk=Dv=128, conv kernel 4), sigmoid output gate, one logical
NeuronCore (two programs). Weights are random at the checkpoint's scales; the states are built
by 12 PyTorch decode steps from zero, then one step runs through both paths from the same pages.
Padded batch rows (block table -1) must leave every page untouched and give zero output.

Needs one Neuron device and NKI; skipped otherwise. Select the NeuronCore with the usual
runtime environment (e.g. ``NEURON_RT_VISIBLE_CORES``). The compiled programs contain collective
operations even on one logical core, so the runtime needs ``NEURON_RT_ROOT_COMM_ID`` to load them
(defaulted below to a local port):

    NEURON_RT_VISIBLE_CORES=0 pytest test/integration/model/qwen4_exp/test_gdn_decode_kernel.py -v -s
"""

import os
import types

import pytest
import torch

# Without it nrt_load rejects the programs ("cannot load NEFF with collective operations").
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:62182")


def _neuron_available() -> bool:
    try:
        import nki  # noqa: F401

        import vllm_neuron  # noqa: F401  (registers the neuron device)
    except ImportError:
        return False
    neuron = getattr(torch, "neuron", None)
    return neuron is not None and neuron.is_available()


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _neuron_available(), reason="needs a Neuron device and NKI"),
]

H, HK, HV, D, K = 2560, 2, 6, 128, 4
# State pages; HALF floats per state half (rows of 128: heads align to the halves).
PAGES, HALF = 48, 65536


def _gdn(dtype, dev):
    from vllm_neuron.model.qwen4_exp.model import Qwen4ExpGatedDeltaNet

    g = torch.Generator().manual_seed(0)
    m = Qwen4ExpGatedDeltaNet.__new__(Qwen4ExpGatedDeltaNet)
    torch.nn.Module.__init__(m)
    m.layer_idx, m.dtype, m.hidden_size, m.world_size, m.fp8 = 0, dtype, H, 1, False
    m.config = types.SimpleNamespace(rms_norm_eps=1e-6, output_gate_type="sigmoid")
    m.Hk, m.Hv, m.Dk, m.Dv, m.K = HK, HV, D, D, K
    m.key_dim, m.value_dim = HK * D, HV * D
    m.conv_dim = 2 * m.key_dim + m.value_dim
    m.conv_numel = m.conv_dim * (K - 1)
    m.ssm_numel = HV * D * D
    m.state_numel = m.conv_numel + m.ssm_numel
    m.state_a = m.state_b = None
    shapes = {
        "in_proj_qkv_weight": ((H, m.conv_dim), H**-0.5),
        "in_proj_z_weight": ((H, m.value_dim), H**-0.5),
        "in_proj_b_weight": ((H, HV), H**-0.5),
        "in_proj_a_weight": ((H, HV), H**-0.5),
        "conv_weight": ((m.conv_dim, K), 0.3),
        "out_proj_weight": ((m.value_dim, H), m.value_dim**-0.5),
    }
    for name, (shape, scale) in shapes.items():
        w = torch.randn(shape, generator=g) * scale
        m.register_parameter(
            name, torch.nn.Parameter(w.to(dtype).to(dev), requires_grad=False)
        )
    vec = {
        "A_log": torch.rand(HV, generator=g) * 2 - 1,
        "dt_bias": torch.randn(HV, generator=g) * 0.5,
        "norm_weight": 1 + 0.1 * torch.randn(D, generator=g),
    }
    for name, v in vec.items():
        dt = dtype if name == "norm_weight" else torch.float32
        m.register_parameter(
            name, torch.nn.Parameter(v.to(dt).to(dev), requires_grad=False)
        )
    return m


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _md(block_table):
    return dict(
        max_query_len=1, decode_token_threshold=1, block_table_tensor=block_table
    )


def _states(m):
    """Both state halves of every page, [PAGES, 2 * HALF] on CPU."""
    return torch.cat(
        [m.state_a.cpu().reshape(PAGES, HALF), m.state_b.cpu().reshape(PAGES, HALF)], 1
    )


@pytest.mark.parametrize("B,pad", [(1, 0), (8, 0), (8, 2), (16, 1)])
def test_gdn_decode_kernel_matches_torch(B, pad):
    from vllm_neuron.envs import get_compile_backend_name

    dev = torch.device("neuron:0")
    m = _gdn(torch.bfloat16, dev)
    ref = _gdn(torch.float32, "cpu")  # same random weights, float32 on CPU
    bt = torch.arange(B, dtype=torch.int32).view(B, 1) * 2 + 3
    if pad:
        bt[B - pad :] = -1
    real = bt[:, 0] >= 0
    pages = bt[real, 0].long()
    backend = get_compile_backend_name()
    f_torch = torch.compile(
        lambda x, t: m._decode_torch(x, _md(t)), backend=backend, fullgraph=True
    )
    f_kernel = torch.compile(
        lambda x, t: m._decode(x, _md(t)), backend=backend, fullgraph=True
    )

    g = torch.Generator().manual_seed(B)
    m.bind_state(torch.zeros(PAGES, HALF).to(dev), torch.zeros(PAGES, HALF).to(dev))
    assert m._use_decode_kernel(
        torch.zeros(B, H, device=dev), max_requests=m.DECODE_KERNEL_MAX_REQUESTS
    )
    for _ in range(12):
        y = f_torch(torch.randn(B, H, generator=g).bfloat16().to(dev), bt.to(dev))
        y.cpu()  # wait: unsynchronised steps can overrun the runtime's execution queue
    x = torch.randn(B, H, generator=g).bfloat16()
    sa0, sb0 = (
        m.state_a.cpu().reshape(PAGES, HALF),
        m.state_b.cpu().reshape(PAGES, HALF),
    )

    y_t = f_torch(x.to(dev), bt.to(dev)).cpu().float()
    st_t = _states(m)
    m.bind_state(sa0.to(dev), sb0.to(dev))
    y_k = f_kernel(x.to(dev), bt.to(dev)).cpu().float()
    st_k = _states(m)
    ref.bind_state(sa0.clone(), sb0.clone())
    y_r = ref._decode_torch(x.float(), _md(bt)).float()
    st_r = _states(ref)

    e_k, e_t = _rel(y_k[real], y_r[real]), _rel(y_t[real], y_r[real])
    s_k, s_t = _rel(st_k[pages], st_r[pages]), _rel(st_t[pages], st_r[pages])
    print(
        f"B={B} pad={pad}: out vs fp32 kernel {e_k:.2e} torch {e_t:.2e}; "
        f"state vs fp32 kernel {s_k:.2e} torch {s_t:.2e}; kernel vs torch {_rel(y_k[real], y_t[real]):.2e}"
    )
    assert torch.isfinite(y_k).all()
    # The kernel keeps float32 where the torch path uses bf16.
    assert e_k < 1e-2 and e_k <= 1.2 * e_t
    assert s_k < 1e-2 and s_k <= 1.2 * s_t
    # Pages not in the batch (incl. the null block 0 for padded rows) are untouched.
    others = torch.ones(PAGES, dtype=torch.bool)
    others[pages] = False
    assert torch.equal(st_k[others], torch.cat([sa0, sb0], 1)[others])
    assert (y_k[~real] == 0).all()
