# SPDX-License-Identifier: Apache-2.0
"""Op lowering capabilities of the compile route.

Model code that picks between equivalent formulations of an op, because one route lowers
it and the other does not (or lowers it badly), asks here instead of testing the backend
directly, so each choice names the observed reason. The native route (torch.compile) is the
plugin's default, so CPU tests take the native branch unless they patch these functions.
"""

from vllm_neuron import envs


def bitcast_lowers() -> bool:
    """Whether ``Tensor.view(dtype)`` (a bitcast) lowers: on the native route, not on the
    XLA route."""
    return envs.is_native_backend()


def index_copy_lowers_to_scatter() -> bool:
    """Whether ``index_copy_`` lowers to an element-wise scatter over the whole indexed slice
    (native route; ~74 MiB of instruction code per batch-8 qwen4_exp verify graph for
    [B, 512, 128] state pages), where ``index_put_`` stays a row copy. The XLA route lowers
    ``index_copy_`` to a row copy."""
    return envs.is_native_backend()
