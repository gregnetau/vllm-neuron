# SPDX-License-Identifier: Apache-2.0
"""
Gated DeltaNet per-request state shapes for Qwen3.5 / Qwen3.8
=============================================================

A GDN layer keeps a conv state stored ``[K-1, conv_dim/tp]`` and a recurrent state stored
``[Hv/tp, Dk, Dv]``, both float32 (``mamba_ssm_dtype``; one dtype keeps the paired-half page
views contiguous, see ``state_cache.paired_half_views``). The storage order is chosen for the
NKI decode kernel: a page is [ssm | conv | pad] in rows of 128 floats; one row is one Dk row of
one head's [Dk, Dv] state tile, or a channel block of one conv tap. Only sizes matter to vLLM.
Page packing lives in :mod:`vllm_neuron.model.state_cache`.
"""

from vllm_neuron.model.state_cache import StatePageLayout, page_layout


def gdn_state_shapes(cfg, tp: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-rank (conv_state, ssm_state) shapes for one linear-attention layer."""
    cfg.validate_tp(tp)
    conv = (cfg.linear_conv_kernel_dim - 1, cfg.conv_dim // tp)
    ssm = (
        cfg.linear_num_value_heads // tp,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
    )
    return conv, ssm


def gdn_page_layout(cfg, tp: int) -> StatePageLayout:
    conv, ssm = gdn_state_shapes(cfg, tp)
    return page_layout((conv, ssm), (cfg.mamba_ssm_dtype, cfg.mamba_ssm_dtype))
