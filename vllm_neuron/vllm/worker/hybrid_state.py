# SPDX-License-Identifier: Apache-2.0
"""Recurrent-state helpers shared by the V1 and V2 Neuron model runners.

Hybrid models (e.g. Qwen3.5 / Qwen3.8 dense: Gated DeltaNet + attention) declare
fixed-size per-request state layers (``KVSpec.state_layers``). The runners report them
to vLLM as ``MambaSpec`` KV groups, keep the state in a dedicated pool per layer when
prefix caching is off, and clear newly allocated attention blocks. This module holds
the pieces that do not depend on either runner's batch bookkeeping, so the V2 runner
can use them without importing the V1 runner. See docs/design/hybrid-state-gdn.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_neuron.model.state_cache import (
    StatePageLayout,
    padded_state_page_bytes,
    page_layout,
    state_half_elems,
)

# Recurrent-state pages are padded to a whole number of this many attention-block tokens:
# vLLM enlarges the attention block to (state page / attention page) * block_size tokens, and
# the decode attention kernels need that to be a multiple of 128.
STATE_PAGE_ALIGN_TOKENS = 256

# Manager blocks cleared per call of the compiled KV-clearing graph (fixed shape: one compile).
KV_CLEAR_CHUNK = 8


def state_page_align_pages(kernel_block: int) -> int:
    """Attention pages per STATE_PAGE_ALIGN_TOKENS tokens (1 if the block does not divide it)."""
    if STATE_PAGE_ALIGN_TOKENS % kernel_block:
        return 1
    return max(1, STATE_PAGE_ALIGN_TOKENS // kernel_block)


def build_state_cache_specs(
    state_layers,
    attn_specs: dict[str, KVCacheSpec],
    *,
    kernel_block: int,
    max_model_len: int,
    prefix_caching: bool,
    dedicated: bool,
    num_spec_blocks: int,
    placeholder_pages: int,
) -> tuple[dict[str, MambaSpec], dict[str, tuple[StatePageLayout, int]]]:
    """MambaSpec per recurrent-state layer, and the real pages of a dedicated pool.

    The state page is padded to an integral multiple of the attention page so vLLM's
    page-size unification can scale the attention block size to match. Without prefix
    caching a request keeps one state page (``1 + num_spec_blocks`` with MTP) for its whole
    context; with it (Mamba "align" mode) the state block spans as many tokens as the
    enlarged attention block, so state checkpoints line up with the attention prefix blocks.

    With a dedicated pool the state never lives in vLLM's shared pool, but vLLM still
    allocates its blocks there. Their spec then reports a placeholder page of
    ``placeholder_pages`` attention pages instead of the real (padded) page, which is
    returned separately for the pool: each unused block costs little, and vLLM enlarges the
    attention block only to the placeholder (the V1 runner uses one 256-token block, the V2
    runner one attention page, so its attention block stays at ``--block-size``).

    Args:
        state_layers: The model's ``StateLayerSpec`` list.
        attn_specs: Attention specs of the same model (for the attention page size).
        kernel_block: Attention kernel block size (``cache_config.block_size``).
        max_model_len: Block size of a state group without prefix caching.
        prefix_caching: Use Mamba "align" mode (shared pool only).
        dedicated: Keep the state in a dedicated pool (placeholder pages in vLLM's pool).
        num_spec_blocks: Extra state blocks per request (MTP draft tokens).
        placeholder_pages: Attention pages per placeholder state page (dedicated only).

    Returns:
        ``(specs, real_pages)``: the MambaSpec per layer, and for a dedicated pool the
        ``(layout, page bytes)`` of each layer's real page (empty otherwise).
    """
    attn_page = max((s.page_size_bytes for s in attn_specs.values()), default=1)
    align_pages = state_page_align_pages(kernel_block)
    specs: dict[str, MambaSpec] = {}
    real_pages: dict[str, tuple[StatePageLayout, int]] = {}
    for layer in state_layers:
        layout = page_layout(layer.shapes, layer.dtypes)
        page = padded_state_page_bytes(layout.page_bytes, attn_page, align_pages)
        # Power-of-two attention pages per state page, so the enlarged attention block
        # divides max_model_len (e.g. Qwen3.8-27B with FP8 KV would otherwise give 1792-token
        # blocks, block tables of 1064 entries, and odd-stride FP8 DMAs neuronx-cc rejects).
        page = attn_page * (1 << (page // attn_page - 1).bit_length())
        shapes, dtypes = tuple(layer.shapes), tuple(layer.dtypes)
        if dedicated:
            real_pages[layer.name] = (layout, page)
            page, shapes, dtypes = (
                attn_page * placeholder_pages,
                ((1,),),
                (torch.float32,),
            )
        if prefix_caching:
            block_size = page // attn_page * kernel_block
            mode = "align"
        else:
            # One state page per request: a single block spans the whole context.
            block_size, mode = max_model_len, "none"
        specs[layer.name] = MambaSpec(
            block_size=block_size,
            shapes=shapes,
            dtypes=dtypes,
            page_size_padded=page,
            mamba_type=MambaAttentionBackendEnum[layer.mamba_type],
            mamba_cache_mode=mode,
            num_speculative_blocks=num_spec_blocks,
        )
    return specs, real_pages


def state_pool_pages(spec: MambaSpec, max_num_reqs: int, dedicated: bool) -> int:
    """Pages per recurrent-state layer in a dedicated pool, or 0 to keep the state in
    vLLM's shared KV pool.

    In the shared pool each raw tensor is bound twice (attention K/V views and state
    views), and neuronx-cc's graph I/O check (NCC_EVRF009) counts it twice, which caps
    the pool at about half of the free HBM. Without prefix caching a request holds its
    1 + num_speculative_blocks state blocks for its whole life, so a dedicated pool of
    one set per batch slot (plus the null page 0) suffices. Prefix caching keeps cached
    state blocks, so it uses the shared pool."""
    if not dedicated or spec.mamba_cache_mode != "none":
        return 0
    return 1 + max_num_reqs * (1 + spec.num_speculative_blocks)


def state_page(
    layer_name: str,
    spec: MambaSpec,
    real_pages: dict[str, tuple[StatePageLayout, int]],
) -> tuple[StatePageLayout, int]:
    """(layout, page bytes) of a recurrent-state layer's real page: the one recorded for a
    dedicated pool, whose spec carries a placeholder page, else the spec's own."""
    real = real_pages.get(layer_name)
    if real is not None:
        return real
    return page_layout(spec.shapes, spec.dtypes), spec.page_size_bytes


def allocate_state_pool(
    layer_pages: dict[str, tuple[StatePageLayout, int]],
    n_pages: int,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    """Zeroed dedicated pools: two contiguous ``[n_pages, page_bytes // 8]`` float32 halves
    per layer (the paired-half layout of state_cache.paired_half_views), so the graphs see
    each pool byte once.

    Args:
        layer_pages: ``(layout, page bytes)`` of each layer's real page.
        n_pages: Pages per layer (state_pool_pages).
        device: Allocation device.

    Returns:
        ``{layer_name: [half_a, half_b]}`` for the model's ``bind_state_cache``.

    Raises:
        ValueError: If a layer's state does not fit its page.
    """
    caches: dict[str, list[torch.Tensor]] = {}
    for name, (layout, page_bytes) in layer_pages.items():
        state_half_elems(layout, page_bytes)  # validates fit
        caches[name] = [
            torch.zeros(n_pages, page_bytes // 8, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
    return caches


def pool_pages_for_slot(slot: int, pages_per_slot: int, width: int) -> np.ndarray:
    """``[width]`` int32 block-table row of a batch slot in a dedicated pool.

    Slot ``s`` owns pages ``1 + s * pages_per_slot + j`` (page 0 is the null page), one per
    state block (``1 + num_speculative_blocks``). Columns past them (a wider table) map to
    the null page.
    """
    row = np.zeros(width, dtype=np.int32)
    n = min(pages_per_slot, width)
    row[:n] = 1 + slot * pages_per_slot + np.arange(n, dtype=np.int32)
    return row


def zero_kv_blocks(
    caches: list[torch.Tensor], kernel_blocks: torch.Tensor
) -> torch.Tensor:
    """Zero kernel blocks [N] of every KV cache tensor [num_kernel_blocks, ...] in place."""
    for cache in caches:
        zeros = torch.zeros(
            (kernel_blocks.shape[0],) + tuple(cache.shape[1:]),
            dtype=cache.dtype,
            device=cache.device,
        )
        cache.index_put_((kernel_blocks,), zeros)
    return kernel_blocks.sum()


def compile_zero_kv_blocks() -> Callable:
    """zero_kv_blocks compiled for the Neuron device (fixed shape, one graph)."""
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(
        zero_kv_blocks, backend=get_compile_backend_name(), dynamic=False
    )


class KVBlockClearer:
    """Clear attention KV blocks when the scheduler allocates them.

    Block ids are global across KV groups, so a block can previously have held float32
    state, which reads as NaN/Inf in bf16 (the decode kernel reads whole blocks). One small
    compiled graph zeroes ``KV_CLEAR_CHUNK`` manager blocks of every attention layer per
    call, in place.

    Args:
        caches: Every attention K and V cache tensor ``[num_kernel_blocks, ...]``.
        ratio: Kernel blocks per manager block (1 unless vLLM enlarged the blocks).
        device: Device of the caches.
        zero_fn: The zeroing function; defaults to the compiled zero_kv_blocks.
    """

    def __init__(
        self,
        caches: list[torch.Tensor],
        ratio: int,
        device: torch.device,
        zero_fn: Callable | None = None,
    ) -> None:
        self.caches = caches
        self.ratio = ratio
        self.device = device
        self.zero_fn = compile_zero_kv_blocks() if zero_fn is None else zero_fn

    def clear_blocks(self, manager_blocks: list[int]) -> None:
        """Zero the given manager blocks (chunks padded with the null block 0)."""
        ratio = self.ratio
        for start in range(0, len(manager_blocks), KV_CLEAR_CHUNK):
            chunk = list(manager_blocks[start : start + KV_CLEAR_CHUNK])
            chunk = chunk + [0] * (KV_CLEAR_CHUNK - len(chunk))
            kernel_blocks = (
                torch.tensor(chunk)[:, None] * ratio + torch.arange(ratio)
            ).reshape(-1)
            self.zero_fn(self.caches, kernel_blocks.to(self.device))

    def clear(self, block_ids: Iterable[int] | None) -> None:
        """Zero newly allocated blocks (``SchedulerOutput.new_block_ids_to_zero``); the null
        block and duplicates are skipped, and nothing runs without any."""
        blocks = set(block_ids or ())
        blocks.discard(0)
        if blocks:
            self.clear_blocks(sorted(blocks))

    def warm_up(self) -> None:
        """Compile the graph (clears the null block). Call after model warmup: executing on
        device earlier would precede the parallel-trace forks."""
        self.clear_blocks([0])
