# SPDX-License-Identifier: Apache-2.0
"""Recurrent-state helpers shared by the V1 and V2 Neuron model runners.

Hybrid models (e.g. Qwen3.5 / Qwen3.8 dense: Gated DeltaNet + attention) declare
fixed-size per-request state layers (``KVSpec.state_layers``). The runners report them
to vLLM as ``MambaSpec`` KV groups, keep the state in a dedicated pool per layer when
prefix caching is off (the V2 runner also with it, copying block-boundary checkpoints
to and from vLLM's shared pool), and clear newly allocated attention blocks. This module
holds the pieces that do not depend on either runner's batch bookkeeping, so the V2 runner
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
    paired_half_views,
    state_half_elems,
)

# Recurrent-state pages are padded to a whole number of this many attention-block tokens:
# vLLM enlarges the attention block to (state page / attention page) * block_size tokens, and
# the decode attention kernels need that to be a multiple of 128.
STATE_PAGE_ALIGN_TOKENS = 256

# Manager blocks cleared per call of the compiled KV-clearing graphs (fixed shapes, one compile
# each): small chunks for the few blocks of a decode step, large ones for a long prompt.
KV_CLEAR_CHUNK = 8
KV_CLEAR_CHUNK_LARGE = 256
# Clearing calls submitted before the clearer waits for them: the runtime's execution queue
# holds a bounded number of in-flight graph executions ("Execution Queue Full").
KV_CLEAR_MAX_INFLIGHT = 32


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
    align_pages: int | None = None,
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
        align_pages: Attention pages the real state page is padded to a multiple of;
            ``None``: :func:`state_page_align_pages` (``STATE_PAGE_ALIGN_TOKENS``
            tokens, which the V1 runner's attention kernels need for an enlarged
            block). The V2 runner splits enlarged blocks into kernel blocks and
            passes 1.

    Returns:
        ``(specs, real_pages)``: the MambaSpec per layer, and for a dedicated pool the
        ``(layout, page bytes)`` of each layer's real page (empty otherwise).
    """
    attn_page = max((s.page_size_bytes for s in attn_specs.values()), default=1)
    if align_pages is None:
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
    state blocks, so the V1 runner uses the shared pool then (the V2 runner keeps the
    running state in a dedicated pool also then, :func:`running_state_pool_pages`)."""
    if not dedicated or spec.mamba_cache_mode != "none":
        return 0
    return running_state_pool_pages(spec, max_num_reqs)


def running_state_pool_pages(spec: MambaSpec, max_num_reqs: int) -> int:
    """Pages per recurrent-state layer of a dedicated running-state pool: the null page
    0 plus ``1 + num_speculative_blocks`` per batch slot. With prefix caching (Mamba
    ``align`` mode) the V2 runner keeps it too, next to the block-boundary checkpoints
    in vLLM's shared pool (:class:`StateCheckpointer`)."""
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
    state, which reads as NaN/Inf in bf16 (the decode kernel reads whole blocks). Two small
    compiled graphs zero ``KV_CLEAR_CHUNK`` or ``KV_CLEAR_CHUNK_LARGE`` manager blocks of
    every attention layer per call, in place: a long prompt with 32-token blocks allocates
    thousands of blocks in one step. The clearer waits for its submissions every
    ``KV_CLEAR_MAX_INFLIGHT`` calls, so a step never overflows the runtime's queue.

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
        """Zero the given manager blocks (chunks padded with the null block 0).

        Large chunks while more than a few small chunks' worth remains, then small ones.
        """
        ratio = self.ratio
        blocks = list(manager_blocks)
        pending = []
        start = 0
        while start < len(blocks):
            remaining = len(blocks) - start
            size = (
                KV_CLEAR_CHUNK_LARGE
                if remaining > 4 * KV_CLEAR_CHUNK
                else KV_CLEAR_CHUNK
            )
            chunk = blocks[start : start + size]
            chunk = chunk + [0] * (size - len(chunk))
            start += size
            kernel_blocks = (
                torch.tensor(chunk)[:, None] * ratio + torch.arange(ratio)
            ).reshape(-1)
            pending.append(self.zero_fn(self.caches, kernel_blocks.to(self.device)))
            if len(pending) >= KV_CLEAR_MAX_INFLIGHT:
                for out in pending:  # wait for the submitted clears
                    if torch.is_tensor(out):
                        out.cpu()
                pending.clear()

    def clear(self, block_ids: Iterable[int] | None) -> None:
        """Zero newly allocated blocks (``SchedulerOutput.new_block_ids_to_zero``); the null
        block and duplicates are skipped, and nothing runs without any."""
        blocks = set(block_ids or ())
        blocks.discard(0)
        if blocks:
            self.clear_blocks(sorted(blocks))

    def warm_up(self) -> None:
        """Compile both graphs (they clear the null block). Call after model warmup:
        executing on device earlier would precede the parallel-trace forks."""
        self.clear_blocks([0])
        self.clear_blocks([0] * (4 * KV_CLEAR_CHUNK + 1))


# Pages per call of the compiled state-checkpoint copy graphs (fixed shape: one compile
# each). A step saves at most one checkpoint per request, and only when it reaches a
# block boundary, which happens once per prefix-cache block of a request.
STATE_CKPT_ROWS = 1


def state_page_rows(half: torch.Tensor) -> torch.Tensor:
    """``[pages, rows, 128]`` view of a ``[pages, half]`` state half (unchanged if the half
    is not a multiple of 128): a page moved as one ``[N, half]`` row is copied with
    per-element descriptors. Make the views outside the graphs, since ``index_copy_``
    through an in-graph view is not lowered reliably."""
    if half.dim() == 2 and half.shape[1] % 128 == 0:
        return half.view(half.shape[0], half.shape[1] // 128, 128)
    return half


def shared_state_views(
    raw_by_layer: dict[str, torch.Tensor], page_bytes: dict[str, int]
) -> dict[str, list[torch.Tensor]]:
    """Row views of state pages in vLLM's shared KV buffers, per state layer.

    Block ``i`` of a raw buffer is state page ``i`` of each state layer on it (the
    paired-half layout, :func:`~vllm_neuron.model.state_cache.paired_half_views`).
    Layers on one raw buffer get the same view objects, so a graph binds each buffer
    once.

    Args:
        raw_by_layer: The raw int8 buffer of each state layer.
        page_bytes: The (shared) page size of each state layer.

    Returns:
        ``{layer: [half_a, half_b]}``, each ``[num_blocks, rows, 128]``.
    """
    views: dict[tuple[int, int], list[torch.Tensor]] = {}
    out: dict[str, list[torch.Tensor]] = {}
    for name, raw in raw_by_layer.items():
        key = (id(raw), page_bytes[name])
        if key not in views:
            views[key] = [
                state_page_rows(h) for h in paired_half_views(raw, page_bytes[name])
            ]
        out[name] = views[key]
    return out


def checkpoint_boundaries(
    num_computed: int, num_scheduled: int, block_size: int, *, slack: int = 0
) -> list[int]:
    """Block boundaries a step can leave a request's state on.

    A step computes positions ``num_computed .. num_computed + num_scheduled - 1``.
    Its running state is checkpointed for the block ending at boundary ``b`` (a
    multiple of ``block_size``) when the step's accepted tokens reach ``b``: its last
    token for a prefill chunk or a plain decode step, any accepted token of a
    speculative verify step. ``slack`` widens the window below ``num_computed`` for a
    host count that may exceed the device count (async scheduling with speculative
    decoding: by at most the in-flight step's rejected drafts).

    Args:
        num_computed: Host computed-token count at the step's start.
        num_scheduled: Tokens of the step, ``1 + drafts`` for a verify row.
        block_size: Prefix-cache block (the Mamba ``align`` block).
        slack: How far the device count may be below ``num_computed``.

    Returns:
        The candidate boundaries, ascending (usually none or one).

    Example:
        >>> checkpoint_boundaries(13, 3, 16)  # a verify step at positions 13 .. 15
        [16]
        >>> checkpoint_boundaries(32, 16, 16)  # a prefill chunk ending at 48
        [48]
    """
    lo = max(num_computed - slack + 1, 1)
    hi = num_computed + num_scheduled
    first = -(-lo // block_size) * block_size
    return list(range(first, hi + 1, block_size))


def _rows_mask(mask: torch.Tensor, dim: int) -> torch.Tensor:
    return mask.view(-1, *([1] * (dim - 1)))


def save_state_checkpoints(
    pool_pages: list[torch.Tensor],
    ckpt_pages: list[torch.Tensor],
    groups: list[int],
    row_slot: torch.Tensor,
    boundary: torch.Tensor,
    dst_blocks: torch.Tensor,
    row_valid: torch.Tensor,
    num_computed: torch.Tensor,
    num_accepted: torch.Tensor,
    pages_per_slot: int,
) -> torch.Tensor:
    """Copy each row's state at a block boundary to its checkpoint block.

    Row ``r`` of slot ``s`` ran a step that left ``num_computed[s]`` tokens computed
    and committed ``c = num_accepted[s]`` tokens (1 for a prefill chunk or a plain
    decode step), so ``n = num_computed[s] - c`` were computed before it, and the
    state after the step's token ``j`` is pool page ``1 + s * pages_per_slot + j``
    (an MTP verify step writes the state after token ``j`` to block ``j``; other
    steps write block 0). If the accepted tokens reach ``boundary[r]``
    (``n < boundary[r] <= n + c``), page ``j = boundary[r] - n - 1`` holds the state
    after exactly ``boundary[r]`` tokens, and it is copied to the shared-pool block
    ``dst_blocks[g, r]`` of every state group ``g``. The counts are the ones the step
    committed on the device, so this is exact under async scheduling, where the host
    knows only an upper bound of ``n`` and not ``c``.

    Rows that do not reach their boundary (and padding rows, ``row_valid`` False)
    copy the null block onto itself: every write is a read-modify-write, so the null
    blocks of both pools keep their contents. 1-D gathers only (a 2-D gather fails
    with NCC_ISMP902), int64 indices.

    Args:
        pool_pages: ``[pages, rows, 128]`` views of the dedicated pool halves.
        ckpt_pages: The matching ``[blocks, rows, 128]`` views of the shared pool.
        groups: State group (row of ``dst_blocks``) of each entry.
        row_slot: ``[N]`` int64 slots.
        boundary: ``[N]`` int64 candidate boundaries.
        dst_blocks: ``[G, N]`` int64 checkpoint blocks (0: none).
        row_valid: ``[N]`` bool.
        num_computed: ``[max_num_reqs]`` int32 device computed counts.
        num_accepted: ``[max_num_reqs]`` int32 device committed counts.
        pages_per_slot: Pool pages per slot (``1 + k``).

    Returns:
        The number of rows that reached their boundary (the graph's output).
    """
    n_after = num_computed.to(torch.int64)[row_slot]
    committed = num_accepted.to(torch.int64)[row_slot]
    n_before = n_after - committed
    reached = row_valid & (boundary > n_before) & (boundary <= n_after)
    page_in_slot = torch.clamp(boundary - n_before - 1, min=0, max=pages_per_slot - 1)
    zeros = torch.zeros_like(row_slot)
    for pool, ckpt, g in zip(pool_pages, ckpt_pages, groups):
        valid = reached & (dst_blocks[g] > 0)
        src = torch.where(valid, 1 + row_slot * pages_per_slot + page_in_slot, zeros)
        dst = torch.where(valid, dst_blocks[g], zeros)
        new = pool.index_select(0, src)
        old = ckpt.index_select(0, dst)
        ckpt.index_copy_(0, dst, torch.where(_rows_mask(valid, new.dim()), new, old))
    return reached.sum()


def restore_state_checkpoints(
    ckpt_pages: list[torch.Tensor],
    pool_pages: list[torch.Tensor],
    groups: list[int],
    src_blocks: torch.Tensor,
    dst_page: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Copy checkpoint blocks ``src_blocks[g, r]`` to pool pages ``dst_page[r]``.

    A request admitted with a cached prefix starts its prefill from the state at the
    prefix's last block boundary. Invalid rows copy the null page onto itself
    (read-modify-write, as in :func:`save_state_checkpoints`).

    Args:
        ckpt_pages: ``[blocks, rows, 128]`` views of the shared pool.
        pool_pages: The matching ``[pages, rows, 128]`` dedicated pool views.
        groups: State group (row of ``src_blocks``) of each entry.
        src_blocks: ``[G, N]`` int64 checkpoint blocks.
        dst_page: ``[N]`` int64 pool pages.
        valid: ``[N]`` bool.

    Returns:
        The number of valid rows (the graph's output).
    """
    zeros = torch.zeros_like(dst_page)
    dst = torch.where(valid, dst_page, zeros)
    for ckpt, pool, g in zip(ckpt_pages, pool_pages, groups):
        src = torch.where(valid, src_blocks[g], zeros)
        new = ckpt.index_select(0, src)
        old = pool.index_select(0, dst)
        pool.index_copy_(0, dst, torch.where(_rows_mask(valid, new.dim()), new, old))
    return valid.sum()


class StateCheckpointer:
    """Prefix-cache checkpoints of recurrent state for the V2 runner (Mamba ``align``).

    The running state stays in the dedicated per-slot pool. vLLM's align mode caches
    one state block for each block boundary that a request's computation reaches;
    those blocks are real state pages in the shared KV pool, so vLLM's block pool
    allocates, caches, evicts and reuses them like attention blocks. This class copies
    a slot's state to such a block after the step that reached the boundary
    (:meth:`save`), and a cached block to the slot of a request admitted with a cached
    prefix (:meth:`restore`), each with one small compiled graph.

    Args:
        pool_pages: ``{layer: [half_a, half_b]}`` row views of the dedicated pools.
        ckpt_pages: ``{layer: [half_a, half_b]}`` row views of the shared pool; layers
            on one raw buffer share the view objects.
        layer_groups: ``[(kv_group_index, [layer names])]`` of the state groups.
        pages_per_slot: Pool pages per slot (``1 + k``).
        device: Device of the pools.
        compiled: Compile the copies for the device; ``False`` runs them eagerly.
    """

    def __init__(
        self,
        pool_pages: dict[str, list[torch.Tensor]],
        ckpt_pages: dict[str, list[torch.Tensor]],
        layer_groups: list[tuple[int, list[str]]],
        pages_per_slot: int,
        device: torch.device,
        compiled: bool = True,
    ) -> None:
        self.group_ids = [g for g, _ in layer_groups]
        self.pool: list[torch.Tensor] = []
        self.ckpt: list[torch.Tensor] = []
        self.entry_groups: list[int] = []
        for k, (_, names) in enumerate(layer_groups):
            for name in names:
                for pool, ckpt in zip(pool_pages[name], ckpt_pages[name]):
                    self.pool.append(pool)
                    self.ckpt.append(ckpt)
                    self.entry_groups.append(k)
        self.pages_per_slot = pages_per_slot
        self.device = device
        self._save_fn: Callable = save_state_checkpoints
        self._restore_fn: Callable = restore_state_checkpoints
        if compiled:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
            self._save_fn = torch.compile(
                save_state_checkpoints, backend=backend, dynamic=False
            )
            self._restore_fn = torch.compile(
                restore_state_checkpoints, backend=backend, dynamic=False
            )
        # Candidate saves / restores submitted (a save copies only if its row reached
        # the boundary on the device).
        self.num_saves = 0
        self.num_restores = 0

    def _tensor(self, values, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype).to(self.device)

    def save(
        self,
        entries: list[tuple[int, int, list[int]]],
        num_computed: torch.Tensor,
        num_accepted: torch.Tensor,
    ) -> None:
        """Checkpoint ``(slot, boundary, [block per state group])`` entries after a step.

        Args:
            entries: Candidates of the step's rows (:func:`checkpoint_boundaries`).
            num_computed: The device computed counts the step committed.
            num_accepted: The device committed counts of the step (ones without
                speculative decoding).
        """
        rows = STATE_CKPT_ROWS
        for start in range(0, len(entries), rows):
            chunk = entries[start : start + rows]
            pad = rows - len(chunk)
            blocks = [
                [e[2][k] for e in chunk] + [0] * pad for k in range(len(self.group_ids))
            ]
            self._save_fn(
                self.pool,
                self.ckpt,
                self.entry_groups,
                self._tensor([e[0] for e in chunk] + [0] * pad),
                self._tensor([e[1] for e in chunk] + [0] * pad),
                self._tensor(blocks),
                self._tensor([True] * len(chunk) + [False] * pad, torch.bool),
                num_computed,
                num_accepted,
                self.pages_per_slot,
            )
            self.num_saves += len(chunk)

    def restore(self, entries: list[tuple[int, list[int]]]) -> None:
        """Copy ``(slot, [checkpoint block per state group])`` to block 0 of each slot."""
        rows = STATE_CKPT_ROWS
        for start in range(0, len(entries), rows):
            chunk = entries[start : start + rows]
            pad = rows - len(chunk)
            blocks = [
                [e[1][k] for e in chunk] + [0] * pad for k in range(len(self.group_ids))
            ]
            self._restore_fn(
                self.ckpt,
                self.pool,
                self.entry_groups,
                self._tensor(blocks),
                self._tensor(
                    [1 + e[0] * self.pages_per_slot for e in chunk] + [0] * pad
                ),
                self._tensor([True] * len(chunk) + [False] * pad, torch.bool),
            )
            self.num_restores += len(chunk)

    def warm_up(self, num_computed: torch.Tensor, num_accepted: torch.Tensor) -> None:
        """Compile both graphs with padding-only inputs (no-op copies of the null pages).
        Call after model warmup, as :meth:`KVBlockClearer.warm_up`."""
        rows = STATE_CKPT_ROWS
        # Distinct tensors for every argument, as at run time: passing one tensor twice
        # makes Dynamo guard on the aliasing ("row_slot is boundary"), and the first real
        # save then recompiled the graph while serving (about 5 s of TTFT).
        slots, boundaries, pages = (self._tensor([0] * rows) for _ in range(3))
        blocks = self._tensor([[0] * rows for _ in self.group_ids])
        invalid = self._tensor([False] * rows, torch.bool)
        self._save_fn(
            self.pool,
            self.ckpt,
            self.entry_groups,
            slots,
            boundaries,
            blocks,
            invalid,
            num_computed,
            num_accepted,
            self.pages_per_slot,
        )
        self._restore_fn(
            self.ckpt,
            self.pool,
            self.entry_groups,
            self._tensor([[0] * rows for _ in self.group_ids]),
            pages,
            self._tensor([False] * rows, torch.bool),
        )
