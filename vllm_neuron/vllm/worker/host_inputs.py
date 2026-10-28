# SPDX-License-Identifier: Apache-2.0
"""Host inputs (``interfaces.HasHostInputs``), shared by the V1 and V2 runners.

A model with host inputs takes extra per-token tensors that the host builds every step
from the batch's tokens (Qwen3.8-Flash-Next's hashed n-gram rows); every compiled graph is
traced with them, zero-filled. See docs/design/host-inputs.md.

This module must not import either runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_neuron.model.interfaces import HasHostInputs

HostInputSpecs = dict[str, tuple[tuple[int, ...], torch.dtype]]


def _host_input_specs_of(model: Any) -> HostInputSpecs | None:
    """Return the ``HasHostInputs`` specs of a loaded model, or None without the capability.

    ``torch.compile`` wraps the module, so test the original. The spec-decode decorator
    patches ``forward`` on the class, so the module itself is the model instance.

    Args:
        model: The loaded model, wrapped or unwrapped.

    Returns:
        A copy of kwarg name -> (per-token trailing shape, dtype), or None.
    """
    model = getattr(model, "_orig_mod", model)
    if model is None or not isinstance(model, HasHostInputs):
        return None
    return {
        name: (tuple(int(d) for d in shape), dtype)
        for name, (shape, dtype) in model.host_input_specs().items()
    }


def host_input_req_rows(
    num_real_tokens: np.ndarray, num_pad_tokens: np.ndarray, num_rows: int
) -> np.ndarray:
    """Map each row of the padded token layout to its ``input_batch`` row.

    Request ``i`` (``input_batch`` row ``i``) owns ``num_real_tokens[i]`` rows followed by
    ``num_pad_tokens[i]`` padding rows (prefill bucket padding). Rows past the last
    request (decode batch padding, mixed-spec padding, DP padding) are padding too.

    Args:
        num_real_tokens: [num_reqs] scheduled (unpadded) token count per request.
        num_pad_tokens: [num_reqs] per-request padding rows appended after the real ones.
        num_rows: Total row count T of the padded ``input_ids``.

    Returns:
        [T] int64: the ``input_batch`` row of each token, -1 on padding rows.

    Examples:
        >>> host_input_req_rows(np.array([2, 1]), np.array([1, 0]), 6).tolist()
        [0, 0, -1, 1, -1, -1]
    """
    num_real = np.asarray(num_real_tokens, dtype=np.int64)
    num_pad = np.asarray(num_pad_tokens, dtype=np.int64)
    num_reqs = num_real.shape[0]
    req_ids = np.arange(num_reqs, dtype=np.int64)
    # Interleave [real_0, pad_0, real_1, pad_1, ...] segments.
    seg_vals = np.stack([req_ids, np.full(num_reqs, -1, dtype=np.int64)], 1).reshape(-1)
    seg_lens = np.stack([num_real, num_pad], 1).reshape(-1)
    rows = np.repeat(seg_vals, seg_lens)
    if rows.shape[0] > num_rows:
        raise ValueError(
            f"Host-input layout has {rows.shape[0]} rows, more than the {num_rows} "
            "padded input_ids rows."
        )
    return np.concatenate([rows, np.full(num_rows - rows.shape[0], -1, dtype=np.int64)])


def host_input_zero_kwargs(
    specs: HostInputSpecs | None, num_tokens: int, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Zero-filled host inputs ``[num_tokens, *shape]`` for warmup, capture and dummy runs.

    Returns an empty dict when ``specs`` is None, so a model without the capability gets
    no extra kwargs.
    """
    if not specs:
        return {}
    return {
        name: torch.zeros((num_tokens, *shape), dtype=dtype, device=device)
        for name, (shape, dtype) in specs.items()
    }


def build_step_host_inputs(
    model: Any,
    specs: HostInputSpecs,
    input_ids: np.ndarray,
    req_rows: np.ndarray,
    positions: np.ndarray,
    token_ids_cpu: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Call ``build_host_inputs`` and check its result against the declared specs.

    Padding rows (``req_rows == -1``) get position 0. A wrong name, shape or dtype raises
    here, on the host, instead of tracing a new graph or failing a compiled-graph guard.

    Args:
        model: The loaded model, wrapped or unwrapped.
        specs: The model's ``host_input_specs()``.
        input_ids: [T] padded token ids.
        req_rows: [T] ``input_batch`` row per token, -1 on padding rows.
        positions: [T] absolute position per token.
        token_ids_cpu: ``input_batch.token_ids_cpu``.

    Returns:
        name -> CPU tensor ``[T, *shape]``.
    """
    model = getattr(model, "_orig_mod", model)
    req_rows = np.asarray(req_rows, dtype=np.int64)
    positions = np.where(req_rows >= 0, np.asarray(positions, dtype=np.int64), 0)
    input_ids = np.asarray(input_ids).astype(np.int64, copy=False)
    num_tokens = input_ids.shape[0]
    if req_rows.shape != (num_tokens,) or positions.shape != (num_tokens,):
        raise ValueError(
            f"Host-input rows mismatch: input_ids {input_ids.shape}, req_rows "
            f"{req_rows.shape}, positions {positions.shape}."
        )
    out = model.build_host_inputs(input_ids, req_rows, positions, token_ids_cpu)
    if set(out) != set(specs):
        raise ValueError(
            f"build_host_inputs returned {sorted(out)}, host_input_specs declares "
            f"{sorted(specs)}."
        )
    for name, (shape, dtype) in specs.items():
        tensor = out[name]
        if tuple(tensor.shape) != (num_tokens, *shape) or tensor.dtype != dtype:
            raise ValueError(
                f"Host input {name!r}: got {tuple(tensor.shape)} {tensor.dtype}, "
                f"expected {(num_tokens, *shape)} {dtype}."
            )
    return out


def build_v2_host_input_layout(
    *,
    num_tokens: int,
    slots: np.ndarray,
    base_positions: np.ndarray,
    num_real_tokens: np.ndarray,
    row_starts: np.ndarray,
    token_ids_cpu: np.ndarray,
    drafts_cpu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild a V2 step's padded token layout on the host.

    Model Runner V2 builds ``input_ids`` and positions inside the graph from device
    state; the host inputs must match that layout row for row. Row ``r`` (stable slot
    ``slots[r]``) owns ``num_real_tokens[r]`` tokens at flat indices ``row_starts[r] +
    j`` with positions ``base_positions[r] + j``. Token ``j`` is
    ``token_ids_cpu[slot, base + j]``, except on a verify row (``drafts_cpu`` given),
    whose tokens after the first are the slot's drafts ``drafts_cpu[slot, j - 1]``.
    Every other index is padding.

    Args:
        num_tokens: Padded token count of the graph.
        slots: ``[rows]`` stable slot per row (the ``token_ids_cpu`` row).
        base_positions: ``[rows]`` position of each row's first token.
        num_real_tokens: ``[rows]`` real tokens per row.
        row_starts: ``[rows]`` flat index of each row's first token.
        token_ids_cpu: ``[max_num_reqs, max_model_len]`` host token history.
        drafts_cpu: ``[max_num_reqs, k]`` draft table of a verify step, else None.

    Returns:
        ``(input_ids, req_rows, positions)``, each ``[num_tokens]`` int64; ``req_rows``
        is the slot of each token, -1 on padding (input id and position 0).

    Example:
        >>> hist = np.arange(20).reshape(2, 10)
        >>> ids, rows, pos = build_v2_host_input_layout(
        ...     num_tokens=6, slots=np.array([1]), base_positions=np.array([3]),
        ...     num_real_tokens=np.array([2]), row_starts=np.array([0]),
        ...     token_ids_cpu=hist, drafts_cpu=np.array([[0], [99]]))
        >>> ids.tolist(), rows.tolist(), pos.tolist()
        ([13, 99, 0, 0, 0, 0], [1, 1, -1, -1, -1, -1], [3, 4, 0, 0, 0, 0])
    """
    input_ids = np.zeros(num_tokens, dtype=np.int64)
    req_rows = np.full(num_tokens, -1, dtype=np.int64)
    positions = np.zeros(num_tokens, dtype=np.int64)
    max_model_len = token_ids_cpu.shape[1]
    for slot, base, n, start in zip(
        np.asarray(slots, dtype=np.int64).tolist(),
        np.asarray(base_positions, dtype=np.int64).tolist(),
        np.asarray(num_real_tokens, dtype=np.int64).tolist(),
        np.asarray(row_starts, dtype=np.int64).tolist(),
    ):
        if n <= 0:
            continue
        if start + n > num_tokens or base + n > max_model_len:
            raise ValueError(
                f"Host-input row of slot {slot} (start {start}, {n} tokens at position "
                f"{base}) does not fit {num_tokens} tokens / max_model_len {max_model_len}"
            )
        flat = slice(start, start + n)
        req_rows[flat] = slot
        positions[flat] = np.arange(base, base + n, dtype=np.int64)
        if drafts_cpu is None:
            input_ids[flat] = token_ids_cpu[slot, base : base + n]
        else:
            input_ids[start] = token_ids_cpu[slot, base]
            input_ids[start + 1 : start + n] = drafts_cpu[slot, : n - 1]
    return input_ids, req_rows, positions
