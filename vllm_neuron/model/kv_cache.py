# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None


@dataclass
class StateLayerSpec:
    """
    Defines a per-request recurrent-state cache for a non-attention layer
    (e.g. a Gated DeltaNet / Mamba-style layer).

    Each request owns one fixed-size "page" per layer, independent of context
    length. A page packs ``shapes[i]`` tensors of ``dtypes[i]`` back to back, in
    order (per-rank shapes, i.e. already divided by the TP degree).
    """

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
    # Recurrent-state layers of hybrid models. Additive: models without such
    # layers leave this empty and are unaffected.
    state_layers: list[StateLayerSpec] = field(default_factory=list)
