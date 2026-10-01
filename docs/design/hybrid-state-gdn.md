# Hybrid Recurrent State for Gated DeltaNet Models

<!-- meta: description: Design of per-request recurrent state (Gated DeltaNet) alongside paged
KV cache in the vLLM Neuron plugin, as used by Qwen3.5 / Qwen3.8 dense models. -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-10-01 -->

## Overview

Qwen3.5 / Qwen3.8 dense models interleave Gated DeltaNet (GDN) linear-attention layers with
full-attention layers (48 + 16 on 27B). A GDN layer carries a fixed-size state per request
instead of a KV cache:

| State | Shape per rank (27B, TP=4) | Dtype |
|---|---|---|
| Conv state | `[K-1, conv_dim/tp]` = `[3, 2560]` | float32 |
| Recurrent state | `[Hv/tp, Dk, Dv]` = `[12, 128, 128]` | float32 |

This document covers how that state is allocated through vLLM's hybrid KV cache manager,
how it is laid out in device memory, and how the model reads and writes it on Neuron.

## KV cache groups

The model reports GDN layers through `KVSpec.state_layers` (`StateLayerSpec`: shapes and
dtypes per layer, see `model/kv_cache.py`). The runner emits one `MambaSpec` per GDN layer
with `mamba_cache_mode="none"`, so vLLM allocates one state page per request per GDN layer
group. A request's page is the first block id of the group's block table; no extra
scheduler input is needed.

vLLM unifies page sizes across groups by enlarging the attention block size. The runner pads
the state page so the enlarged attention block is a whole number of 256-token blocks, and
the attention metadata splits each manager block back into kernel-sized blocks
(`block_size` 32). With one KV head per rank, a manager block is a pure reshape of its kernel
blocks. The runner logs the result, for example
`KV cache block_size resolved: cache_config=32, per_group=[1024, 2048, 2048, 2048]`.

## Page layout

Each raw KV buffer is shared by one layer of every group. The attention view of a buffer is
`(2, num_blocks, ...)` for K and V, so a state page is stored as two contiguous float32
halves aligned with that layout (`state_cache.paired_half_views`). The model binds each
half as `[pages, rows, 128]` outside the compiled graph (`Qwen3_5GatedDeltaNet.bind_state`).

A page is `[ssm | conv | pad]` in rows of 128 floats:

| Rows | Contents |
|---|---|
| `h*128 + dk` | recurrent state of head `h`, row `dk`, columns `dv` |
| `Hv*128 + j*NCB + f` | conv state tap `j`, channels `f*128 .. f*128+127` |

With 1024-row halves no head straddles the two halves, which lets one DMA move all of a
program's heads.

## Reads and writes

- **Prefill** (one request per step, padded to the bucket): the GDN layer reads its page,
  runs the causal conv and the chunked delta rule (`gdn_ops.chunk_gated_delta_rule`,
  chunk 64), and writes the final state back. Padding tokens are identity updates
  (`g = beta = 0`); a token is valid iff its slot is not the null block.
- **Decode**: `gdn_kernels.gdn_decode` runs the conv step, q/k L2 norm, gating, the gated
  delta rule, the gated RMSNorm and the in-place page update in one NKI kernel. Heads are
  split across the two programs of a logical core; page DMAs use a runtime page offset with
  hardware descriptor generation. Other cases use the PyTorch ops in `gdn_ops`.
- State writes are in place (`index_copy_` on the bound views, or the kernel's aliased
  outputs). The runner's block tables use `-1` for unused entries; the model maps them to
  the null block before indexing.

## Recycled blocks

Block ids are global across groups, so a KV block can previously have held float32 state.
Read as bf16, those bytes include NaN and Inf, and the decode attention kernel reads whole
blocks (a masked `0 * NaN` is still NaN). The model clears a KV block on its first write:

- at prefill, every block at or after the cached prefix;
- at decode, the block of a token at block offset 0, unless every block of a request is
  allocated at prefill (one attention block covers `max_model_len`), which the runner
  signals through `set_kv_blocks_cover_context`.

Unused block-table entries are passed to the decode kernel as the null block rather than
`-1`: the kernel skips out-of-bounds reads but keeps the previous SBUF contents.

## Compiler constraints

The model avoids patterns that lower incorrectly or slowly on Neuron:

| Pattern | Replacement |
|---|---|
| `torch.split` with a size list | `torch.tensor_split` or slicing |
| Column slices of a fused projection at unaligned offsets | separate projections |
| Grouped `conv1d` | sum of K shifted multiplies |
| `index_copy_` through an in-graph `.view()` | views created at bind time |
| A page moved as one `[B, half]` row | `[B, rows, 128]` gathers and scatters |
| `solve_triangular` | matmul-only unit-lower inverse |
| Writing the KV cache ahead of `attention_decode` | clear only when required |

## Static FP8

With `neuron_config.quantization="fp8"`, weights are quantized per tensor at load
(`scale = amax(W) / 240` over the full tensor, so every rank uses the same scale) and
activations use static scales from an offline calibration file
(`neuron_config.fp8_activation_scales_path`).

| Module | Kernel |
|---|---|
| MLP | `NF.mlp` STATIC, post-attention RMSNorm fused (as in the Llama static-FP8 path) |
| GDN `in_proj_qkv`, `in_proj_z` | `NF.qkv_proj` STATIC as a generic projection (one scale for all parts) |
| GDN `out_proj` | `NF.o_proj` STATIC |

`NF.mlp` falls back to PyTorch, without quantization, where its kernel does not apply (CTE
with intermediate size per rank above 4096 and hidden size below 7168); the FP8 MLP then
runs in BF16 on dequantized weights.
