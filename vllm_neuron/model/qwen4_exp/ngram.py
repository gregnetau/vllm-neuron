# SPDX-License-Identifier: Apache-2.0
"""
Host-side n-gram (PLE) embedding for Qwen3.8-Flash-Next
========================================================

<-- MODEL-SPECIFIC: the per-layer n-gram embedding (HF ``Qwen4ExpTextNGramEmbedding``) is
computed entirely on the host. The device never sees the 320M x 160 BF16 table (102 GB) or the
int64 hash (neuronx-cc downcasts integers to int32). Each step the model runner calls
:class:`HostPLEInputs` with the padded batch and passes the returned ``[T, H * D]`` BF16 tensor
(``H = (ngram_size - 1) * heads_per_ngram = 16`` heads of ``D = 160``, so 2560 wide) into the
graph. The device-side PLE layer (key/value projections, gating, dilated conv) consumes it.

Pipeline (pure torch / numpy, no vLLM or Neuron imports, so it is unit-testable on CPU):

  1. :func:`context_tokens`  ``[T]`` current tokens -> ``shifted [ngram_size, T]`` int64:
     ``shifted[s]`` is the token ``s`` positions back, with HF's EOS rule.
  2. :class:`NGramHasher`    ``shifted`` -> ``ids [T, H]`` int64 global table rows (exact HF hash:
     wrapping int64 multiply, xor, ``torch.remainder`` per head, plus the head offset).
  3. :class:`HostNGramTable` ``ids`` -> ``[T, H * D]`` BF16 (heads concatenated in HF order:
     8 bigram heads then 8 trigram heads). The table is a zero-copy mmap of the checkpoint shards.

HF semantics reproduced (``Qwen4ExpTextModel.forward`` / ``_shift_right_ignore_eos``):
  * every sequence is preceded by ``ngram_size - 1`` EOS tokens of context (negative positions
    read as EOS); HF padding tokens are replaced by EOS through ``conv_mask``;
  * ``shifted[s]`` at position ``p`` is ``token[p - s]`` unless ``p - s < 0`` or any of
    ``token[p - s .. p - 1]`` is EOS, in which case it is EOS (n-grams never span an EOS; the
    current token itself being EOS does not matter);
  * the 2-gram heads hash ``shifted[0..1]``, the 3-gram heads ``shifted[0..2]``.

Checkpoint layout (``Qwen4ExpForConditionalGeneration``)::

  model.language_model.layers.1.ple.ple_embedding.layer_multipliers        I64 [3]
  model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes  I64 [16]
  model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets      I64 [16]
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight
                                                                           BF16 [2500012, 160]

HF concatenates the shards along rows (``split_ngram_parts``); we keep them as separate mmapped
views and route each id to its shard.
"""

import ctypes
import json
import math
import mmap
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import torch

__all__ = [
    "HostNGramTable",
    "HostPLEInputs",
    "NGramHasher",
    "build_head_vocab",
    "build_layer_multipliers",
    "context_tokens",
]

TEXT_PREFIX = "model.language_model."


def _eos_id(cfg: Any) -> int:
    eos = cfg.eos_token_id
    return int(eos[0] if isinstance(eos, (list, tuple)) else eos)


# The per-step host math runs in numpy: single-threaded and predictable. torch's intra-op pool
# (one thread per core by default) showed 50-80 ms stalls on these ~1 ms ops on a loaded host.
def _as_np_long(x) -> np.ndarray:
    """tensor / numpy / list -> int64 numpy array (zero-copy when already int64)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.int64, copy=False)


def _as_long(x) -> torch.Tensor:
    """numpy / list / tensor -> int64 CPU tensor (zero-copy when already int64)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return torch.as_tensor(x).to(device="cpu", dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. Context tokens (HF _shift_right_ignore_eos on the vLLM batch layout)
# ---------------------------------------------------------------------------
def context_tokens(
    input_ids,
    req_rows,
    positions,
    token_ids_cpu,
    eos_token_id: int,
    ngram_size: int = 3,
) -> torch.Tensor:
    """Gather each token's n-gram context from the batch and the per-request history.

    Args:
        input_ids: ``[T]`` int, the padded batch (the current tokens).
        req_rows: ``[T]`` int, row of ``token_ids_cpu`` for each token; ``-1`` marks padding.
        positions: ``[T]`` int, absolute position of each token in its request.
        token_ids_cpu: ``[R, L]`` int (vLLM ``input_batch.token_ids_cpu``), per-request history.
        eos_token_id: EOS id (also the context of negative positions and padding rows).
        ngram_size: n of the largest n-gram (3).

    Returns:
        ``shifted [ngram_size, T]`` int64. ``shifted[0]`` is the current token; ``shifted[s]`` is the
        token at ``position - s``, or EOS if ``position - s < 0`` or an EOS occurs in
        ``[position - s, position - 1]``. Padding rows are all EOS.

    The token at ``p - s`` comes from the batch itself when another row of the same request has
    position ``p - s`` (prefill chunks, speculative-decode verify rows: their tokens are not in
    ``token_ids_cpu`` yet); otherwise from ``token_ids_cpu[row, p - s]``. Matching is by
    ``(row, position)`` key via sort + searchsorted, so no ordering of the batch is assumed.
    """
    ids = _as_np_long(input_ids).reshape(-1)
    rows = _as_np_long(req_rows).reshape(-1)
    pos = _as_np_long(positions).reshape(-1)
    # token_ids_cpu is large ([max_num_reqs, max_model_len] int32): index it first, cast after.
    hist = (
        token_ids_cpu.numpy()
        if isinstance(token_ids_cpu, torch.Tensor)
        else np.asarray(token_ids_cpu)
    )
    T = ids.shape[0]
    valid = rows >= 0
    out = np.full((ngram_size, T), eos_token_id, dtype=np.int64)
    out[0] = np.where(valid, ids, eos_token_id)
    if ngram_size > 1 and T > 0:
        # Batch lookup table: key = row * stride + position (positions are < L <= stride).
        stride = max(int(hist.shape[1]), int(pos.max()) + 1) + ngram_size
        keys = np.where(valid, rows * stride + pos, -1)
        order = np.argsort(keys)
        sorted_keys = keys[order]
        row_c = np.maximum(rows, 0)
        invalid = ~valid
        for s in range(1, ngram_size):
            src = pos - s
            qkey = keys - s
            idx = np.minimum(np.searchsorted(sorted_keys, qkey), T - 1)
            in_batch = (sorted_keys[idx] == qkey) & valid & (src >= 0)
            tok = np.where(
                in_batch,
                ids[order[idx]],
                hist[row_c, np.clip(src, 0, hist.shape[1] - 1)],
            )
            invalid = invalid | (src < 0) | (tok == eos_token_id)
            out[s] = np.where(invalid, eos_token_id, tok)
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# 2. Hash (exact HF: int64 wrapping multiply, xor, torch.remainder)
# ---------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    """HF ``_build_layer_multipliers``: odd splitmix64 multipliers below ``2**63 / vocab``."""
    half_bound = max(1, ((1 << 63) - 1) // max(unigram_vocab_size, 1) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound)
        + 1
        for i in range(ngram_size)
    ]


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def build_head_vocab(
    ngram_vocab_size_base: int, num_heads: int, ple_layer_index: int
) -> tuple[list[int], list[int]]:
    """HF head sizes / offsets: head ``h`` of PLE layer ``i`` has the ``(i * H + h + 1)``-th prime
    ``>= ngram_vocab_size_base``; heads are laid out back to back."""
    primes, p = [], ngram_vocab_size_base - 1
    while len(primes) < (ple_layer_index + 1) * num_heads:
        p += 1
        while not _is_prime(p):
            p += 1
        primes.append(p)
    sizes = primes[ple_layer_index * num_heads :]
    offsets = [sum(sizes[:h]) for h in range(num_heads)]
    return sizes, offsets


class NGramHasher:
    """``shifted [n, T]`` -> global table rows ``[T, H]`` (heads in HF order: the 2-gram heads,
    then the 3-gram heads, ``heads_per_ngram`` each)."""

    def __init__(
        self, layer_multipliers, head_vocab_sizes, head_offsets, heads_per_ngram: int
    ):
        self.layer_multipliers = _as_long(layer_multipliers).reshape(-1)
        self.head_vocab_sizes = _as_long(head_vocab_sizes).reshape(-1)
        self.head_offsets = _as_long(head_offsets).reshape(-1)
        self.ngram_size = self.layer_multipliers.numel()
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (self.ngram_size - 1) * heads_per_ngram
        if (
            self.head_vocab_sizes.numel() != self.num_heads
            or self.head_offsets.numel() != self.num_heads
        ):
            raise ValueError(
                f"expected {self.num_heads} head sizes/offsets, got "
                f"{self.head_vocab_sizes.numel()}/{self.head_offsets.numel()}"
            )

    @classmethod
    def from_buffers(
        cls,
        layer_multipliers,
        head_vocab_sizes,
        head_offsets,
        ngram_size: int,
        heads_per_ngram: int,
    ):
        """From the checkpoint's int64 buffers (``layer_multipliers`` [n], ``ngram_heads_*`` [H])."""
        h = cls(layer_multipliers, head_vocab_sizes, head_offsets, heads_per_ngram)
        if h.ngram_size != ngram_size:
            raise ValueError(
                f"layer_multipliers has {h.ngram_size} entries, expected ngram_size={ngram_size}"
            )
        return h

    @classmethod
    def from_config(cls, cfg: Any, ple_layer_index: int = 0):
        """Derive the buffers like HF (primes after ``ngram_vocab_size_base``, splitmix multipliers).
        ``cfg``: HF ``Qwen4ExpTextConfig`` or the plugin ``Qwen4ExpConfig``."""
        n, hpn = cfg.ngram_size, cfg.heads_per_ngram
        mult = build_layer_multipliers(cfg.vocab_size, n, ple_layer_index, cfg.seed)
        sizes, offsets = build_head_vocab(
            cfg.ngram_vocab_size_base, (n - 1) * hpn, ple_layer_index
        )
        return cls.from_buffers(mult, sizes, offsets, n, hpn)

    @property
    def total_vocab_size(self) -> int:
        return int(self.head_offsets[-1] + self.head_vocab_sizes[-1])

    def ids(self, shifted: torch.Tensor) -> torch.Tensor:
        """``shifted [ngram_size, T]`` int64 -> ``[T, H]`` int64 rows of the concatenated table."""
        shifted = _as_np_long(shifted)
        m = self.layer_multipliers.numpy()
        mixed = shifted[0] * m[0]  # int64 multiply wraps like HF
        per_order = []
        for s in range(1, self.ngram_size):
            mixed = mixed ^ (shifted[s] * m[s])
            per_order.append(mixed)
        # [T, n-1] -> [T, H]: each n-gram order feeds heads_per_ngram heads.
        mixed = np.repeat(np.stack(per_order, axis=-1), self.heads_per_ngram, axis=-1)
        # np.remainder with a positive divisor is floor-mod (non-negative), like torch.remainder.
        return torch.from_numpy(
            np.remainder(mixed, self.head_vocab_sizes.numpy())
            + self.head_offsets.numpy()
        )


# ---------------------------------------------------------------------------
# 3. Table (zero-copy mmap of the checkpoint shards)
# ---------------------------------------------------------------------------
_MADV_RANDOM = 1
_MADV_SEQUENTIAL = 2
_MADV_WILLNEED = 3
_MADV_POPULATE_READ = 22  # Linux >= 5.14
_libc = None


def _madvise(addr: int, length: int, advice: int) -> int:
    """libc madvise through ctypes (releases the GIL, unlike ``mmap.madvise``). Returns errno."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    page = mmap.PAGESIZE
    start = addr // page * page
    return (
        0
        if _libc.madvise(start, length + addr - start, advice) == 0
        else ctypes.get_errno()
    )


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Returns (tensor metadata, absolute byte offset of the data section)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class _MappedFile:
    """Read-only file mapped MAP_PRIVATE with PROT_WRITE so torch gets a writable buffer without
    copying (nothing writes into it, so no page is ever copied-on-write)."""

    def __init__(self, path: str):
        self.path = path
        fd = os.open(path, os.O_RDONLY)
        try:
            self.size = os.fstat(fd).st_size
            self.mm = mmap.mmap(
                fd,
                self.size,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)
        self._anchor = ctypes.c_char.from_buffer(
            self.mm
        )  # pins the mapping; gives its address
        self.addr = ctypes.addressof(self._anchor)
        # Gathers touch random 320-byte rows: no readahead on page faults.
        _madvise(self.addr, self.size, _MADV_RANDOM)

    def tensor(
        self, offset: int, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        count = math.prod(shape)
        return torch.frombuffer(self.mm, dtype=dtype, count=count, offset=offset).view(
            *shape
        )


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


# Below this many gathered rows (tokens x heads) ``HostNGramTable.gather`` stays single-threaded.
_MIN_PARALLEL_GATHER_ROWS = 4096


class HostNGramTable:
    """Row-sharded ``[V, D]`` n-gram table in host memory; ``gather`` returns concatenated heads.

    ``shards`` are 2-D CPU tensors with equal ``D`` whose row concatenation is the HF
    ``ngram_embedding.weight``. They are views into mmapped checkpoint files
    (:meth:`from_checkpoint`) or into an in-memory tensor (:meth:`from_tensor`).
    """

    def __init__(
        self, shards: Sequence[torch.Tensor], num_threads: int = 8, _files=None
    ):
        if not shards:
            raise ValueError("no table shards")
        self.shards = list(shards)
        self.dim = int(self.shards[0].shape[1])
        self.dtype = self.shards[0].dtype
        if any(
            s.dim() != 2 or s.shape[1] != self.dim or s.dtype != self.dtype
            for s in self.shards
        ):
            raise ValueError("shards must be 2-D with a common width and dtype")
        if not all(s.is_contiguous() for s in self.shards):
            raise ValueError("shards must be row-contiguous")
        rows = [int(s.shape[0]) for s in self.shards]
        self.shard_starts = np.cumsum([0] + rows[:-1]).astype(np.int64)
        self.num_rows = sum(rows)
        # Uniform shards (the checkpoint layout): route by division instead of searchsorted.
        self._uniform_rows = (
            rows[0]
            if all(r == rows[0] for r in rows[:-1]) and rows[-1] <= rows[0]
            else None
        )
        # One numpy record per table row: fancy indexing then memcpys whole rows (much faster than
        # torch's per-element bf16 index ops) and releases the GIL, so shards gather in parallel.
        self._row_dtype = np.dtype((np.void, self.dim * self.shards[0].element_size()))
        self._np_shards = [self._records(s) for s in self.shards]
        self._files = list(_files or [])
        self._regions = []  # (mapped file, byte offset, byte length) per shard, for prefetch
        self._num_threads = max(1, num_threads)
        self._pool = (
            ThreadPoolExecutor(self._num_threads, thread_name_prefix="ngram-gather")
            if num_threads > 1
            else None
        )

    def _records(self, t: torch.Tensor) -> np.ndarray:
        """Zero-copy ``[rows]`` record view of a contiguous 2-D CPU tensor."""
        return t.view(torch.uint8).numpy().view(self._row_dtype).reshape(-1)

    @classmethod
    def from_tensor(
        cls, weight: torch.Tensor, num_shards: int = 1, num_threads: int = 1
    ):
        """Wrap an in-memory ``[V, D]`` table (tests); ``num_shards`` row splits exercise routing."""
        rows = -(-weight.shape[0] // num_shards)
        return cls(
            list(torch.split(weight.detach().contiguous(), rows, dim=0)),
            num_threads=num_threads,
        )

    @staticmethod
    def tensor_prefix(layer_idx: int) -> str:
        return f"{TEXT_PREFIX}layers.{layer_idx}.ple.ple_embedding."

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        cfg: Any,
        layer_idx: int | None = None,
        num_threads: int = 8,
    ):
        """mmap ``ngram_embedding.shard_*.weight`` of 0-based decoder layer ``layer_idx`` (default:
        the first PLE layer). Uses ``model.safetensors.index.json`` when present, else scans
        ``*.safetensors``. No data is read here: rows are paged in on first touch (or by
        :meth:`prefetch`)."""
        if layer_idx is None:
            layer_idx = sorted(cfg.ple_layer_ids)[0] - 1
        prefix = cls.tensor_prefix(layer_idx) + "ngram_embedding.shard_"
        pat = re.compile(re.escape(prefix) + r"(\d+)\.weight$")
        files = _tensor_files(checkpoint_dir, lambda name: pat.match(name) is not None)
        found = {}
        mapped = []
        for fname in sorted(set(files.values())):
            path = os.path.join(checkpoint_dir, fname)
            header, data_start = _read_safetensors_header(path)
            mf = None
            for name, meta in header.items():
                m = pat.match(name)
                if m is None:
                    continue
                mf = mf or _MappedFile(path)
                begin, end = meta["data_offsets"]
                t = mf.tensor(
                    data_start + begin, meta["shape"], _ST_DTYPES[meta["dtype"]]
                )
                assert t.numel() * t.element_size() == end - begin, name
                found[int(m.group(1))] = (t, mf, data_start + begin, end - begin)
            if mf is not None:
                mapped.append(mf)
        n = len(found)
        expected = getattr(cfg, "split_ngram_parts", n)
        if n == 0 or sorted(found) != list(range(n)) or n != expected:
            raise ValueError(
                f"expected shards 0..{expected - 1} under {prefix}*, found {sorted(found)[:8]}... ({n})"
            )
        table = cls(
            [found[i][0] for i in range(n)], num_threads=num_threads, _files=mapped
        )
        table._regions = [(found[i][1], found[i][2], found[i][3]) for i in range(n)]
        return table

    @property
    def nbytes(self) -> int:
        return self.num_rows * self.dim * self.shards[0].element_size()

    def gather(self, ids, out: torch.Tensor | None = None) -> torch.Tensor:
        """``ids [T, H]`` int rows of the concatenated table -> ``[T, H * D]`` in the table dtype
        (row ``(t, h)`` lands at columns ``h * D : (h + 1) * D``). Negative ids give zero rows.
        ``out`` (contiguous ``[T, H * D]``) may be passed to reuse a buffer."""
        flat = ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
        T, H = flat.shape
        flat = flat.reshape(-1).astype(np.int64, copy=False)
        if out is None:
            out = torch.empty(T, H * self.dim, dtype=self.dtype)
        if (
            tuple(out.shape) != (T, H * self.dim)
            or out.dtype != self.dtype
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous [{T}, {H * self.dim}] {self.dtype}"
            )
        if flat.size == 0:
            return out
        dst = self._records(out.view(T * H, self.dim))
        if int(flat.max()) >= self.num_rows:
            raise IndexError(
                f"n-gram id {int(flat.max())} out of range [0, {self.num_rows})"
            )
        S = len(self.shards)
        if self._uniform_rows is not None:
            shard = flat // self._uniform_rows  # -1 for negative ids
        else:
            shard = np.searchsorted(self.shard_starts, flat, side="right") - 1
        shard = np.maximum(shard, -1).astype(np.int16 if S < 2**15 else np.int32)
        order = np.argsort(shard, kind="stable")  # radix sort on small ints
        shard_sorted = shard[order]
        bounds = np.searchsorted(
            shard_sorted, np.arange(-1, S + 1)
        ).tolist()  # bucket k: k + 1
        local = flat[order]
        dst[order[: bounds[1]]] = np.zeros((), self._row_dtype)  # negative ids
        work = [k for k in range(S) if bounds[k + 2] > bounds[k + 1]]

        def run(ks):
            for k in ks:
                a, b = bounds[k + 1], bounds[k + 2]
                dst[order[a:b]] = self._np_shards[k][local[a:b] - self.shard_starts[k]]

        # A decode step gathers a few hundred rows: the pool's submit/wake-up and GIL hand-offs
        # cost more than the copies, so gather small batches on the calling thread.
        if self._pool is None or len(work) < 2 or flat.size < _MIN_PARALLEL_GATHER_ROWS:
            run(work)
        else:
            n = min(self._num_threads, len(work))
            for f in [self._pool.submit(run, work[i::n]) for i in range(n)]:
                f.result()
        return out

    def prefetch(
        self, num_threads: int = 32, block: bool = False, chunk_bytes: int = 1 << 28
    ) -> threading.Thread | None:
        """Warm the page cache and populate this process's page tables for the whole table.

        Per chunk: ``MADV_SEQUENTIAL`` + ``MADV_WILLNEED`` (large async readahead; the mapping is
        otherwise ``MADV_RANDOM``, which would make population read 4 KB at a time), then
        ``MADV_POPULATE_READ`` (Linux >= 5.14; maps the pages so later gathers take no faults),
        then back to ``MADV_RANDOM``. Runs in a background thread unless ``block``; returns the
        thread. No-op for in-memory tables."""
        regions = self._regions
        if not regions:
            return None
        jobs = []
        for mf, off, length in regions:
            for a in range(0, length, chunk_bytes):
                jobs.append((mf.addr + off + a, min(chunk_bytes, length - a)))

        def one(job):
            _madvise(*job, _MADV_SEQUENTIAL)
            _madvise(*job, _MADV_WILLNEED)
            _madvise(
                *job, _MADV_POPULATE_READ
            )  # EINVAL on old kernels: WILLNEED still warms the cache
            _madvise(*job, _MADV_RANDOM)

        def run():
            with ThreadPoolExecutor(
                num_threads, thread_name_prefix="ngram-prefetch"
            ) as pool:
                list(pool.map(one, jobs))

        if block:
            run()
            return None
        t = threading.Thread(target=run, name="ngram-prefetch", daemon=True)
        t.start()
        return t


def _tensor_files(checkpoint_dir: str, want) -> dict[str, str]:
    """Tensor name -> safetensors file for the tensors ``want(name)`` selects."""
    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        return {k: v for k, v in weight_map.items() if want(k)}
    out = {}
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith(".safetensors"):
            header, _ = _read_safetensors_header(os.path.join(checkpoint_dir, fname))
            out.update({k: fname for k in header if want(k)})
    return out


def load_hasher_buffers(checkpoint_dir: str, layer_idx: int) -> dict[str, torch.Tensor]:
    """The stored int64 ``layer_multipliers`` / ``ngram_heads_vocab_sizes`` / ``ngram_heads_offsets``."""
    from safetensors import safe_open

    prefix = HostNGramTable.tensor_prefix(layer_idx)
    names = [
        prefix + n
        for n in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
    ]
    files = _tensor_files(checkpoint_dir, lambda k: k in names)
    out = {}
    for name in names:
        if name not in files:
            raise KeyError(f"{name} not found in {checkpoint_dir}")
        with safe_open(os.path.join(checkpoint_dir, files[name]), framework="pt") as f:
            out[name[len(prefix) :]] = f.get_tensor(name)
    return out


# ---------------------------------------------------------------------------
# 4. What the model holds
# ---------------------------------------------------------------------------
class HostPLEInputs:
    """Per-step host computation of the PLE layer's n-gram input.

    ``__call__(input_ids [T], req_rows [T], positions [T], token_ids_cpu [R, L])`` returns
    ``[T, H * D]`` BF16 (table dtype): the concatenated n-gram rows of each token, zeros for
    padding rows (``req_rows == -1``). See :func:`context_tokens` for the argument semantics.
    """

    def __init__(self, hasher: NGramHasher, table: HostNGramTable, eos_token_id: int):
        if hasher.total_vocab_size > table.num_rows:
            raise ValueError(
                f"hash range {hasher.total_vocab_size} exceeds table rows {table.num_rows}"
            )
        self.hasher = hasher
        self.table = table
        self.eos_token_id = int(eos_token_id)
        self.ngram_size = hasher.ngram_size
        self.embed_dim = hasher.num_heads * table.dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        cfg: Any,
        ple_layer_index: int = 0,
        prefetch: bool = True,
        num_threads: int = 8,
    ) -> "HostPLEInputs":
        """mmap the table of PLE layer ``ple_layer_index`` and load the stored hash buffers
        (falling back to :meth:`NGramHasher.from_config` if the checkpoint lacks them).
        ``prefetch`` starts warming the page cache in the background (see the module notes)."""
        layer_idx = sorted(cfg.ple_layer_ids)[ple_layer_index] - 1
        table = HostNGramTable.from_checkpoint(
            checkpoint_dir, cfg, layer_idx, num_threads=num_threads
        )
        try:
            b = load_hasher_buffers(checkpoint_dir, layer_idx)
            hasher = NGramHasher.from_buffers(
                b["layer_multipliers"],
                b["ngram_heads_vocab_sizes"],
                b["ngram_heads_offsets"],
                cfg.ngram_size,
                cfg.heads_per_ngram,
            )
        except KeyError:
            hasher = NGramHasher.from_config(cfg, ple_layer_index)
        if prefetch:
            table.prefetch()
        return cls(hasher, table, _eos_id(cfg))

    def ids(self, input_ids, req_rows, positions, token_ids_cpu) -> torch.Tensor:
        """``[T, H]`` int64 table rows (padding rows hash an all-EOS context)."""
        shifted = context_tokens(
            input_ids,
            req_rows,
            positions,
            token_ids_cpu,
            self.eos_token_id,
            self.ngram_size,
        )
        return self.hasher.ids(shifted)

    def __call__(
        self,
        input_ids,
        req_rows,
        positions,
        token_ids_cpu,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out``: optional preallocated ``[T, H * D]`` buffer (saves ~10k page faults per 8k-token
        step); the caller must not reuse it while the device copy of the previous step is pending."""
        ids = self.ids(input_ids, req_rows, positions, token_ids_cpu)
        ids[_as_np_long(req_rows).reshape(-1) < 0] = -1  # padding rows -> zeros
        return self.table.gather(ids, out=out)
