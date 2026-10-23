# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8-Flash-Next sparse MoE MLP (every decoder layer and the MTP layer)
=========================================================================

<-- MODEL-SPECIFIC: HF ``Qwen3NextSparseMoeBlock`` (``Qwen4ExpTextSparseMoeBlock``):

    probs   = softmax(fp32(x @ gate.weight.T))             # 512 experts (logits in fp32 here)
    w, idx  = topk(probs, 10); w /= sum(w); w = bf16(w)     # norm_topk_prob=True
    routed  = sum_k w_k * down_e(silu(gate_e(x)) * up_e(x))  # I = 640 per expert
    shared  = sigmoid(x @ shared_expert_gate.T) * down(silu(gate(x)) * up(x))
    out     = routed + shared

There is no norm inside the block: the input is the hyper-connection read.

Parallelism (TP group of ``tp`` ranks):

  * Routed experts are expert-parallel: rank r holds experts [r*E_L, (r+1)*E_L) at the full
    intermediate size (E_L = 512 / 8 = 64 at TP=8). Tokens are replicated (gathered for prefill),
    so each rank computes its local experts' contribution for every token. The partial sums are
    combined by the reduce-scatter (prefill, back to the [T/tp, H] sequence shard) or the
    all-reduce (decode).
  * The shared expert is TP-sharded on its intermediate dim (640 / tp per rank), so its partial
    sums ride the same reduction. Its sigmoid gate and the router are replicated.
  * The local-expert base is ``rank_in_group_tensor * E_L`` (an SPMD graph input), so all ranks
    trace one graph.

Kernels (trn2; the dense torch path is the CPU / correctness baseline):

  * Prefill: blockwise ``moe_cte`` over the local experts with the token-to-block mapping from
    ``build_blockwise_mapping`` (as GPT-OSS EP): ``shard_on_block`` below 1k tokens,
    ``shard_on_i_hybrid`` (dynamic loop over only the non-empty blocks) above.
  * Decode, T <= SPARSE_MAX_TOKENS (16): ``moe_tkg_sparse`` (vllm_neuron.functional), a dynamic
    loop over only the local experts that some token selected: one weight load per distinct
    selected local expert (about T * top_k / tp of them), split over the two physical cores.
  * Decode fallbacks (and larger T): ``moe_tkg`` selective mode for T * top_k <=
    SELECTIVE_MAX_SLOTS (it reads one expert per (token, expert) slot, local or not: non-local
    slots point at a local expert the token did not select, affinity 0), else all-expert mode
    (reads all E_L experts once, masked affinities; T <= 128).

FP8 (``config.fp8_quantized("layers.{i}.mlp.experts")``): routed-expert weights are stored in
float8_e4m3fn with per-output-channel fp32 scales, quantized at load from the BF16 checkpoint
(values within +-240, as the rest of the trn2 FP8 path). Decode runs it as weight-only FP8 in
moe_tkg_sparse / moe_tkg (BF16 activations, FP8 weight tiles, per-channel dequant; ROW in
moe_tkg), so no activation calibration is needed. Prefill dequantizes to BF16 in-graph (see
_experts_cte). Router, shared expert and its gate stay BF16.
"""

import threading

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

from .config import Qwen4ExpConfig

FP8_MAX = 240.0  # trn2 float8_e4m3 range (bytes identical to e4m3fn within +-240)
# Prefill kernel: None = by token count, or force "hybrid" (shard_on_i_hybrid) / "block"
# (shard_on_block). Measured at TP=8 (BF16, per layer): block 3.0 / 6.5 ms, hybrid 4.3 / 5.3 ms
# for T = 512 / 2048 gathered tokens.
CTE_IMPL = None
CTE_HYBRID_MIN_TOKENS = 1024
CTE_HYBRID_BLOCK_SIZE = 256  # shard_on_i kernels need a multiple of 256
CTE_BLOCK_SIZE = 128
TKG_MAX_TOKENS = 128
# Decode uses moe_tkg_sparse up to this many tokens. Measured at TP=8 on trn2 (FP8, routed
# experts per layer): T=1/4/8/16 -> 0.067/0.186/0.272/0.336 ms (selective: see below).
SPARSE_MAX_TOKENS = 16
SPARSE_UNROLL = 2  # experts per core per loop trip; 1 and 4 measured slower
# Decode switches from moe_tkg selective to all-expert above this many (token, expert) slots
# (T * top_k). Measured at TP=8 on trn2, routed experts per layer:
#   selective  FP8: T=1/4/8/16 -> 0.28/0.58/1.03/1.97 ms   BF16: 0.37/0.71/1.29/2.49 ms
#   all-expert FP8: T=8/32 -> 1.07/1.18 ms                 BF16: T=4/8/32 -> 1.96/1.78/1.84 ms
SELECTIVE_MAX_SLOTS = {"fp8": 80, "bf16": 110}

__all__ = [
    "Qwen4ExpSparseMoe",
    "attach_moe_loaders",
    "moe_weight_mappings",
    "fp8_quantize_per_channel",
]


# =============================================================================
# Section 1: FP8 weight transforms (per-output-channel, weight-only)
# =============================================================================
def fp8_quantize_per_channel(w: torch.Tensor, reduce_dim: int):
    """Quantize ``w`` with one scale per slice along every dim except ``reduce_dim`` (the input
    dim of the projection). Returns (float8_e4m3fn weight, fp32 scale with ``reduce_dim``
    squeezed); W ~= W_fp8 * scale."""
    w = w.float()
    amax = w.abs().amax(dim=reduce_dim, keepdim=True)
    scale = amax.clamp(min=1e-12) / FP8_MAX
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(reduce_dim)


def experts_gate_up_device(t: torch.Tensor) -> torch.Tensor:
    """HF ``experts.gate_up_proj`` slice [E_L, 2I, H] (gate rows, then up rows) -> kernel
    layout [E_L, H, 2, I]."""
    E, I2, H = t.shape
    return t.reshape(E, 2, I2 // 2, H).permute(0, 3, 1, 2).contiguous()


def experts_down_device(t: torch.Tensor) -> torch.Tensor:
    """HF ``experts.down_proj`` slice [E_L, H, I] -> kernel layout [E_L, I, H]."""
    return t.transpose(1, 2).contiguous()


# =============================================================================
# Section 2: the MoE block
# =============================================================================
class Qwen4ExpSparseMoe(nn.Module):
    """Router + expert-parallel routed experts + TP-sharded sigmoid-gated shared expert."""

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = tp = self.tp_group.world_size
        self.layer_idx = layer_idx
        H, E = config.hidden_size, config.num_experts
        I, Is = config.moe_intermediate_size, config.shared_expert_intermediate_size
        if E % tp or Is % tp:
            raise ValueError(
                f"num_experts={E} / shared I={Is} not divisible by tp={tp}"
            )
        self.hidden_size, self.num_experts = H, E
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.num_local_experts = E_L = E // tp
        self.intermediate_size = I
        dtype = config.torch_dtype

        # Replicated router and shared-expert gate, stored [in, out].
        self.gate_weight = nn.Parameter(
            torch.empty(H, E, dtype=dtype), requires_grad=False
        )
        self.shared_expert_gate_weight = nn.Parameter(
            torch.empty(H, 1, dtype=dtype), requires_grad=False
        )
        # Shared expert, intermediate-sharded: [H, gate shard | up shard] and [Is/tp, H].
        self.shared_gate_up_weight = nn.Parameter(
            torch.empty(H, 2 * (Is // tp), dtype=dtype), requires_grad=False
        )
        self.shared_down_weight = nn.Parameter(
            torch.empty(Is // tp, H, dtype=dtype), requires_grad=False
        )
        # Local routed experts in the moe_tkg / moe_cte layout.
        self.fp8 = config.fp8_quantized(f"layers.{layer_idx}.mlp.experts")
        wdt = torch.float8_e4m3fn if self.fp8 else dtype
        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(E_L, H, 2, I, dtype=wdt), requires_grad=False
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(E_L, I, H, dtype=wdt), requires_grad=False
        )
        if self.fp8:  # per-output-channel dequant scales
            self.experts_gate_up_scale = nn.Parameter(
                torch.ones(E_L, 2, I, dtype=torch.float32), requires_grad=False
            )
            self.experts_down_scale = nn.Parameter(
                torch.ones(E_L, H, dtype=torch.float32), requires_grad=False
            )

    # ---------------------------------------------------------------- forward
    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: prefill gathers the sequence shard; decode is replicated <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        x = hidden_states
        routed = self._routed_experts(x, *self._route(x), is_prefill)
        output = routed.to(x.dtype) + self._shared_expert(x)
        # >>> PARALLELISM: sum the per-rank partials (local experts + shared shard) <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output

    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial of sigmoid(x . w_g) * shared_expert(x)."""
        gate, up = (x @ self.shared_gate_up_weight).chunk(2, dim=-1)
        out = (F.silu(gate) * up) @ self.shared_down_weight
        return torch.sigmoid(x @ self.shared_expert_gate_weight) * out

    def _local_expert_base(self, device) -> torch.Tensor:
        """First global expert id on this rank, int32 [1] (SPMD: from the stamped rank tensor)."""
        if self.world_size == 1:
            return torch.zeros(1, dtype=torch.int32, device=device)
        rank = self.tp_group.rank_in_group_tensor.to(device=device, dtype=torch.int32)
        return rank.reshape(1) * self.num_local_experts

    def _route(self, x: torch.Tensor):
        """Top-k routing in torch (the nkilib router asserts k <= 8). Returns
        ``(affinities [T, E_L] fp32, local_idx [T, K] int32, is_local [T, K] bool)``:
        dense local affinities (zero for unselected experts; the values are the bf16-rounded
        HF routing weights) and the top-k expert ids relative to this rank's first expert."""
        # fp32 logits (HF: bf16 F.linear, then fp32 softmax): bf16 logits tie often enough at
        # the 10th/11th expert that device and CPU pick different experts.
        logits = x.float() @ self.gate_weight.float()  # [T, E]
        probs = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(x.dtype).float()
        local = top_i.to(torch.int32) - self._local_expert_base(x.device)  # [T, K]
        is_local = (local >= 0) & (local < self.num_local_experts)
        experts = torch.arange(
            self.num_local_experts, device=x.device, dtype=torch.int32
        )
        onehot = (local.unsqueeze(-1) == experts).float()  # [T, K, E_L]
        affinities = (onehot * top_w.unsqueeze(-1)).sum(dim=1)  # [T, E_L]
        return affinities, local, is_local

    # ---------------------------------------------------------- routed experts
    def _routed_experts(self, x, affinities, local, is_local, is_prefill: bool):
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        T = x.shape[0]
        if not can_run_kernel(x):
            return self._experts_dense(x, affinities)
        if is_prefill or T > TKG_MAX_TOKENS:
            return self._experts_cte(x, affinities)
        if T <= SPARSE_MAX_TOKENS:
            return self._experts_tkg_sparse(x, affinities)
        if T * self.top_k <= SELECTIVE_MAX_SLOTS["fp8" if self.fp8 else "bf16"]:
            return self._experts_tkg_selective(x, affinities, local, is_local)
        return self._experts_tkg_all(x, affinities, local, is_local)

    def _weights_scales(self):
        if self.fp8:
            return (
                self.experts_gate_up_weight,
                self.experts_down_weight,
                self.experts_gate_up_scale,
                self.experts_down_scale,
            )
        return self.experts_gate_up_weight, self.experts_down_weight, None, None

    def _dequantized_weights(self, dtype):
        """[E_L, H, 2I] and [E_L, I, H] in ``dtype`` (FP8 weights dequantized)."""
        gu, dn, gus, dns = self._weights_scales()
        E_L, H, _, I = gu.shape
        if self.fp8:
            gu = gu.to(torch.float32) * gus.unsqueeze(1)
            dn = dn.to(torch.float32) * dns.unsqueeze(1)
        return gu.reshape(E_L, H, 2 * I).to(dtype), dn.to(dtype)

    def _experts_dense(self, x, affinities):
        """Correctness baseline: every local expert on every token, weighted by the (zero for
        unselected) affinities. Per-expert outputs are rounded to x.dtype before the affinity
        product, as HF does."""
        gu, dn = self._dequantized_weights(x.dtype)
        gate, up = torch.einsum("th,ehi->eti", x, gu).chunk(2, dim=-1)  # [E_L, T, I]
        y = torch.bmm(F.silu(gate) * up, dn)  # [E_L, T, H]
        w = affinities.t().to(x.dtype).unsqueeze(-1)  # [E_L, T, 1]
        return (y * w).float().sum(dim=0)

    def _experts_cte(self, x, affinities):
        """Prefill: blockwise moe_cte over the local experts (dropless).

        * ``shard_on_block`` (128-token blocks): runs every block of the worst-case block count
          (all of a token's top-k on this rank) statically. Faster for short prompts.
        * ``shard_on_i_hybrid`` (256-token blocks, I split over the two physical cores): with
          ``num_static_block=0`` a dynamic loop runs exactly the non-empty blocks (the mapping
          puts them first). Faster from ~1k tokens.

        Neither runs FP8 here: shard_on_block ignores the scales and the hybrid kernel's
        per-channel FP8 path fails to compile (NCC_IGCA108), so FP8 experts are dequantized to
        BF16 in-graph first (weight-only FP8 storage; one extra pass over the local weights)."""
        import nki.language as nl
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

        import vllm_neuron.functional as NF

        gu, dn, gus, dns = self._weights_scales()
        E_L, H = gu.shape[0], gu.shape[1]
        T = x.shape[0]
        if self.fp8:
            gu = (gu.to(torch.float32) * gus.unsqueeze(1)).to(x.dtype)
            dn = (dn.to(torch.float32) * dns.unsqueeze(1)).to(x.dtype)
            gus = dns = None
        impl = CTE_IMPL or ("hybrid" if T >= CTE_HYBRID_MIN_TOKENS else "block")
        hybrid = impl == "hybrid"
        block = CTE_HYBRID_BLOCK_SIZE if hybrid else CTE_BLOCK_SIZE
        masked, pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=E_L,
            num_experts_per_token=self.top_k,
            block_size=block,
            moe_group=self.tp_group,
            tp_degree=1,  # expert parallel: no intermediate sharding
        )
        common = dict(
            gate_up_proj_weight=gu,
            down_proj_weight=dn,
            gate_up_proj_scale=None if gus is None else gus.reshape(E_L, 1, -1),
            down_proj_scale=None if dns is None else dns.reshape(E_L, 1, H),
            block_to_expert=block_to_expert,
            conditions=conditions,
            block_size=block,
            activation_function=ActFnType.SiLU,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )
        if not hybrid:
            return NF.moe_cte(
                implementation=MoECTEImplementation.shard_on_block,
                hidden_states=x,
                expert_affinities_masked=masked,
                token_position_to_id=pos_to_id,
                skip_token=True,  # padding slots are -1
                skip_weight=True,  # padding blocks all map to one expert: load it once
                **common,
            )
        # shard_on_i kernels: padding slots point at an extra zero row T (affinity 0).
        pos_to_id = torch.where(pos_to_id < 0, T, pos_to_id).to(torch.int32)
        x_pad = torch.cat([x, x.new_zeros(1, H)])
        masked = torch.cat([masked, masked.new_zeros(E_L, 1)])
        out = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i_hybrid,
            hidden_states=x_pad,
            expert_affinities_masked=masked,
            token_position_to_id=pos_to_id,
            num_static_block=0,
            **common,
        )
        return out[:T]

    def _tkg_kwargs(self):
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

        gu, dn, gus, dns = self._weights_scales()
        return dict(
            expert_gate_up_weights=gu,
            expert_down_weights=dn,
            expert_gate_up_weights_scale=gus,  # [E_L, 2, I]: ROW (per-channel) FP8
            expert_down_weights_scale=dns,  # [E_L, H]
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )

    def _experts_tkg_sparse(self, x, affinities):
        """Decode, few tokens: moe_tkg_sparse over the selected local experts only. The plan
        (compact expert list, per-entry token weights) is built with dense ops; the kernel's
        loop trip count comes from it at run time. Returns fp32 [T, H]."""
        from vllm_neuron.functional.moe import moe_tkg_sparse as S

        gu, dn, gus, dns = self._weights_scales()
        T = x.shape[0]
        expert_list, num_iters, aff = S.sparse_expert_plan(
            affinities, min(self.num_local_experts, T * self.top_k), SPARSE_UNROLL
        )
        return S.moe_tkg_sparse(
            x, gu, dn, expert_list, num_iters, aff, gus, dns, unroll=SPARSE_UNROLL
        )

    def _experts_tkg_selective(self, x, affinities, local, is_local):
        """Decode, few tokens: moe_tkg selective mode over T * top_k (token, expert) slots.
        Non-local slots point at a local expert the token did not select (its affinity is 0),
        so they contribute nothing."""
        import vllm_neuron.functional as NF

        unselected = (affinities == 0).float()  # >= E_L - top_k ones per row
        filler = torch.argmax(unselected, dim=-1, keepdim=True).to(torch.int32)
        index = torch.where(is_local, local, filler)
        return NF.moe_tkg(
            hidden_input=x,
            expert_affinities=affinities,  # fp32 (the kernel scales by an fp32 operand)
            expert_index=index,
            is_all_expert=False,
            **self._tkg_kwargs(),
        )

    def _experts_tkg_all(self, x, affinities, local, is_local):
        """Decode: moe_tkg all-expert mode (affinities already local and masked)."""
        import vllm_neuron.functional as NF

        return NF.moe_tkg(
            hidden_input=x,
            expert_affinities=affinities,  # fp32 (the kernel scales by an fp32 operand)
            expert_index=torch.where(is_local, local, 0),
            is_all_expert=True,
            rank_id=torch.zeros((1, 1), dtype=torch.int32, device=x.device),
            mask_unselected_experts=False,
            **self._tkg_kwargs(),
        )


# =============================================================================
# Section 3: weight loading
# =============================================================================
def _slice_loader(fn) -> SafetensorsWeightLoader:
    """Loader from ``fn(slices, rank) -> Tensor`` over lazy safetensors slices (index them to
    read only what this rank needs)."""
    return SafetensorsWeightLoader(transform=fn)


class _ExpertsQuantCache:
    """The FP8 weight and its scale come from the same BF16 slice: the first loader to run
    quantizes it and leaves the other result here, so each slice is read once."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[tuple, torch.Tensor] = {}

    def get(self, key, part: int, compute):
        with self._lock:
            hit = self._pending.pop((key, part), None)
        if hit is not None:
            return hit
        q, s = compute()
        with self._lock:
            self._pending[(key, 1 - part)] = (q, s)[1 - part]
        return (q, s)[part]


def attach_moe_loaders(moe: Qwen4ExpSparseMoe, cfg: Qwen4ExpConfig, tp: int) -> None:
    """Per-parameter transforms (HF checkpoint slices -> this rank's layout) for one MoE block;
    the checkpoint keys come from :func:`moe_weight_mappings`."""
    E_L = moe.num_local_experts
    Is_tp = cfg.shared_expert_intermediate_size // tp

    def experts(s, r):  # this rank's [E_L, ...] only
        return s[0][r * E_L : (r + 1) * E_L]

    def rows(s, r):
        return s[r * Is_tp : (r + 1) * Is_tp]

    set_weight_loader(moe.gate_weight, _slice_loader(lambda s, r: s[0][:].t()))
    set_weight_loader(
        moe.shared_expert_gate_weight, _slice_loader(lambda s, r: s[0][:].t())
    )
    # [gate shard | up shard] columns from shared_expert.{gate,up}_proj.weight [Is, H].
    set_weight_loader(
        moe.shared_gate_up_weight,
        _slice_loader(lambda s, r: torch.cat([rows(s[0], r), rows(s[1], r)]).t()),
    )
    # shared_expert.down_proj.weight [H, Is] -> this rank's [Is/tp, H].
    set_weight_loader(
        moe.shared_down_weight,
        _slice_loader(lambda s, r: s[0][:, r * Is_tp : (r + 1) * Is_tp].t()),
    )

    def gate_up(s, r):
        return experts_gate_up_device(experts(s, r))

    def down(s, r):
        return experts_down_device(experts(s, r))

    if not moe.fp8:
        set_weight_loader(moe.experts_gate_up_weight, _slice_loader(gate_up))
        set_weight_loader(moe.experts_down_weight, _slice_loader(down))
        return
    cache = _ExpertsQuantCache()
    # gate_up [E_L, H, 2, I]: reduce over H (dim 1); down [E_L, I, H]: reduce over I (dim 1).
    for wname, sname, dev in (
        ("experts_gate_up_weight", "experts_gate_up_scale", gate_up),
        ("experts_down_weight", "experts_down_scale", down),
    ):
        for name, part in ((wname, 0), (sname, 1)):

            def load(s, r, wname=wname, part=part, dev=dev):
                return cache.get(
                    (wname, r), part, lambda: fp8_quantize_per_channel(dev(s, r), 1)
                )

            set_weight_loader(getattr(moe, name), _slice_loader(load))


def moe_weight_mappings(moe: Qwen4ExpSparseMoe, pre: str, ck: str) -> dict:
    """Param name -> checkpoint key(s) for one MoE block. ``pre`` is the module's parameter
    prefix (e.g. "model.layers.3.mlp."), ``ck`` the checkpoint prefix
    (e.g. "model.language_model.layers.3.mlp.")."""
    m = {
        pre + "gate_weight": ck + "gate.weight",
        pre + "shared_expert_gate_weight": ck + "shared_expert_gate.weight",
        pre + "shared_gate_up_weight": [
            ck + "shared_expert.gate_proj.weight",
            ck + "shared_expert.up_proj.weight",
        ],
        pre + "shared_down_weight": ck + "shared_expert.down_proj.weight",
        pre + "experts_gate_up_weight": ck + "experts.gate_up_proj",
        pre + "experts_down_weight": ck + "experts.down_proj",
    }
    if moe.fp8:
        m[pre + "experts_gate_up_scale"] = ck + "experts.gate_up_proj"
        m[pre + "experts_down_scale"] = ck + "experts.down_proj"
    return m
