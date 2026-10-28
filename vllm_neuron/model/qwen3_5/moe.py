# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 sparse MoE FFN (``qwen3_5_moe``: Qwen3.5-35B-A3B, 397B-A17B), BF16 experts
==================================================================================

Derived from the contrib Qwen3.5 model of vllm-project/vllm-neuron PR #68 (whn09,
``contrib/models/qwen3_5/src/moe.py``), adapted to this model's conventions: weights in the
kernels' ``[in, out]`` layouts with pure-torch device-layout helpers (``weights.moe_*``),
loaders attached by ``model._attach_layer_loaders``, the zero-centered post-attention norm
already folded to ``gamma = 1 + w`` at load time, and the decoder layer's prefill
sequence-parallel layout ([T/tp, H] residual).

The block replaces ``Qwen3_5MLP``: ``num_experts`` routed SwiGLU experts, top
``num_experts_per_tok`` per token, plus one shared SwiGLU expert scaled by
``sigmoid(shared_expert_gate(x))``. Like the static-FP8 MLP it takes the residual *before*
the post-attention norm (``ln_w`` = folded gamma), because the decode kernel fuses the norm.

  * prefill: norm, router (``NF.router``, softmax over all experts -> top-k -> L1
    renormalise, as ``Qwen3_5MoeTopKRouter``) on the sequence-parallel shard; all-gather
    tokens + affinities; ``NF.build_blockwise_mapping`` + ``NF.moe_cte`` (blockwise,
    ``BLOCK_SIZE`` tokens of one expert per block; padding tokens are masked out);
    shared expert; reduce-scatter.
  * decode: ``NF.moe_block_tkg`` (norm + router + top-k + experts fused; below the
    selective-loading threshold only the selected experts' weights are read); shared
    expert in torch; all-reduce. top-k > 8 (397B, the kernel's router asserts K <= 8)
    routes in torch and runs the experts in ``NF.moe_tkg``. More than
    ``MAX_TKG_TOKENS`` decode tokens (large batch x MTP) go through the blockwise path.

The shared expert stays outside the fused kernel: its sigmoid gate is input dependent and
the kernel adds a shared-expert branch unscaled.

>>> PARALLELISM: TP (default) or EP <<< Under TP every rank holds every expert with the
intermediate dim sharded (moe_intermediate_size / tp; the decode kernel needs a multiple of
128). With vLLM's ``enable_expert_parallel`` each rank of an EP partition holds
num_experts / ep_degree experts, sharded on I over the ep_degree's TP sub-group. Either way
each rank produces a partial sum, so the combine is the same all-reduce / reduce-scatter.
The router and the shared-expert gate are replicated; the shared expert is TP-sharded.
"""

from typing import ClassVar

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from .config import Qwen3_5Config

# Tokens per expert block in prefill (moe_cte needs a multiple of 128). The block count is
# ceil((T * top_k - (E - 1)) / BLOCK_SIZE) + E - 1, so with 256 experts the E - 1 term
# dominates and larger blocks only add padding (PR #68: at T = 2048, 381 blocks at 128 vs
# 318 at 256, 34% vs 20% of the slots real).
BLOCK_SIZE = 128
# moe_block_tkg reads every local expert once T * top_k reaches this fraction of the local
# experts (batch 32 at top-8 / 256 experts); below it only the selected experts.
SELECTIVE_LOADING_THRESHOLD = 1.0
# nkilib router_topk (inside moe_block_tkg) supports top-k <= 8.
MAX_KERNEL_TOP_K = 8
# moe_block_tkg (BF16) handles at most 128 tokens per call.
MAX_TKG_TOKENS = 128


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Same arithmetic as ``model.Qwen3_5RMSNorm`` (gamma already folded to 1 + w)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (gamma.float() * xf).to(x.dtype)


def _expert_parallel_degree() -> tuple[int, int, object]:
    """(ep_degree, ep_rank, ep_tp_group) from vLLM's enable_expert_parallel; (1, 0, None)
    without an engine context (CPU tests, offline compile probes)."""
    try:
        from vllm.config import get_current_vllm_config

        parallel_config = get_current_vllm_config().parallel_config
        ep_enabled = bool(parallel_config.enable_expert_parallel)
        dp_size = int(parallel_config.data_parallel_size)
    except (AssertionError, AttributeError, ImportError, RuntimeError):
        return 1, 0, None
    if not ep_enabled:
        return 1, 0, None
    if dp_size > 1:
        # The combine reduces over the TP group only: experts spread over data-parallel
        # replicas would silently drop the other replicas' tokens.
        raise NotImplementedError(
            "Qwen3.5 MoE expert parallelism across data-parallel replicas is not supported; "
            "keep expert parallelism inside one tensor-parallel group."
        )
    from vllm_neuron.parallel.neuron_parallel_state import (
        get_neuron_ep_degree,
        get_neuron_ep_rank,
        get_neuron_ep_tp_group,
    )

    return get_neuron_ep_degree(), get_neuron_ep_rank(), get_neuron_ep_tp_group()


class Qwen3_5SparseMoeBlock(nn.Module):
    """Routed experts + gated shared expert. ``forward(x, is_prefill, ln_w, token_mask)``
    takes the residual before the post-attention norm."""

    FUSES_NORM: ClassVar[bool] = True

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = H = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size

        # >>> PARALLELISM: TP or EP <<<
        self.ep_degree, self.ep_rank, ep_tp_group = _expert_parallel_degree()
        self.ep_tp_group = ep_tp_group if ep_tp_group is not None else self.tp_group
        self.tp_degree = self.ep_tp_group.world_size
        if self.num_experts % self.ep_degree:
            raise ValueError(
                f"num_experts={self.num_experts} is not divisible by ep_degree={self.ep_degree}"
            )
        self.num_local_experts = self.num_experts // self.ep_degree
        if self.intermediate_size % self.tp_degree:
            raise ValueError(
                f"moe_intermediate_size={self.intermediate_size} is not divisible by "
                f"the expert TP degree {self.tp_degree}"
            )
        self.inter_per_rank = I_r = self.intermediate_size // self.tp_degree
        S_r = config.shared_expert_intermediate_size // self.world_size

        E_l = self.num_local_experts
        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(E_l, H, 2, I_r, dtype=self.dtype)
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(E_l, I_r, H, dtype=self.dtype)
        )
        # Replicated and global (every rank scores every expert, also under EP), [H, E].
        self.router_weight = nn.Parameter(
            torch.empty(H, self.num_experts, dtype=self.dtype)
        )
        self.shared_gate_proj_weight = nn.Parameter(
            torch.empty(H, S_r, dtype=self.dtype)
        )
        self.shared_up_proj_weight = nn.Parameter(torch.empty(H, S_r, dtype=self.dtype))
        self.shared_down_proj_weight = nn.Parameter(
            torch.empty(S_r, H, dtype=self.dtype)
        )
        self.shared_expert_gate_weight = nn.Parameter(
            torch.empty(H, 1, dtype=self.dtype)
        )

    @property
    def local_experts(self) -> list[int]:
        """Global ids of this rank's experts (linear placement, as gpt_oss)."""
        lo = self.ep_rank * self.num_local_experts
        return list(range(lo, lo + self.num_local_experts))

    # -- helpers -------------------------------------------------------------------------
    def _ep_rank_tensor(self, device) -> torch.Tensor:
        """EP partition index as a traced int32 [1] tensor (SPMD: a Python-int rank would
        give every rank its own graph)."""
        if self.ep_degree <= 1:
            return torch.zeros(1, dtype=torch.int32, device=device)
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_group,
            try_get_rank_in_group_tensor,
        )

        stamped = try_get_rank_in_group_tensor(get_neuron_ep_group())
        if stamped is None:  # direct (non-SPMD) entry points
            return torch.tensor([self.ep_rank], dtype=torch.int32, device=device)
        return stamped.reshape(1).to(torch.int32)

    def _local_affinities(self, affinities: torch.Tensor) -> torch.Tensor:
        """[T, E] global affinities -> [T, E_local] for this rank's experts."""
        if self.ep_degree <= 1:
            return affinities
        idx = (
            torch.arange(
                self.num_local_experts, device=affinities.device, dtype=torch.int32
            )
            + self._ep_rank_tensor(affinities.device) * self.num_local_experts
        )
        return NF.get_local_expert_affinities(affinities, idx)

    def _route(self, normed: torch.Tensor) -> torch.Tensor:
        """[T, H] normed -> [T, E] affinities: softmax over all experts, top-k, renormalise."""
        from vllm_neuron.functional.moe.router import RouterComputationOrder

        return NF.router(
            hidden_states=normed,
            router_weights=self.router_weight,
            top_k=self.top_k,
            router_bias=None,
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        )

    def _shared_expert(self, normed: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of sigmoid(gate(x)) * SwiGLU(x). The gate is computed from
        the replicated weight, so scaling each partial before the reduction is exact."""
        out = NF.mlp(
            normed,
            self.shared_gate_proj_weight,
            self.shared_up_proj_weight,
            self.shared_down_proj_weight,
        )
        gate = torch.sigmoid((normed @ self.shared_expert_gate_weight).float())
        return (out.float() * gate).to(out.dtype)

    def _routed_blockwise(self, normed, affinities, token_mask):
        """Blockwise experts (moe_cte) over all T tokens: normed [T, H], affinities [T, E]."""
        from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

        affinities = self._local_affinities(affinities)
        (
            affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=BLOCK_SIZE,
            # Under TP the block construction (not the expert set) is split over the ranks;
            # under pure EP this is a one-rank group.
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=token_mask,
        )
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=normed,
            expert_affinities_masked=affinities_masked,
            gate_up_proj_weight=self.experts_gate_up_weight,
            down_proj_weight=self.experts_down_weight,
            activation_function=ActFnType.SiLU,
            block_size=BLOCK_SIZE,
            token_position_to_id=token_position_to_id.to(torch.int32),
            block_to_expert=block_to_expert.to(torch.int32),
            # The affinity scales the expert output (Qwen3_5MoeExperts).
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
        )

    def _routed_tkg(self, x, ln_w):
        """Fused decode kernel on the pre-norm residual x [T, H] (top-k <= 8, T <= 128)."""
        from nkilib.core.utils.common_types import (
            ActFnType,
            ExpertAffinityScaleMode,
            RouterActFnType,
        )

        T = x.shape[0]
        is_all_expert = (
            T * self.top_k / self.num_local_experts >= SELECTIVE_LOADING_THRESHOLD
            or self.ep_degree > 1
        )
        rank_id = (
            self._ep_rank_tensor(x.device).reshape(1, 1) if is_all_expert else None
        )
        out = NF.moe_block_tkg(
            inp=x.unsqueeze(0),
            gamma=ln_w.float().view(1, -1),
            router_weights=self.router_weight,
            expert_gate_up_weights=self.experts_gate_up_weight,
            expert_down_weights=self.experts_down_weight,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,  # softmax over all experts, then top-k
            norm_topk_prob=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            is_all_expert=is_all_expert,
            skip_router_logits=True,
        )
        return out[0] if isinstance(out, tuple) else out

    def _routed_unfused(self, normed):
        """top-k > 8: router in torch, experts in moe_tkg on the local affinities (blockwise
        experts where NKI kernels cannot run)."""
        from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

        probs = torch.softmax(normed.float() @ self.router_weight.float(), dim=-1)
        top_values, top_index = torch.topk(probs, self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        affinities = torch.zeros_like(probs).scatter(1, top_index, top_values)
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if not can_run_kernel(normed):  # moe_tkg has no PyTorch fallback
            return self._routed_blockwise(normed, affinities, None)
        return NF.moe_tkg(
            hidden_input=normed,
            expert_gate_up_weights=self.experts_gate_up_weight,
            expert_down_weights=self.experts_down_weight,
            expert_affinities=self._local_affinities(affinities)
            .contiguous()
            .to(self.dtype),
            expert_index=top_index.to(torch.int32),
            # The affinities are already local, so the kernel's rank offset is 0.
            is_all_expert=True,
            rank_id=torch.zeros((1, 1), dtype=torch.int32, device=normed.device),
            mask_unselected_experts=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )

    # -- forward -------------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        ln_w: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: pre-norm residual ([T/tp, H] in prefill, [T, H] in decode);
        token_mask [T] (prefill, all tokens): True for real tokens."""
        x = hidden_states.to(self.dtype)
        if is_prefill:
            normed = _rms_norm(x, ln_w, self.eps)
            # Router on the sequence-parallel shard: gathering [T, E] scores is cheaper than
            # routing the gathered [T, H].
            affinities = self._route(normed)
            if self.world_size > 1:  # >>> PARALLELISM: all-gather from SP <<<
                affinities = self.tp_group.all_gather(affinities, dim=0)
                normed = self.tp_group.all_gather(normed, dim=0)
            out = self._routed_blockwise(normed, affinities, token_mask)
            out = out.to(self.dtype) + self._shared_expert(normed)
            if self.world_size > 1:  # >>> PARALLELISM: reduce-scatter to SP <<<
                out = self.tp_group.reduce_scatter(out, dim=0)
            return out

        normed = _rms_norm(x, ln_w, self.eps)
        if x.shape[0] > MAX_TKG_TOKENS:
            routed = self._routed_blockwise(normed, self._route(normed), None)
        elif self.top_k > MAX_KERNEL_TOP_K:
            routed = self._routed_unfused(normed)
        else:
            routed = self._routed_tkg(x, ln_w)
        out = routed.to(self.dtype) + self._shared_expert(normed)
        if self.world_size > 1:  # >>> PARALLELISM: TP all-reduce <<<
            self.tp_group.all_reduce(out)
        return out
