# SPDX-License-Identifier: Apache-2.0
"""Model Runner V2's inputs drive the Qwen3.5 model to the HF reference (CPU).

The KV cache config comes from vLLM's grouping and KV cache manager, the block tables, slot
mappings, state-pool pages and cached_seq_len from the real V2 runner
(initialize_kv_cache, update_states, _prepare_model_input, _build_forward_adapter_kwargs).
The model is the real Qwen3.5 module with the NKI ops stubbed in torch (test_model_smoke);
attention decode runs each row as a one-token chunk through the torch prefill path.

The schedule interleaves two chunked prefills with decode steps: a request sits out steps
while the other prefills, decodes run in a padded 4-row bucket and in both row orders, and
the state pages start dirty (a first chunk must start from zero state).
"""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("transformers.models.qwen3_5")

BLOCK, MAX_MODEL_LEN, BUCKET, MAX_REQS = 16, 128, 32, 4
ATTN_LAYERS = (3, 7)


class _SpecModel:
    """Stands in for the model while the runner builds the KV cache (phase A)."""

    def __init__(self):
        from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, StateLayerSpec

        layers, state_layers = [], []
        for i in range(8):
            if i in ATTN_LAYERS:
                layers.append(LayerSpec(f"layers.{i}.self_attn", 2, 32, torch.float32))
            else:
                state_layers.append(
                    StateLayerSpec(
                        f"layers.{i}.linear_attn",
                        ((3, 128), (4, 16, 16)),
                        (torch.float32, torch.float32),
                    )
                )
        self.kv_spec = KVSpec(layers=layers, state_layers=state_layers)

    def get_kv_spec(self):
        return self.kv_spec

    def bind_kv_cache(self, kv_caches):
        pass

    def bind_state_cache(self, state_caches):
        pass


def _runner():
    from vllm_neuron.vllm.worker.neuron_model_runner_v2 import NeuronModelRunnerV2
    from vllm_neuron.vllm.worker.neuron_request_state import NeuronRequestState

    runner = object.__new__(NeuronModelRunnerV2)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    runner.model = _SpecModel()
    runner.max_num_reqs, runner.max_model_len = MAX_REQS, MAX_MODEL_LEN
    runner.neuron_config = SimpleNamespace(
        fp8_packed_kv=False, num_batched_tokens_buckets=[BUCKET]
    )
    runner.num_seqs_buckets = [1, 4]
    runner.kv_segment_size = 512
    runner._has_state_layers = True
    runner.num_speculative_tokens = 0
    runner.draft_model = None
    runner._state_real_pages = {}
    runner._kv_cache_config = None
    runner._inflight_graphs, runner._quarantined_slots = [], set()
    runner._dp_sync_shape = lambda tokens, rows: (tokens, rows)
    runner._rank_tensor_cache = torch.zeros(1, dtype=torch.int32)
    runner._next_token_by_slot = torch.zeros(MAX_REQS, 1, dtype=torch.int32)
    runner.request_state = NeuronRequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_MODEL_LEN,
        vocab_size=128,
        device=runner.device,
        block_sizes=[BLOCK],
    )
    return runner


def _build_steps(prompts, plan):
    """Phase A: run the plan through vLLM's KV cache manager and the V2 runner."""
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.request import Request

    runner = _runner()
    vllm_config = VllmConfig()
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    groups = get_kv_cache_groups(vllm_config, runner.get_kv_cache_spec())
    attn_page = 2 * BLOCK * 2 * 32 * 4
    config = get_kv_cache_config_from_groups(
        vllm_config, groups, 64 * attn_page * len(groups[0].layer_names)
    )
    runner.initialize_kv_cache(config)
    manager = KVCacheManager(
        config,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=MAX_MODEL_LEN,
        hash_block_size=MAX_MODEL_LEN,
        enable_caching=False,
    )
    requests = {
        r: Request(r, p, SamplingParams(max_tokens=8), None) for r, p in prompts.items()
    }
    steps = []
    for sched in plan:
        new, cached = (
            [],
            SimpleNamespace(
                req_ids=[],
                num_computed_tokens=[],
                new_block_ids=[],
                resumed_req_ids=set(),
            ),
        )
        padded = {}
        for r, n in sched.items():
            req = requests[r]
            block_ids = manager.allocate_slots(req, n).get_block_ids()
            if req.num_computed_tokens == 0:
                new.append(
                    SimpleNamespace(
                        req_id=r,
                        prompt_token_ids=prompts[r],
                        prefill_token_ids=prompts[r],
                        sampling_params=req.sampling_params,
                        num_computed_tokens=0,
                        block_ids=block_ids,
                    )
                )
            else:
                cached.req_ids.append(r)
                cached.num_computed_tokens.append(req.num_computed_tokens)
                cached.new_block_ids.append(block_ids if any(block_ids) else None)
            padded[r] = BUCKET if req.num_computed_tokens < len(prompts[r]) else 1
        step = SimpleNamespace(
            scheduled_new_reqs=new,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=dict(sched),
            num_scheduled_tokens_padded=padded,
            total_num_scheduled_tokens=sum(sched.values()),
            finished_req_ids=set(),
            preempted_req_ids=None,
        )
        runner.update_states(step)
        model_input = runner._prepare_model_input(step)
        kwargs = runner._build_forward_adapter_kwargs(model_input)
        order = [n.req_id for n in new] + cached.req_ids
        steps.append((model_input, kwargs["attn_metadata"], order))
        for r, n in sched.items():
            requests[r].num_computed_tokens += n
    return runner, steps


# Shapes shared by the dense and the sparse MoE (qwen3_5_moe) tiny models.
_TINY = {
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "vocab_size": 128,
    "dtype": "float32",
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 2],
        "mrope_interleaved": True,
    },
}
TINY_DENSE = dict(_TINY, intermediate_size=96)
# 8 experts, top 2: a verify bucket of 4 rows x (1 + k) tokens reads every expert
# (moe_block_tkg's all-expert mode), a single decode row only the selected ones.
TINY_MOE = dict(
    _TINY,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=48,
)


def _hf_and_model(moe=False):
    """Phase B: the HF reference and the Neuron model with the same random weights.
    ``moe``: Qwen3_5MoeForCausalLM, with the plugin's MoE ops on their PyTorch
    fallbacks (test_moe)."""
    import test_model_smoke as smoke

    if moe:
        import test_moe
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForCausalLM,
        )

        mod = test_moe._install_stubs_with_moe_ops()
        hf_cfg = Qwen3_5MoeTextConfig(**TINY_MOE)
        hf_cls = Qwen3_5MoeForCausalLM
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )

        mod = smoke._install_stubs()
        hf_cfg = Qwen3_5TextConfig(**TINY_DENSE)
        hf_cls = smoke.hf.Qwen3_5ForCausalLM
    hf_model = hf_cls(hf_cfg).eval()
    with torch.no_grad():
        for n, p in hf_model.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.1))
            elif "linear_attn.norm" in n:
                p.copy_(torch.rand_like(p) + 0.5)
            elif "norm" in n:
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    sd = {
        (
            k.replace("model.", "model.language_model.", 1)
            if k.startswith("model.")
            else k
        ): v
        for k, v in hf_model.state_dict().items()
    }
    model = mod.Qwen3_5ForCausalLM.from_configs(hf_cfg.to_dict(), None)
    assert model.config.is_moe == moe
    wl = sys.modules["vllm_neuron.utils.weight_loader"]
    params = dict(model.named_parameters())
    for name, keys in model._weight_mappings().items():
        keys = [keys] if isinstance(keys, str) else keys
        loaded = wl.get_weight_loader(params[name]).load(
            [smoke.FakeSlice(sd[k]) for k in keys], 0
        )
        params[name].data = loaded.to(params[name].dtype)
    for i in ATTN_LAYERS:
        _torch_attention(model.model.layers[i].self_attn)
    return hf_model, model


def _torch_attention(attn):
    """NF.segmented_attention / attention_decode are not stubbed: use the torch paths."""
    attn._segmented_kernel_ok = lambda block_size: False
    name = f"layers.{attn.layer_idx}.self_attn"

    def forward_decode(hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[name]
        cos, sin = position_embeddings
        outs = []
        for b in range(md["block_table_tensor"].shape[0]):
            row = dict(
                md,
                block_table_tensor=md["block_table_tensor"][b : b + 1],
                slot_mapping=md["slot_mapping"][b : b + 1],
                cached_seq_len=positions[b].view(1, 1).to(torch.int32),
                kv_segment_size=1,
            )
            outs.append(
                attn.forward_prefill(
                    hidden_states[b : b + 1],
                    positions[b : b + 1],
                    (cos[b : b + 1], sin[b : b + 1]),
                    {name: row},
                )
            )
        return torch.cat(outs)

    attn.forward_decode = forward_decode


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_v2_inputs_match_hf_through_interleaved_chunked_prefills(moe):
    torch.manual_seed(3)
    prompts = {
        "a": torch.randint(1, 128, (50,)).tolist(),
        "b": torch.randint(1, 128, (40,)).tolist(),
    }
    plan = [
        {"a": 32},  # a, first chunk
        {"b": 32},  # b, first chunk (a sits out)
        {"a": 18},  # a, last chunk: samples
        {"a": 1},  # a decodes while b is mid-prefill
        {"b": 8},  # b, last chunk
        {"a": 1, "b": 1},  # padded 4-row decode
        {"b": 1, "a": 1},  # other row order
    ]
    runner, steps = _build_steps(prompts, plan)
    hf_model, model = _hf_and_model(moe)
    model.bind_kv_cache(runner.kv_caches)
    model.bind_state_cache(runner.state_caches)
    for halves in runner.state_caches.values():  # recycled pages hold stale state
        for half in halves:
            half.fill_(7.0)

    tokens = {r: list(p) for r, p in prompts.items()}
    state_rows = []
    for model_input, md, order in steps:
        positions = model_input.positions
        if model_input.is_prefill:
            start, n = int(positions[0]), plan[len(state_rows)][order[0]]
            ids = torch.zeros(model_input.padded_num_tokens, dtype=torch.long)
            ids[:n] = torch.tensor(tokens[order[0]][start : start + n])
            sampled = [(order[0], start + n - 1, 0)]
        else:
            ids = torch.zeros(model_input.padded_num_reqs, dtype=torch.long)
            for row, r in enumerate(order):
                ids[row] = tokens[r][int(positions[row])]
            sampled = [(r, int(positions[row]), row) for row, r in enumerate(order)]
        with torch.no_grad():
            logits = model(
                input_ids=ids,
                positions=positions,
                attn_metadata=md,
                sampling_positions=model_input.sampling_positions,
                sampling_params=None,
                logit_mask=None,
                rank=None,
            )
        for r, pos, row in sampled:
            with torch.no_grad():
                ref = hf_model(torch.tensor([tokens[r][: pos + 1]])).logits[0, -1]
            torch.testing.assert_close(logits[row], ref, atol=1e-3, rtol=1e-3)
            if pos + 1 == len(tokens[r]):  # past the prompt: append the greedy token
                tokens[r].append(int(ref.argmax()))
        state_rows.append(
            md["layers.0.linear_attn"]["block_table_tensor"][:, 0].tolist()
        )
    # Slots 0 and 1 own pool pages 1 and 2 across every step; padded rows are -1.
    assert state_rows == [[1], [2], [1], [1], [2], [1, 2, -1, -1], [2, 1, -1, -1]]
