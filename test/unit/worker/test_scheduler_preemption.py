# SPDX-License-Identifier: Apache-2.0
"""NeuronScheduler with VLLM_NEURON_MAX_KV_CONCURRENT above the KV capacity (CPU): requests
outgrow the cache, are preempted and recomputed, and every request finishes."""

import pytest
import torch

pytest.importorskip("vllm_neuron.vllm.core.scheduler")

BLOCK_SIZE = 16
NUM_BLOCKS = 40  # 640 tokens: 4 requests of 100 + 200 tokens need 1,200
MAX_MODEL_LEN = 512


def _scheduler(tmp_path):
    from transformers import LlamaConfig
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=1000,
        max_position_embeddings=4096,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=MAX_MODEL_LEN,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=32, dtype=torch.bfloat16
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layers.0"], spec)],
    )
    return NeuronScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )


def test_overcommitted_requests_preempt_and_finish(monkeypatch, tmp_path):
    """Requests outgrow the cache and are preempted. A preempted request cannot re-prefill
    while the decodes hold the cache: that prefill step schedules nothing, and the scheduler
    must run the decodes instead of retrying it forever. Its recompute (prompt and output
    tokens) runs in prefill steps of its own."""
    from vllm import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    monkeypatch.setenv("VLLM_NEURON_MAX_KV_CONCURRENT", "4")
    scheduler = _scheduler(tmp_path)
    assert scheduler._max_kv_concurrent == 4
    requests = [
        Request(
            f"r{i}", [1] * 100, SamplingParams(max_tokens=200, ignore_eos=True), None
        )
        for i in range(4)
    ]
    for request in requests:
        scheduler.add_request(request)

    preempted, steps, empty_steps = set(), 0, 0
    while scheduler.has_requests():
        steps += 1
        assert steps < 2000, f"no progress: {empty_steps} empty steps"
        out = scheduler.schedule()
        preempted |= set(out.preempted_req_ids or ())
        req_ids = list(out.num_scheduled_tokens)
        empty_steps += not req_ids
        if any(out.num_scheduled_tokens[r] > 1 for r in req_ids):
            # A resumed request recomputes its output tokens as a prefill too.
            assert len(req_ids) == 1, (
                f"prefill mixed with decode: {out.num_scheduled_tokens}"
            )
        # schedule() has advanced num_computed_tokens: sample where it reached the end.
        sampled = [
            [7]
            if scheduler.requests[r].num_computed_tokens
            >= scheduler.requests[r].num_tokens
            else []
            for r in req_ids
        ]
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: k for k, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
            ),
        )
    assert preempted, "the KV cache was expected to run out"
    assert empty_steps < 10
    assert [len(r.output_token_ids) for r in requests] == [200] * 4


def test_partial_prefill_ids_resumed_recompute_but_not_verify_steps():
    """V1 runner: a request resumed after preemption is a partial prefill until its output
    tokens are recomputed too, but a verify step (scheduled drafts) never is, even when the
    runner's token count is ahead of it (async spec decode counts the in-flight steps'
    accepted tokens): flagging it dropped the step's tokens."""
    from types import SimpleNamespace

    import numpy as np

    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    def state(prompt, outputs):
        return SimpleNamespace(
            num_prompt_tokens=prompt, num_tokens=prompt + outputs
        )

    runner = SimpleNamespace(
        requests={
            "verify": state(100, 12),  # 3 tokens ahead of computed + scheduled
            "resumed": state(100, 40),  # recomputing prompt and outputs
            "resumed_last": state(100, 40),
        },
        input_batch=SimpleNamespace(
            req_id_to_index={"verify": 0, "resumed": 1, "resumed_last": 2},
            num_computed_tokens_cpu=np.array([106, 64, 128]),
        ),
    )
    so = SimpleNamespace(
        num_scheduled_tokens={"verify": 3, "resumed": 64, "resumed_last": 12},
        scheduled_spec_decode_tokens={"verify": [7, 8]},
    )
    partial = NeuronModelRunner._get_partial_prefill_req_ids(
        runner, so, ["verify", "resumed", "resumed_last"]
    )
    assert partial == {"resumed"}
