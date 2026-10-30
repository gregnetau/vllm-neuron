# SPDX-License-Identifier: Apache-2.0
"""Logit-validation accuracy test for Qwen3.5 / Qwen3.8 dense, served through vLLM on Neuron.

Follows the contrib template (contrib/models/template/test/integration/test_model.py): the
per-token raw logits of a greedy decode are compared against a Hugging Face reference (an FP32
baseline plus a BF16 pass teacher-forced along the FP32 greedy sequence).

The reference runs the Hugging Face text model on CPU. The FP32 pass of a 27B model needs about
110 GB of host memory, so run this test with no other model loaded. The references are cached on
disk (``VLLM_NEURON_GOLDEN_CACHE_DIR``), so only the first run computes them.

Running (trn2.3xlarge, TP=4):

    QWEN3_5_MODEL_PATH=/path/to/Qwen3.8-27B \
    pytest test/integration/model/qwen3_5/test_model.py -v -s --timeout=10800

Qwen3.5-2B: QWEN3_5_TP=2.
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get("QWEN3_5_MODEL_PATH", "Qwen/Qwen3.8-27B")
# Tensor parallelism: 4 for Qwen3.8-27B on trn2.3xlarge, 2 for Qwen3.5-2B.
TP = int(os.environ.get("QWEN3_5_TP", "4"))

PROMPTS = [
    "The color of the sky is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "The three primary colors of light are",
    "Translate to French: The weather is nice today.\nFrench:",
]

OUTPUT_LENGTH = 16

# The argmax lower bound stays at 1e-5: a divergence is a correctness bug. The per-K upper
# bounds are the template's BF16 dense convention.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

# Prompts validated with wider top-K bounds. The fibonacci prompt's 12th generated token ('0'
# after "return ") is ill-conditioned in BF16: a small deviation at its input grows about 15x
# in layer 62 (a GDN layer), and the Neuron layer matches Hugging Face's layer on the same
# inputs. Hugging Face's own BF16 model, run over the same sequence, misses the FP32 logits
# there by 0.19 (L-inf / max), more than Neuron (0.14 to 0.17). Its other tokens pass the
# template bounds.
ILL_CONDITIONED_PROMPTS = {1}
ILL_CONDITIONED_TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.07),
    "1000": (1e-5, 0.12),
    "all": (1e-5, 0.2),
}

torch.manual_seed(42)


def _load_text_model(model_checkpoint, dtype, config=None):
    """Load only the text tower (Qwen3_5ForCausalLM) of the published checkpoint; the vision
    tower and MTP head do not take part in the logits and would add host memory."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    text_config = AutoConfig.from_pretrained(model_checkpoint).text_config
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_checkpoint, config=text_config, dtype=dtype, low_cpu_mem_usage=True
    )
    return model.eval()


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
        (4, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int, monkeypatch) -> None:
    """Serve Qwen3.8 (BF16, BF16 KV, TP=4) through vLLM and validate its logits."""
    # Same defaults as examples/vllm_neuron/models/qwen3_5/run.py (unless already set).
    for name, value in (
        ("NEURON_SKIP_EFA_AFFINITY", "1"),  # trn2.3xlarge has no EFA
        ("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.15"),
    ):
        if name not in os.environ:
            monkeypatch.setenv(name, value)
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}")

    # The golden key and dtype come from the text config (BF16).
    model_config = AutoConfig.from_pretrained(MODEL_PATH).text_config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        output_length=OUTPUT_LENGTH,
        golden_variant="text_tower",
        model_loader=_load_text_model,
    )

    # max_logprobs=-1, logprobs_mode="raw_logits", async_scheduling=False and all_greedy are
    # required by logit validation (see the template).
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.65,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        generate_fn = create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH)
        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
        ]
        for tol_map, indices in (
            (
                TOL_MAP,
                [i for i in range(len(PROMPTS)) if i not in ILL_CONDITIONED_PROMPTS],
            ),
            (ILL_CONDITIONED_TOL_MAP, sorted(ILL_CONDITIONED_PROMPTS)),
        ):
            result = multi_prompt_logit_validation(
                prompts_input_ids=[prompts_input_ids[i] for i in indices],
                generate_fn=generate_fn,
                prompts_expected_logits=[
                    cached_goldens["dtype_logits"][i] for i in indices
                ],
                prompts_baseline_logits=[
                    cached_goldens["fp32_logits"][i] for i in indices
                ],
                colorize=True,
                replicate_to_batch_size=batch_size,
                tol_map=tol_map,
            )
            assert result.passed, (
                f"Logit validation failed for parameters: {batch_size=}, {seq_len=}, "
                f"prompts {indices}"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}")
