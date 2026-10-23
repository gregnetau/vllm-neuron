# SPDX-License-Identifier: Apache-2.0
"""Logit-validation accuracy test for Qwen3.8-Flash-Next (qwen4_exp), served through vLLM on Neuron.

Follows test/integration/model/qwen3_5/test_model.py: the per-token raw logits of a greedy decode
are compared against a Hugging Face reference (an FP32 baseline plus a BF16 pass teacher-forced
along the FP32 greedy sequence).

The reference is the Hugging Face text tower (Qwen4ExpForCausalLM) on CPU, optionally truncated
to the first ``QWEN4_EXP_NUM_LAYERS`` decoder layers (the final hyper-connection mixer and
lm_head are kept). The references come from the shared golden cache
(``VLLM_NEURON_GOLDEN_CACHE_DIR``) and are computed on a cache miss. Computing them needs about
650 GB of host RAM for the full 48-layer FP32 reference, and roughly 200 GB for the truncated
runs, because of the 102 GB n-gram table, so pre-populating the cache on a large host is
recommended. The cache key includes the model path string and the rendered prompts, so use the
same ``QWEN4_EXP_MODEL_PATH`` the cache was populated with.

The prompts are user messages rendered through the checkpoint's chat template.

Running (trn2.48xlarge, TP=8):

    QWEN4_EXP_MODEL_PATH=/path/to/Qwen3.8-Flash-Next QWEN4_EXP_NUM_LAYERS=4 \
    VLLM_NEURON_GOLDEN_CACHE_DIR=/path/to/golden_cache \
    pytest test/integration/model/qwen4_exp/test_model.py -v -s --timeout=10800
"""

import os

import pytest
import torch

MODEL_PATH = os.environ.get("QWEN4_EXP_MODEL_PATH", "Qwen/Qwen3.8-Flash-Next")
NUM_LAYERS = int(os.environ.get("QWEN4_EXP_NUM_LAYERS", "48"))
TP_DEGREE = int(os.environ.get("QWEN4_EXP_TP", "8"))
# "fp8": FP8 routed experts (needed for the full 48 layers to fit 8 NeuronCores).
QUANTIZATION = os.environ.get("QWEN4_EXP_QUANTIZATION") or None
GPU_MEM_UTIL = float(os.environ.get("QWEN4_EXP_GPU_MEM_UTIL", "0.9"))

# The rendered prompts are part of the golden cache key: any edit invalidates cached goldens.
MESSAGES = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    (
        "The following Python function is supposed to return the k most frequent words in a text, "
        "ties broken alphabetically, but it has bugs. Find them and give a corrected version.\n\n"
        "```python\n"
        "import collections\n\n"
        "def top_k_words(text: str, k: int) -> list[str]:\n"
        "    counts = collections.Counter()\n"
        "    for word in text.split(' '):\n"
        "        word = word.strip('.,!?;:').lower\n"
        "        if word:\n"
        "            counts[word] += 1\n"
        "    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))\n"
        "    return [w for w, _ in ranked[:k]]\n"
        "```"
    ),
    (
        "A train leaves Lyon at 09:40 travelling at 180 km/h towards Paris, 465 km away. A second "
        "train leaves Paris at 10:10 travelling at 220 km/h towards Lyon on the same line. At what "
        "time do they meet, and how far from Paris? Show your reasoning step by step."
    ),
    (
        "Read the following notes and write a concise summary in three bullet points.\n\n"
        "Meeting notes, infrastructure review, week 42.\n"
        "1. The storage cluster in region B reached 87% utilisation. The team agreed to add four "
        "nodes before the end of the month and to enable automatic tiering of objects older than "
        "ninety days to cold storage, which should free roughly a fifth of the hot tier.\n"
        "2. Latency on the public API regressed by about 30 ms at the 99th percentile after the "
        "last deployment. The cause was traced to a new logging middleware that serialises the "
        "full request body. A fix that samples one request in a hundred is in review.\n"
        "3. The on-call rotation will move from weekly to twice-weekly handovers to reduce "
        "fatigue, starting in November. Each handover gets a fifteen-minute written summary.\n"
        "4. The database migration to the new major version is postponed by two weeks because "
        "two client libraries still rely on a deprecated authentication method. Owners for both "
        "libraries were assigned and will report progress on Thursday.\n"
        "5. Budget: cloud spend is 6% under forecast, mainly due to reserved-instance renewals "
        "negotiated in September."
    ),
]

OUTPUT_LENGTH = 16

# The template's BF16 dense bounds.
TOL_MAP = {
    "5": (1e-5, 0.03),
    "50": (1e-5, 0.05),
    "1000": (1e-5, 0.08),
    "all": (1e-5, 0.09),
}

torch.manual_seed(42)


def _text_config(model_checkpoint):
    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(model_checkpoint).text_config
    if NUM_LAYERS != text_config.num_hidden_layers:
        text_config.layer_types = list(text_config.layer_types[:NUM_LAYERS])
        text_config.num_hidden_layers = NUM_LAYERS
    return text_config


def _golden_variant():
    return "text_tower" if NUM_LAYERS == 48 else f"text_tower_L{NUM_LAYERS}"


def _render_prompts(tokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True
        )
        for m in MESSAGES
    ]


def _load_text_model(model_checkpoint, dtype, config=None):
    """Load only the text tower (Qwen4ExpForCausalLM) of the published checkpoint, truncated to
    ``QWEN4_EXP_NUM_LAYERS``; used only on a golden-cache miss."""
    from transformers import Qwen4ExpForCausalLM

    model = Qwen4ExpForCausalLM.from_pretrained(
        model_checkpoint, config=config or _text_config(model_checkpoint), dtype=dtype
    )
    return model.eval()


@pytest.mark.integration
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 2048),
    ],
)
def test_model_accuracy(batch_size: int, seq_len: int) -> None:
    """Serve Qwen3.8-Flash-Next (BF16, TP=8) through vLLM and validate its logits."""
    from transformers import AutoTokenizer
    from vllm import LLM

    from vllm_neuron.accuracy import (
        create_offline_vllm_generate_fn,
        get_cached_reference_goldens,
        multi_prompt_logit_validation,
    )

    print(f"Testing model with parameters: {batch_size=}, {seq_len=}, {NUM_LAYERS=}")

    model_config = _text_config(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = _render_prompts(tokenizer)
    cached_goldens = get_cached_reference_goldens(
        model_id=MODEL_PATH,
        model_checkpoint=MODEL_PATH,
        model_config=model_config,
        tokenizer=tokenizer,
        prompts=prompts,
        output_length=OUTPUT_LENGTH,
        golden_variant=_golden_variant(),
        model_loader=_load_text_model,
    )

    hf_overrides = {}
    if NUM_LAYERS != 48:
        hf_overrides = {
            "text_config": {
                "num_hidden_layers": NUM_LAYERS,
                "layer_types": list(model_config.layer_types),
            }
        }
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_DEGREE,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=GPU_MEM_UTIL,
        limit_mm_per_prompt={"image": 0, "video": 0},
        hf_overrides=hf_overrides,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
                **({"quantization": QUANTIZATION} if QUANTIZATION else {}),
            }
        },
    )
    try:
        generate_fn = create_offline_vllm_generate_fn(llm, OUTPUT_LENGTH)
        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in cached_goldens["input_ids"]
        ]
        result = multi_prompt_logit_validation(
            prompts_input_ids=prompts_input_ids,
            generate_fn=generate_fn,
            prompts_expected_logits=cached_goldens["dtype_logits"],
            prompts_baseline_logits=cached_goldens["fp32_logits"],
            colorize=True,
            replicate_to_batch_size=batch_size,
            tol_map=TOL_MAP,
        )
        assert result.passed, (
            f"Logit validation failed for parameters: {batch_size=}, {seq_len=}, {NUM_LAYERS=}"
        )
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(f"Test passed for parameters: {batch_size=}, {seq_len=}, {NUM_LAYERS=}")
