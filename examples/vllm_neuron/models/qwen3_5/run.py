# SPDX-License-Identifier: Apache-2.0
"""Offline text generation example for Qwen3.5 / Qwen3.8 dense (27B) on Neuron.

Serves the text tower of ``Qwen3_5ForConditionalGeneration`` with TP=4 on one Trainium2
chip (trn2.3xlarge). ``--fp8`` quantizes the MLPs to per-tensor static FP8 at load time,
using activation scales from ``calibrate_fp8.py``.

Usage:
    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.8-27B

    # Static FP8 MLPs
    python examples/vllm_neuron/models/qwen3_5/run.py \
        --model <path-to-checkpoint>/Qwen3.8-27B \
        --fp8 --fp8-activation-scales <path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json
"""

import argparse
import os

os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")  # trn2.3xlarge has no EFA
os.environ.setdefault("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.15")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "def fibonacci(n):",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prefill-bucket", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-activation-scales", default=None)
    args = parser.parse_args()

    neuron_config = {
        "num_batched_tokens_buckets": [args.prefill_bucket],
        "num_seqs_buckets": [1],
    }
    if args.fp8:
        neuron_config["quantization"] = "fp8"
        if args.fp8_activation_scales:
            neuron_config["fp8_activation_scales_path"] = args.fp8_activation_scales

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.prefill_bucket,
        gpu_memory_utilization=0.65,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"neuron_config": neuron_config},
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
    for prompt, output in zip(PROMPTS, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
