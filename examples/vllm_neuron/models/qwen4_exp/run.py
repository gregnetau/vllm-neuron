# SPDX-License-Identifier: Apache-2.0
"""Offline text generation for Qwen3.8-Flash-Next (qwen4_exp) on Neuron (trn2.48xlarge).

Serves the text tower with TP=8 (one replicated KV head per rank, 64 experts per rank).
``--fp8`` stores the routed experts in FP8 (quantized at load from the BF16 checkpoint); the
full 48-layer model needs it to fit 8 NeuronCores. ``--num-layers`` truncates the model for
bring-up (only those layers are loaded and compiled).

Usage:
    python examples/vllm_neuron/models/qwen4_exp/run.py \
        --model <path-to-checkpoint>/Qwen3.8-Flash-Next --fp8
    python examples/vllm_neuron/models/qwen4_exp/run.py \
        --model <path-to-checkpoint>/Qwen3.8-Flash-Next --num-layers 4
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.15")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")  # no EFA on this instance

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "def fibonacci(n):",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prefill-bucket", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--fp8", action="store_true")
    # The full model's weights take about 16.5 GiB of each 24 GiB core (FP8 experts).
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prompts", default=None, help="JSON list of prompts")
    parser.add_argument("--mtp", type=int, default=0, help="MTP draft tokens (0: off)")
    args = parser.parse_args()

    neuron_config = {
        "num_batched_tokens_buckets": [args.prefill_bucket],
        "num_seqs_buckets": [args.max_num_seqs],
    }
    if args.fp8:
        neuron_config["quantization"] = "fp8"
    hf_overrides = {"text_config": {}}
    if args.num_layers:
        hf_overrides["text_config"]["num_hidden_layers"] = args.num_layers
        with open(os.path.join(args.model, "config.json")) as f:
            types = json.load(f)["text_config"]["layer_types"]
        hf_overrides["text_config"]["layer_types"] = types[: args.num_layers]
    spec = {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
    llm = LLM(
        speculative_config=spec,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.prefill_bucket,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=not args.mtp,  # the MTP summary reads the spec-decode metrics
        limit_mm_per_prompt={"image": 0, "video": 0},
        hf_overrides=hf_overrides,
        additional_config={"neuron_config": neuron_config},
    )
    prompts = json.loads(args.prompts) if args.prompts else PROMPTS
    outputs = llm.generate(
        prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    )
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}\n")
    if args.mtp:
        m = {x.name: x for x in llm.get_metrics()}
        drafts = m.get("vllm:spec_decode_num_drafts")
        acc = m.get("vllm:spec_decode_num_accepted_tokens")
        if drafts is not None and acc is not None and drafts.value:
            print(
                f"MTP: {acc.value / drafts.value:.2f} accepted tokens per draft step "
                f"(of {args.mtp}); mean accepted length {1 + acc.value / drafts.value:.2f}"
            )


if __name__ == "__main__":
    main()
