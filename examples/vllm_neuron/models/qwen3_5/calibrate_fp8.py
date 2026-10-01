# SPDX-License-Identifier: Apache-2.0
"""Static-FP8 activation calibration for Qwen3.5 / Qwen3.8 dense checkpoints (CPU).

Runs the Hugging Face BF16 model over a small, mixed prompt set and records the input
amax of every ``nn.Linear`` in the language model. The output is the file referenced by
``neuron_config.fp8_activation_scales_path`` when serving with ``quantization="fp8"``.
Weight scales are computed from the BF16 weights at load time and are not stored here.

Usage:
    python examples/vllm_neuron/models/qwen3_5/calibrate_fp8.py \
        --model <path-to-checkpoint>/Qwen3.8-27B \
        --output <path-to-checkpoint>/Qwen3.8-27B/fp8_act_amax.json

Output format: ``{"layers.<i>.mlp.down_proj": {"amax": float, "per_prompt": [...]}, ...,
"_meta": {...}}`` with module names relative to the language model.
"""

import argparse
import json
import time

import torch
import transformers

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. Rome is the capital of",
    "Write a haiku about the sea and explain the imagery you chose in two sentences.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n",
    "Solve step by step: a train leaves at 3:15 pm travelling 84 km/h; another leaves at 4:00 pm at 102 km/h. When does the second catch up?",
    "Explique en français la différence entre la photosynthèse et la respiration cellulaire.",
    "请用中文简要介绍一下长城的历史和它的建造目的。",
    "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2024-01-01' GROUP BY customer_id ORDER BY total DESC LIMIT 10;",
    "In 1905, Albert Einstein published four papers that changed physics. The first concerned the photoelectric effect,",
    "Here is a JSON object describing a user: {\"id\": 4821, \"name\": \"Ada\", \"roles\": [\"admin\", \"editor\"], \"active\": true}. Convert it to YAML.",
    "Translate to German: 'The weather was terrible, so we stayed inside and played board games all afternoon.'",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,",
    "Summarize the plot of Romeo and Juliet in three bullet points, then list the main characters.",
    "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 10; ++i) {\n        printf(\"%d\\n\", i * i);\n    }\n",
    "Q: What is the derivative of x^3 * sin(x)? A: Using the product rule,",
    "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation, which",
    "Dear hiring manager, I am writing to apply for the position of senior software engineer at your company.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    chats = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True)
        for p in PROMPTS[:4]
    ]
    texts = PROMPTS + chats
    t0 = time.time()
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    lm = model.model.language_model
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    stats, current = {}, {}
    for name, mod in lm.named_modules():
        if isinstance(mod, torch.nn.Linear):
            def hook(_m, inputs, name=name):
                amax = inputs[0].detach().abs().amax().float().item()
                current[name] = max(current.get(name, 0.0), amax)
            mod.register_forward_pre_hook(hook)

    n_tokens = 0
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tok(text, return_tensors="pt").input_ids
            n_tokens += ids.shape[1]
            current.clear()
            model(input_ids=ids, use_cache=False)
            for k, v in current.items():
                s = stats.setdefault(k, {"amax": 0.0, "per_prompt": []})
                s["amax"] = max(s["amax"], v)
                s["per_prompt"].append(v)
            print(f"prompt {i + 1}/{len(texts)}: {ids.shape[1]} tokens", flush=True)
    stats["_meta"] = {"n_prompts": len(texts), "n_tokens": n_tokens, "checkpoint": args.model}
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=0)
    print(f"saved {len(stats) - 1} modules to {args.output}")


if __name__ == "__main__":
    main()
