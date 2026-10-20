# SPDX-License-Identifier: Apache-2.0
"""Greedy-output agreement between two serving configurations.

``capture`` stores 64-token greedy completions (with token ids) for a fixed prompt set from
a running server; ``compare`` reports, per prompt, the length of the identical token prefix
between two captures (e.g. BF16 vs FP8, or two implementations).

Usage:
    python examples/vllm_neuron/models/qwen3_5/compare_greedy.py capture --model Qwen3.8-27B bf16.json
    python examples/vllm_neuron/models/qwen3_5/compare_greedy.py capture --model Qwen3.8-27B fp8.json
    python examples/vllm_neuron/models/qwen3_5/compare_greedy.py compare bf16.json fp8.json
"""

import argparse
import json
import urllib.request

PROMPTS = [
    "The history of the Roman Empire can be divided into",
    "To make a good cup of coffee, you should",
    "import numpy as np\n\ndef softmax(x):\n",
    "The three laws of thermodynamics are",
    "Once upon a time, in a small village by the sea,",
    "Q: If a rectangle has sides 7 and 12, what is its area? A:",
    "Les avantages des énergies renouvelables sont",
    "A binary search tree is a data structure that",
    "The difference between TCP and UDP is",
    "My favourite book is",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n",
    "请用中文简要介绍一下长城的历史。",
]


def capture(url: str, model: str, max_tokens: int, out: str) -> None:
    results = []
    for prompt in PROMPTS:
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "return_token_ids": True}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        choice = json.load(urllib.request.urlopen(req))["choices"][0]
        results.append({"prompt": prompt, "text": choice["text"], "token_ids": choice.get("token_ids")})
    with open(out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"saved {len(results)} completions to {out}")


def compare(a_path: str, b_path: str) -> None:
    with open(a_path) as fa, open(b_path) as fb:
        a, b = json.load(fa), json.load(fb)
    prefixes = []
    for x, y in zip(a, b):
        ta, tb = x["token_ids"] or list(x["text"]), y["token_ids"] or list(y["text"])
        n = next((i for i, (u, v) in enumerate(zip(ta, tb)) if u != v), min(len(ta), len(tb)))
        prefixes.append((n, len(ta)))
        print(f"{n:3d}/{len(ta)} identical  {x['prompt'][:40]!r}")
    full = sum(n == total for n, total in prefixes)
    mean = sum(n for n, _ in prefixes) / len(prefixes)
    print(f"mean identical prefix {mean:.1f} tokens; {full}/{len(prefixes)} fully identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("out")
    cap.add_argument("--model", required=True, help="served model name")
    cap.add_argument("--url", default="http://localhost:8000/v1/completions")
    cap.add_argument("--max-tokens", type=int, default=64)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.url, args.model, args.max_tokens, args.out)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
