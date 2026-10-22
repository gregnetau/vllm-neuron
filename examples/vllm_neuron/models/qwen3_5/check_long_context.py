# SPDX-License-Identifier: Apache-2.0
"""Long-context retrieval check (passkey in a haystack) against a running server.

Hides a random 6-digit passkey at several depths of a filler document of about
``--context`` tokens and asks for it with greedy chat completions (thinking disabled).
Reports, per depth, whether the answer contains the passkey and the time to the first
token. Exits non-zero if any retrieval fails.

Usage:
    python examples/vllm_neuron/models/qwen3_5/check_long_context.py \
        --tokenizer <path-to-checkpoint>/Qwen3.8-27B --model Qwen3.8-27B \
        --context 30000 --depths 0.1,0.5,0.9
"""

import argparse
import json
import random
import sys
import time
import urllib.request

import transformers

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back "
    "again. "
)
QUESTION = "What is the passkey mentioned in the document? Answer with the number only."


def build_prompt(tok, context: int, depth: float, passkey: str) -> str:
    unit = len(tok.encode(FILLER, add_special_tokens=False))
    n = max(1, context // unit)
    needle = f"The passkey is {passkey}. Remember it. The passkey is {passkey}. "
    at = int(n * depth)
    return FILLER * at + needle + FILLER * (n - at)


def ask(url: str, model: str, prompt: str, max_tokens: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + "\n\n" + QUESTION}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0, ttft, text, usage = time.perf_counter(), None, "", {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content") or ""
                if delta and ttft is None:
                    ttft = time.perf_counter() - t0
                text += delta
    return text, ttft, usage.get("prompt_tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--context", type=int, default=30000, help="approximate prompt tokens"
    )
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    failures = 0
    for depth in (float(d) for d in args.depths.split(",")):
        passkey = str(rng.randint(100000, 999999))
        text, ttft, n_prompt = ask(
            args.url,
            args.model,
            build_prompt(tok, args.context, depth, passkey),
            args.max_tokens,
        )
        ok = passkey in text
        failures += not ok
        print(
            f"context {n_prompt} tokens, depth {depth:.2f}: {'OK  ' if ok else 'FAIL'} "
            f"passkey {passkey} answer {text.strip()!r} TTFT {ttft:.2f} s",
            flush=True,
        )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
