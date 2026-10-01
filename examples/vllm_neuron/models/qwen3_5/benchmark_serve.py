# SPDX-License-Identifier: Apache-2.0
"""Single-stream latency benchmark against a running OpenAI-compatible server.

Sends an exact ``--input-len``-token prompt as token ids with ``ignore_eos`` and streams
``--output-len`` tokens; reports mean TTFT, TPOT (time per output token after the first)
and end-to-end latency over ``--rounds`` timed rounds after one warmup round.

Usage:
    python examples/vllm_neuron/models/qwen3_5/benchmark_serve.py \
        --tokenizer <path-to-checkpoint>/Qwen3.8-27B --model Qwen3.8-27B \
        --input-len 1024 --output-len 128 --json results.json
"""

import argparse
import json
import statistics
import time
import urllib.request

import transformers

TEXT = (
    "The history of science is the study of the development of science and scientific "
    "knowledge, including both the natural and social sciences. Science is a body of "
    "empirical, theoretical, and practical knowledge about the natural world, produced by "
    "scientists who emphasize the observation, explanation, and prediction of real-world "
    "phenomena. "
)


def _post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def timed_request(url: str, model: str, ids: list[int], output_len: int) -> dict:
    body = {"model": model, "prompt": ids, "max_tokens": output_len, "temperature": 0,
            "ignore_eos": True, "stream": True}
    start = time.perf_counter()
    first, chunks = None, 0
    with _post(url, body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            if json.loads(line[5:])["choices"][0].get("text") is not None:
                first = first or time.perf_counter()
                chunks += 1
    end = time.perf_counter()
    return {"ttft_s": first - start, "tpot_s": (end - first) / max(output_len - 1, 1),
            "e2e_s": end - start, "chunks": chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(TEXT * (args.input_len // 40 + 1)).input_ids[: args.input_len]
    if len(ids) != args.input_len:
        raise ValueError(f"prompt has {len(ids)} tokens, expected {args.input_len}")

    timed_request(args.url, args.model, ids, args.output_len)  # warmup
    runs = [timed_request(args.url, args.model, ids, args.output_len) for _ in range(args.rounds)]
    result = {
        "input_len": args.input_len, "output_len": args.output_len, "rounds": args.rounds,
        "ttft_ms": round(statistics.mean(r["ttft_s"] for r in runs) * 1e3, 1),
        "tpot_ms": round(statistics.mean(r["tpot_s"] for r in runs) * 1e3, 2),
        "e2e_ms": round(statistics.mean(r["e2e_s"] for r in runs) * 1e3, 1),
        "runs": runs,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
