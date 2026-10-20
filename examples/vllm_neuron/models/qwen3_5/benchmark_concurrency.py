# SPDX-License-Identifier: Apache-2.0
"""Concurrent streaming benchmark against a running OpenAI-compatible server.

Starts ``--concurrency`` identical requests at once (exact ``--input-len``-token prompts as
token ids, ``ignore_eos``, ``--output-len`` tokens each) and reports the aggregate output
tokens/s over the whole run, plus per-request TTFT (including the wait for the other
requests' prefills) and TPOT. With MTP a streamed chunk carries several tokens, so token
counts come from ``--output-len`` (exact with ``ignore_eos``), not from the chunks.

Usage:
    python examples/vllm_neuron/models/qwen3_5/benchmark_concurrency.py \
        --tokenizer <path-to-checkpoint>/Qwen3.8-27B --model Qwen3.8-27B \
        --concurrency 4 --input-len 1000 --output-len 512
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

import transformers

TEXT = "The history of science is the study of the development of scientific knowledge. "


def stream(url, model, prompt_ids, output_len, out, i):
    body = {"model": model, "prompt": prompt_ids, "max_tokens": output_len, "temperature": 0,
            "ignore_eos": True, "stream": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, times = time.perf_counter(), []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                if json.loads(line[6:])["choices"][0].get("text"):
                    times.append(time.perf_counter())
    out[i] = (t0, times)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=1000)
    ap.add_argument("--output-len", type=int, default=512)
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok.encode(TEXT * (args.input_len // 10 + 1), add_special_tokens=False)
    out = [None] * args.concurrency
    threads = []
    for i in range(args.concurrency):
        # Distinct first token per request: no shared prefix between requests.
        prompt = [ids[i]] + ids[args.concurrency:args.concurrency + args.input_len - 1]
        threads.append(threading.Thread(target=stream, args=(args.url, args.model, prompt,
                                                             args.output_len, out, i)))
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.perf_counter()
    n_tokens = args.concurrency * args.output_len
    ttft = [times[0] - t0 for t0, times in out]
    tpot = [(times[-1] - times[0]) / max(1, args.output_len - 1) for _, times in out]
    print(json.dumps({
        "concurrency": args.concurrency, "input_len": args.input_len, "output_len": args.output_len,
        "aggregate_tokens_per_s": round(n_tokens / (end - start), 1),
        "ttft_s": [round(x, 2) for x in ttft],
        "tpot_ms_mean": round(1e3 * statistics.mean(tpot), 2),
        "tpot_ms_range": [round(1e3 * min(tpot), 2), round(1e3 * max(tpot), 2)],
    }, indent=1))


if __name__ == "__main__":
    main()
