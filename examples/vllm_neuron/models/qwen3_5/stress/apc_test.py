# SPDX-License-Identifier: Apache-2.0
"""Prefix-caching correctness / benefit test against a running vLLM server.

  ref:  capture greedy outputs (server WITHOUT prefix caching) for a fixed prompt set
  test: (server WITH prefix caching) run the same prompts sequentially, concurrently and
        repeated, compare token ids with the reference; teacher-force each divergence and
        classify it (tie vs CORRUPT); report TTFT cold vs warm and the server's prefix hit rate.

Prompt set: G groups; each group shares a long prefix (several cache blocks) and has M
distinct suffixes of varied length, so hits land on block boundaries of different depths.
Use a --prefix-len of several prefix-cache blocks (the block is 1024 or 2048 tokens for
Qwen3.8-27B, see the recipe). Prompts are slices of tokenized GSM8K questions (``--data``,
see ``stress_common.py``); ref and test must use the same --data, --seed and prompt options.
The server's ``Prefix cache hit rate`` is in its log.

Usage:
    D=examples/vllm_neuron/models/qwen3_5/stress
    # server without PREFIX_CACHING=1
    python $D/apc_test.py ref --ref apc_ref.json --data gsm8k_test.json
    # same configuration with PREFIX_CACHING=1; exit code 1 if any CORRUPT
    python $D/apc_test.py test --ref apc_ref.json --data gsm8k_test.json
"""

import argparse
import asyncio
import json
import random
import sys
import time

import aiohttp
from stress_common import add_data_arg, load_gsm8k, read_json, write_json


def build_prompts(base, seed, groups, members, prefix_len, max_suffix):
    rnd = random.Random(seed)
    out = []
    for g in range(groups):
        off = rnd.randint(0, len(base) - prefix_len - 1)
        prefix = base[off : off + prefix_len]
        for m in range(members):
            sl = rnd.randint(8, max_suffix)
            so = rnd.randint(0, len(base) - sl - 1)
            out.append({"group": g, "ids": prefix + base[so : so + sl]})
    return out


async def main(a):
    U = a.url
    data = load_gsm8k(a.data)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        m = (await (await s.get(f"{U}/v1/models")).json())["data"][0]["id"]

        async def tok(t):
            async with s.post(f"{U}/tokenize", json={"model": m, "prompt": t}) as r:
                return (await r.json())["tokens"]

        async def req(ids, mt, lp=None, delay=0.0):
            await asyncio.sleep(delay)
            b = {
                "model": m,
                "prompt": ids,
                "max_tokens": mt,
                "temperature": 0,
                "ignore_eos": True,
                "return_token_ids": True,
            }
            if lp:
                b["logprobs"] = lp
            t0 = time.perf_counter()
            async with s.post(f"{U}/v1/completions", json=b) as r:
                j = await r.json()
            if "choices" not in j:
                raise RuntimeError(str(j)[:300])
            c = j["choices"][0]
            c["_lat"] = time.perf_counter() - t0
            return c

        async def ttft(ids):
            b = {
                "model": m,
                "prompt": ids,
                "max_tokens": 2,
                "temperature": 0,
                "stream": True,
            }
            t0 = time.perf_counter()
            async with s.post(f"{U}/v1/completions", json=b) as r:
                async for line in r.content:
                    if line.startswith(b"data: ") and b"[DONE]" not in line:
                        return time.perf_counter() - t0

        base = await tok("\n\n".join(d["question"] for d in data[:600]))
        prompts = build_prompts(
            base, a.seed, a.groups, a.members, a.prefix_len, a.max_suffix
        )

        if a.cmd == "ref":
            res = [await req(p["ids"], a.max_tokens) for p in prompts]
            write_json(
                a.ref, [{**p, "out": r["token_ids"]} for p, r in zip(prompts, res)]
            )
            print(f"saved {len(res)} refs")
            return

        ref = read_json(a.ref)
        assert [r["ids"] for r in ref] == [p["ids"] for p in prompts], (
            "prompt set changed"
        )

        async def classify(i, got):
            want = ref[i]["out"]
            n = next((k for k, (x, y) in enumerate(zip(want, got)) if x != y), None)
            if n is None:
                return ("match" if len(want) == len(got) else "CORRUPT"), None
            r = await req(ref[i]["ids"] + want[:n], 1, 20)
            top = r["logprobs"]["top_logprobs"][0]
            async with s.post(
                f"{U}/detokenize", json={"model": m, "tokens": [got[n]]}
            ) as rr:
                ct = (await rr.json())["prompt"]
            lp = top.get(ct)
            gap = None if lp is None else max(top.values()) - lp
            kind = "tie" if gap is not None and gap <= a.tie else "CORRUPT"
            return kind, (n, ct, gap)

        async def phase(name, outs):
            st = {"match": 0, "tie": 0, "CORRUPT": 0}
            for i, o in enumerate(outs):
                kind, info = await classify(i, o["token_ids"])
                st[kind] += 1
                if kind == "CORRUPT":
                    print(f"   {name} prompt {i} (group {ref[i]['group']}): {info}")
            print(json.dumps({"phase": name, **st}), flush=True)
            return st["CORRUPT"]

        bad = 0
        # cold TTFT on a fresh group-specific prefix vs warm (same prefix, new suffix)
        fresh = build_prompts(base, a.seed + 99, 1, 2, a.prefix_len, a.max_suffix)
        cold = await ttft(fresh[0]["ids"])
        warm = await ttft(fresh[1]["ids"])
        print(
            json.dumps(
                {"ttft_cold_ms": round(cold * 1e3), "ttft_warm_ms": round(warm * 1e3)}
            )
        )
        bad += await phase(
            "sequential", [await req(p["ids"], a.max_tokens) for p in prompts]
        )
        bad += await phase(
            "concurrent",
            await asyncio.gather(
                *[
                    req(p["ids"], a.max_tokens, None, random.random() * a.jitter)
                    for p in prompts
                ]
            ),
        )
        bad += await phase(
            "repeat", [await req(p["ids"], a.max_tokens) for p in prompts]
        )
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_arg(p)
    p.add_argument("cmd", choices=["ref", "test"])
    p.add_argument("--ref", required=True)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--prefix-len", type=int, default=2500)
    p.add_argument("--max-suffix", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--jitter", type=float, default=2.0)
    p.add_argument("--tie", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(main(p.parse_args()))
