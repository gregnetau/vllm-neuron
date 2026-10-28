# SPDX-License-Identifier: Apache-2.0
"""Randomized near-max_model_len / arrival-timing stress with a corruption classifier.

Many greedy requests whose prompt + max_tokens reach max_model_len (or close), arriving
with random jitter, plus short "disturber" requests whose prefills hold the others back.
Each long request's concurrent output is compared with its sequential output; at the
first divergence the prompt + sequential prefix is teacher-forced (prefill, logprobs) and
the divergence is classified:
  tie      - the concurrent token is within --tie nats of the argmax (numerics)
  CORRUPT  - the concurrent token is far below the argmax (or not in the top 20)
Exit code 1 if any CORRUPT.

With --mid most long requests stop well short of max_model_len (mid-sequence requests),
and --n-near of them still run to it. Prompts are slices of tokenized GSM8K questions
(``--data``, see ``stress_common.py``). ``--max-len`` must equal the server's max_model_len.

Usage:
    python examples/vllm_neuron/models/qwen3_5/stress/nearmax_random.py \
        --data gsm8k_test.json --max-len 4096 --n-long 6 --n-short 6 --reps 3
    # mid-sequence requests, two of them running to max_model_len
    python examples/vllm_neuron/models/qwen3_5/stress/nearmax_random.py \
        --data gsm8k_test.json --max-len 4096 --mid --n-near 2

With FP8 weights raise --tie (for example 0.8): FP8 graphs differ more at near-ties.
"""

import argparse
import asyncio
import json
import random
import sys

import aiohttp
from stress_common import add_data_arg, load_gsm8k


async def main(a):
    rnd = random.Random(a.seed)
    data = load_gsm8k(a.data)
    U = a.url
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        m = (await (await s.get(f"{U}/v1/models")).json())["data"][0]["id"]

        async def tok(t):
            async with s.post(f"{U}/tokenize", json={"model": m, "prompt": t}) as r:
                return (await r.json())["tokens"]

        base = await tok("\n\n".join(d["question"] for d in data[:400]))

        async def req(p, mt, lp=None, delay=0.0):
            await asyncio.sleep(delay)
            b = {
                "model": m,
                "prompt": p,
                "max_tokens": mt,
                "temperature": 0,
                "ignore_eos": True,
                "return_token_ids": True,
            }
            if lp:
                b["logprobs"] = lp
            async with s.post(f"{U}/v1/completions", json=b) as r:
                j = await r.json()
                if "choices" not in j:
                    raise RuntimeError(str(j)[:300])
                return j["choices"][0]

        longs = []
        for i in range(a.n_long):
            if a.mid and i >= a.n_near:
                plen = rnd.randint(200, a.max_len // 2)
                off = rnd.randint(0, len(base) - plen - 1)
                longs.append((base[off : off + plen], rnd.randint(300, 700)))
                continue
            remaining = rnd.choice([12, 20, 30, 45, 60, 90, 150])
            plen = rnd.randint(a.max_len // 2, a.max_len - remaining - 1)
            off = rnd.randint(0, len(base) - plen - 1)
            longs.append(
                (
                    base[off : off + plen],
                    a.max_len - plen
                    if rnd.random() < 0.8
                    else a.max_len - plen - rnd.randint(1, 5),
                )
            )
        seq = [(await req(p, mt))["token_ids"] for p, mt in longs]
        print("sequential done", flush=True)
        bad = 0
        for rep in range(a.reps):
            tasks = [req(p, mt, None, rnd.random() * a.jitter) for p, mt in longs]
            # disturbers: short prompts arriving throughout, small outputs
            for _ in range(a.n_short):
                plen = rnd.randint(20, 1500)
                off = rnd.randint(0, len(base) - plen - 1)
                tasks.append(
                    req(
                        base[off : off + plen],
                        rnd.choice([1, 4, 16]),
                        None,
                        rnd.random() * a.jitter * 1.5,
                    )
                )
            res = await asyncio.gather(*tasks)
            con = [r["token_ids"] for r in res[: a.n_long]]
            stats = {"match": 0, "tie": 0, "CORRUPT": 0}
            for i, (sq, cn) in enumerate(zip(seq, con)):
                n = next((k for k, (x, y) in enumerate(zip(sq, cn)) if x != y), None)
                if n is None and len(sq) == len(cn):
                    stats["match"] += 1
                    continue
                if n is None:
                    stats["CORRUPT"] += 1
                    print(f"  rep{rep} req{i}: length {len(sq)} vs {len(cn)}")
                    continue
                r = await req(longs[i][0] + sq[:n], 1, 20)
                top = r["logprobs"]["top_logprobs"][0]
                # map token ids via a second call is costly; compare by detokenized strings
                async with s.post(
                    f"{U}/detokenize", json={"model": m, "tokens": [cn[n]]}
                ) as rr:
                    con_tok = (await rr.json())["prompt"]
                best = max(top.values())
                lp = top.get(con_tok)
                gap = None if lp is None else best - lp
                kind = "tie" if gap is not None and gap <= a.tie else "CORRUPT"
                stats[kind] += 1
                if kind == "CORRUPT":
                    print(
                        f"  rep{rep} req{i}: diverges at {n}/{len(sq)} (plen {len(longs[i][0])}) "
                        f"con={con_tok!r} gap={gap} top={ {k: round(v, 2) for k, v in list(top.items())[:4]} }"
                    )
            print(json.dumps({"rep": rep, **stats}), flush=True)
            bad += stats["CORRUPT"]
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_arg(p)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument(
        "--max-len", type=int, default=4096, help="the server's max_model_len"
    )
    p.add_argument("--n-long", type=int, default=6)
    p.add_argument("--n-short", type=int, default=6)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--jitter", type=float, default=6.0)
    p.add_argument("--tie", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mid", action="store_true", help="stop well short of max_model_len"
    )
    p.add_argument(
        "--n-near",
        type=int,
        default=0,
        help="with --mid: this many requests still run to max_model_len",
    )
    asyncio.run(main(p.parse_args()))
