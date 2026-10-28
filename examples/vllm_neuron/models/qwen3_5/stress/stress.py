# SPDX-License-Identifier: Apache-2.0
"""Stress / validation client for a vLLM OpenAI-compatible server (Qwen3.5 / Qwen3.8 on
Neuron).

Subcommands:
  gsm8k        accuracy at a given concurrency (greedy, chat API, thinking off)
  ref          capture greedy token ids for a prompt set of mixed lengths, one request at
               a time (the reference)
  consistency  the same prompts at concurrency C with jittered arrivals, compared token for
               token with the reference
  chaos        random lengths, sampling parameters (incl. seeds, logprobs, n > 1, stop
               strings) and client aborts; then optionally a greedy re-check against a
               reference. Exits 1 on server errors or malformed responses.

Prompts are built from GSM8K questions (``--data``, see ``stress_common.py``). Run against a
server started with ``examples/vllm_neuron/models/qwen3_5/serve.sh`` (``srv.sh`` here
starts and stops one); requires ``aiohttp``.

Usage:
    D=examples/vllm_neuron/models/qwen3_5/stress
    python $D/stress.py --data gsm8k_test.json gsm8k --n 1319 --concurrency 8 --out gsm8k.json
    python $D/stress.py --data gsm8k_test.json ref --n 32 --out ref.json
    python $D/stress.py --data gsm8k_test.json consistency --ref ref.json --concurrency 8
    python $D/stress.py --data gsm8k_test.json chaos --n 200 --concurrency 16 --ref ref.json

A divergence under ``consistency`` is not necessarily a bug: FP8 graphs can pick different
tokens at near-ties. Classify divergences by teacher forcing (``nearmax_random.py`` and
``apc_test.py`` do this on the server; ``hf_ref_pos.py`` checks a position against the
Hugging Face BF16 model on CPU).
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time

import aiohttp
from stress_common import add_data_arg, load_gsm8k, read_json, write_json

URL = "http://localhost:8000"


async def model_name(s):
    async with s.get(f"{URL}/v1/models") as r:
        return (await r.json())["data"][0]["id"]


async def complete(s, model, prompt, max_tokens, chat=False, timeout=None, **kw):
    body = {"model": model, "max_tokens": max_tokens, **kw}
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
        path = "/v1/completions"
    t0 = time.time()
    async with s.post(
        f"{URL}{path}", json=body, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as r:
        txt = await r.text()
        if r.status != 200:
            return {"error": f"HTTP {r.status}: {txt[:300]}", "lat": time.time() - t0}
        j = json.loads(txt)
    c = j["choices"][0]
    out = {
        "text": c["message"]["content"] if chat else c["text"],
        "finish": c.get("finish_reason"),
        "token_ids": c.get("token_ids"),
        "usage": j.get("usage"),
        "lat": time.time() - t0,
    }
    return out


def gsm_answer(text):
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if not m:
        m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not m:
        m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    v = m[-1].replace(",", "").strip().rstrip(".")
    try:
        return float(v)
    except ValueError:
        return None


async def run_pool(items, conc, fn, jitter=0.0):
    sem = asyncio.Semaphore(conc)
    res = [None] * len(items)

    async def one(i, it):
        if jitter:
            await asyncio.sleep(random.random() * jitter)
        async with sem:
            res[i] = await fn(it)

    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return res


async def cmd_gsm8k(a):
    data = load_gsm8k(a.data)[: a.n]
    async with aiohttp.ClientSession() as s:
        model = await model_name(s)
        t0 = time.time()

        async def fn(d):
            p = (
                d["question"]
                + "\nSolve step by step, then give the final answer on the last line "
                "as '#### <number>'."
            )
            return await complete(s, model, p, a.max_tokens, chat=True, temperature=0)

        res = await run_pool(data, a.concurrency, fn)
        dt = time.time() - t0
    ok = errs = 0
    rows = []
    for d, r in zip(data, res):
        gold = float(d["answer"].split("####")[-1].strip().replace(",", ""))
        if "error" in r:
            errs += 1
            rows.append({"q": d["question"], "error": r["error"]})
            continue
        pred = gsm_answer(r["text"])
        good = pred is not None and abs(pred - gold) < 1e-6
        ok += good
        rows.append(
            {
                "q": d["question"],
                "gold": gold,
                "pred": pred,
                "ok": good,
                "text": r["text"],
                "finish": r["finish"],
            }
        )
    toks = sum(
        (r.get("usage") or {}).get("completion_tokens", 0)
        for r in res
        if "error" not in r
    )
    summary = {
        "n": len(data),
        "acc": ok / len(data),
        "errors": errs,
        "concurrency": a.concurrency,
        "secs": round(dt, 1),
        "out_tok_s": round(toks / dt, 1),
    }
    print(json.dumps(summary))
    if a.out:
        write_json(a.out, {"summary": summary, "rows": rows}, indent=1)


def consistency_prompts(data, n, seed=0, max_words=1500):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        # mixed lengths: short, medium, long (concatenated questions)
        k = rnd.choice([1, 1, 1, 3, 8, 20, 40])
        qs = [rnd.choice(data)["question"] for _ in range(k)]
        text = "\n\n".join(qs)
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
        out.append(
            "Read the following and then answer the last question.\n\n"
            + text
            + "\nAnswer:"
        )
    return out


async def cmd_ref(a):
    prompts = consistency_prompts(load_gsm8k(a.data), a.n, a.seed, a.max_words)
    async with aiohttp.ClientSession() as s:
        model = await model_name(s)
        res = await run_pool(
            prompts,
            1,
            lambda p: complete(
                s, model, p, a.max_tokens, temperature=0, return_token_ids=True
            ),
        )
    write_json(a.out, [{"prompt": p, **r} for p, r in zip(prompts, res)])
    print(f"saved {len(res)} refs to {a.out}; errors={sum('error' in r for r in res)}")


def compare(ref, res):
    exact = 0
    divs = []
    bad = []
    for i, (r0, r1) in enumerate(zip(ref, res)):
        if "error" in r1:
            bad.append((i, r1["error"]))
            continue
        a, b = r0["token_ids"], r1["token_ids"]
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        if n == len(a) == len(b):
            exact += 1
        else:
            divs.append((i, n, len(a), len(b)))
    return exact, divs, bad


async def cmd_consistency(a):
    ref = read_json(a.ref)
    prompts = [r["prompt"] for r in ref]
    async with aiohttp.ClientSession() as s:
        model = await model_name(s)
        for rep in range(a.reps):
            t0 = time.time()
            res = await run_pool(
                prompts,
                a.concurrency,
                lambda p: complete(
                    s, model, p, a.max_tokens, temperature=0, return_token_ids=True
                ),
                jitter=a.jitter,
            )
            exact, divs, bad = compare(ref, res)
            early = [d for d in divs if d[1] < a.early]
            print(
                json.dumps(
                    {
                        "rep": rep,
                        "conc": a.concurrency,
                        "n": len(prompts),
                        "exact": exact,
                        "diverged": len(divs),
                        f"early_div(<{a.early})": len(early),
                        "errors": len(bad),
                        "secs": round(time.time() - t0, 1),
                    }
                )
            )
            for d in sorted(divs, key=lambda x: x[1])[: a.show]:
                i, n, la, lb = d
                print(
                    f"   prompt {i} diverges at {n} (ref len {la}, got {lb}): "
                    f"ref={ref[i]['text'][:80]!r} got={res[i]['text'][:80]!r}"
                )
            for i, e in bad[:5]:
                print("   ERR", i, e)
            if a.out:
                write_json(f"{a.out}.{rep}.json", res)


async def cmd_chaos(a):
    rnd = random.Random(a.seed)
    data = load_gsm8k(a.data)
    stats = {"ok": 0, "aborted": 0, "error": 0, "bad": 0}
    errors = []

    async def one(s, model, i):
        k = rnd.choice([1, 1, 2, 5, 15, 40, 80])
        prompt = "\n\n".join(rnd.choice(data)["question"] for _ in range(k))
        words = prompt.split()
        prompt = " ".join(words[: a.max_words])
        mt = rnd.choice([1, 2, 8, 64, 256, a.max_tokens])
        kw = rnd.choice(
            [
                {"temperature": 0},
                {"temperature": 0.8, "top_p": 0.95},
                {"temperature": 1.0, "top_k": 20},
                {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "seed": i},
                {"temperature": 0, "logprobs": 5},
                {"temperature": 0.7, "presence_penalty": 1.0},
                {"temperature": 0, "stop": ["\n\n"]},
                {"temperature": 0.9, "n": 2},
            ]
        )
        chat = rnd.random() < 0.5
        if "logprobs" in kw and chat:
            kw = {"temperature": 0, "logprobs": True, "top_logprobs": 5}
        abort_after = rnd.random() * 3.0 if rnd.random() < a.abort_frac else None
        try:
            coro = complete(s, model, prompt, mt, chat=chat, **kw)
            if abort_after is not None:
                r = await asyncio.wait_for(coro, abort_after)
            else:
                r = await asyncio.wait_for(coro, 1800)
        except TimeoutError:
            stats["aborted"] += 1
            return
        except Exception as e:  # noqa: BLE001
            stats["error"] += 1
            errors.append(repr(e)[:300])
            return
        if "error" in r:
            # 400s for over-length prompts are fine
            if "HTTP 400" in r["error"]:
                stats["ok"] += 1
            else:
                stats["error"] += 1
                errors.append(r["error"])
            return
        u = r.get("usage") or {}
        if r["finish"] not in ("stop", "length", "abort") or (
            r["finish"] == "length"
            and u.get("completion_tokens", 0) < mt * kw.get("n", 1)
        ):
            stats["bad"] += 1
            errors.append(f"bad finish {r['finish']} usage={u} mt={mt}")
        elif "!!!!!!!!" in (r["text"] or ""):
            stats["bad"] += 1
            errors.append(f"garbage text: {r['text'][:200]!r}")
        else:
            stats["ok"] += 1

    async with aiohttp.ClientSession() as s:
        model = await model_name(s)
        t0 = time.time()
        sem = asyncio.Semaphore(a.concurrency)

        async def guarded(i):
            await asyncio.sleep(rnd.random() * a.duration_spread)
            async with sem:
                await one(s, model, i)

        await asyncio.gather(*(guarded(i) for i in range(a.n)))
        print(json.dumps({"chaos": stats, "secs": round(time.time() - t0, 1)}))
        for e in errors[:20]:
            print("   ", e)
        if a.ref:
            ref = read_json(a.ref)
            res = await run_pool(
                [r["prompt"] for r in ref],
                a.recheck_conc,
                lambda p: complete(
                    s,
                    model,
                    p,
                    len(ref[0]["token_ids"]),
                    temperature=0,
                    return_token_ids=True,
                ),
            )
            exact, divs, bad = compare(ref, res)
            print(
                json.dumps(
                    {
                        "post_chaos_recheck": {
                            "exact": exact,
                            "n": len(ref),
                            "diverged": [(d[0], d[1]) for d in divs],
                            "errors": len(bad),
                        }
                    }
                )
            )
    sys.exit(1 if stats["error"] or stats["bad"] else 0)


def main():
    global URL
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=URL)
    add_data_arg(p)
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gsm8k")
    g.add_argument("--n", type=int, default=1319)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--max-tokens", type=int, default=768)
    g.add_argument("--out")
    r = sp.add_parser("ref")
    r.add_argument("--n", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-words", type=int, default=1500)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--out", required=True)
    c = sp.add_parser("consistency")
    c.add_argument("--ref", required=True)
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--jitter", type=float, default=5.0)
    c.add_argument("--reps", type=int, default=1)
    c.add_argument("--early", type=int, default=16)
    c.add_argument("--show", type=int, default=5)
    c.add_argument("--out")
    h = sp.add_parser("chaos")
    h.add_argument("--n", type=int, default=200)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--concurrency", type=int, default=16)
    h.add_argument("--max-words", type=int, default=1500)
    h.add_argument("--max-tokens", type=int, default=512)
    h.add_argument("--abort-frac", type=float, default=0.25)
    h.add_argument("--duration-spread", type=float, default=60.0)
    h.add_argument("--ref")
    h.add_argument("--recheck-conc", type=int, default=4)
    a = p.parse_args()
    URL = a.url
    asyncio.run(
        {
            "gsm8k": cmd_gsm8k,
            "ref": cmd_ref,
            "consistency": cmd_consistency,
            "chaos": cmd_chaos,
        }[a.cmd](a)
    )


if __name__ == "__main__":
    main()
