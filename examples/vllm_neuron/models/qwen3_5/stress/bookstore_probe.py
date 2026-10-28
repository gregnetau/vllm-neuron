# SPDX-License-Identifier: Apache-2.0
"""Repeat the Aider Polyglot book-store exercise directly against the server and measure
long-thinking behaviour: completion length, max_tokens hits, whether the thinking closes
(``</think>``), and for Python whether the produced solution passes the exercise's tests.

Thinking mode with Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20), as
in ``aider-model-settings.yml``. Compare two server configurations (for example with and
without prefix caching) by running it against each with a different --tag.

Needs a checkout of https://github.com/Aider-AI/polyglot-benchmark (``--polyglot-dir``).
The Python tests run the model's generated code locally (``python3 -I -m unittest`` in a
temporary directory): run this in a disposable environment.

Usage:
    python examples/vllm_neuron/models/qwen3_5/stress/bookstore_probe.py \
        --polyglot-dir <aider>/tmp.benchmarks/polyglot-benchmark --tag apc --samples 8 \
        --max-tokens 32768
"""

import argparse
import asyncio
import json
import pathlib
import re
import subprocess
import tempfile
import time

import aiohttp
from stress_common import write_json

BASE = None  # --polyglot-dir
LANGS = {
    "python": ("python/exercises/practice/book-store", "book_store.py"),
    "javascript": ("javascript/exercises/practice/book-store", "book-store.js"),
}

SYSTEM = (
    "Act as an expert software developer. Always use best practices when coding. "
    "Respect and use existing conventions, libraries, etc that are already present in the code base."
)


def prompt(lang):
    d, fname = LANGS[lang]
    ex = pathlib.Path(BASE, d)
    ins = (ex / ".docs" / "instructions.md").read_text()
    stub = (ex / fname).read_text()
    return (
        f"{ins}\n\nUse the above instructions to modify the supplied files: {fname}\n"
        "Don't change the names of existing functions or classes, as they may be referenced "
        "from other code like unit tests, etc.\nOnly use standard libraries, don't suggest "
        f"installing any packages.\n\n{fname}\n```\n{stub}\n```\n\n"
        f"Reply with the complete new content of {fname} in a single fenced code block."
    )


def extract_code(text):
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


def run_python_tests(code):
    d, fname = LANGS["python"]
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / fname).write_text(code)
        test = pathlib.Path(BASE, d, "book_store_test.py").read_text()
        (tmp / "book_store_test.py").write_text(test)
        r = subprocess.run(
            ["python3", "-I", "-m", "unittest", "-q", "book_store_test"],
            check=False,
            cwd=td,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.returncode == 0


async def main(a):
    global BASE
    BASE = a.polyglot_dir
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        m = (await (await s.get(f"{a.url}/v1/models")).json())["data"][0]["id"]

        async def one(lang, i):
            body = {
                "model": m,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(lang)},
                ],
                "max_tokens": a.max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            }
            t0 = time.time()
            async with s.post(f"{a.url}/v1/chat/completions", json=body) as r:
                j = await r.json()
            c = j["choices"][0]
            msg = c["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            full = reasoning + content
            closed = ("</think>" in full) or bool(reasoning and content)
            code = extract_code(content)
            ok = run_python_tests(code) if (lang == "python" and code) else None
            return {
                "lang": lang,
                "i": i,
                "tokens": j["usage"]["completion_tokens"],
                "finish": c["finish_reason"],
                "closed_think": closed,
                "has_code": code is not None,
                "tests_pass": ok,
                "secs": round(time.time() - t0),
            }

        tasks = [one(lang, i) for lang in LANGS for i in range(a.samples)]
        res = await asyncio.gather(*tasks)
    out = a.out or f"bookstore_{a.tag}.json"
    write_json(out, res, indent=1)
    for lang in LANGS:
        rs = [r for r in res if r["lang"] == lang]
        toks = sorted(r["tokens"] for r in rs)
        print(
            json.dumps(
                {
                    "tag": a.tag,
                    "lang": lang,
                    "n": len(rs),
                    "median_tokens": toks[len(toks) // 2],
                    "max_tokens_hit": sum(r["finish"] == "length" for r in rs),
                    "with_code": sum(r["has_code"] for r in rs),
                    "tests_pass": sum(1 for r in rs if r["tests_pass"]),
                    "tokens": toks,
                }
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--polyglot-dir",
        required=True,
        help="checkout of Aider-AI/polyglot-benchmark",
    )
    p.add_argument("--out", help="result JSON (default: bookstore_<tag>.json)")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--tag", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32768)
    asyncio.run(main(p.parse_args()))
