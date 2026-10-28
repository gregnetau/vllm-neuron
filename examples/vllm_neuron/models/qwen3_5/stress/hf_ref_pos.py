# SPDX-License-Identifier: Apache-2.0
"""Next-token log-probabilities of the Hugging Face BF16 model at one position of a
reference output, on CPU.

Use it to judge a greedy divergence found by the stress scripts: if the Hugging Face BF16
model also rates the two diverging tokens as near-equal, the divergence is numerics (FP8
graphs can pick different tokens at near-ties); a token it gives a very low log-probability
is corruption.

The reference is a JSON list written by ``apc_test.py ref`` (records with ``ids``, the
prompt token ids, and ``out``, the generated ids) or by ``stress.py ref`` (``prompt`` text
and ``token_ids``). The context is the prompt plus the first ``position`` generated tokens.
The model is loaded in BF16 (2 bytes of host memory per parameter).

Usage:
    python examples/vllm_neuron/models/qwen3_5/stress/hf_ref_pos.py \
        --model <path-to-checkpoint>/Qwen3.8-27B --ref apc_ref.json --index 3 \
        --position 41 --candidates " GSM" "A"
"""

import argparse
import json
import time

import torch
import transformers


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Hugging Face checkpoint directory")
    p.add_argument(
        "--ref", required=True, help="reference JSON (apc_test / stress ref)"
    )
    p.add_argument("--index", type=int, required=True, help="record in the reference")
    p.add_argument(
        "--position", type=int, required=True, help="generated tokens before the probe"
    )
    p.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="strings whose first token's log-probability to print",
    )
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()

    with open(a.ref) as f:
        rec = json.load(f)[a.index]
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    if "ids" in rec:
        prompt, out = rec["ids"], rec["out"]
    else:
        prompt, out = tok.encode(rec["prompt"]), rec["token_ids"]
    ids = prompt + out[: a.position]

    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16
    )
    model.eval()
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f} s", flush=True)
    with torch.no_grad():
        logits = model(torch.tensor([ids]), logits_to_keep=1).logits
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    top = torch.topk(lp, a.top)
    print(
        f"HF BF16 next token after {a.position} generated tokens:",
        [
            (tok.decode([int(t)]), round(float(v), 2))
            for v, t in zip(top.values, top.indices)
        ],
    )
    if a.position < len(out):
        ref_tok = out[a.position]
        print(f"reference token {tok.decode([ref_tok])!r}: {float(lp[ref_tok]):.2f}")
    for s in a.candidates:
        tid = tok.encode(s, add_special_tokens=False)
        print(f"{s!r} {tid}: {float(lp[tid[0]]):.2f}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
