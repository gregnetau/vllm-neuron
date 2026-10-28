# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the stress scripts: the GSM8K loader and JSON files.

The scripts use GSM8K questions for accuracy runs and as filler text for prompts. Pass the
test split with ``--data`` as one of:

- a ``.json`` file holding a list of ``{"question": ..., "answer": ...}`` records;
- a ``.jsonl`` file with one such record per line;
- a ``.parquet`` file (needs ``pyarrow``).

Without ``--data`` the split is loaded with the ``datasets`` package, if installed.

To obtain the files:

    hf download openai/gsm8k --repo-type dataset --local-dir gsm8k
    # gsm8k/main/test-00000-of-00001.parquet; optionally convert it to JSON:
    python -c "import json, pyarrow.parquet as pq; json.dump(pq.read_table(
        'gsm8k/main/test-00000-of-00001.parquet').to_pylist(), open('gsm8k_test.json', 'w'))"
"""

import json
import pathlib


def load_gsm8k(path=None, split="test"):
    """Return the GSM8K records of ``split`` as a list of dicts (question, answer)."""
    if path is None:
        try:
            import datasets
        except ImportError:
            raise SystemExit(
                "GSM8K: pass --data <gsm8k test .json/.jsonl/.parquet> or install "
                "`datasets` (see stress_common.py for how to download the split)"
            ) from None
        return list(datasets.load_dataset("openai/gsm8k", "main", split=split))
    p = pathlib.Path(path)
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit(
                f"GSM8K: reading {p} needs pyarrow; convert it to JSON instead "
                "(see stress_common.py)"
            ) from None
        return pq.read_table(p).to_pylist()
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def add_data_arg(parser):
    parser.add_argument(
        "--data",
        default=None,
        help="GSM8K test split (.json list, .jsonl or .parquet); default: load "
        "openai/gsm8k with the `datasets` package",
    )


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
