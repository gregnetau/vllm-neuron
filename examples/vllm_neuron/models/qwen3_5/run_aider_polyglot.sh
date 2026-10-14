#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Aider Polyglot benchmark against a running Qwen3.8 server (serve.sh), in Aider's benchmark
# Docker image. Runs detached from the terminal; all output goes to LOG.
#
# Setup (once):
#   git clone https://github.com/Aider-AI/aider && cd aider
#   mkdir -p tmp.benchmarks
#   git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
#   ./benchmark/docker_build.sh
#
# Usage:
#   AIDER_DIR=<path>/aider examples/vllm_neuron/models/qwen3_5/run_aider_polyglot.sh
#   tail -f "$LOG"                                    # progress (default below)
#   CONT=1 AIDER_DIR=... .../run_aider_polyglot.sh    # resume the same run after a stop
#
# Knobs: RUN_NAME (default qwen3.8-27b-neuron), THREADS (default 8, at most the server's
# MAX_NUM_SEQS), NUM_TESTS (default all 225), TRIES (default 2), MODEL_NAME (default
# openai/qwen3.8-27b: openai/<served model name>), API_BASE (default the host's port 8000),
# LOG (default /root/runs/aider_polyglot.log).
set -euo pipefail

AIDER_DIR=${AIDER_DIR:?set AIDER_DIR to the aider checkout}
RUN_NAME=${RUN_NAME:-qwen3.8-27b-neuron}
LOG=${LOG:-/root/runs/aider_polyglot.log}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cp "${HERE}/aider-model-settings.yml" "${AIDER_DIR}/aider-model-settings.yml"
mkdir -p "$(dirname "${LOG}")"

BENCH_ARGS=(--model "${MODEL_NAME:-openai/qwen3.8-27b}" --read-model-settings aider-model-settings.yml
            --edit-format diff --threads "${THREADS:-8}" --tries "${TRIES:-2}"
            --exercises-dir polyglot-benchmark)
[[ -z "${NUM_TESTS:-}" ]] || BENCH_ARGS+=(--num-tests "${NUM_TESTS}")
[[ "${CONT:-0}" != 1 ]] || BENCH_ARGS+=(--cont)

nohup docker run --rm --name aider-polyglot --memory=12g --memory-swap=12g \
    --add-host=host.docker.internal:host-gateway \
    -v "${AIDER_DIR}":/aider -v "${AIDER_DIR}/tmp.benchmarks/.":/benchmarks \
    -e OPENAI_API_KEY=none -e OPENAI_API_BASE="${API_BASE:-http://host.docker.internal:8000/v1}" \
    -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks -e PYTHONUNBUFFERED=1 aider-benchmark \
    bash -c "pip install -q -e .[dev] && ./benchmark/benchmark.py ${RUN_NAME} ${BENCH_ARGS[*]}" \
    >> "${LOG}" 2>&1 &

echo "Aider Polyglot started (container aider-polyglot); log: ${LOG}"
echo "Results so far: docker exec aider-polyglot ./benchmark/benchmark.py --stats /benchmarks/*--${RUN_NAME}"
