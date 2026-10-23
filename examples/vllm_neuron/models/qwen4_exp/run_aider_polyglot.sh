#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Aider Polyglot benchmark against a running Qwen3.8-Flash-Next OpenAI-compatible server
# (vllm-neuron), in Aider's benchmark Docker image. Runs detached from the terminal; all output
# goes to LOG.
#
# Setup (once):
#   git clone https://github.com/Aider-AI/aider && cd aider
#   git checkout 5dc9490                              # Aider 0.86.3.dev, as for qwen3_5
#   mkdir -p tmp.benchmarks
#   git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
#   ./benchmark/docker_build.sh
#
# Usage:
#   AIDER_DIR=<path>/aider examples/vllm_neuron/models/qwen4_exp/run_aider_polyglot.sh
#   tail -f "$LOG"                                    # progress (default below)
#   CONT=1 AIDER_DIR=... .../run_aider_polyglot.sh    # resume the same run after a stop
#   STATS=1 AIDER_DIR=... .../run_aider_polyglot.sh   # print the run's results so far
#   docker stop aider-polyglot                        # stop the run (resume with CONT=1)
#
# Knobs: RUN_NAME (default qwen3.8-flash-next-neuron), THREADS (default 8, at most the server's
# MAX_NUM_SEQS), NUM_TESTS (default all 225), TRIES (default 2), MODEL_NAME (default
# openai/qwen3.8-flash-next: openai/<served model name>), API_BASE (default the host's port 8000),
# LOG (default <AIDER_DIR>/aider_polyglot.log).
# Reasoning controls, passed as chat_template_kwargs (unset: the chat template's default):
#   REASONING_EFFORT   xhigh (default) | medium | low; ignored when thinking is off.
#   ENABLE_THINKING    true (default) | false. With false, also consider the model card's
#                      non-thinking sampling, which aider-model-settings.yml does not switch to.
#   PRESERVE_THINKING  true (default) | false: keep <think> blocks of earlier assistant turns.
#                      Aider resends earlier turns without their thinking, so with true the
#                      template renders them with an empty <think></think> block.
# Use a different RUN_NAME per reasoning configuration: CONT and STATS find the run by name.
set -euo pipefail

AIDER_DIR=${AIDER_DIR:?set AIDER_DIR to the aider checkout}
AIDER_DIR=$(cd "${AIDER_DIR}" && pwd)
RUN_NAME=${RUN_NAME:-qwen3.8-flash-next-neuron}
MODEL_NAME=${MODEL_NAME:-openai/qwen3.8-flash-next}
LOG=${LOG:-${AIDER_DIR}/aider_polyglot.log}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER=aider-polyglot
DOCKER_ARGS=(-v "${AIDER_DIR}":/aider -v "${AIDER_DIR}/tmp.benchmarks/.":/benchmarks
             -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks -e PYTHONUNBUFFERED=1)

# Results so far: run benchmark.py --stats in the running benchmark container, or in a new one
# after the run has finished. benchmark.py finds the run directory by its name.
if [[ "${STATS:-0}" == 1 ]]; then
    STATS_CMD="./benchmark/benchmark.py --stats ${RUN_NAME}"
    if docker container inspect "${CONTAINER}" > /dev/null 2>&1; then
        exec docker exec "${CONTAINER}" bash -c "${STATS_CMD}"
    fi
    exec docker run --rm "${DOCKER_ARGS[@]}" aider-benchmark \
        bash -c "pip install -q -e . > /dev/null 2>&1 && ${STATS_CMD}"
fi

if docker container inspect "${CONTAINER}" > /dev/null 2>&1; then
    echo "Container ${CONTAINER} exists (a run is in progress?); docker stop ${CONTAINER} first" >&2
    exit 1
fi

# The settings file, with the model name and the reasoning controls of this run.
SETTINGS="${AIDER_DIR}/aider-model-settings.yml"
sed "s|^- name: .*|- name: ${MODEL_NAME}|" "${HERE}/aider-model-settings.yml" > "${SETTINGS}"
KWARGS=()
case "${REASONING_EFFORT:-}" in
    "") ;; xhigh|medium|low) KWARGS+=("reasoning_effort: ${REASONING_EFFORT}") ;;
    *) echo "REASONING_EFFORT must be xhigh, medium or low" >&2; exit 1 ;;
esac
for KNOB in ENABLE_THINKING PRESERVE_THINKING; do
    case "${!KNOB:-}" in
        "") ;; true|false) KWARGS+=("${KNOB,,}: ${!KNOB}") ;;
        *) echo "${KNOB} must be true or false" >&2; exit 1 ;;
    esac
done
if (( ${#KWARGS[@]} )); then                # appended under extra_body, the file's last block
    echo "      chat_template_kwargs:" >> "${SETTINGS}"
    printf '        %s\n' "${KWARGS[@]}" >> "${SETTINGS}"
fi
mkdir -p "$(dirname "${LOG}")"

BENCH_ARGS=(--model "${MODEL_NAME}" --read-model-settings aider-model-settings.yml
            --edit-format diff --threads "${THREADS:-8}" --tries "${TRIES:-2}"
            --exercises-dir polyglot-benchmark)
[[ -z "${NUM_TESTS:-}" ]] || BENCH_ARGS+=(--num-tests "${NUM_TESTS}")
[[ "${CONT:-0}" != 1 ]] || BENCH_ARGS+=(--cont)

nohup docker run --rm --name "${CONTAINER}" --memory=12g --memory-swap=12g \
    --add-host=host.docker.internal:host-gateway "${DOCKER_ARGS[@]}" \
    -e OPENAI_API_KEY=none -e OPENAI_API_BASE="${API_BASE:-http://host.docker.internal:8000/v1}" \
    aider-benchmark \
    bash -c "pip install -q -e .[dev] && ./benchmark/benchmark.py ${RUN_NAME} ${BENCH_ARGS[*]}" \
    >> "${LOG}" 2>&1 &

echo "Aider Polyglot started (container ${CONTAINER}); log: ${LOG}"
echo "Results so far: STATS=1 AIDER_DIR=${AIDER_DIR} RUN_NAME=${RUN_NAME} $0"
