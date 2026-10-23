#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Data-parallel Qwen3.8-Flash-Next on trn2.48xlarge: N independent TP=8 replicas (serve.sh, one
# per 8 logical NeuronCores, ports BASE_PORT+1..BASE_PORT+N) behind dp_proxy.py on BASE_PORT.
# Everything runs in the background (setsid), with one log per replica in LOG_DIR.
#
# Usage:
#   MODEL=<path>/Qwen3.8-Flash-Next REPLICAS=8 examples/vllm_neuron/models/qwen4_exp/run_replicas.sh
#   FIRST_CORE=32 REPLICAS=2 ...        # use cores 32-47 only
#   STOP=1 examples/vllm_neuron/models/qwen4_exp/run_replicas.sh   # stop all replicas + proxy
#
# serve.sh knobs (MAX_MODEL_LEN, PREFILL_BUCKETS, ...) are passed through. Replicas start
# STAGGER_S seconds apart (default 60) so they do not all read the checkpoint at once; the first
# one populates the compile cache, which later replicas reuse.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-${PWD}/qwen4_exp_replicas}
BASE_PORT=${BASE_PORT:-8000}
mkdir -p "${LOG_DIR}"

if [[ "${STOP:-0}" == 1 ]]; then
  for f in "${LOG_DIR}"/*.pid; do
    [[ -e "$f" ]] || continue
    kill -- -"$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
  echo "stopped"
  exit 0
fi

MODEL=${MODEL:?set MODEL to the checkpoint directory}
REPLICAS=${REPLICAS:-8}
FIRST_CORE=${FIRST_CORE:-0}
BACKENDS=()
for ((i = 0; i < REPLICAS; i++)); do
  c0=$((FIRST_CORE + 8 * i)); port=$((BASE_PORT + 1 + i))
  BACKENDS+=("http://127.0.0.1:${port}")
  MODEL="${MODEL}" CORES="${c0}-$((c0 + 7))" PORT="${port}" \
    setsid "${HERE}/serve.sh" > "${LOG_DIR}/replica${i}.log" 2>&1 &
  echo $! > "${LOG_DIR}/replica${i}.pid"
  echo "replica ${i}: cores ${c0}-$((c0 + 7)), port ${port}, log ${LOG_DIR}/replica${i}.log"
  [[ $i -eq $((REPLICAS - 1)) ]] || sleep "${STAGGER_S:-60}"
done
IFS=,; setsid python "${HERE}/dp_proxy.py" --port "${BASE_PORT}" --backends "${BACKENDS[*]}" \
  > "${LOG_DIR}/proxy.log" 2>&1 &
echo $! > "${LOG_DIR}/proxy.pid"
echo "proxy on port ${BASE_PORT} -> ${BACKENDS[*]}"
echo "ready when every replica answers: curl -s localhost:${BASE_PORT}/proxy/stats"
