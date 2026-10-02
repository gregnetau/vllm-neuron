#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# OpenAI-compatible server for Qwen3.5 / Qwen3.8 dense (27B), TP=4 on trn2.3xlarge.
#
# Usage:
#   MODEL=<path-to-checkpoint>/Qwen3.8-27B examples/vllm_neuron/models/qwen3_5/serve.sh
#   # Static FP8 MLPs:
#   MODEL=... FP8_SCALES=<path>/fp8_act_amax.json examples/vllm_neuron/models/qwen3_5/serve.sh
#   # neuronx-cc optimization level (default 1; 3 speeds up prefill, longer compile):
#   MODEL=... OPTLEVEL=3 examples/vllm_neuron/models/qwen3_5/serve.sh
#   # FP8 KV cache (unit KV scales):
#   MODEL=... KV_CACHE_DTYPE=fp8 examples/vllm_neuron/models/qwen3_5/serve.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL to the checkpoint directory}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
PREFILL_BUCKET=${PREFILL_BUCKET:-1024}

export NEURON_SKIP_EFA_AFFINITY=1                   # trn2.3xlarge has no EFA
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15  # caps the KV/state pool (24 GB HBM per core)

NEURON_CONFIG="\"num_batched_tokens_buckets\": [${PREFILL_BUCKET}], \"num_seqs_buckets\": [1]"
if [[ -n "${HLO2TENSORIZER_OPTIONS+set}" ]]; then  # "" = whole-graph compilation (no modular flow)
  NEURON_CONFIG="${NEURON_CONFIG}, \"hlo2tensorizer_options\": \"${HLO2TENSORIZER_OPTIONS}\""
fi
if [[ -n "${FP8_SCALES:-}" ]]; then
  NEURON_CONFIG="${NEURON_CONFIG}, \"quantization\": \"fp8\", \"fp8_activation_scales_path\": \"${FP8_SCALES}\""
fi

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "${MODEL}")}" \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${PREFILL_BUCKET}" \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --optimization-level "${OPTLEVEL:-1}" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --additional-config "{\"neuron_config\": {${NEURON_CONFIG}}}"
